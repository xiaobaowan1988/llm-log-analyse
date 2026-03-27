"""
triage/pipeline.py
==================
告警分级主流程：串联规则引擎 → 聚合去重 → LLM 分类 → 告警路由。

                ┌──────────────────────────────────────────┐
  原始日志 ───▶ │              TriagePipeline              │
                │                                          │
                │  LogAggregator   ──▶  RuleEngine         │
                │  (去重 + 聚合)         (快速过滤)         │
                │       │                    │             │
                │       │ 未命中规则          │ 命中规则     │
                │       ▼                    ▼             │
                │  LLMClassifier         直接输出          │
                │  (Few-Shot)                              │
                │       │                                  │
                │       ▼                                  │
                │   AlertRouter (P0→唤醒 / P3→静默)        │
                └──────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .aggregator import LogAggregator, LogGroup
from .classifier import LLMClassifier
from .models import Severity, TriageResult, ClassifySource
from .router import AlertRouter
from .rules import RuleEngine


# ---------------------------------------------------------------------------
# 批量分析统计
# ---------------------------------------------------------------------------

@dataclass
class TriageStats:
    total: int = 0
    by_severity: dict = field(default_factory=lambda: {s.value: 0 for s in Severity})
    by_source: dict = field(default_factory=lambda: {"rule": 0, "llm": 0, "default": 0})
    llm_calls: int = 0
    rule_hits: int = 0
    total_elapsed_ms: float = 0.0

    @property
    def rule_hit_rate(self) -> float:
        return self.rule_hits / self.total if self.total else 0.0

    @property
    def avg_llm_ms(self) -> float:
        return self.total_elapsed_ms / self.llm_calls if self.llm_calls else 0.0

    def update(self, result: TriageResult) -> None:
        self.total += 1
        self.by_severity[result.severity.value] = \
            self.by_severity.get(result.severity.value, 0) + 1
        self.by_source[result.source.value] = \
            self.by_source.get(result.source.value, 0) + 1
        if result.source == ClassifySource.RULE:
            self.rule_hits += 1
        elif result.source == ClassifySource.LLM:
            self.llm_calls += 1
            self.total_elapsed_ms += result.elapsed_ms

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("  告警分级统计摘要")
        print("=" * 55)
        print(f"  总计日志组: {self.total}")
        print()
        for sev in Severity:
            if sev == Severity.UNKNOWN:
                continue
            cnt = self.by_severity.get(sev.value, 0)
            bar = "█" * min(cnt, 30)
            print(f"  {sev.emoji} {sev.value:<14} {cnt:>4}  {bar}")
        print()
        print(f"  规则命中率 : {self.rule_hit_rate:.0%}  ({self.rule_hits} 条规则直接处理)")
        print(f"  LLM 调用数 : {self.llm_calls}")
        if self.llm_calls:
            print(f"  LLM 平均耗时: {self.avg_llm_ms:.0f} ms/条")
        print("=" * 55)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

class TriagePipeline:
    """
    告警分级流水线。

    Parameters
    ----------
    classifier : LLMClassifier
        LLM 分类器（必须）
    router : AlertRouter | None
        告警路由器；None 则使用默认路由（ConsoleHandler + FileHandler）
    use_rules : bool
        是否启用规则引擎预过滤（默认 True）
    use_aggregation : bool
        是否启用日志聚合去重（默认 True）
    window_seconds : int
        聚合时间窗口（秒）
    flush_threshold : int
        触发立即处理的频率阈值（同一指纹在窗口内出现 N 次时立即处理）
    """

    def __init__(
        self,
        classifier: LLMClassifier,
        router: Optional[AlertRouter] = None,
        use_rules: bool = True,
        use_aggregation: bool = True,
        window_seconds: int = 60,
        flush_threshold: int = 50,
    ):
        self.classifier = classifier
        self.router = router or AlertRouter.default()
        self.rule_engine = RuleEngine() if use_rules else None
        self.aggregator = LogAggregator(
            window_seconds=window_seconds,
            flush_threshold=flush_threshold,
        ) if use_aggregation else None
        self.stats = TriageStats()

    # ------------------------------------------------------------------
    # 流式处理（逐条输入）
    # ------------------------------------------------------------------

    def process(
        self,
        raw_log: str,
        service_name: str = "",
        ts: Optional[float] = None,
    ) -> Optional[TriageResult]:
        """
        处理单条日志。

        Parameters
        ----------
        raw_log : str
            原始日志文本
        service_name : str
            服务名（可选，用于路由和展示）
        ts : float | None
            日志时间戳（None = 当前时间）

        Returns
        -------
        TriageResult | None
            - 返回 TriageResult：本条日志已完成分级
            - 返回 None：已聚合入窗口，等待 flush
        """
        # ── Step 1: 聚合去重 ────────────────────────────────────────────
        if self.aggregator:
            group = self.aggregator.add(raw_log, service_name=service_name, ts=ts)
            if group is None:
                return None  # 已聚合，等待窗口
        else:
            from .aggregator import LogGroup, generate_fingerprint
            now = ts or time.time()
            group = LogGroup(
                fingerprint=generate_fingerprint(raw_log),
                representative_log=raw_log,
                service_name=service_name,
                first_seen=now,
                last_seen=now,
            )

        return self._classify_and_route(group)

    def _classify_and_route(self, group: LogGroup) -> TriageResult:
        # ── Step 2: 规则引擎快速匹配 ────────────────────────────────────
        if self.rule_engine:
            result = self.rule_engine.match(
                group.representative_log,
                service_name=group.service_name,
            )
            if result:
                result.fingerprint = group.fingerprint
                result.count = group.count
                result.first_seen = group.first_seen
                result.last_seen = group.last_seen
                self.stats.update(result)
                self.router.route(result)
                return result

        # ── Step 3: LLM 分类 ────────────────────────────────────────────
        result = self.classifier.classify(group)
        self.stats.update(result)
        self.router.route(result)
        return result

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------

    def process_batch(
        self,
        logs: list[str],
        service_name: str = "",
        flush_remaining: bool = True,
    ) -> list[TriageResult]:
        """
        批量处理日志列表。

        Parameters
        ----------
        logs : list[str]
            日志文本列表
        service_name : str
            统一的服务名
        flush_remaining : bool
            处理完毕后是否 flush 聚合窗口中的剩余日志

        Returns
        -------
        list[TriageResult]
        """
        results = []
        for log in logs:
            r = self.process(log, service_name=service_name)
            if r:
                results.append(r)

        # Flush 窗口中尚未输出的聚合组
        if flush_remaining and self.aggregator:
            for group in self.aggregator.flush_all():
                r = self._classify_and_route(group)
                results.append(r)

        return results

    def process_file(
        self,
        log_file_path: str,
        service_name: str = "",
    ) -> list[TriageResult]:
        """
        从日志文件逐行读取并处理。

        每行日志独立处理，相邻的多行 Stack Trace 通过聚合器的指纹机制自动关联。
        """
        results = []
        with open(log_file_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                r = self.process(line, service_name=service_name)
                if r:
                    results.append(r)

        # Flush 剩余
        if self.aggregator:
            for group in self.aggregator.flush_all():
                r = self._classify_and_route(group)
                results.append(r)

        return results
