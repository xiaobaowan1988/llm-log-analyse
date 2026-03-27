"""
aggregator.py
=============
日志聚合去重模块：模拟 Kafka + Flink 流处理层的聚合行为。

核心思路：
  同一时间窗口内，相同「指纹」的日志合并为一条，附带出现频率。
  避免大模型被 1000 条相同错误淹没，也让「爆炸半径」信息可量化。

指纹生成策略（按优先级）：
  1. 提取错误类型 + 报错位置（文件:行号）→ 最精准
  2. 提取错误类型 + 关键词       → 次精准
  3. 日志文本的模糊哈希           → 兜底
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .models import LogGroup


# ---------------------------------------------------------------------------
# 指纹生成
# ---------------------------------------------------------------------------

# 需要从日志中抹去的「时变」部分（时间戳、ID、IP、数字）
_NOISE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),  # 时间戳
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),   # IP:port
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # UUID
    re.compile(r"\b[0-9a-f]{24,64}\b"),                     # Hex ID / commit sha
    re.compile(r'"[^"]{0,8}\d{4,}[^"]{0,8}"'),             # 含数字的 JSON 值
    re.compile(r"\b\d{5,}\b"),                               # 纯数字 ID
    re.compile(r"batch_\w+|request_id[=:]\s*\S+"),          # 业务流水号
]

# 提取结构化错误特征的模式
_JAVA_ERROR  = re.compile(r"([\w.$]+(?:Exception|Error|Throwable))")
_PYTHON_ERROR = re.compile(r"(\w+(?:Error|Exception)):")
_GO_ERROR    = re.compile(r"(panic:|runtime error:[^,\n]+)")
_FILE_LINE   = re.compile(r"([\w/.-]+\.(?:java|py|go|js|ts))(?::(\d+))?")


def _normalize(text: str) -> str:
    """抹去时变噪音，生成稳定的文本用于指纹计算"""
    for pat in _NOISE_PATTERNS:
        text = pat.sub("<VAR>", text)
    # 压缩连续空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_fingerprint(log_text: str) -> str:
    """
    生成日志指纹（相同错误产生相同指纹）。

    算法：
      1. 尝试提取「错误类型 + 第一个报错文件:行号」
      2. 退回到「错误类型 + 错误消息前 80 字符」
      3. 最终退回到规范化文本的 MD5
    """
    # 尝试提取错误类型
    error_type = ""
    for pat in (_JAVA_ERROR, _PYTHON_ERROR, _GO_ERROR):
        m = pat.search(log_text)
        if m:
            error_type = m.group(1)
            break

    # 尝试提取文件:行号
    file_loc = ""
    m_file = _FILE_LINE.search(log_text)
    if m_file:
        fname = m_file.group(1).split("/")[-1]  # 只取文件名
        lineno = m_file.group(2) or ""
        file_loc = f"{fname}:{lineno}" if lineno else fname

    if error_type and file_loc:
        key = f"{error_type}@{file_loc}"
    elif error_type:
        normalized = _normalize(log_text)
        key = f"{error_type}:{normalized[:80]}"
    else:
        key = _normalize(log_text)[:120]

    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 内存滑动窗口聚合器
# ---------------------------------------------------------------------------

@dataclass
class _WindowEntry:
    """单个指纹在时间窗口内的聚合状态"""
    fingerprint: str
    first_log: str          # 第一条日志（作为代表性样本）
    service_name: str
    error_type: str
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_group(self) -> LogGroup:
        return LogGroup(
            fingerprint=self.fingerprint,
            representative_log=self.first_log,
            count=self.count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            service_name=self.service_name,
            error_type=self.error_type,
        )


class LogAggregator:
    """
    基于滑动时间窗口的内存日志聚合器。

    在真实生产环境中，这一层由 Kafka + Flink / Spark Streaming 承担。
    本模块提供一个单机版实现，用于 Demo 和小规模场景。

    参数
    ----
    window_seconds : int
        时间窗口大小（秒）。窗口内相同指纹的日志合并为一条。
    flush_threshold : int
        窗口内相同错误超过此次数，立即触发（不等窗口结束）。
        用于快速响应「错误风暴」场景。

    示例
    ----
        agg = LogAggregator(window_seconds=60, flush_threshold=50)

        for raw_log in log_stream:
            group = agg.add(raw_log, service_name="payment-service")
            if group:                   # 返回 None 表示已聚合，等待窗口
                triage_pipeline.analyse(group)
    """

    def __init__(
        self,
        window_seconds: int = 60,
        flush_threshold: int = 50,
    ):
        self.window_seconds = window_seconds
        self.flush_threshold = flush_threshold
        self._window: dict[str, _WindowEntry] = {}

    def add(
        self,
        log_text: str,
        service_name: str = "",
        ts: Optional[float] = None,
    ) -> Optional[LogGroup]:
        """
        接收一条日志，进行聚合。

        Returns
        -------
        LogGroup | None
            - 返回 LogGroup：该条日志是新指纹，或已触发 flush 阈值，应立即处理
            - 返回 None：已聚合到已有条目，等待窗口到期
        """
        now = ts or time.time()
        fp = generate_fingerprint(log_text)
        error_type = self._extract_error_type(log_text)

        if fp in self._window:
            entry = self._window[fp]
            # 检查窗口是否已过期
            if now - entry.first_seen > self.window_seconds:
                group = entry.to_group()
                del self._window[fp]
                # 重新开始新窗口
                self._window[fp] = _WindowEntry(
                    fingerprint=fp,
                    first_log=log_text,
                    service_name=service_name,
                    error_type=error_type,
                    first_seen=now,
                    last_seen=now,
                )
                return group
            else:
                # 同一窗口内，累加计数
                entry.count += 1
                entry.last_seen = now
                # 超过阈值立即 flush
                if entry.count >= self.flush_threshold:
                    group = entry.to_group()
                    del self._window[fp]
                    return group
                return None
        else:
            # 新指纹，直接返回（count=1，让 LLM 先分析）
            self._window[fp] = _WindowEntry(
                fingerprint=fp,
                first_log=log_text,
                service_name=service_name,
                error_type=error_type,
                first_seen=now,
                last_seen=now,
            )
            group = LogGroup(
                fingerprint=fp,
                representative_log=log_text,
                count=1,
                first_seen=now,
                last_seen=now,
                service_name=service_name,
                error_type=error_type,
            )
            return group

    def flush_all(self) -> list[LogGroup]:
        """手动 flush 所有待处理窗口（用于批量处理场景）"""
        groups = [e.to_group() for e in self._window.values()]
        self._window.clear()
        return groups

    def flush_expired(self, now: Optional[float] = None) -> list[LogGroup]:
        """flush 所有已超时的窗口条目"""
        now = now or time.time()
        expired = [
            fp for fp, entry in self._window.items()
            if now - entry.first_seen > self.window_seconds
        ]
        groups = []
        for fp in expired:
            groups.append(self._window.pop(fp).to_group())
        return groups

    @staticmethod
    def _extract_error_type(log_text: str) -> str:
        for pat in (_JAVA_ERROR, _PYTHON_ERROR, _GO_ERROR):
            m = pat.search(log_text)
            if m:
                return m.group(1)
        return ""

    @property
    def pending_count(self) -> int:
        """当前窗口内待处理的指纹数"""
        return len(self._window)
