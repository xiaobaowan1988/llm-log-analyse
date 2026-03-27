"""
classifier.py
=============
基于 LLM 的日志严重程度分类器（Few-Shot Prompting）。

核心设计：
  1. Few-Shot 示例覆盖 P0~P3 各级别的典型场景（含边界案例）
  2. 强制 JSON 输出，避免自由文本不可解析
  3. 三维判断框架：错误类型 + 发生位置 + 爆炸半径
  4. 解析失败时的降级策略（默认 P1，宁可误报不漏报）
"""

from __future__ import annotations

import json
import re
import sys
import os
import time
from typing import Optional

# 复用 RAG 模块的 LLM 客户端
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.llm_client import LLMClient

from .models import Severity, ClassifySource, TriageResult, LogGroup


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一名资深的 SRE（站点可靠性工程师），专精于生产告警分级（Alert Triage）。

你的任务：分析给定的 ERROR 日志，判断其严重程度，输出标准化 JSON。

## 严重程度等级定义

| 级别 | 名称 | 响应时效 | 触发条件 |
|------|------|---------|---------|
| P0_CRITICAL | 紧急 | 立即唤醒 | 核心业务中断、进程崩溃（panic/OOM）、节点宕机、磁盘满、证书过期 |
| P1_HIGH | 高危 | 1小时内 | 部分业务受损、核心接口 5xx、数据库死锁、熔断器开启 |
| P2_WARNING | 警告 | 工作时间 | 边缘功能异常、性能下降但未中断、非核心依赖失败 |
| P3_IGNORE | 忽略 | 无需处理 | 客户端参数错误、记录不存在、健康检查噪音、偶发超时已重试成功、爬虫访问 |

## 三维判断框架

判断时必须从以下三个维度分析，不能只看 ERROR 关键字：

1. **错误类型**：底层崩溃（OOM/panic）vs 业务异常（记录不存在）？
2. **发生位置**：核心交易链路 vs 后台任务 vs 健康探针？
3. **爆炸半径**：单次偶发 vs 持续高频（1分钟 N 次）？频率高会升级严重程度。

## 输出格式

必须且只能输出合法 JSON，不要有任何额外文字：
{
  "severity": "P0_CRITICAL|P1_HIGH|P2_WARNING|P3_IGNORE",
  "reason": "简洁说明判断依据（1-2句话）",
  "action": "建议立即执行的操作（1-2句话）",
  "blast_radius": "single|partial|full",
  "confidence": "high|medium|low"
}"""


# ---------------------------------------------------------------------------
# Few-Shot 示例（覆盖典型边界场景）
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = [
    # ── P0 示例 ─────────────────────────────────────────────────────────────
    {
        "log": "2024-01-15 03:22:18 ERROR [payment-service] goroutine 47 panic: runtime error: nil pointer dereference\n    at github.com/myco/payment/processor.go:87\n[聚合信息] 该错误在过去 5s 内出现 1 次（12次/分钟）",
        "output": {
            "severity": "P0_CRITICAL",
            "reason": "支付服务核心 goroutine 发生 nil pointer panic，处理协程已崩溃，支付请求无法完成。",
            "action": "立即检查 payment-service Pod 状态，查看是否需要重启；分析 processor.go:87 的空指针来源。",
            "blast_radius": "full",
            "confidence": "high",
        },
    },
    {
        "log": "2024-01-16 08:30:00 CRITICAL [db-service] HikariPool-1 - Connection is not available, request timed out after 30001ms.\npool.size=20, active=20, idle=0, waiting=156\n[聚合信息] 该错误在过去 60s 内出现 892 次（892次/分钟）",
        "output": {
            "severity": "P0_CRITICAL",
            "reason": "数据库连接池完全耗尽（20/20 占用，156 个请求等待），且在 1 分钟内爆发 892 次——所有需要数据库的请求均会失败。",
            "action": "立即执行 SHOW PROCESSLIST 找到慢查询并 KILL；临时扩大连接池至 50；重启应用释放泄漏连接。",
            "blast_radius": "full",
            "confidence": "high",
        },
    },
    # ── P1 示例 ─────────────────────────────────────────────────────────────
    {
        "log": "2024-01-16 14:05:33 ERROR [api-gateway] upstream order-service (10.0.1.23:8080) timeout after 5000ms, retry 3/3 failed\n[聚合信息] 该错误在过去 120s 内出现 47 次（23次/分钟）",
        "output": {
            "severity": "P1_HIGH",
            "reason": "API 网关对 order-service 的调用持续超时，且重试 3 次均失败，订单查询业务受损，但其他服务不受影响。",
            "action": "立即检查 10.0.1.23:8080 进程状态；在负载均衡器摘除该节点；查看 order-service 最近是否有发布。",
            "blast_radius": "partial",
            "confidence": "high",
        },
    },
    # ── P2 示例 ─────────────────────────────────────────────────────────────
    {
        "log": "2024-01-17 22:10:15 ERROR [report-service] Failed to generate monthly report for tenant_id=9821: template rendering timeout after 30s",
        "output": {
            "severity": "P2_WARNING",
            "reason": "月报生成超时，属于后台非核心任务，不影响用户实时交互流程，仅影响该租户的报表功能。",
            "action": "检查报表数据量是否异常增大；调大渲染超时阈值；分析是否需要异步化处理。",
            "blast_radius": "single",
            "confidence": "high",
        },
    },
    {
        "log": "2024-01-17 09:00:03 ERROR [notification-service] Failed to send marketing email to user@example.com: SMTP connection refused by relay.smtp.example.com:587",
        "output": {
            "severity": "P2_WARNING",
            "reason": "营销邮件发送失败，属于非核心功能，不影响用户登录、交易等核心链路。",
            "action": "检查 SMTP 服务器连接状态；确认邮件服务商配额是否耗尽；核心事务邮件是否走同一 relay。",
            "blast_radius": "single",
            "confidence": "medium",
        },
    },
    # ── P3 示例（边界：看似严重，实为噪音）─────────────────────────────────
    {
        "log": "2024-01-15 14:22:31 ERROR [user-service] Failed to fetch user profile, err: record not found, user_id=deleted_user_19283",
        "output": {
            "severity": "P3_IGNORE",
            "reason": "查询已删除用户的 Profile，系统正常返回 not found，属于客户端请求了无效数据，服务行为正确。",
            "action": "无需处理；如频繁出现，可检查客户端是否有过期缓存。",
            "blast_radius": "single",
            "confidence": "high",
        },
    },
    {
        "log": "2024-01-16 03:15:42 ERROR [file-service] S3 GetObject timeout (attempt 1/3): connection reset by peer. Retrying in 2s...\n2024-01-16 03:15:44 INFO  [file-service] S3 GetObject success on attempt 2/3, latency=1847ms",
        "output": {
            "severity": "P3_IGNORE",
            "reason": "S3 连接偶发超时，日志已明确显示重试后成功，对用户请求无影响。",
            "action": "无需处理；若 S3 超时频率持续升高（>10次/分钟），则升级为 P2 调查网络问题。",
            "blast_radius": "single",
            "confidence": "high",
        },
    },
    # ── 边界案例：频率放大严重程度 ─────────────────────────────────────────
    {
        "log": "2024-01-18 11:00:01 ERROR [inventory-service] StockCheckException: item 'SKU-98821' stock insufficient\n[聚合信息] 该错误在过去 60s 内出现 3421 次（3421次/分钟）",
        "output": {
            "severity": "P1_HIGH",
            "reason": "库存不足错误本身是业务异常（正常），但 3421次/分钟的极高频率表明大规模秒杀或超卖场景正在发生，可能触发级联故障。",
            "action": "立即检查是否有超卖异常；确认限流策略是否生效；查看数据库锁争用情况。",
            "blast_radius": "partial",
            "confidence": "medium",
        },
    },
]


def _build_few_shot_block() -> str:
    """将 Few-Shot 示例格式化为 Prompt 文本块"""
    parts = []
    for i, ex in enumerate(_FEW_SHOT_EXAMPLES, 1):
        output_str = json.dumps(ex["output"], ensure_ascii=False, indent=2)
        parts.append(
            f"【示例 {i}】\n"
            f"输入日志：\n```\n{ex['log']}\n```\n"
            f"输出：\n```json\n{output_str}\n```"
        )
    return "\n\n".join(parts)


_FEW_SHOT_BLOCK = _build_few_shot_block()  # 模块加载时预编译，避免重复计算


# ---------------------------------------------------------------------------
# JSON 输出解析
# ---------------------------------------------------------------------------

_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_RE_BARE_JSON  = re.compile(r"\{[^{}]*\"severity\"[^{}]*\}", re.DOTALL)

_VALID_SEVERITIES = {s.value for s in Severity if s != Severity.UNKNOWN}


def _parse_llm_output(raw: str) -> Optional[dict]:
    """从 LLM 的原始输出中提取并验证 JSON 字段"""
    # 尝试 markdown 代码块
    m = _RE_JSON_BLOCK.search(raw)
    if not m:
        # 尝试裸 JSON
        m = _RE_BARE_JSON.search(raw)
    if not m:
        # 整个输出当作 JSON 直接解析
        candidate = raw.strip()
    else:
        candidate = m.group(1) if hasattr(m, "group") and m.lastindex else m.group(0)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    # 验证 severity 字段合法性
    if data.get("severity") not in _VALID_SEVERITIES:
        return None

    return data


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

class LLMClassifier:
    """
    基于 LLM 的日志严重程度分类器。

    Parameters
    ----------
    llm : LLMClient
        复用 rag.llm_client 中的客户端（Ollama / vLLM / OpenAI 兼容）
    fallback_severity : Severity
        LLM 调用或解析失败时的降级级别（默认 P1，宁可误报）
    max_log_chars : int
        传给 LLM 的日志文本最大字符数（防止超出上下文窗口）
    """

    def __init__(
        self,
        llm: LLMClient,
        fallback_severity: Severity = Severity.P1_HIGH,
        max_log_chars: int = 1500,
    ):
        self.llm = llm
        self.fallback_severity = fallback_severity
        self.max_log_chars = max_log_chars

    def classify(
        self,
        log_group: LogGroup,
    ) -> TriageResult:
        """
        对一个日志组进行 LLM 分类。

        Parameters
        ----------
        log_group : LogGroup
            来自 aggregator 的聚合日志组

        Returns
        -------
        TriageResult
        """
        t_start = time.time()

        # 构建日志文本（含聚合频率信息）
        log_text = log_group.context_summary()
        if len(log_text) > self.max_log_chars:
            log_text = log_text[:self.max_log_chars] + "\n...[日志截断]"

        user_prompt = self._build_user_prompt(log_text)

        try:
            response = self.llm.chat(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.05,   # 极低随机性，分类任务要确定性
                max_tokens=512,     # 分类输出不需要太长
            )
            raw_output = response.content
            parsed = _parse_llm_output(raw_output)
        except Exception as e:
            raw_output = f"[LLM 调用失败] {e}"
            parsed = None

        elapsed_ms = (time.time() - t_start) * 1000

        if parsed:
            severity = Severity(parsed["severity"])
            reason  = parsed.get("reason", "")
            action  = parsed.get("action", "")
            source  = ClassifySource.LLM
        else:
            # 降级：解析失败，保守地返回 P1
            severity = self.fallback_severity
            reason   = f"LLM 输出解析失败，降级为 {self.fallback_severity.value}（需人工确认）"
            action   = "请人工查看原始日志"
            source   = ClassifySource.DEFAULT

        return TriageResult(
            raw_log=log_group.representative_log,
            service_name=log_group.service_name,
            fingerprint=log_group.fingerprint,
            severity=severity,
            reason=reason,
            action=action,
            source=source,
            count=log_group.count,
            first_seen=log_group.first_seen,
            last_seen=log_group.last_seen,
            model=getattr(self.llm, "model", ""),
            llm_raw_output=raw_output,
            elapsed_ms=elapsed_ms,
        )

    def _build_user_prompt(self, log_text: str) -> str:
        return (
            f"{_FEW_SHOT_BLOCK}\n\n"
            "──────────────────────────────────────\n"
            "【待分析日志】\n"
            "请分析以下日志并输出 JSON（不要有任何额外文字）：\n"
            f"```\n{log_text}\n```"
        )
