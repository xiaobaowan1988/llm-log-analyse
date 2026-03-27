"""
models.py
=========
告警分级系统的核心数据结构定义。

严重程度等级定义（与 PagerDuty / OpsGenie 业界标准对齐）：
  P0_CRITICAL  — 核心业务中断，立即唤醒 on-call，分钟级响应
  P1_HIGH      — 部分业务受损，1 小时内响应
  P2_WARNING   — 边缘功能异常，工作时间内处理
  P3_IGNORE    — 可自愈或无影响，仅记录，阻断告警通知
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


# ---------------------------------------------------------------------------
# 严重程度枚举
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH     = "P1_HIGH"
    P2_WARNING  = "P2_WARNING"
    P3_IGNORE   = "P3_IGNORE"
    UNKNOWN     = "UNKNOWN"       # LLM 输出异常时的兜底值

    @property
    def level(self) -> int:
        """数字级别，越小越严重（方便比较）"""
        return {"P0_CRITICAL": 0, "P1_HIGH": 1,
                "P2_WARNING": 2, "P3_IGNORE": 3, "UNKNOWN": 9}[self.value]

    @property
    def emoji(self) -> str:
        return {"P0_CRITICAL": "🔴", "P1_HIGH": "🟠",
                "P2_WARNING": "🟡", "P3_IGNORE": "🟢", "UNKNOWN": "⚪"}[self.value]

    @property
    def label_cn(self) -> str:
        return {"P0_CRITICAL": "紧急", "P1_HIGH": "高危",
                "P2_WARNING": "警告", "P3_IGNORE": "忽略", "UNKNOWN": "未知"}[self.value]

    def __lt__(self, other: "Severity") -> bool:
        return self.level < other.level


# ---------------------------------------------------------------------------
# 分类来源
# ---------------------------------------------------------------------------

class ClassifySource(str, Enum):
    RULE    = "rule"    # 规则引擎快速匹配
    LLM     = "llm"     # 大模型推理
    DEFAULT = "default" # 兜底默认值（LLM 调用失败）


# ---------------------------------------------------------------------------
# 单条日志的分级结果
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """单条（或聚合后）日志的分级结果"""

    # 输入
    raw_log: str                  # 原始日志文本
    service_name: str = ""        # 服务名
    fingerprint: str = ""         # 去重指纹（相同错误的标识）

    # 分级输出
    severity: Severity = Severity.UNKNOWN
    reason: str = ""              # 判断理由（用于解释给 on-call 工程师）
    action: str = ""              # 建议动作
    source: ClassifySource = ClassifySource.DEFAULT

    # 聚合信息（来自 aggregator）
    count: int = 1                # 时间窗口内相同错误的出现次数
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    window_seconds: int = 60      # 统计窗口（秒）

    # 模型信息
    model: str = ""
    llm_raw_output: str = ""      # 模型原始输出，方便调试
    elapsed_ms: float = 0.0

    @property
    def frequency(self) -> float:
        """每分钟出现频率"""
        duration = max(self.last_seen - self.first_seen, 1)
        return self.count / duration * 60

    def to_dict(self) -> dict:
        return {
            "severity":    self.severity.value,
            "reason":      self.reason,
            "action":      self.action,
            "source":      self.source.value,
            "service":     self.service_name,
            "count":       self.count,
            "frequency":   round(self.frequency, 1),
            "fingerprint": self.fingerprint,
        }

    def summary_line(self) -> str:
        freq_str = f"  频率: {self.frequency:.0f}次/分" if self.count > 1 else ""
        src_str = f"[{self.source.value}]"
        return (
            f"{self.severity.emoji} {self.severity.value:<12} {src_str:<8}"
            f"  {self.service_name or '?':<20}"
            f"  {self.reason[:60]}{freq_str}"
        )


# ---------------------------------------------------------------------------
# 聚合后的日志组
# ---------------------------------------------------------------------------

@dataclass
class LogGroup:
    """
    聚合去重后的日志组：相同指纹的多条日志合并为一组。
    传给 LLM 时附带频率信息，辅助判断「爆炸半径」。
    """
    fingerprint: str
    representative_log: str       # 取一条代表性日志送给 LLM
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    service_name: str = ""
    error_type: str = ""

    @property
    def frequency_per_min(self) -> float:
        duration = max(self.last_seen - self.first_seen, 1)
        return self.count / duration * 60

    def context_summary(self) -> str:
        """生成传给 LLM 的上下文摘要（含频率信息）"""
        lines = [self.representative_log]
        if self.count > 1:
            lines.append(
                f"[聚合信息] 该错误在过去 {int(self.last_seen - self.first_seen)}s 内"
                f"出现 {self.count} 次（{self.frequency_per_min:.0f}次/分钟）"
            )
        return "\n".join(lines)
