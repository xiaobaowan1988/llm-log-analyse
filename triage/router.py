"""
router.py
=========
告警路由器：根据 TriageResult 的严重程度，决定通知目标和方式。

内置路由目标（均为可插拔的 Handler）：
  - ConsoleHandler     — 打印到终端（默认，开发/调试用）
  - FileHandler        — 写入本地文件（审计日志）
  - WebhookHandler     — 发送 HTTP POST 到任意 Webhook（钉钉/飞书/Slack/PagerDuty）
  - ElasticsearchHandler — 将 P3 日志写入 ES 并打 ai_ignored 标签

路由规则（可通过 config/triage_config.yaml 覆盖）：
  P0 → ConsoleHandler (RED) + WebhookHandler (PagerDuty / 电话告警)
  P1 → ConsoleHandler + WebhookHandler (Slack/钉钉 告警频道)
  P2 → ConsoleHandler + FileHandler
  P3 → FileHandler (ai_ignored 标签) + ElasticsearchHandler
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .models import Severity, TriageResult


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class AlertHandler(ABC):
    """所有告警 Handler 的统一接口"""

    @abstractmethod
    def send(self, result: TriageResult) -> bool:
        """
        发送告警。

        Returns
        -------
        bool
            True = 发送成功；False = 失败（不抛异常，避免影响主流程）
        """


# ---------------------------------------------------------------------------
# ConsoleHandler：彩色终端输出
# ---------------------------------------------------------------------------

# ANSI 颜色码
_COLORS = {
    Severity.P0_CRITICAL: "\033[1;31m",   # 粗体红
    Severity.P1_HIGH:     "\033[1;33m",   # 粗体黄
    Severity.P2_WARNING:  "\033[1;34m",   # 粗体蓝
    Severity.P3_IGNORE:   "\033[0;32m",   # 绿
    Severity.UNKNOWN:     "\033[0;37m",   # 灰
}
_RESET = "\033[0m"


class ConsoleHandler(AlertHandler):
    """将告警打印到终端，P0 使用红色高亮"""

    def __init__(self, show_raw_log: bool = False, use_color: bool = True):
        self.show_raw_log = show_raw_log
        self.use_color = use_color and os.environ.get("NO_COLOR") is None

    def send(self, result: TriageResult) -> bool:
        color  = _COLORS.get(result.severity, "") if self.use_color else ""
        reset  = _RESET if self.use_color else ""
        width  = 70

        print(f"\n{color}{'─' * width}")
        print(f"  {result.severity.emoji}  {result.severity.value}  |  {result.service_name or '未知服务'}")
        print(f"{'─' * width}{reset}")
        print(f"  {color}原因{reset}: {result.reason}")
        print(f"  {color}操作{reset}: {result.action}")
        print(f"  来源: {result.source.value}  |  计数: {result.count}  |  "
              f"耗时: {result.elapsed_ms:.0f}ms")

        if self.show_raw_log:
            log_preview = result.raw_log[:200].replace("\n", " ")
            print(f"  日志: {log_preview}...")

        if result.severity == Severity.P0_CRITICAL:
            print(f"\n{color}  ⚡ 需立即处理！请查看服务状态和监控面板{reset}")

        return True


# ---------------------------------------------------------------------------
# FileHandler：写入本地文件
# ---------------------------------------------------------------------------

class FileHandler(AlertHandler):
    """
    将分级结果以 JSONL 格式写入文件，用于：
      - 审计日志（所有级别）
      - P3 的 ai_ignored 归档（阻断告警，但保留记录）
    """

    def __init__(self, file_path: str, append: bool = True):
        self.file_path = file_path
        self.mode = "a" if append else "w"
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    def send(self, result: TriageResult) -> bool:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **result.to_dict(),
            "raw_log_preview": result.raw_log[:300],
        }
        try:
            with open(self.file_path, self.mode, encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as e:
            print(f"[FileHandler] 写入失败: {e}")
            return False


# ---------------------------------------------------------------------------
# WebhookHandler：HTTP POST 通知
# ---------------------------------------------------------------------------

@dataclass
class WebhookConfig:
    url: str
    # 消息模板，支持 {severity}, {reason}, {action}, {service}, {count} 占位符
    message_template: str = (
        "🚨 [{severity}] {service}\n"
        "原因: {reason}\n"
        "操作: {action}\n"
        "出现次数: {count} 次"
    )
    method: str = "POST"
    headers: dict = None
    timeout: int = 10

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


class WebhookHandler(AlertHandler):
    """
    发送 HTTP POST Webhook 通知。

    兼容格式：
      - 钉钉机器人：设置 body_template 为钉钉格式
      - 飞书机器人：同上
      - Slack Incoming Webhook：同上
      - PagerDuty Events API v2：同上
      - 通用 JSON Webhook（默认）

    示例（钉钉机器人）：
        handler = WebhookHandler(
            config=WebhookConfig(
                url="https://oapi.dingtalk.com/robot/send?access_token=xxx",
                headers={"Content-Type": "application/json"},
            ),
            body_builder=dingtalk_body_builder,
        )
    """

    def __init__(
        self,
        config: WebhookConfig,
        body_builder=None,
    ):
        self.config = config
        # body_builder: (TriageResult) -> dict，自定义消息体格式
        self.body_builder = body_builder or self._default_body

    def send(self, result: TriageResult) -> bool:
        body = self.body_builder(result)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            self.config.url,
            data=data,
            method=self.config.method,
            headers=self.config.headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                status = resp.status
            if status < 300:
                return True
            print(f"[WebhookHandler] HTTP {status}")
            return False
        except Exception as e:
            print(f"[WebhookHandler] 发送失败: {e}")
            return False

    def _default_body(self, result: TriageResult) -> dict:
        """默认消息体（通用 JSON 格式）"""
        text = self.config.message_template.format(
            severity=result.severity.value,
            service=result.service_name or "unknown",
            reason=result.reason,
            action=result.action,
            count=result.count,
        )
        return {"text": text, "severity": result.severity.value}


def make_dingtalk_body(result: TriageResult) -> dict:
    """钉钉机器人 Markdown 消息格式"""
    emoji = result.severity.emoji
    text = (
        f"## {emoji} {result.severity.value} — {result.service_name}\n\n"
        f"**原因**: {result.reason}\n\n"
        f"**操作**: {result.action}\n\n"
        f"> 出现次数: {result.count}  来源: {result.source.value}"
    )
    return {
        "msgtype": "markdown",
        "markdown": {"title": f"{result.severity.value} 告警", "text": text},
    }


def make_slack_body(result: TriageResult) -> dict:
    """Slack Incoming Webhook Block Kit 格式"""
    color_map = {
        Severity.P0_CRITICAL: "#FF0000",
        Severity.P1_HIGH:     "#FF8800",
        Severity.P2_WARNING:  "#FFDD00",
        Severity.P3_IGNORE:   "#00AA00",
    }
    return {
        "attachments": [{
            "color": color_map.get(result.severity, "#888888"),
            "title": f"{result.severity.emoji} {result.severity.value} — {result.service_name}",
            "text": f"*原因*: {result.reason}\n*操作*: {result.action}",
            "footer": f"次数: {result.count} | 来源: {result.source.value}",
        }]
    }


# ---------------------------------------------------------------------------
# ElasticsearchHandler：P3 日志归档
# ---------------------------------------------------------------------------

class ElasticsearchHandler(AlertHandler):
    """
    将 P3 日志写入 Elasticsearch，打上 ai_ignored=true 标签。
    便于后续统计分析噪音日志的分布，持续优化规则和模型。
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index: str = "ai-triage-ignored",
        api_key: str = "",
    ):
        self.es_url = es_url.rstrip("/")
        self.index = index
        self.api_key = api_key or os.getenv("ES_API_KEY", "")

    def send(self, result: TriageResult) -> bool:
        doc = {
            "@timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ai_ignored": True,
            "ai_severity": result.severity.value,
            "ai_reason": result.reason,
            "service": result.service_name,
            "fingerprint": result.fingerprint,
            "count": result.count,
            "raw_log": result.raw_log[:500],
        }
        url = f"{self.es_url}/{self.index}/_doc"
        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        data = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 300
        except Exception as e:
            print(f"[ESHandler] 写入失败: {e}")
            return False


# ---------------------------------------------------------------------------
# 路由器
# ---------------------------------------------------------------------------

class AlertRouter:
    """
    根据 TriageResult 的严重程度，分发到对应的 Handler 列表。

    示例：
        router = AlertRouter()
        router.add_handler(Severity.P0_CRITICAL, ConsoleHandler())
        router.add_handler(Severity.P0_CRITICAL, WebhookHandler(pagerduty_config))
        router.add_handler(Severity.P1_HIGH,     WebhookHandler(slack_config))
        router.add_handler(Severity.P3_IGNORE,   FileHandler("logs/ai_ignored.jsonl"))

        router.route(triage_result)
    """

    def __init__(self):
        # severity -> list of handlers
        self._handlers: dict[Severity, list[AlertHandler]] = {s: [] for s in Severity}
        # 默认添加 ConsoleHandler 用于调试
        console = ConsoleHandler()
        for sev in (Severity.P0_CRITICAL, Severity.P1_HIGH, Severity.P2_WARNING):
            self._handlers[sev].append(console)

    def add_handler(self, severity: Severity, handler: AlertHandler) -> "AlertRouter":
        """添加 Handler（支持链式调用）"""
        self._handlers[severity].append(handler)
        return self

    def route(self, result: TriageResult) -> list[bool]:
        """
        分发 TriageResult 到对应 Handler。

        Returns
        -------
        list[bool]
            每个 Handler 的发送结果
        """
        handlers = self._handlers.get(result.severity, [])
        return [h.send(result) for h in handlers]

    @classmethod
    def default(
        cls,
        audit_log_path: str = "output/triage_audit.jsonl",
    ) -> "AlertRouter":
        """
        创建包含默认配置的路由器：
          - 所有级别 → ConsoleHandler
          - 所有级别 → FileHandler（审计日志）
          - P3 → 额外的 FileHandler（ai_ignored 专属文件）
        """
        router = cls()
        audit = FileHandler(audit_log_path)
        for sev in Severity:
            if sev != Severity.UNKNOWN:
                router.add_handler(sev, audit)

        ignored_log = audit_log_path.replace(".jsonl", "_p3_ignored.jsonl")
        router.add_handler(Severity.P3_IGNORE, FileHandler(ignored_log))

        return router
