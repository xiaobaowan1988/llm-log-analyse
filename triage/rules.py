"""
rules.py
========
规则引擎：在调用 LLM 之前，用轻量级正则规则对日志进行快速预筛选。

设计原则：
  - 只处理「高置信度」的确定性场景（99% 准确率以上才写规则）
  - P0 规则：宁可误报也不漏报（false negative 代价极高）
  - P3 规则：只过滤完全确定无影响的噪音日志
  - 其余情况一律交给 LLM，不要过度依赖规则

规则优先级：P0 > P1 > P3（P2 不写规则，全部交 LLM）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import Severity, ClassifySource, TriageResult


# ---------------------------------------------------------------------------
# 规则定义
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    name: str
    severity: Severity
    pattern: re.Pattern
    reason: str
    action: str
    # 反向模式：匹配此 pattern 则跳过本规则（用于排除误匹配）
    exclude_pattern: Optional[re.Pattern] = None


# ---------------------------------------------------------------------------
# P0 规则：进程崩溃 / 核心服务中断
# ---------------------------------------------------------------------------

P0_RULES: list[Rule] = [
    Rule(
        name="go_panic",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"panic:|goroutine\s+\d+\s+\[running\]|"
            r"runtime error:\s*(invalid memory|nil pointer|index out of range)",
            re.IGNORECASE,
        ),
        reason="Go 程序发生 panic，当前 goroutine 已崩溃，可能导致服务不可用",
        action="立即查看服务健康状态，检查是否需要重启；分析 stack trace 定位 nil pointer 位置",
    ),
    Rule(
        name="jvm_oom",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"java\.lang\.OutOfMemoryError|"
            r"java\.lang\.StackOverflowError",
            re.IGNORECASE,
        ),
        reason="JVM 内存耗尽（OOM），进程即将崩溃或已崩溃",
        action="立即检查堆内存使用率，考虑重启服务并分析内存泄漏；调整 -Xmx 参数",
    ),
    Rule(
        name="process_killed",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"Killed process|OOMKilled|oom-kill-event|"
            r"signal:\s*killed|exit status 137",
            re.IGNORECASE,
        ),
        reason="进程被系统 OOM Killer 杀死（内存超限），服务实例已宕机",
        action="立即检查 K8s Pod 状态，重新调度；分析内存增长趋势，扩大资源限制",
    ),
    Rule(
        name="db_connection_exhausted",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"connection pool exhausted|"
            r"too many connections|"
            r"max_connections reached|"
            r"HikariPool-\d+ - Connection is not available",
            re.IGNORECASE,
        ),
        exclude_pattern=re.compile(r"retry|retrying|backoff", re.IGNORECASE),
        reason="数据库连接池耗尽，所有新请求将失败，核心业务受阻",
        action="立即检查慢查询和连接泄漏；临时扩大连接池；必要时重启应用释放连接",
    ),
    Rule(
        name="k8s_node_not_ready",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"Node .+ condition changed.*NotReady|"
            r"node .+ is not ready|"
            r"Failed to connect to .+ NodeNotReady",
            re.IGNORECASE,
        ),
        reason="Kubernetes 节点进入 NotReady 状态，该节点上所有 Pod 将被驱逐",
        action="立即检查节点状态（kubectl describe node）；确认是否需要人工干预或等待自动恢复",
    ),
    Rule(
        name="disk_full",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"No space left on device|"
            r"disk quota exceeded|"
            r"ENOSPC",
            re.IGNORECASE,
        ),
        reason="磁盘空间耗尽，文件写入失败，服务无法正常运行",
        action="立即清理日志和临时文件；紧急扩容磁盘；检查是否有日志轮转配置",
    ),
    Rule(
        name="ssl_cert_expired",
        severity=Severity.P0_CRITICAL,
        pattern=re.compile(
            r"certificate has expired|"
            r"SSL certificate problem.*expired|"
            r"certificate verify failed.*expired",
            re.IGNORECASE,
        ),
        reason="TLS 证书已过期，HTTPS 连接全部失败，影响所有使用该服务的客户端",
        action="立即续签证书并热更新；紧急联系证书管理员",
    ),
]


# ---------------------------------------------------------------------------
# P1 规则：部分功能受损
# ---------------------------------------------------------------------------

P1_RULES: list[Rule] = [
    Rule(
        name="repeated_5xx",
        severity=Severity.P1_HIGH,
        pattern=re.compile(
            r"HTTP 5[0-9]{2}|status[= :]5[0-9]{2}|"
            r"upstream.*(500|502|503|504)",
            re.IGNORECASE,
        ),
        exclude_pattern=re.compile(r"health.?check|probe|ping", re.IGNORECASE),
        reason="核心接口返回 5xx 错误，部分用户请求失败",
        action="检查上游服务健康状态；查看是否有最近发布；分析错误率趋势",
    ),
    Rule(
        name="deadlock",
        severity=Severity.P1_HIGH,
        pattern=re.compile(
            r"Deadlock found|deadlock detected|"
            r"Lock wait timeout exceeded",
            re.IGNORECASE,
        ),
        reason="数据库死锁，受影响的事务失败，可能造成用户操作失败",
        action="分析死锁日志（SHOW ENGINE INNODB STATUS）；统一加锁顺序；考虑乐观锁改造",
    ),
    Rule(
        name="circuit_breaker_open",
        severity=Severity.P1_HIGH,
        pattern=re.compile(
            r"circuit.?breaker.*(open|trip)|"
            r"CircuitBreakerOpenException|"
            r"half.?open.*(fail|reject)",
            re.IGNORECASE,
        ),
        reason="熔断器开启，该服务调用已被自动降级，影响相关业务链路",
        action="检查被熔断的下游服务是否恢复；分析错误率是否下降",
    ),
]


# ---------------------------------------------------------------------------
# P3 规则：噪音，可安全忽略
# ---------------------------------------------------------------------------

P3_RULES: list[Rule] = [
    Rule(
        name="record_not_found",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"record not found|"
            r"404 Not Found|"
            r"no rows in result set|"
            r"EntityNotFoundException",
            re.IGNORECASE,
        ),
        exclude_pattern=re.compile(
            r"user|order|payment|transaction|auth",  # 核心业务资源 404 不能忽略
            re.IGNORECASE,
        ),
        reason="查询到不存在的记录，通常为客户端请求了无效数据（如过期链接），系统正常",
        action="无需处理；如频繁出现可检查客户端是否存在缓存失效问题",
    ),
    Rule(
        name="health_check_noise",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"(GET|POST) /health(?:z|/check|/live|/ready)?.* (200|204|404)|"
            r"health.?check.*(pass|ok|success|200)|"
            r"liveness.?probe|readiness.?probe",
            re.IGNORECASE,
        ),
        reason="健康检查探针请求，属于正常监控行为",
        action="无需处理",
    ),
    Rule(
        name="bot_crawler",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"Googlebot|Baiduspider|Bingbot|Sogou|YandexBot|"
            r"curl/|python-requests|Go-http-client|wget/",
            re.IGNORECASE,
        ),
        reason="爬虫或自动化工具访问产生的异常，非真实用户请求",
        action="无需处理；如爬虫流量过高可配置限流或 robots.txt",
    ),
    Rule(
        name="client_validation_error",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"validation.?error|invalid.?parameter|"
            r"missing.?required.?field|"
            r"IllegalArgumentException.*invalid|"
            r"400 Bad Request.*client",
            re.IGNORECASE,
        ),
        reason="客户端传入了无效参数，服务端校验拒绝，系统行为正常",
        action="无需处理；可统计分析是否需要优化 API 文档或客户端参数校验",
    ),
    Rule(
        name="s3_transient_timeout",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"(?:S3|OSS|GCS|blob).{0,30}(?:timeout|connection reset|"
            r"temporarily unavailable).{0,50}(?:retry|retried|success)",
            re.IGNORECASE,
        ),
        reason="对象存储偶发超时，已自动重试成功，对业务无影响",
        action="无需处理；如超时频率持续升高则升级为 P2 调查",
    ),
    Rule(
        name="jwt_expired",
        severity=Severity.P3_IGNORE,
        pattern=re.compile(
            r"JWT.*expired|token.*expired|"
            r"TokenExpiredException|ExpiredJwtException",
            re.IGNORECASE,
        ),
        reason="用户 Token 自然过期，属于正常鉴权行为，客户端将重新登录",
        action="无需处理",
    ),
]


# ---------------------------------------------------------------------------
# 规则引擎
# ---------------------------------------------------------------------------

ALL_RULES: list[Rule] = P0_RULES + P1_RULES + P3_RULES  # 优先级顺序


class RuleEngine:
    """
    轻量级规则引擎，用于在 LLM 推理之前快速预筛选。

    命中规则 → 直接返回 TriageResult，跳过 LLM 调用
    未命中  → 返回 None，由 LLM 分类器处理
    """

    def __init__(self, rules: list[Rule] = None):
        self.rules = rules if rules is not None else ALL_RULES

    def match(self, log_text: str, service_name: str = "") -> Optional[TriageResult]:
        """
        对单条日志文本进行规则匹配。

        Returns
        -------
        TriageResult | None
            命中规则返回结果；未命中返回 None
        """
        for rule in self.rules:
            if not rule.pattern.search(log_text):
                continue
            # 检查排除模式（避免误匹配）
            if rule.exclude_pattern and rule.exclude_pattern.search(log_text):
                continue

            return TriageResult(
                raw_log=log_text,
                service_name=service_name,
                severity=rule.severity,
                reason=f"[规则:{rule.name}] {rule.reason}",
                action=rule.action,
                source=ClassifySource.RULE,
            )

        return None  # 未命中任何规则，交给 LLM

    def stats(self) -> dict:
        return {
            "total_rules": len(self.rules),
            "p0_rules": sum(1 for r in self.rules if r.severity == Severity.P0_CRITICAL),
            "p1_rules": sum(1 for r in self.rules if r.severity == Severity.P1_HIGH),
            "p3_rules": sum(1 for r in self.rules if r.severity == Severity.P3_IGNORE),
        }
