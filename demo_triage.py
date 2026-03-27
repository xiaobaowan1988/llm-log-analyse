#!/usr/bin/env python3
"""
demo_triage.py
==============
告警分级系统 Demo 入口。

演示完整的「规则引擎 → 聚合去重 → LLM 分类 → 告警路由」流水线。

======================================================================
快速运行（3 种方式）
======================================================================

方式 A：dry-run（只测试规则引擎，无需 LLM）
  python demo_triage.py --dry-run

方式 B：Ollama 本地 LLM 分类
  ollama pull qwen2.5:7b
  python demo_triage.py --backend ollama --model qwen2.5:7b

方式 C：从日志文件分析
  python demo_triage.py --log /var/log/app/error.log --backend ollama

======================================================================
"""

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage.models import Severity
from triage.rules import RuleEngine
from triage.aggregator import LogAggregator, generate_fingerprint, LogGroup
from triage.router import AlertRouter, ConsoleHandler, FileHandler


# ---------------------------------------------------------------------------
# 内置测试日志（覆盖 P0~P3 各级别，含边界场景）
# ---------------------------------------------------------------------------

DEMO_LOGS = [
    # P0：Go panic（规则直接命中）
    (
        "payment-service",
        "2024-01-15 03:22:18 ERROR [payment-service] goroutine 47 panic: "
        "runtime error: nil pointer dereference\n"
        "    github.com/myco/payment/pkg/processor.go:87",
    ),
    # P0：数据库连接池耗尽（规则直接命中）
    (
        "order-service",
        "2024-01-15 08:45:00 ERROR [order-service] HikariPool-1 - Connection is not available, "
        "request timed out after 30001ms.\npool.size=20, active=20, idle=0, waiting=89",
    ),
    # P1：API 网关超时（LLM 分类）
    (
        "api-gateway",
        "2024-01-16 14:05:33 ERROR [api-gateway] upstream order-service timeout after 5000ms, "
        "retry 3/3 failed\naffected endpoints: /api/v1/orders\nError rate: 78%",
    ),
    # P2：后台任务失败（LLM 分类）
    (
        "report-service",
        "2024-01-17 22:10:15 ERROR [report-service] Failed to generate monthly report for "
        "tenant_id=9821: template rendering timeout after 30s",
    ),
    # P3：记录不存在（规则直接命中 → 排除核心资源，交 LLM）
    (
        "user-service",
        "2024-01-15 11:23:45 ERROR [user-service] err: record not found, "
        "requested_id=deleted_config_v2",
    ),
    # P3：S3 偶发超时已重试成功（规则直接命中）
    (
        "file-service",
        "2024-01-16 09:12:33 ERROR [file-service] S3 GetObject timeout (attempt 1/3). "
        "Retrying... SUCCESS on attempt 2 with 1847ms latency.",
    ),
    # P3：JWT 过期（规则直接命中）
    (
        "auth-service",
        "2024-01-17 08:30:01 ERROR [auth-service] ExpiredJwtException: JWT expired at "
        "2024-01-17T08:25:33Z. User will need to re-authenticate.",
    ),
    # 边界：营销邮件发送失败（P2，LLM 分类）
    (
        "notification-service",
        "2024-01-18 09:00:03 ERROR [notification-service] Failed to send marketing email "
        "to user@example.com via SMTP relay.smtp.example.com:587: Connection refused",
    ),
    # 边界：频率极高的业务异常（本身 P3，但频率使其升级为 P1）
    (
        "inventory-service",
        "2024-01-18 11:00:01 ERROR [inventory-service] StockCheckException: SKU-98821 "
        "stock insufficient\n[聚合信息] 该错误在过去 60s 内出现 3421 次（3421次/分钟）",
    ),
    # P0：磁盘满（规则直接命中）
    (
        "log-collector",
        "2024-01-19 02:30:00 ERROR [log-collector] Failed to write log file: "
        "No space left on device (ENOSPC)\n/data usage: 499.8GB/500GB",
    ),
]


# ---------------------------------------------------------------------------
# CLI 参数
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="告警分级系统 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例：
          python demo_triage.py --dry-run
          python demo_triage.py --backend ollama --model qwen2.5:7b --stream
          python demo_triage.py --log /var/log/app/error.log --backend ollama
        """),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只运行规则引擎，跳过 LLM 调用")
    parser.add_argument("--log", metavar="FILE",
                        help="从日志文件读取（每行一条日志）")
    parser.add_argument("--service", default="",
                        help="指定服务名（配合 --log 使用）")
    parser.add_argument("--backend", choices=["ollama", "vllm", "openai", "env"],
                        default="env")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-aggregation", action="store_true",
                        help="禁用日志聚合去重")
    parser.add_argument("--no-rules", action="store_true",
                        help="禁用规则引擎（全部走 LLM）")
    parser.add_argument("--output", metavar="FILE",
                        help="审计日志保存路径（默认 output/triage_audit.jsonl）")
    parser.add_argument("--window", type=int, default=60,
                        help="聚合时间窗口（秒），默认 60")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 65)
    print("  🔍 告警分级系统 — 基于 LLM 的 Alert Triage Demo")
    print("=" * 65)

    rule_engine = None if args.no_rules else RuleEngine()

    # ── dry-run 模式：只测试规则引擎 ──────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] 仅运行规则引擎，不调用 LLM\n")
        if rule_engine:
            print(f"规则引擎统计: {rule_engine.stats()}\n")

        logs_to_test = DEMO_LOGS
        if args.log:
            with open(args.log, encoding="utf-8", errors="replace") as f:
                logs_to_test = [(args.service, line.rstrip()) for line in f if line.strip()]

        hits, misses = 0, 0
        for service, log_text in logs_to_test:
            result = rule_engine.match(log_text, service_name=service) if rule_engine else None
            if result:
                hits += 1
                print(result.summary_line())
            else:
                misses += 1
                preview = log_text[:80].replace("\n", " ")
                print(f"⚪ [LLM 待处理]  {service:<20}  {preview}...")

        total = hits + misses
        print(f"\n规则命中: {hits}/{total} ({hits/total:.0%})  待 LLM 处理: {misses}/{total}")
        return

    # ── 构建 LLM 客户端 ────────────────────────────────────────────────────
    from rag.llm_client import (OllamaClient, VLLMClient,
                                OpenAICompatClient, create_client_from_env)
    from triage.classifier import LLMClassifier
    from triage.pipeline import TriagePipeline

    if args.backend == "ollama":
        llm = OllamaClient(
            model=args.model or "qwen2.5:7b",
            base_url=args.base_url or "http://localhost:11434",
        )
    elif args.backend == "vllm":
        llm = VLLMClient(
            model=args.model or "Qwen/Qwen2.5-7B-Instruct",
            base_url=args.base_url or "http://localhost:8000",
        )
    elif args.backend == "openai":
        llm = OpenAICompatClient(
            model=args.model or "deepseek-chat",
            base_url=args.base_url,
            api_key=args.api_key,
        )
    else:
        llm = create_client_from_env()

    print(f"[INFO] LLM 后端: {getattr(llm, 'model', '?')}")
    if rule_engine:
        print(f"[INFO] 规则引擎: {rule_engine.stats()['total_rules']} 条规则")

    # ── 构建路由器 ─────────────────────────────────────────────────────────
    audit_path = args.output or "output/triage_audit.jsonl"
    router = AlertRouter.default(audit_log_path=audit_path)

    # ── 构建流水线 ─────────────────────────────────────────────────────────
    classifier = LLMClassifier(llm=llm)
    pipeline = TriagePipeline(
        classifier=classifier,
        router=router,
        use_rules=not args.no_rules,
        use_aggregation=not args.no_aggregation,
        window_seconds=args.window,
    )

    # ── 运行分析 ────────────────────────────────────────────────────────────
    if args.log:
        print(f"\n[INFO] 从文件读取日志: {args.log}\n")
        results = pipeline.process_file(args.log, service_name=args.service)
    else:
        print(f"\n[INFO] 分析 {len(DEMO_LOGS)} 条内置演示日志\n")
        logs_text = [log for _, log in DEMO_LOGS]
        results = pipeline.process_batch(logs_text)

    # ── 打印统计 ────────────────────────────────────────────────────────────
    pipeline.stats.print_summary()
    print(f"\n[INFO] 审计日志已保存到: {audit_path}")


if __name__ == "__main__":
    main()
