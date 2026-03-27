#!/usr/bin/env python3
"""
demo_rag.py
===========
RAG 根因分析 Demo 入口。

该脚本演示「日志 + 代码」结合的完整分析流程：
  1. 解析 Stack Trace，提取报错文件和行号
  2. 从 Git 仓库自动检索相关代码片段
  3. 组装 Prompt，调用本地大模型生成分析报告

======================================================================
快速运行（3 种方式）
======================================================================

方式 A：Ollama（最简单，推荐入门）
  # 1. 安装并启动 Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:7b
  # 2. 运行 Demo（使用内置模拟日志，无需真实代码仓库）
  python demo_rag.py --mock

方式 B：指定真实代码仓库
  python demo_rag.py \\
      --repo /home/user/my-java-service \\
      --log  /var/log/my-service/error.log \\
      --backend ollama \\
      --model qwen2.5:7b

方式 C：通过环境变量配置后端
  export LLM_BACKEND=vllm
  export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
  export LLM_BASE_URL=http://localhost:8000
  python demo_rag.py --repo /home/user/my-service --log /tmp/error.log

======================================================================
"""

import argparse
import os
import sys
import textwrap

# 确保 rag 包可以被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.llm_client import OllamaClient, VLLMClient, OpenAICompatClient, create_client_from_env
from rag.code_retriever import LocalGitRetriever, GitLabRetriever
from rag.pipeline import RCAPipeline


# ---------------------------------------------------------------------------
# 内置模拟数据（--mock 模式用）
# ---------------------------------------------------------------------------

# 模拟日志 1：Java NullPointerException（模拟 PaymentProcessor 业务代码）
MOCK_LOG_JAVA = """\
2024-01-15 03:22:18 ERROR [payment-service] Unhandled exception in payment processing thread
java.lang.NullPointerException: Cannot invoke "com.example.model.PaymentMethod.getCardNumber()" because "paymentMethod" is null
    at com.example.service.PaymentProcessor.validatePayment(PaymentProcessor.java:87)
    at com.example.service.PaymentProcessor.processBatch(PaymentProcessor.java:52)
    at com.example.controller.PaymentController.submitBatch(PaymentController.java:134)
    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
堆内存使用率: 72% | 活跃线程数: 48 | 失败订单数: 1,247"""

# 模拟日志 2：Python KeyError（模拟数据管道代码）
MOCK_LOG_PYTHON = """\
2024-01-16 14:30:45 ERROR [data-pipeline] ETL transform stage failed
Traceback (most recent call last):
  File "pipeline/etl/transformer.py", line 203, in normalize_field
    value = record["user_profile"]["preferences"]["language"]
  File "pipeline/etl/transformer.py", line 178, in transform_batch
    normalized = self.normalize_field(record)
  File "pipeline/etl/runner.py", line 89, in run_stage
    results = stage.transform_batch(batch_data)
KeyError: 'user_profile'
处理批次: batch_20240116_143000 | 已处理: 1,872,451 条 | 失败位置: 第 1,872,452 条"""

# 模拟日志 3：Go panic（模拟 Kubernetes Controller 代码）
MOCK_LOG_GO = """\
2024-01-17 09:15:33 CRITICAL [k8s-controller] goroutine panic recovered
panic: assignment to entry in nil map

goroutine 47 [running]:
runtime/debug.Stack(...)
    /usr/local/go/src/runtime/debug/stack.go:24
github.com/myorg/k8s-operator/pkg/controller.(*ReconcileWebhook).handleEvent(...)
    /home/runner/go/src/github.com/myorg/k8s-operator/pkg/controller/webhook_handler.go:128 +0x2a4
github.com/myorg/k8s-operator/pkg/controller.(*ReconcileWebhook).Reconcile(...)
    /home/runner/go/src/github.com/myorg/k8s-operator/pkg/controller/reconciler.go:67 +0x1b8
影响: Webhook 处理停止 | 积压事件: 3,421 个"""

# 模拟代码文件（用于演示无真实仓库时的 Prompt 构建）
MOCK_CODE_JAVA = """\
package com.example.service;

import com.example.model.PaymentMethod;
import com.example.model.Order;
import java.util.List;

public class PaymentProcessor {

    private final PaymentGateway gateway;
    private final OrderRepository orderRepo;

    public PaymentProcessor(PaymentGateway gateway, OrderRepository orderRepo) {
        this.gateway = gateway;
        this.orderRepo = orderRepo;
    }

    // 批量处理支付订单
    public BatchResult processBatch(List<Order> orders) {
        BatchResult result = new BatchResult();
        for (Order order : orders) {
            try {
                PaymentMethod method = orderRepo.findPaymentMethod(order.getId());
                // 注意：findPaymentMethod 可能返回 null（订单关联支付方式被删除）
                result.add(validatePayment(order, method));  // Line 52
            } catch (Exception e) {
                result.addFailure(order, e);
            }
        }
        return result;
    }

    // 验证支付方式
    private ValidationResult validatePayment(Order order, PaymentMethod paymentMethod) {
        // ⚠️ 此处未对 paymentMethod 进行 null 检查
        String cardNumber = paymentMethod.getCardNumber();  // Line 87 — NPE 触发点
        String expiry = paymentMethod.getExpiryDate();

        if (!isValidCard(cardNumber)) {
            return ValidationResult.invalid("无效的卡号");
        }
        if (isExpired(expiry)) {
            return ValidationResult.invalid("支付卡已过期");
        }
        return ValidationResult.valid();
    }
}"""


# ---------------------------------------------------------------------------
# Mock 代码检索器（演示用，不依赖真实仓库）
# ---------------------------------------------------------------------------

class MockCodeRetriever:
    """演示用检索器，返回硬编码的模拟代码片段"""

    def retrieve_from_log(self, parsed, ref="HEAD", max_files=5):
        from rag.code_retriever import CodeSnippet

        if parsed.error_type in ("NullPointerException",):
            lines = MOCK_CODE_JAVA.splitlines()
            numbered = "\n".join(f"{i+1:>6} | {l}" for i, l in enumerate(lines))
            return [CodeSnippet(
                file_path="com/example/service/PaymentProcessor.java",
                start_line=1,
                end_line=len(lines),
                content=numbered,
                highlight_lines=[52, 87],
                language="java",
                commit_sha="a3f8c2d",
            )]
        # 其他语言返回空，演示"无代码时的分析"
        return []


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="日志 + 代码 RAG 根因分析 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例：
          # 使用内置模拟数据（无需真实仓库和 LLM）
          python demo_rag.py --mock --dry-run

          # 使用 Ollama 分析模拟日志
          python demo_rag.py --mock --backend ollama --model qwen2.5:7b

          # 分析真实日志文件
          python demo_rag.py --repo /path/to/repo --log /var/log/app/error.log

          # 指定 Git 分支
          python demo_rag.py --repo /path/to/repo --log error.log --ref main
        """),
    )

    # 数据来源
    src = parser.add_argument_group("数据来源")
    src.add_argument("--mock", action="store_true",
                     help="使用内置模拟日志（无需真实仓库）")
    src.add_argument("--log", metavar="FILE",
                     help="日志文件路径（不指定则从 stdin 读取）")
    src.add_argument("--log-text", metavar="TEXT",
                     help="直接传入日志文本（适合脚本调用）")

    # 代码仓库
    repo = parser.add_argument_group("代码仓库")
    repo.add_argument("--repo", metavar="PATH",
                      help="本地 Git 仓库根目录")
    repo.add_argument("--gitlab-url", metavar="URL",
                      help="GitLab 服务地址（如 https://gitlab.company.com）")
    repo.add_argument("--gitlab-project", metavar="ID",
                      help="GitLab 项目 ID 或 group/name")
    repo.add_argument("--ref", default="HEAD",
                      help="Git 引用（分支/tag/commit sha），默认 HEAD")
    repo.add_argument("--context-lines", type=int, default=40,
                      help="报错行上下各取几行，默认 40")

    # LLM 配置
    llm = parser.add_argument_group("LLM 配置")
    llm.add_argument("--backend", choices=["ollama", "vllm", "openai", "env"],
                     default="env",
                     help="推理后端（默认从环境变量读取）")
    llm.add_argument("--model", metavar="NAME",
                     help="模型名称")
    llm.add_argument("--base-url", metavar="URL",
                     help="推理服务地址")
    llm.add_argument("--api-key", metavar="KEY",
                     help="API Key（云端服务需要）")
    llm.add_argument("--max-tokens", type=int, default=2048,
                     help="最大输出 Token 数，默认 2048")
    llm.add_argument("--temperature", type=float, default=0.1,
                     help="采样温度，默认 0.1")

    # 运行选项
    opt = parser.add_argument_group("运行选项")
    opt.add_argument("--stream", action="store_true",
                     help="流式输出 LLM 结果（边生成边打印）")
    opt.add_argument("--dry-run", action="store_true",
                     help="只解析日志和检索代码，不调用 LLM（用于调试）")
    opt.add_argument("--output", metavar="FILE",
                     help="将报告保存为 Markdown 文件")
    opt.add_argument("--max-files", type=int, default=5,
                     help="最多检索几个源文件，默认 5")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- 1. 确定日志来源 -------------------------------------------------
    if args.mock:
        # 依次演示 3 种语言的日志
        logs_to_analyse = [
            ("Java NullPointerException", MOCK_LOG_JAVA),
            ("Python KeyError", MOCK_LOG_PYTHON),
            ("Go Panic", MOCK_LOG_GO),
        ]
    elif args.log_text:
        logs_to_analyse = [("命令行输入", args.log_text)]
    elif args.log:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            logs_to_analyse = [("日志文件", f.read())]
    else:
        print("从 stdin 读取日志（输入完成后按 Ctrl+D）：")
        logs_to_analyse = [("stdin", sys.stdin.read())]

    # ---- 2. 构建代码检索器 -----------------------------------------------
    if args.mock and not args.repo:
        retriever = MockCodeRetriever()
        print("[INFO] 使用内置模拟代码检索器")
    elif args.repo:
        retriever = LocalGitRetriever(args.repo, context_lines=args.context_lines)
        print(f"[INFO] 使用本地仓库: {args.repo}")
    elif args.gitlab_url and args.gitlab_project:
        retriever = GitLabRetriever(
            base_url=args.gitlab_url,
            project_id=args.gitlab_project,
            access_token=args.api_key or os.getenv("GITLAB_TOKEN", ""),
            context_lines=args.context_lines,
        )
        print(f"[INFO] 使用 GitLab: {args.gitlab_url} / {args.gitlab_project}")
    else:
        print("[WARN] 未指定代码仓库，将只分析日志文本（无代码片段）")
        retriever = MockCodeRetriever()

    # ---- 3. 构建 LLM 客户端 ----------------------------------------------
    if args.dry_run:
        llm = None
        print("[INFO] --dry-run 模式，跳过 LLM 调用")
    elif args.backend == "ollama":
        llm = OllamaClient(
            model=args.model or "qwen2.5:7b",
            base_url=args.base_url or "http://localhost:11434",
        )
        print(f"[INFO] 使用 Ollama: model={llm.model}, url={llm.base_url}")
    elif args.backend == "vllm":
        llm = VLLMClient(
            model=args.model or "Qwen/Qwen2.5-7B-Instruct",
            base_url=args.base_url or "http://localhost:8000",
        )
        print(f"[INFO] 使用 vLLM: model={llm.model}, url={llm.base_url}")
    elif args.backend == "openai":
        if not args.base_url:
            print("[ERROR] 使用 openai 后端时必须指定 --base-url")
            sys.exit(1)
        llm = OpenAICompatClient(
            model=args.model or "deepseek-chat",
            base_url=args.base_url,
            api_key=args.api_key or "",
        )
    else:
        llm = create_client_from_env()
        print(f"[INFO] 从环境变量读取 LLM 配置: backend={os.getenv('LLM_BACKEND', 'ollama')}")

    # ---- 4. 构建并运行 Pipeline ------------------------------------------
    from rag.log_parser import parse_log, summarize_parsed_log
    from rag.prompt_builder import build_analysis_prompt

    reports = []
    for label, raw_log in logs_to_analyse:
        print(f"\n{'='*70}")
        print(f"  分析: {label}")
        print(f"{'='*70}\n")

        if args.dry_run:
            # Dry-run：只展示解析结果和 Prompt，不调用 LLM
            parsed = parse_log(raw_log)
            print("【日志解析结果】")
            print(summarize_parsed_log(parsed))
            print(f"\n检测到 {len(parsed.frames)} 个栈帧，涉及文件：")
            for f_path in parsed.unique_files[:5]:
                print(f"  - {f_path}")

            snippets = retriever.retrieve_from_log(
                parsed, ref=args.ref, max_files=args.max_files
            )
            print(f"\n成功检索 {len(snippets)} 个代码片段")

            _, user_prompt = build_analysis_prompt(parsed, snippets)
            print(f"\n【Prompt 预览】（前 800 字符）")
            print("-" * 60)
            print(user_prompt[:800] + ("..." if len(user_prompt) > 800 else ""))
            print("-" * 60)
            continue

        pipeline = RCAPipeline(
            retriever=retriever,
            llm=llm,
            max_files=args.max_files,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            git_ref=args.ref,
        )

        report = pipeline.analyse(raw_log, stream=args.stream)
        reports.append((label, report))

        if not args.stream:
            report.print_summary()

        # 保存报告
        if args.output:
            output_path = args.output if len(logs_to_analyse) == 1 \
                else f"{args.output}_{len(reports)}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# 根因分析报告：{label}\n\n")
                f.write(report.markdown)
            print(f"\n[INFO] 报告已保存到: {output_path}")

    print("\n[INFO] 分析完成！")


if __name__ == "__main__":
    main()
