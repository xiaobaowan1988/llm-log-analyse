"""
pipeline.py
===========
主流程：将 log_parser / code_retriever / prompt_builder / llm_client 串联起来，
提供一个简洁的 RCAPipeline 类供外部调用。

典型用法：
    from rag.pipeline import RCAPipeline
    from rag.code_retriever import LocalGitRetriever
    from rag.llm_client import OllamaClient

    pipeline = RCAPipeline(
        retriever=LocalGitRetriever("/home/user/my-service"),
        llm=OllamaClient(model="qwen2.5:7b"),
    )
    report = pipeline.analyse(raw_log)
    print(report.markdown)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .code_retriever import CodeRetriever, CodeSnippet
from .llm_client import LLMClient, LLMResponse
from .log_parser import ParsedLog, parse_log
from .prompt_builder import build_analysis_prompt


# ---------------------------------------------------------------------------
# 分析报告数据结构
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """根因分析完整报告"""
    # 输入
    raw_log: str
    parsed_log: ParsedLog
    snippets: list[CodeSnippet]

    # 大模型输出
    markdown: str               # 完整 Markdown 格式报告
    model: str = ""             # 使用的模型名称
    usage: dict = field(default_factory=dict)   # Token 用量

    # 元信息
    elapsed_seconds: float = 0.0    # 总耗时
    git_ref: str = "HEAD"


    def print_summary(self) -> None:
        """打印分析报告摘要到终端"""
        width = 70
        print("=" * width)
        print("  根因分析报告".center(width))
        print("=" * width)
        print(f"服务: {self.parsed_log.service_name or '未知'}")
        print(f"错误: {self.parsed_log.error_type}: {self.parsed_log.error_message[:60]}")
        print(f"检索文件: {len(self.snippets)} 个")
        print(f"模型: {self.model}  |  耗时: {self.elapsed_seconds:.1f}s")
        if self.usage:
            pt = self.usage.get("prompt_tokens", self.usage.get("prompt_eval_count", "?"))
            ct = self.usage.get("completion_tokens", self.usage.get("eval_count", "?"))
            print(f"Token 用量: prompt={pt}, completion={ct}")
        print("-" * width)
        print(self.markdown)
        print("=" * width)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

class RCAPipeline:
    """
    日志根因分析流水线。

    Parameters
    ----------
    retriever : CodeRetriever
        代码检索器（LocalGitRetriever / GitLabRetriever / GitHubRetriever）
    llm : LLMClient
        大模型客户端（OllamaClient / VLLMClient / OpenAICompatClient）
    max_files : int
        每次分析最多检索几个源文件（控制 Prompt 长度）
    max_prompt_chars : int
        代码片段部分的字符数上限
    temperature : float
        LLM 采样温度，根因分析推荐低温（0.1）
    max_tokens : int
        LLM 最大输出 Token 数
    git_ref : str
        默认从哪个 Git 引用检索代码（分支/tag/commit sha）
    """

    def __init__(
        self,
        retriever: CodeRetriever,
        llm: LLMClient,
        max_files: int = 5,
        max_prompt_chars: int = 12_000,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        git_ref: str = "HEAD",
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_files = max_files
        self.max_prompt_chars = max_prompt_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.git_ref = git_ref

    # ------------------------------------------------------------------
    # 核心分析接口
    # ------------------------------------------------------------------

    def analyse(
        self,
        raw_log: str,
        git_ref: Optional[str] = None,
        stream: bool = False,
    ) -> AnalysisReport:
        """
        完整分析一条日志，返回 AnalysisReport。

        Parameters
        ----------
        raw_log : str
            原始日志文本（包含 Stack Trace）
        git_ref : str | None
            覆盖默认 git_ref（None 则使用初始化时的值）
        stream : bool
            是否以流式方式打印 LLM 输出（不影响返回值）

        Returns
        -------
        AnalysisReport
        """
        t_start = time.time()
        ref = git_ref or self.git_ref

        # Step 1: 解析日志
        parsed = parse_log(raw_log)
        print(f"[Step 1/3] 日志解析完成 — 错误类型: {parsed.error_type or '未知'}, "
              f"检测到 {len(parsed.frames)} 个栈帧")

        # Step 2: 检索相关代码片段
        snippets = []
        if parsed.frames:
            snippets = self.retriever.retrieve_from_log(
                parsed,
                ref=ref,
                max_files=self.max_files,
            )
            print(f"[Step 2/3] 代码检索完成 — 成功检索 {len(snippets)}/{len(parsed.unique_files[:self.max_files])} 个文件")
        else:
            print("[Step 2/3] 未检测到 Stack Trace，跳过代码检索")

        # Step 3: 构建 Prompt 并调用 LLM
        system_prompt, user_prompt = build_analysis_prompt(
            parsed,
            snippets,
            max_chars=self.max_prompt_chars,
        )

        total_chars = len(system_prompt) + len(user_prompt)
        print(f"[Step 3/3] 开始 LLM 推理 — Prompt 约 {total_chars:,} 字符")

        if stream:
            # 流式模式：边生成边打印
            print("\n" + "─" * 60 + "\n")
            content_parts = []
            for token in self.llm.stream_chat(
                system_prompt, user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                print(token, end="", flush=True)
                content_parts.append(token)
            print("\n" + "─" * 60)
            markdown = "".join(content_parts)
            llm_response = LLMResponse(content=markdown, model=getattr(self.llm, "model", ""))
        else:
            llm_response = self.llm.chat(
                system_prompt, user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            markdown = llm_response.content

        elapsed = time.time() - t_start

        return AnalysisReport(
            raw_log=raw_log,
            parsed_log=parsed,
            snippets=snippets,
            markdown=markdown,
            model=getattr(self.llm, "model", ""),
            usage=llm_response.usage,
            elapsed_seconds=elapsed,
            git_ref=ref,
        )

    def analyse_batch(
        self,
        logs: list[str],
        git_ref: Optional[str] = None,
    ) -> list[AnalysisReport]:
        """
        批量分析多条日志（顺序处理）。

        Parameters
        ----------
        logs : list[str]
            日志文本列表
        git_ref : str | None
            Git 引用

        Returns
        -------
        list[AnalysisReport]
        """
        reports = []
        for i, log in enumerate(logs, 1):
            print(f"\n{'='*60}")
            print(f"  分析日志 {i}/{len(logs)}")
            print(f"{'='*60}")
            try:
                report = self.analyse(log, git_ref=git_ref)
                reports.append(report)
            except Exception as e:
                print(f"[ERROR] 分析失败: {e}")
        return reports
