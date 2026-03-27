"""
prompt_builder.py
=================
将「日志解析结果」+「代码片段」组装成结构化的 Prompt，送给大模型。

设计目标：
  - Prompt 格式清晰，让模型能精准定位「哪段代码导致了什么问题」
  - 控制 Token 总量（预估字符数），避免超出上下文窗口
  - 输出格式固定（Markdown），方便后续解析成结构化报告
"""

from __future__ import annotations

from .log_parser import ParsedLog, summarize_parsed_log
from .code_retriever import CodeSnippet


# ---------------------------------------------------------------------------
# 系统提示词（角色设定）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名资深的软件工程师和 SRE（站点可靠性工程师），专精于生产环境故障根因分析（Root Cause Analysis）。

你的分析必须：
1. 紧密结合用户提供的「源代码片段」，不要只靠通用知识猜测
2. 精确到代码行级别（例如：handler.go 第 128 行）
3. 给出可立即执行的修复方案（如有可能，直接提供修复后的代码）
4. 区分「直接原因」和「根本原因」
5. 如果代码片段不足以定位问题，明确指出还需要哪些信息

输出格式要求（严格遵守 Markdown）：
## 直接原因
[1-2句话，描述触发异常的直接代码逻辑]

## 根本原因
[1-3句话，描述导致直接原因的更深层设计或数据问题]

## 问题代码定位
[列出具体文件名和行号，说明问题所在]

## 修复方案
### 紧急处理（立即可做）
[步骤列表]

### 代码修复
```[语言]
[修复后的代码片段，只改有问题的部分]
```

### 长期优化
[预防同类问题的架构/规范建议]

## 置信度
[高 / 中 / 低，并说明原因]"""


# ---------------------------------------------------------------------------
# Prompt 构建器
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    parsed: ParsedLog,
    snippets: list[CodeSnippet],
    max_chars: int = 12_000,
) -> tuple[str, str]:
    """
    构建用于根因分析的完整 Prompt。

    Parameters
    ----------
    parsed : ParsedLog
        log_parser.parse_log() 返回的解析结果
    snippets : list[CodeSnippet]
        code_retriever.retrieve_from_log() 返回的代码片段列表
    max_chars : int
        代码片段部分的最大字符数（防止 Prompt 过长，默认 12000 字符）
        对应约 3000~4000 Token，适合大多数 8K+ 上下文模型

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
        分别对应 LLM API 的 system 和 user 角色消息
    """
    # ---- Part 1: 日志摘要 ------------------------------------------------
    log_section = _build_log_section(parsed)

    # ---- Part 2: 代码片段（按字符预算裁剪）--------------------------------
    code_section = _build_code_section(snippets, budget=max_chars)

    # ---- Part 3: 历史上下文（如有）----------------------------------------
    # 此处预留接口，后续可接 RAG 向量检索的历史排障记录
    history_section = ""

    # ---- 组装 User Prompt ------------------------------------------------
    parts = [log_section]
    if code_section:
        parts.append(code_section)
    if history_section:
        parts.append(history_section)
    parts.append(_TASK_INSTRUCTION)

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


def _build_log_section(parsed: ParsedLog) -> str:
    lines = ["## 错误日志"]

    # 结构化摘要
    lines.append("### 解析摘要")
    lines.append(summarize_parsed_log(parsed))

    # 原始日志（截断防止过长）
    raw = parsed.raw_log
    if len(raw) > 3000:
        raw = raw[:2800] + "\n... [日志截断，已保留关键部分]"

    lines.append("### 原始日志")
    lines.append(f"```\n{raw}\n```")

    return "\n".join(lines)


def _build_code_section(snippets: list[CodeSnippet], budget: int) -> str:
    if not snippets:
        return ""

    lines = ["## 相关源代码片段"]
    lines.append("*以下代码由系统从 Git 仓库中自动检索，报错行用「>>>」标注*\n")

    used = 0
    for i, snippet in enumerate(snippets, 1):
        # 在代码内容中标注报错行
        annotated = _annotate_snippet(snippet)
        block = f"### 片段 {i}: `{snippet.file_path}`\n{annotated}"

        if used + len(block) > budget:
            lines.append(f"*（剩余 {len(snippets) - i + 1} 个文件因 Token 限制省略）*")
            break

        lines.append(block)
        used += len(block)

    return "\n".join(lines)


def _annotate_snippet(snippet: CodeSnippet) -> str:
    """在代码片段中用 >>> 标注报错行"""
    highlight_set = set(snippet.highlight_lines or [])
    annotated_lines = []

    for raw_line in snippet.content.splitlines():
        # raw_line 格式: "    42 | code here"
        try:
            ln_str = raw_line.split("|")[0].strip()
            ln = int(ln_str)
            prefix = ">>>" if ln in highlight_set else "   "
            annotated_lines.append(f"{prefix} {raw_line}")
        except (ValueError, IndexError):
            annotated_lines.append(f"    {raw_line}")

    lang = snippet.language or ""
    return f"```{lang}\n" + "\n".join(annotated_lines) + "\n```"


_TASK_INSTRUCTION = """## 分析任务

请结合上方的**错误日志**和**源代码片段**，进行深度根因分析。

重点关注：
1. 报错行（用 `>>>` 标注）的代码逻辑是否存在空指针、越界、竞态、资源泄漏等问题
2. 调用链上下文——调用该方法的代码是否传入了非法参数
3. 并发/异步场景下是否存在状态共享问题

请严格按照系统要求的 Markdown 格式输出分析报告。"""
