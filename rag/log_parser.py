"""
log_parser.py
=============
解析错误日志，从 Stack Trace 中提取结构化的代码定位信息。

支持的日志格式：
  - Java  (at com.example.Foo.bar(Foo.java:42))
  - Python (File "app/server.py", line 87, in handle_request)
  - Go     (goroutine panic: runtime/debug.Stack at server.go:128)
  - Node   (at Object.<anonymous> (/app/src/handler.js:34:12))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StackFrame:
    """Stack Trace 中单个栈帧的结构化信息"""
    language: str           # java / python / go / node
    file_path: str          # 文件路径，如 com/example/Foo.java 或 app/server.py
    line_number: int        # 报错行号
    function_name: str      # 函数 / 方法名
    class_name: str = ""    # Java 类名（其他语言为空）
    raw: str = ""           # 原始文本行，方便调试


@dataclass
class ParsedLog:
    """完整日志的解析结果"""
    raw_log: str                            # 原始日志全文
    error_type: str = ""                    # 错误类型，如 NullPointerException
    error_message: str = ""                 # 错误描述信息
    frames: list[StackFrame] = field(default_factory=list)
    severity: str = "ERROR"                 # ERROR / WARN / CRITICAL
    service_name: str = ""                  # 服务名，从日志前缀提取

    @property
    def top_frame(self) -> Optional[StackFrame]:
        """最顶层（最直接报错）的栈帧"""
        return self.frames[0] if self.frames else None

    @property
    def unique_files(self) -> list[str]:
        """去重后的文件路径列表（按出现顺序）"""
        seen: set[str] = set()
        result: list[str] = []
        for f in self.frames:
            if f.file_path not in seen:
                seen.add(f.file_path)
                result.append(f.file_path)
        return result


# ---------------------------------------------------------------------------
# 正则表达式
# ---------------------------------------------------------------------------

# 日志头部：级别 + 服务名
_RE_LOG_HEADER = re.compile(
    r"(?P<severity>ERROR|WARN|WARNING|CRITICAL|FATAL|INFO)"
    r"\s+\[(?P<service>[^\]]+)\]",
    re.IGNORECASE,
)

# Java: at com.example.service.Foo.bar(Foo.java:42)
_RE_JAVA = re.compile(
    r"^\s+at\s+"
    r"(?P<class_method>[\w\.$]+)\."
    r"(?P<method>[\w<>$]+)"
    r"\((?P<file>[\w$.]+\.java):(?P<line>\d+)\)"
)

# Python: File "app/server.py", line 87, in handle_request
_RE_PYTHON = re.compile(
    r'^\s+File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>\S+)'
)

# Go: goroutine / panic 堆栈，如 github.com/foo/bar/pkg.Handler(...)
#     /home/user/go/src/github.com/foo/bar/pkg/handler.go:128
_RE_GO_FUNC = re.compile(
    r"^(?P<pkg>[\w./\-]+)\((?:.*)\)$"
)
_RE_GO_FILE = re.compile(
    r"^\s+(?P<file>[^:]+\.go):(?P<line>\d+)"
)

# Node.js: at Object.<anonymous> (/app/src/handler.js:34:12)
#          at processTicksAndRejections (internal/process/task_queues.js:95:5)
_RE_NODE = re.compile(
    r"^\s+at\s+(?P<func>.+?)\s+\((?P<file>[^)]+):(?P<line>\d+):\d+\)"
    r"|"
    r"^\s+at\s+(?P<file2>[^:]+):(?P<line2>\d+):\d+"  # 匿名调用
)

# Java / Python 错误类型行
_RE_JAVA_ERROR = re.compile(
    r"^(?P<type>[\w.$]+(?:Exception|Error|Throwable|Fault))"
    r"(?::\s*(?P<msg>.+))?"
)
_RE_PYTHON_ERROR = re.compile(
    r"^(?P<type>\w+(?:Error|Exception|Warning)):\s*(?P<msg>.+)"
)


# ---------------------------------------------------------------------------
# 核心解析逻辑
# ---------------------------------------------------------------------------

def _java_class_to_path(class_method: str) -> str:
    """将 Java 全限定类名转换为文件路径片段
    例：com.example.service.PaymentProcessor -> com/example/service/PaymentProcessor.java
    """
    # 去掉内部类标记（$）和方法名
    parts = class_method.split(".")
    # 过滤掉全小写的最后一段（方法名）
    class_parts = [p for p in parts if p and p[0].isupper() or "$" in p]
    if not class_parts:
        class_parts = parts
    return "/".join(class_parts[:]) + ".java" if class_parts else class_method


def _parse_java_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    frames: list[StackFrame] = []
    error_type = ""
    error_message = ""

    for line in lines:
        m_err = _RE_JAVA_ERROR.match(line.strip())
        if m_err and not error_type:
            error_type = m_err.group("type")
            error_message = (m_err.group("msg") or "").strip()
            continue

        m = _RE_JAVA.match(line)
        if m:
            class_method = m.group("class_method")
            file_path = _java_class_to_path(class_method)
            frames.append(StackFrame(
                language="java",
                file_path=file_path,
                line_number=int(m.group("line")),
                function_name=m.group("method"),
                class_name=class_method,
                raw=line.rstrip(),
            ))

    return frames, error_type, error_message


def _parse_python_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    frames: list[StackFrame] = []
    error_type = ""
    error_message = ""

    i = 0
    while i < len(lines):
        m = _RE_PYTHON.match(lines[i])
        if m:
            # 下一行是实际代码，再下一行可能是函数名
            func_name = m.group("func")
            frames.append(StackFrame(
                language="python",
                file_path=m.group("file"),
                line_number=int(m.group("line")),
                function_name=func_name,
                raw=lines[i].rstrip(),
            ))
        else:
            m_err = _RE_PYTHON_ERROR.match(lines[i].strip())
            if m_err and not error_type:
                error_type = m_err.group("type")
                error_message = m_err.group("msg").strip()
        i += 1

    return frames, error_type, error_message


def _parse_go_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    """解析 Go goroutine dump / panic 堆栈"""
    frames: list[StackFrame] = []
    error_type = ""
    error_message = ""
    pending_func = ""

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("panic:"):
            error_type = "panic"
            error_message = stripped[6:].strip()
            continue

        m_func = _RE_GO_FUNC.match(stripped)
        if m_func:
            pending_func = m_func.group("pkg")
            continue

        m_file = _RE_GO_FILE.match(line)
        if m_file and pending_func:
            file_path = m_file.group("file").lstrip("/")
            # 只保留项目内路径（过滤掉 GOROOT）
            if "runtime/" not in file_path and "vendor/" not in file_path:
                func_short = pending_func.split(".")[-1].split("(")[0]
                frames.append(StackFrame(
                    language="go",
                    file_path=file_path,
                    line_number=int(m_file.group("line")),
                    function_name=func_short,
                    class_name=pending_func,
                    raw=line.rstrip(),
                ))
            pending_func = ""

    return frames, error_type, error_message


def _parse_node_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    frames: list[StackFrame] = []
    error_type = ""
    error_message = ""

    for line in lines:
        # 错误首行：TypeError: Cannot read properties of null
        if not error_type and re.match(r"^\w+Error:", line.strip()):
            parts = line.strip().split(":", 1)
            error_type = parts[0]
            error_message = parts[1].strip() if len(parts) > 1 else ""
            continue

        m = _RE_NODE.match(line)
        if not m:
            continue

        if m.group("func"):
            file_path = m.group("file")
            line_num = int(m.group("line"))
            func_name = m.group("func").strip()
        else:
            file_path = m.group("file2")
            line_num = int(m.group("line2"))
            func_name = "<anonymous>"

        # 过滤 node_modules 和 Node 内部路径
        if "node_modules" in file_path or "internal/" in file_path:
            continue

        frames.append(StackFrame(
            language="node",
            file_path=file_path.lstrip("/"),
            line_number=line_num,
            function_name=func_name,
            raw=line.rstrip(),
        ))

    return frames, error_type, error_message


def _detect_language(text: str) -> str:
    """根据 Stack Trace 特征判断编程语言"""
    if re.search(r"^\s+at\s+[\w.$]+\([\w$.]+\.java:\d+\)", text, re.MULTILINE):
        return "java"
    if re.search(r'^\s+File\s+"[^"]+\.py",\s+line\s+\d+', text, re.MULTILINE):
        return "python"
    if re.search(r"^\s+/[^\s]+\.go:\d+", text, re.MULTILINE):
        return "go"
    if re.search(r"^\s+at\s+.+\([^)]+\.(?:js|ts|mjs):\d+:\d+\)", text, re.MULTILINE):
        return "node"
    return "unknown"


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def parse_log(raw_log: str) -> ParsedLog:
    """
    解析日志文本，返回结构化的 ParsedLog 对象。

    Parameters
    ----------
    raw_log : str
        原始日志文本（可包含多行 Stack Trace）

    Returns
    -------
    ParsedLog
        包含错误类型、服务名、栈帧列表等信息
    """
    result = ParsedLog(raw_log=raw_log)

    # 提取日志级别和服务名
    m_header = _RE_LOG_HEADER.search(raw_log)
    if m_header:
        result.severity = m_header.group("severity").upper()
        result.service_name = m_header.group("service")

    lines = raw_log.splitlines()
    language = _detect_language(raw_log)

    if language == "java":
        result.frames, result.error_type, result.error_message = _parse_java_frames(lines)
    elif language == "python":
        result.frames, result.error_type, result.error_message = _parse_python_frames(lines)
    elif language == "go":
        result.frames, result.error_type, result.error_message = _parse_go_frames(lines)
    elif language == "node":
        result.frames, result.error_type, result.error_message = _parse_node_frames(lines)

    return result


def summarize_parsed_log(parsed: ParsedLog) -> str:
    """将 ParsedLog 转换为人类可读的摘要字符串"""
    lines = [
        f"服务: {parsed.service_name or '未知'}",
        f"级别: {parsed.severity}",
        f"错误: {parsed.error_type or '未知'}: {parsed.error_message or ''}",
        f"涉及文件数: {len(parsed.unique_files)}",
    ]
    if parsed.top_frame:
        f = parsed.top_frame
        lines.append(
            f"最顶层报错: {f.file_path}:{f.line_number} in {f.function_name}()"
        )
    return "\n".join(lines)
