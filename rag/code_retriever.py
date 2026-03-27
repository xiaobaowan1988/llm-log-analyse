"""
code_retriever.py
=================
根据 Stack Trace 解析结果，从代码仓库中检索相关代码片段。

支持三种来源：
  1. LocalGitRetriever  — 本地 Git 仓库（最常用，无网络依赖）
  2. GitLabRetriever    — GitLab 私有部署（通过 REST API）
  3. GitHubRetriever    — GitHub（通过 REST API，适合开源项目）

设计原则：
  - 所有 Retriever 实现同一接口 (CodeRetriever)，pipeline 无感知切换
  - 每个报错文件取「报错行 ± context_lines」行的代码窗口
  - 对同一文件多处报错，合并为一个连续窗口避免重复
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .log_parser import ParsedLog, StackFrame


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CodeSnippet:
    """从仓库中检索到的代码片段"""
    file_path: str          # 仓库内相对路径
    start_line: int         # 片段起始行（含，从 1 计数）
    end_line: int           # 片段结束行（含）
    content: str            # 带行号的代码文本
    highlight_lines: list[int] = None   # 报错行，用于 Prompt 中标注
    language: str = ""      # 编程语言（java/python/go/node）
    commit_sha: str = ""    # 对应的 Git commit hash（可选）

    def __post_init__(self):
        if self.highlight_lines is None:
            self.highlight_lines = []

    def to_markdown(self) -> str:
        """生成带标注的 Markdown 代码块，方便嵌入 Prompt"""
        lang_hint = self.language or ""
        header = (
            f"**文件**: `{self.file_path}`  "
            f"**行**: {self.start_line}–{self.end_line}"
        )
        if self.highlight_lines:
            header += f"  **报错行**: {', '.join(map(str, self.highlight_lines))}"

        return f"{header}\n```{lang_hint}\n{self.content}\n```"


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class CodeRetriever(ABC):
    """所有代码检索器的统一接口"""

    def __init__(self, context_lines: int = 40):
        """
        Parameters
        ----------
        context_lines : int
            报错行上下各取多少行作为上下文（默认 40 行）
        """
        self.context_lines = context_lines

    @abstractmethod
    def fetch_snippet(
        self,
        file_path: str,
        line_number: int,
        ref: str = "HEAD",
    ) -> Optional[CodeSnippet]:
        """
        获取指定文件指定行的代码片段。

        Parameters
        ----------
        file_path : str
            仓库内文件路径（相对路径）
        line_number : int
            目标行号（从 1 计数）
        ref : str
            Git 引用（分支名、tag、commit sha），默认 HEAD

        Returns
        -------
        CodeSnippet | None
            成功返回代码片段，文件不存在返回 None
        """

    def retrieve_from_log(
        self,
        parsed: ParsedLog,
        ref: str = "HEAD",
        max_files: int = 5,
    ) -> list[CodeSnippet]:
        """
        根据解析后的日志批量检索相关代码片段。

        Parameters
        ----------
        parsed : ParsedLog
            log_parser.parse_log() 的返回值
        ref : str
            Git 引用
        max_files : int
            最多检索几个文件（避免 Prompt 过长）

        Returns
        -------
        list[CodeSnippet]
            检索到的代码片段列表（同一文件合并窗口）
        """
        # 按文件分组，合并同文件多处报错的行号
        file_frames: dict[str, list[StackFrame]] = {}
        for frame in parsed.frames:
            key = frame.file_path
            file_frames.setdefault(key, []).append(frame)

        snippets: list[CodeSnippet] = []
        for file_path, frames in list(file_frames.items())[:max_files]:
            # 合并多处报错为一个连续窗口
            all_lines = sorted({f.line_number for f in frames})
            center_line = all_lines[0]  # 以最顶层报错行为中心

            snippet = self.fetch_snippet(file_path, center_line, ref=ref)
            if snippet:
                snippet.highlight_lines = all_lines
                snippet.language = frames[0].language
                snippets.append(snippet)

        return snippets


# ---------------------------------------------------------------------------
# 实现 1：本地 Git 仓库
# ---------------------------------------------------------------------------

class LocalGitRetriever(CodeRetriever):
    """
    从本地 Git 仓库检索代码片段。

    适用场景：
      - 本地开发调试
      - CI/CD 环境中已 checkout 代码
      - 安全要求极高、不能访问外网的企业内网

    示例：
        retriever = LocalGitRetriever("/home/user/my-service")
        snippet = retriever.fetch_snippet("src/main/java/com/example/Foo.java", 42)
    """

    def __init__(self, repo_path: str, context_lines: int = 40):
        super().__init__(context_lines)
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"路径 {self.repo_path} 不是有效的 Git 仓库")

    def _git(self, *args: str) -> str:
        """执行 git 命令并返回标准输出"""
        result = subprocess.run(
            ["git", "-C", str(self.repo_path), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git 命令失败: {' '.join(args)}\n{result.stderr.strip()}"
            )
        return result.stdout

    def fetch_snippet(
        self,
        file_path: str,
        line_number: int,
        ref: str = "HEAD",
    ) -> Optional[CodeSnippet]:
        # 用 git show 获取指定 ref 下的文件内容（不依赖工作区状态）
        try:
            content = self._git("show", f"{ref}:{file_path}")
        except RuntimeError:
            # 尝试在工作区直接读取（适配未提交的改动）
            local_path = self.repo_path / file_path
            if not local_path.exists():
                return None
            content = local_path.read_text(encoding="utf-8", errors="replace")

        all_lines = content.splitlines()
        total = len(all_lines)

        start = max(1, line_number - self.context_lines)
        end = min(total, line_number + self.context_lines)

        # 生成带行号的代码文本
        numbered_lines = [
            f"{ln:>6} | {all_lines[ln - 1]}"
            for ln in range(start, end + 1)
        ]

        # 获取 commit sha（仅做记录，失败不影响主流程）
        commit_sha = ""
        try:
            commit_sha = self._git("rev-parse", "--short", ref).strip()
        except RuntimeError:
            pass

        return CodeSnippet(
            file_path=file_path,
            start_line=start,
            end_line=end,
            content="\n".join(numbered_lines),
            commit_sha=commit_sha,
        )


# ---------------------------------------------------------------------------
# 实现 2：GitLab REST API
# ---------------------------------------------------------------------------

class GitLabRetriever(CodeRetriever):
    """
    通过 GitLab REST API 检索代码片段。

    适用场景：
      - 企业私有 GitLab 部署
      - 无法在本地 clone 完整仓库时的远程检索

    示例：
        retriever = GitLabRetriever(
            base_url="https://gitlab.your-company.com",
            project_id="42",          # 或 "group/project-name"
            access_token="glpat-xxx",
        )
    """

    def __init__(
        self,
        base_url: str,
        project_id: str,
        access_token: str = "",
        context_lines: int = 40,
    ):
        super().__init__(context_lines)
        self.base_url = base_url.rstrip("/")
        self.project_id = quote(str(project_id), safe="")
        self.access_token = access_token or os.getenv("GITLAB_TOKEN", "")

    def fetch_snippet(
        self,
        file_path: str,
        line_number: int,
        ref: str = "HEAD",
    ) -> Optional[CodeSnippet]:
        try:
            import urllib.request
            import json

            encoded_path = quote(file_path, safe="")
            url = (
                f"{self.base_url}/api/v4/projects/{self.project_id}"
                f"/repository/files/{encoded_path}/raw?ref={ref}"
            )

            req = urllib.request.Request(url)
            if self.access_token:
                req.add_header("PRIVATE-TOKEN", self.access_token)

            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")

        except Exception as e:
            print(f"[GitLabRetriever] 无法获取 {file_path}: {e}")
            return None

        all_lines = content.splitlines()
        total = len(all_lines)
        start = max(1, line_number - self.context_lines)
        end = min(total, line_number + self.context_lines)

        numbered_lines = [
            f"{ln:>6} | {all_lines[ln - 1]}"
            for ln in range(start, end + 1)
        ]

        return CodeSnippet(
            file_path=file_path,
            start_line=start,
            end_line=end,
            content="\n".join(numbered_lines),
        )


# ---------------------------------------------------------------------------
# 实现 3：GitHub REST API
# ---------------------------------------------------------------------------

class GitHubRetriever(CodeRetriever):
    """
    通过 GitHub REST API 检索代码片段。

    适用场景：
      - 开源项目、公共仓库
      - GitHub Enterprise

    示例：
        retriever = GitHubRetriever(
            owner="my-org",
            repo="my-service",
            token="ghp_xxx",          # 可选，提升 API 速率限制
        )
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str = "",
        context_lines: int = 40,
    ):
        super().__init__(context_lines)
        self.owner = owner
        self.repo = repo
        self.token = token or os.getenv("GITHUB_TOKEN", "")

    def fetch_snippet(
        self,
        file_path: str,
        line_number: int,
        ref: str = "HEAD",
    ) -> Optional[CodeSnippet]:
        try:
            import urllib.request
            import base64
            import json

            url = (
                f"https://api.github.com/repos/{self.owner}/{self.repo}"
                f"/contents/{file_path}?ref={ref}"
            )

            req = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            # GitHub 返回 base64 编码内容
            content = base64.b64decode(data["content"]).decode(
                "utf-8", errors="replace"
            )
            commit_sha = data.get("sha", "")[:7]

        except Exception as e:
            print(f"[GitHubRetriever] 无法获取 {file_path}: {e}")
            return None

        all_lines = content.splitlines()
        total = len(all_lines)
        start = max(1, line_number - self.context_lines)
        end = min(total, line_number + self.context_lines)

        numbered_lines = [
            f"{ln:>6} | {all_lines[ln - 1]}"
            for ln in range(start, end + 1)
        ]

        return CodeSnippet(
            file_path=file_path,
            start_line=start,
            end_line=end,
            content="\n".join(numbered_lines),
            commit_sha=commit_sha,
        )
