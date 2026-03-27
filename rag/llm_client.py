"""
llm_client.py
=============
统一的大模型调用客户端，支持多种本地/云端推理后端。

支持后端：
  1. OllamaClient   — 本地 Ollama 服务（最简单，推荐开发测试用）
  2. VLLMClient     — 本地 vLLM OpenAI 兼容接口（生产环境推荐）
  3. OpenAIClient   — OpenAI / 兼容 API（如 DeepSeek API、Qwen API）

所有客户端实现相同的 chat() 接口，pipeline 无感知切换。

快速启动：
  # Ollama（最简单）
  ollama pull qwen2.5:7b
  ollama serve   # 自动在 11434 端口启动

  # vLLM
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-7B-Instruct \
      --port 8000
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Iterator


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class LLMResponse:
    """大模型返回结果的统一封装"""

    def __init__(self, content: str, model: str = "", usage: dict | None = None):
        self.content = content
        self.model = model
        self.usage = usage or {}

    def __str__(self) -> str:
        return self.content


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class LLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        发送一次对话请求并返回完整回复。

        Parameters
        ----------
        system : str
            系统提示词（角色设定）
        user : str
            用户输入（日志 + 代码 Prompt）
        temperature : float
            采样温度，根因分析推荐 0.1（低随机性）
        max_tokens : int
            最大输出 Token 数

        Returns
        -------
        LLMResponse
        """

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """流式输出（默认实现：一次性返回，子类可覆盖）"""
        response = self.chat(system, user, temperature, max_tokens)
        yield response.content


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    """发送 HTTP POST 请求，返回解析后的 JSON"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} 错误: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接到服务: {url}\n原因: {e.reason}\n"
            "请确认推理服务已启动。"
        ) from e


# ---------------------------------------------------------------------------
# 实现 1：Ollama
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """
    调用本地 Ollama 服务。

    前置条件：
        ollama pull qwen2.5:7b   # 或其他模型
        ollama serve             # 默认监听 localhost:11434

    示例：
        client = OllamaClient(model="qwen2.5:7b")
        resp = client.chat(system="...", user="...")
        print(resp.content)
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        result = _http_post(
            f"{self.base_url}/api/chat",
            payload,
            timeout=300,  # Ollama 首次加载模型可能较慢
        )

        content = result.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens":     result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
        }
        return LLMResponse(content=content, model=self.model, usage=usage)

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# 实现 2：vLLM（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

class VLLMClient(LLMClient):
    """
    调用 vLLM 的 OpenAI 兼容 API。

    前置条件：
        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen2.5-7B-Instruct \\
            --host 0.0.0.0 --port 8000

    示例：
        client = VLLMClient(model="Qwen/Qwen2.5-7B-Instruct")
        resp = client.chat(system="...", user="...")
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        result = _http_post(
            f"{self.base_url}/v1/chat/completions",
            payload,
            headers=headers,
            timeout=180,
        )

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return LLMResponse(content=content, model=self.model, usage=usage)


# ---------------------------------------------------------------------------
# 实现 3：OpenAI 兼容 API（DeepSeek / Qwen API / 其他）
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    """
    调用 OpenAI 兼容 API（DeepSeek、通义千问、Moonshot 等均支持）。

    示例（DeepSeek API）：
        client = OpenAICompatClient(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
        )

    示例（通义千问 API）：
        client = OpenAICompatClient(
            base_url="https://dashscope.aliyuncs.com/compatible-mode",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-max",
        )

    注意：云端 API 会发送日志和代码到外部服务器，企业敏感数据请使用本地部署。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        result = _http_post(
            f"{self.base_url}/v1/chat/completions",
            payload,
            headers=headers,
            timeout=120,
        )

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return LLMResponse(content=content, model=self.model, usage=usage)


# ---------------------------------------------------------------------------
# 工厂函数（根据环境变量自动选择后端）
# ---------------------------------------------------------------------------

def create_client_from_env() -> LLMClient:
    """
    根据环境变量自动创建合适的 LLM 客户端。

    优先级：vLLM > Ollama > OpenAI 兼容 API

    环境变量：
        LLM_BACKEND     : ollama / vllm / openai  （默认 ollama）
        LLM_MODEL       : 模型名称
        LLM_BASE_URL    : 服务地址
        LLM_API_KEY     : API Key（仅云端需要）

    示例：
        export LLM_BACKEND=ollama
        export LLM_MODEL=qwen2.5:7b
        python demo_rag.py
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    model = os.getenv("LLM_MODEL", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")

    if backend == "vllm":
        return VLLMClient(
            model=model or "Qwen/Qwen2.5-7B-Instruct",
            base_url=base_url or "http://localhost:8000",
        )
    elif backend == "openai":
        if not base_url:
            raise ValueError("使用 openai 后端时必须设置 LLM_BASE_URL")
        return OpenAICompatClient(
            model=model or "deepseek-chat",
            base_url=base_url,
            api_key=api_key,
        )
    else:  # 默认 ollama
        return OllamaClient(
            model=model or "qwen2.5:7b",
            base_url=base_url or "http://localhost:11434",
        )
