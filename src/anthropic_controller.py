"""Claude Messages API adapter for the starter agent loop.

The API key is read from the ANTHROPIC_API_KEY environment variable, or from a
.env file in the project root (the file is git-ignored). The key is never printed
or written to traces.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.ollama_controller import _SYSTEM_PROMPT

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-5-5"

_TOOLS = [
    {
        "name": "search",
        "description": "Retrieve relevant chunks for a question or unresolved detail.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "final_answer",
        "description": "Return an evidence-grounded answer or abstain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "citations"],
        },
    },
]


def load_api_key(env_file: Path = Path(".env")) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "ANTHROPIC_API_KEY":
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("Thiếu ANTHROPIC_API_KEY: đặt trong biến môi trường hoặc file .env")
    return key


class AnthropicController:
    def __init__(self, *, model: str = DEFAULT_MODEL, max_tokens: int = 512,
                 timeout: float = 120, api_key: str | None = None):
        if not model.strip() or max_tokens < 1 or timeout <= 0:
            raise ValueError("model phải khác rỗng; max_tokens và timeout phải lớn hơn 0")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key or load_api_key()

    def _post(self, request: urllib.request.Request, attempts: int = 3) -> dict[str, Any]:
        """Retry transient network errors; HTTP errors are reported by status only."""
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                # Do not retry or echo the body: it can contain request details.
                raise RuntimeError(f"Claude API lỗi HTTP {error.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == attempts:
                    raise
                time.sleep(5 * attempt)
        raise RuntimeError("unreachable")

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = [f"Question: {question}"]
        if history:
            previous = history[-1]
            prompt.append("Searches already performed: " + json.dumps(
                previous.get("queries", []), ensure_ascii=False))
            prompt.append("Available evidence (cite the bracketed chunk IDs):\n" +
                          str(previous.get("context", "")))
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "tools": _TOOLS,
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": "\n\n".join(prompt)}],
        }).encode("utf-8")
        request = urllib.request.Request(API_URL, data=payload, method="POST", headers={
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
        })
        data = self._post(request)

        calls = [block for block in data.get("content", []) if block.get("type") == "tool_use"]
        if len(calls) != 1:
            raise ValueError("Claude phải trả đúng một tool_use search hoặc final_answer")
        name, arguments = calls[0].get("name"), calls[0].get("input")
        if not isinstance(arguments, dict):
            raise ValueError("Claude tool_use input phải là object")
        if name == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Claude search query phải là chuỗi không rỗng")
            return {"action": "search", "query": query}
        if name == "final_answer":
            answer, citations = arguments.get("answer"), arguments.get("citations")
            if (not isinstance(answer, str) or not isinstance(citations, list)
                    or any(not isinstance(item, str) for item in citations)):
                raise ValueError("Claude final_answer arguments không hợp lệ")
            return {"action": "answer", "answer": answer, "citations": citations}
        raise ValueError(f"Claude trả về tool không được hỗ trợ: {name}")
