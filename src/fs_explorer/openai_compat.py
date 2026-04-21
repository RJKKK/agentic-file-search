"""
Thin compatibility wrapper around OpenAI-compatible chat completions APIs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from google.genai.types import Content, Part
from openai import AsyncOpenAI, OpenAI


def _content_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    parts = getattr(item, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is not None:
            texts.append(str(text))
    return "\n".join(texts).strip()


def _to_messages(contents: Any) -> list[dict[str, Any]]:
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]

    messages: list[dict[str, Any]] = []
    for item in contents or []:
        role = getattr(item, "role", "user") or "user"
        if role == "model":
            role = "assistant"
        text = _content_to_text(item)
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _extract_choice_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content or "")


def _to_compat_response(response: Any) -> Any:
    text = _extract_choice_text(response).strip()
    usage = getattr(response, "usage", None)
    usage_metadata = SimpleNamespace(
        prompt_token_count=getattr(usage, "prompt_tokens", 0) or 0,
        candidates_token_count=getattr(usage, "completion_tokens", 0) or 0,
        total_token_count=getattr(usage, "total_tokens", 0) or 0,
    )
    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                content=Content(role="model", parts=[Part.from_text(text=text)])
            )
        ],
        usage_metadata=usage_metadata,
    )


class _SyncModels:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: Any, config: dict[str, Any] | None = None) -> Any:
        messages = _to_messages(contents)
        effective_config = dict(config or {})
        system_instruction = effective_config.pop("system_instruction", None)
        if system_instruction:
            messages = [{"role": "system", "content": str(system_instruction)}, *messages]
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(effective_config.pop("temperature", 0)),
        }
        if effective_config.get("response_mime_type") == "application/json":
            request["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**request)
        except Exception:
            request.pop("response_format", None)
            response = self._client.chat.completions.create(**request)
        return _to_compat_response(response)


class _AsyncModels:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: dict[str, Any] | None = None,
    ) -> Any:
        messages = _to_messages(contents)
        effective_config = dict(config or {})
        system_instruction = effective_config.pop("system_instruction", None)
        if system_instruction:
            messages = [{"role": "system", "content": str(system_instruction)}, *messages]
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(effective_config.pop("temperature", 0)),
        }
        if effective_config.get("response_mime_type") == "application/json":
            request["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception:
            request.pop("response_format", None)
            response = await self._client.chat.completions.create(**request)
        return _to_compat_response(response)


class OpenAICompatClient:
    """Expose a minimal `.models` and `.aio.models` surface compatible with existing code."""

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.models = _SyncModels(OpenAI(**client_kwargs))
        self.aio = SimpleNamespace(models=_AsyncModels(AsyncOpenAI(**client_kwargs)))
