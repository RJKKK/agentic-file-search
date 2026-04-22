"""
Helpers for lazy image semantic enhancement.
"""

from __future__ import annotations

import base64
from typing import Any

from openai import OpenAI

from .document_parsing import ImageSemanticEnhancer
from .model_config import resolve_vision_config


class OpenAIImageSemanticEnhancer(ImageSemanticEnhancer):
    """Describe page images using an OpenAI-compatible multimodal model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        config = resolve_vision_config(
            api_key=api_key,
            model_name=model,
            base_url=base_url,
        )
        if not config.api_key and client is None:
            raise ValueError(
                "Vision model is not configured. Set VISION_API_KEY "
                "(or TEXT_API_KEY / OPENAI_API_KEY) and optionally VISION_BASE_URL."
            )
        self.model = config.model_name

        if client is not None:
            self._client = client
            return

        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = OpenAI(**client_kwargs)

    def describe_image(
        self,
        *,
        file_path: str,
        page_no: int,
        image_index: int,
        image_bytes: bytes,
        mime_type: str | None,
    ) -> tuple[str, str | None]:
        prompt = (
            "Describe this document image for retrieval and QA. "
            "Focus on charts, tables, labels, legends, axes, and any text visible. "
            "Keep it concise but specific."
        )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        media_type = mime_type or "image/png"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if isinstance(content, str):
            text = content.strip()
        else:
            chunks: list[str] = []
            for item in content or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                else:
                    item_text = getattr(item, "text", None)
                    if item_text is not None:
                        chunks.append(str(item_text))
            text = "\n".join(chunk for chunk in chunks if chunk).strip()

        if text:
            return text, self.model

        fallback = (
            f"Image {image_index} on page {page_no} of {file_path} was processed, "
            "but the vision model returned an empty description."
        )
        return fallback, self.model


def build_image_semantic_enhancer() -> ImageSemanticEnhancer | None:
    """Return a best-effort image enhancer when runtime dependencies are available."""
    try:
        return OpenAIImageSemanticEnhancer()
    except ValueError:
        return None
