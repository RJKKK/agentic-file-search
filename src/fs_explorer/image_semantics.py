"""
Helpers for lazy image semantic enhancement.
"""

from __future__ import annotations

import os
from typing import Any

from .document_parsing import ImageSemanticEnhancer

_DEFAULT_VISION_MODEL = "gemini-2.5-flash"


class GeminiImageSemanticEnhancer(ImageSemanticEnhancer):
    """Describe page images using a Gemini multimodal model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "FS_EXPLORER_VISION_MODEL",
            _DEFAULT_VISION_MODEL,
        )

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Provide api_key or set the environment variable."
            )

        try:
            from google.genai import Client as GenAIClient
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ValueError(
                "google-genai is not installed. Run `python -m pip install -e .` first."
            ) from exc

        self._client = GenAIClient(api_key=resolved_key)

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
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                {
                    "inline_data": {
                        "mime_type": mime_type or "image/png",
                        "data": image_bytes,
                    }
                },
            ],
        )
        text = getattr(response, "text", "") or ""
        if text.strip():
            return text.strip(), self.model

        fallback = (
            f"Image {image_index} on page {page_no} of {file_path} was processed, "
            "but the vision model returned an empty description."
        )
        return fallback, self.model


def build_image_semantic_enhancer() -> ImageSemanticEnhancer | None:
    """Return a best-effort image enhancer when runtime dependencies are available."""
    try:
        return GeminiImageSemanticEnhancer()
    except ValueError:
        return None
