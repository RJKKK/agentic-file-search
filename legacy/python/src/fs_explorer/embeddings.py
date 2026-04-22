"""
Embedding provider for OpenAI-compatible semantic search.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .model_config import resolve_embedding_config

_DEFAULT_DIM = 768
_DEFAULT_BATCH_SIZE = 50


class EmbeddingProvider:
    """Generate text embeddings via an OpenAI-compatible embeddings API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
        client: Any | None = None,
    ) -> None:
        config = resolve_embedding_config(
            api_key=api_key,
            model_name=model,
            base_url=base_url,
        )
        self.model = config.model_name
        self.base_url = config.base_url
        self.dim = dim or int(
            os.getenv(
                "FS_EXPLORER_EMBEDDING_DIM",
                os.getenv("EMBEDDING_DIM", str(_DEFAULT_DIM)),
            )
        )
        self.batch_size = batch_size or int(
            os.getenv(
                "FS_EXPLORER_EMBEDDING_BATCH_SIZE",
                os.getenv("EMBEDDING_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)),
            )
        )

        if client is not None:
            self._client = client
        else:
            if config.api_key is None:
                raise ValueError(
                    "Embedding model is not configured. Set EMBEDDING_API_KEY "
                    "(or TEXT_API_KEY / OPENAI_API_KEY) and optionally EMBEDDING_BASE_URL."
                )
            client_kwargs = {"api_key": config.api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self._client = OpenAI(**client_kwargs)

    def _embed_with_legacy_client(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dim,
            },
        )
        return [list(emb.values) for emb in result.embeddings]

    def _embed_with_openai_client(self, texts: list[str]) -> list[list[float]]:
        request: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dim:
            request["dimensions"] = self.dim
        response = self._client.embeddings.create(**request)
        return [list(item.embedding) for item in response.data]

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if hasattr(self._client, "models") and hasattr(self._client.models, "embed_content"):
                batch_embeddings = self._embed_with_legacy_client(batch, task_type=task_type)
            else:
                batch_embeddings = self._embed_with_openai_client(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text for retrieval."""
        embeddings = self.embed_texts([query], task_type="RETRIEVAL_QUERY")
        return embeddings[0]
