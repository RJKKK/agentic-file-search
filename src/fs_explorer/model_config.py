"""
Model endpoint configuration helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_TEXT_MODEL = "gpt-4o-mini"
_DEFAULT_VISION_MODEL = "gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class ModelEndpointConfig:
    """Resolved configuration for one model endpoint."""

    model_name: str
    api_key: str | None
    base_url: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model_name)


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def resolve_text_config(
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> ModelEndpointConfig:
    return ModelEndpointConfig(
        model_name=_first_non_empty(
            model_name,
            os.getenv("FS_EXPLORER_TEXT_MODEL"),
            os.getenv("TEXT_MODEL_NAME"),
            _DEFAULT_TEXT_MODEL,
        )
        or _DEFAULT_TEXT_MODEL,
        api_key=_first_non_empty(
            api_key,
            os.getenv("FS_EXPLORER_TEXT_API_KEY"),
            os.getenv("TEXT_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ),
        base_url=_first_non_empty(
            base_url,
            os.getenv("FS_EXPLORER_TEXT_BASE_URL"),
            os.getenv("TEXT_BASE_URL"),
            os.getenv("OPENAI_BASE_URL"),
        ),
    )


def resolve_vision_config(
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> ModelEndpointConfig:
    text_config = resolve_text_config()
    return ModelEndpointConfig(
        model_name=_first_non_empty(
            model_name,
            os.getenv("FS_EXPLORER_VISION_MODEL"),
            os.getenv("VISION_MODEL_NAME"),
            text_config.model_name,
            _DEFAULT_VISION_MODEL,
        )
        or _DEFAULT_VISION_MODEL,
        api_key=_first_non_empty(
            api_key,
            os.getenv("FS_EXPLORER_VISION_API_KEY"),
            os.getenv("VISION_API_KEY"),
            text_config.api_key,
        ),
        base_url=_first_non_empty(
            base_url,
            os.getenv("FS_EXPLORER_VISION_BASE_URL"),
            os.getenv("VISION_BASE_URL"),
            text_config.base_url,
        ),
    )


def resolve_embedding_config(
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> ModelEndpointConfig:
    text_config = resolve_text_config()
    return ModelEndpointConfig(
        model_name=_first_non_empty(
            model_name,
            os.getenv("FS_EXPLORER_EMBEDDING_MODEL"),
            os.getenv("EMBEDDING_MODEL_NAME"),
            _DEFAULT_EMBEDDING_MODEL,
        )
        or _DEFAULT_EMBEDDING_MODEL,
        api_key=_first_non_empty(
            api_key,
            os.getenv("FS_EXPLORER_EMBEDDING_API_KEY"),
            os.getenv("EMBEDDING_API_KEY"),
            text_config.api_key,
        ),
        base_url=_first_non_empty(
            base_url,
            os.getenv("FS_EXPLORER_EMBEDDING_BASE_URL"),
            os.getenv("EMBEDDING_BASE_URL"),
            text_config.base_url,
        ),
    )


def resolve_langextract_config(
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> ModelEndpointConfig:
    text_config = resolve_text_config()
    return ModelEndpointConfig(
        model_name=_first_non_empty(
            model_name,
            os.getenv("FS_EXPLORER_LANGEXTRACT_MODEL"),
            os.getenv("LANGEXTRACT_MODEL_NAME"),
            text_config.model_name,
            _DEFAULT_TEXT_MODEL,
        )
        or _DEFAULT_TEXT_MODEL,
        api_key=_first_non_empty(
            api_key,
            os.getenv("FS_EXPLORER_LANGEXTRACT_API_KEY"),
            os.getenv("LANGEXTRACT_API_KEY"),
            text_config.api_key,
        ),
        base_url=_first_non_empty(
            base_url,
            os.getenv("FS_EXPLORER_LANGEXTRACT_BASE_URL"),
            os.getenv("LANGEXTRACT_BASE_URL"),
            text_config.base_url,
        ),
    )


def configured_text_costs() -> tuple[float, float]:
    """Return optional input/output price hints for reporting."""
    input_cost = float(os.getenv("TEXT_INPUT_COST_PER_MILLION", "0"))
    output_cost = float(os.getenv("TEXT_OUTPUT_COST_PER_MILLION", "0"))
    return input_cost, output_cost
