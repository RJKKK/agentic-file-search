"""
FsExplorer Agent for filesystem exploration using an OpenAI-compatible text model.
"""

import contextvars
import json
import os
import re
from pathlib import Path
from typing import Callable, Any, cast, get_args
from dataclasses import dataclass

from dotenv import load_dotenv
from google.genai.types import Content, Part
from pydantic import ValidationError

from .context_budget import ContextBudgetManager
from .context_state import ContextState, compress_unit_ranges, render_ranges
from .models import Action, ActionType, StopAction, ToolCallAction, Tools
from .fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)
from .blob_store import LocalBlobStore
from .document_pages import find_page_by_path, load_document_pages, resolve_pages_directory
from .embeddings import EmbeddingProvider
from .document_parsing import PARSER_VERSION, ParseSelector, enhance_page_image_semantics, reconstruct_parsed_document
from .document_cache import format_parse_result, get_or_parse_document_units, resolve_document_by_path
from .document_library import LIBRARY_CORPUS_ROOT
from .image_semantics import build_image_semantic_enhancer
from .index_config import resolve_db_path
from .model_config import configured_text_costs, resolve_text_config
from .openai_compat import OpenAICompatClient
from .search import (
    IndexedQueryEngine,
    MetadataFilterParseError,
    SearchHit,
    supported_filter_syntax,
)
from .storage import PostgresStorage

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


_PAGE_BLOB_STORE = LocalBlobStore()


# =============================================================================
# Token Usage Tracking
# =============================================================================

# Optional text model pricing hints (per million tokens)
TEXT_INPUT_COST_PER_MILLION, TEXT_OUTPUT_COST_PER_MILLION = configured_text_costs()


@dataclass
class TokenUsage:
    """
    Track token usage and costs across the session.

    Maintains running totals of API calls, token counts, and provides
    cost estimates based on configured text-model pricing hints.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    # Track content sizes
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    def add_api_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        """Record metrics from a tool execution."""
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            # Count documents in scan result by counting document markers
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate estimated costs based on configured pricing hints."""
        input_cost = (
            self.prompt_tokens / 1_000_000
        ) * TEXT_INPUT_COST_PER_MILLION
        output_cost = (
            self.completion_tokens / 1_000_000
        ) * TEXT_OUTPUT_COST_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    def summary(self) -> str:
        """Generate a formatted summary of token usage and costs."""
        input_cost, output_cost, total_cost = self._calculate_cost()

        return f"""
═══════════════════════════════════════════════════════════════
                      TOKEN USAGE SUMMARY
═══════════════════════════════════════════════════════════════
  API Calls:           {self.api_calls}
  Prompt Tokens:       {self.prompt_tokens:,}
  Completion Tokens:   {self.completion_tokens:,}
  Total Tokens:        {self.total_tokens:,}
───────────────────────────────────────────────────────────────
  Documents Scanned:   {self.documents_scanned}
  Documents Parsed:    {self.documents_parsed}
  Tool Result Chars:   {self.tool_result_chars:,}
───────────────────────────────────────────────────────────────
  Est. Cost:
    Input:  ${input_cost:.4f}
    Output: ${output_cost:.4f}
    Total:  ${total_cost:.4f}
═══════════════════════════════════════════════════════════════
"""


# =============================================================================
# Tool Registry
# =============================================================================


@dataclass(frozen=True)
class IndexContext:
    """Execution context for indexed retrieval tools."""

    db_path: str
    root_folder: str | None = None
    document_ids: tuple[str, ...] = ()
    scope_label: str | None = None


RuntimeEventCallback = Callable[[str, dict[str, Any]], None]


_INDEX_CONTEXT_VAR: contextvars.ContextVar[IndexContext | None] = contextvars.ContextVar(
    "_INDEX_CONTEXT_VAR", default=None
)
_EMBEDDING_PROVIDER_VAR: contextvars.ContextVar[EmbeddingProvider | None] = (
    contextvars.ContextVar("_EMBEDDING_PROVIDER_VAR", default=None)
)
_IMAGE_SEMANTIC_ENHANCER_VAR: contextvars.ContextVar[Any | None] = (
    contextvars.ContextVar("_IMAGE_SEMANTIC_ENHANCER_VAR", default=None)
)
_FIELD_CATALOG_SHOWN_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_FIELD_CATALOG_SHOWN_VAR", default=False
)
_ENABLE_SEMANTIC_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENABLE_SEMANTIC_VAR", default=False
)
_ENABLE_METADATA_VAR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ENABLE_METADATA_VAR", default=False
)
_RUNTIME_EVENT_CALLBACK_VAR: contextvars.ContextVar[RuntimeEventCallback | None] = (
    contextvars.ContextVar("_RUNTIME_EVENT_CALLBACK_VAR", default=None)
)
_LAST_FOCUS_ANCHOR_VAR: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("_LAST_FOCUS_ANCHOR_VAR", default=None)
)
_CONTEXT_BUDGET_STATS_VAR: contextvars.ContextVar[dict[str, Any]] = (
    contextvars.ContextVar("_CONTEXT_BUDGET_STATS_VAR", default={})
)
_CONTEXT_STATE_VAR: contextvars.ContextVar[ContextState | None] = (
    contextvars.ContextVar("_CONTEXT_STATE_VAR", default=None)
)
_LAZY_INDEXING_STATS_VAR: contextvars.ContextVar[dict[str, Any]] = (
    contextvars.ContextVar(
        "_LAZY_INDEXING_STATS_VAR",
        default={
            "triggered": False,
            "indexed_documents": 0,
            "chunks_written": 0,
            "embeddings_written": 0,
        },
    )
)


def set_search_flags(
    *, enable_semantic: bool = False, enable_metadata: bool = False
) -> None:
    """Configure which indexed retrieval paths are active."""
    _ENABLE_SEMANTIC_VAR.set(enable_semantic)
    _ENABLE_METADATA_VAR.set(enable_metadata)


def get_search_flags() -> tuple[bool, bool]:
    """Return (enable_semantic, enable_metadata)."""
    return _ENABLE_SEMANTIC_VAR.get(), _ENABLE_METADATA_VAR.get()


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    """Set the embedding provider for vector search in indexed tools."""
    _EMBEDDING_PROVIDER_VAR.set(provider)


def set_image_semantic_enhancer(enhancer) -> None:
    """Override the lazy image semantic enhancer, primarily for tests."""
    _IMAGE_SEMANTIC_ENHANCER_VAR.set(enhancer)


def set_runtime_event_callback(callback: RuntimeEventCallback | None) -> None:
    """Register an optional callback for runtime cache/enhancement events."""
    _RUNTIME_EVENT_CALLBACK_VAR.set(callback)


def initialize_context_state(
    *,
    task: str,
    documents: list[dict[str, str]],
    collection_name: str | None = None,
) -> None:
    """Initialize the structured evidence context for the active session."""
    state = ContextState(task=task, collection_name=collection_name)
    state.register_documents(documents)
    _CONTEXT_STATE_VAR.set(state)


def get_context_state() -> ContextState | None:
    """Return the active structured context state."""
    return _CONTEXT_STATE_VAR.get()


def get_context_state_snapshot() -> dict[str, Any]:
    """Return a JSON-friendly snapshot of the active structured context state."""
    state = _CONTEXT_STATE_VAR.get()
    if state is None:
        return {
            "context_scope": {
                "active_document_id": None,
                "active_file_path": None,
                "active_ranges": [],
            },
            "coverage_by_document": {},
            "working_summary": [],
            "open_gaps": [],
            "compaction_actions": [],
            "promoted_evidence_units": [],
            "evidence_units": [],
        }
    return state.snapshot()


def set_index_context(
    folder: str | None = None,
    db_path: str | None = None,
    document_ids: list[str] | None = None,
    scope_label: str | None = None,
) -> None:
    """Enable indexed tools for a folder corpus or a selected document scope."""
    _INDEX_CONTEXT_VAR.set(
        IndexContext(
            db_path=resolve_db_path(db_path),
            root_folder=str(Path(folder).resolve()) if folder else None,
            document_ids=tuple(document_ids or ()),
            scope_label=scope_label,
        )
    )
    # Auto-create embedding provider if API key available
    if _EMBEDDING_PROVIDER_VAR.get() is None:
        try:
            _EMBEDDING_PROVIDER_VAR.set(EmbeddingProvider())
        except ValueError:
            pass
    if _IMAGE_SEMANTIC_ENHANCER_VAR.get() is None:
        _IMAGE_SEMANTIC_ENHANCER_VAR.set(build_image_semantic_enhancer())


def clear_index_context() -> None:
    """Disable indexed tools for the current process."""
    _INDEX_CONTEXT_VAR.set(None)
    _EMBEDDING_PROVIDER_VAR.set(None)
    _IMAGE_SEMANTIC_ENHANCER_VAR.set(None)
    _RUNTIME_EVENT_CALLBACK_VAR.set(None)
    _LAST_FOCUS_ANCHOR_VAR.set(None)
    _CONTEXT_BUDGET_STATS_VAR.set({})
    _CONTEXT_STATE_VAR.set(None)
    _LAZY_INDEXING_STATS_VAR.set(
        {
            "triggered": False,
            "indexed_documents": 0,
            "chunks_written": 0,
            "embeddings_written": 0,
        }
    )
    _FIELD_CATALOG_SHOWN_VAR.set(False)
    _ENABLE_SEMANTIC_VAR.set(False)
    _ENABLE_METADATA_VAR.set(False)


def _get_index_storage_and_corpus() -> tuple[
    PostgresStorage | None, str | None, str | None
]:
    index_context = _INDEX_CONTEXT_VAR.get()
    if index_context is None:
        return None, None, "Index context is not configured."

    storage = PostgresStorage(index_context.db_path)
    corpus_root = index_context.root_folder or LIBRARY_CORPUS_ROOT
    corpus_id = storage.get_corpus_id(corpus_root)
    if corpus_id is None:
        return (
            None,
            None,
            "No index found for the active document scope.",
        )
    return storage, corpus_id, None


def _clean_excerpt(text: str, max_chars: int = 320) -> str:
    squashed = re.sub(r"\s+", " ", text).strip()
    if len(squashed) <= max_chars:
        return squashed
    return f"{squashed[:max_chars]}..."


def get_last_focus_anchor() -> dict[str, Any] | None:
    """Return the latest anchor inferred from retrieval hits."""
    anchor = _LAST_FOCUS_ANCHOR_VAR.get()
    if anchor is None:
        return None
    return dict(anchor)


def _set_last_focus_anchor(anchor: dict[str, Any] | None) -> None:
    """Persist the latest focus anchor metadata for the current session."""
    _LAST_FOCUS_ANCHOR_VAR.set(dict(anchor) if anchor is not None else None)


def get_lazy_indexing_stats() -> dict[str, Any]:
    """Return latest lazy-indexing stats for the active session."""
    return dict(_LAZY_INDEXING_STATS_VAR.get())


def get_context_budget_stats() -> dict[str, Any]:
    """Return stats captured during the most recent context compaction."""
    return dict(_CONTEXT_BUDGET_STATS_VAR.get())


def _normalize_lookup_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _emit_runtime_event(event_type: str, **data: Any) -> None:
    """Fan out an optional runtime event if a callback is configured."""
    callback = _RUNTIME_EVENT_CALLBACK_VAR.get()
    if callback is not None:
        callback(event_type, data)


def _unit_to_context_dict(unit: Any) -> dict[str, Any]:
    return {
        "unit_no": int(unit.unit_no),
        "source_locator": unit.source_locator,
        "heading": unit.heading,
        "markdown": unit.markdown,
    }


def _build_parse_receipt(
    *,
    file_path: str,
    summary: dict[str, Any],
) -> str:
    returned_ranges = render_ranges(list(summary.get("returned_ranges") or []))
    total_units = summary.get("total_units")
    return (
        "Parse receipt: "
        f"path={file_path}; returned={returned_ranges}; "
        f"new_units={summary.get('new_units_added', 0)}; "
        f"total_units={total_units if total_units is not None else '?'}; "
        "structured evidence has been stored for the next reasoning step."
    )


def _scoped_documents(
    storage: PostgresStorage,
    index_context: IndexContext,
) -> list[dict[str, Any]]:
    if index_context.document_ids:
        return storage.list_documents_by_ids(
            doc_ids=list(index_context.document_ids),
            include_deleted=False,
        )
    corpus_id = storage.get_corpus_id(index_context.root_folder or LIBRARY_CORPUS_ROOT) or ""
    return storage.list_documents(corpus_id=corpus_id, include_deleted=False)


def _resolve_document_page_scope(
    *,
    storage: PostgresStorage,
    index_context: IndexContext,
    target: str,
) -> tuple[dict[str, Any], str] | None:
    normalized_target = str(Path(target).resolve()).lower()
    for document in _scoped_documents(storage, index_context):
        pages_prefix = str(document.get("pages_prefix") or "")
        if not pages_prefix:
            continue
        pages_dir = resolve_pages_directory(blob_store=_PAGE_BLOB_STORE, pages_prefix=pages_prefix)
        source_path = str(Path(str(document["absolute_path"])).resolve()).lower()
        if normalized_target in {
            source_path,
            str(Path(pages_dir).resolve()).lower(),
        }:
            return document, pages_dir
    return None


def _render_search_output(
    *,
    query: str,
    filters: str | None,
    hits: list[SearchHit],
    enhanced_images: int,
    cached_image_pages: int,
    lazy_indexing_stats: dict[str, Any],
    storage: PostgresStorage,
    corpus_id: str,
) -> str:
    if not hits:
        if filters:
            return f"No indexed matches found for query={query!r} with filters={filters!r}."
        return f"No indexed matches found for query: {query!r}"

    lines = [
        "=== INDEXED SEARCH RESULTS ===",
        f"Query: {query}",
    ]
    if filters:
        lines.append(f"Filters: {filters}")
    lines.append("")
    for idx, hit in enumerate(hits, start=1):
        position = hit.position if hit.position is not None else "<metadata>"
        lines.extend(
            [
                f"[{idx}] doc_id: {hit.doc_id}",
                f"    path: {hit.absolute_path}",
                f"    match: {hit.matched_by}",
                f"    chunk_position: {position}",
                f"    source_unit_no: {hit.source_unit_no}",
                f"    semantic_score: {hit.semantic_score}",
                f"    metadata_score: {hit.metadata_score}",
                f"    score: {hit.score:.2f}",
                f"    excerpt: {_clean_excerpt(hit.text)}",
                "",
            ]
        )
    lines.append(
        "Use get_document(doc_id=...) to read full content for the most relevant documents."
    )
    if enhanced_images or cached_image_pages:
        lines.append("")
        lines.append(
            "Image semantics: "
            f"enhanced {enhanced_images} images, reused cached semantics for {cached_image_pages} pages."
        )
    if bool(lazy_indexing_stats.get("triggered")):
        lines.append("")
        lines.append(
            "Lazy indexing: "
            f"indexed {lazy_indexing_stats.get('indexed_documents', 0)} documents, "
            f"wrote {lazy_indexing_stats.get('chunks_written', 0)} chunks."
        )

    if not _FIELD_CATALOG_SHOWN_VAR.get():
        active_schema = storage.get_active_schema(corpus_id=corpus_id)
        if active_schema is not None:
            schema_fields = active_schema.schema_def.get("fields")
            if isinstance(schema_fields, list) and schema_fields:
                field_names = [
                    str(f["name"])
                    for f in schema_fields
                    if isinstance(f, dict) and isinstance(f.get("name"), str)
                ]
                field_values = storage.get_metadata_field_values(
                    corpus_id=corpus_id,
                    field_names=field_names,
                )
                field_descs: list[str] = []
                for field in schema_fields:
                    if not isinstance(field, dict) or not isinstance(
                        field.get("name"), str
                    ):
                        continue
                    name = field["name"]
                    ftype = field.get("type", "string")
                    desc = field.get("description", "")
                    entry = f"{name} ({ftype})"
                    if desc:
                        entry += f": {desc}"
                    vals = field_values.get(name, [])
                    if ftype == "boolean":
                        entry += " Values: true, false"
                    elif ftype in {"integer", "number"} and vals:
                        nums = []
                        for value in vals:
                            try:
                                nums.append(float(value))
                            except (TypeError, ValueError):
                                pass
                        if nums:
                            entry += f" Range: {min(nums):.6g}-{max(nums):.6g}"
                    elif vals:
                        if "enum" in field:
                            entry += f" Values: {field['enum']}"
                        else:
                            entry += f" Values: {', '.join(repr(v) for v in vals)}"
                    elif "enum" in field:
                        entry += f" Values: {field['enum']}"
                    field_descs.append(entry)
                if field_descs:
                    lines.append("")
                    lines.append(
                        "Available filter fields for semantic_search(filters=...):"
                    )
                    for desc in field_descs:
                        lines.append(f"  - {desc}")
                _FIELD_CATALOG_SHOWN_VAR.set(True)
    return "\n".join(lines)


def _pending_image_hashes_for_unit(
    storage: PostgresStorage,
    unit: dict[str, Any],
) -> list[str]:
    images = unit.get("images")
    if not isinstance(images, list) or not images:
        return []
    image_hashes = [
        str(image["image_hash"])
        for image in images
        if isinstance(image, dict) and isinstance(image.get("image_hash"), str)
    ]
    if not image_hashes:
        return []
    semantics = storage.get_image_semantics(image_hashes=image_hashes)
    return [
        image_hash
        for image_hash in image_hashes
        if image_hash not in semantics or not semantics[image_hash].get("semantic_text")
    ]


def _select_relevant_page_numbers(
    units: list[dict[str, Any]],
    excerpt: str,
    *,
    max_pages: int = 2,
) -> list[int]:
    if not units:
        return []

    excerpt_text = _normalize_lookup_text(excerpt)
    if not excerpt_text:
        return [int(units[0]["page_no"])]

    excerpt_terms = [
        term
        for term in re.findall(r"[a-z0-9]{4,}", excerpt_text)
        if term not in {"with", "from", "that", "this"}
    ]

    scored: list[tuple[int, int]] = []
    for unit in units:
        page_text = _normalize_lookup_text(str(unit.get("markdown", "")))
        score = 0
        if excerpt_text and excerpt_text in page_text:
            score += 100
        score += sum(1 for term in excerpt_terms if term in page_text)
        if score > 0:
            scored.append((int(unit["page_no"]), score))

    if scored:
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [page_no for page_no, _ in scored[:max_pages]]

    if len(units) == 1:
        return [int(units[0]["page_no"])]
    return []


def _maybe_enhance_images_for_hits(
    storage: PostgresStorage,
    hits,
) -> tuple[int, int]:
    enhancer = _IMAGE_SEMANTIC_ENHANCER_VAR.get()
    if enhancer is None:
        return 0, 0

    enhanced_images = 0
    cached_pages = 0
    seen_pages: set[tuple[str, int]] = set()

    for hit in hits:
        units = storage.list_parsed_units(
            document_id=hit.doc_id,
            parser_version=PARSER_VERSION,
        )
        for page_no in _select_relevant_page_numbers(units, hit.text):
            page_key = (hit.doc_id, page_no)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            unit = next(
                (candidate for candidate in units if int(candidate["page_no"]) == page_no),
                None,
            )
            if unit is None:
                continue
            pending_hashes = _pending_image_hashes_for_unit(storage, unit)
            if pending_hashes:
                _emit_runtime_event(
                    "image_enhance_started",
                    doc_id=hit.doc_id,
                    absolute_path=hit.absolute_path,
                    page_no=page_no,
                    image_hashes=pending_hashes,
                    pending_images=len(pending_hashes),
                )
                page_enhanced = enhance_page_image_semantics(
                    storage=storage,
                    document_id=hit.doc_id,
                    page_no=page_no,
                    enhancer=enhancer,
                )
                enhanced_images += page_enhanced
                _emit_runtime_event(
                    "image_enhance_done",
                    doc_id=hit.doc_id,
                    absolute_path=hit.absolute_path,
                    page_no=page_no,
                    image_hashes=pending_hashes,
                    enhanced_images=page_enhanced,
                )
            elif unit.get("images"):
                cached_pages += 1
                _emit_runtime_event(
                    "cache_hit",
                    cache_kind="image_semantics",
                    doc_id=hit.doc_id,
                    absolute_path=hit.absolute_path,
                    page_no=page_no,
                    cached_images=len(unit["images"]),
                )

    return enhanced_images, cached_pages


def _format_cached_image_semantics(
    storage: PostgresStorage,
    *,
    document_id: str,
) -> str:
    units = storage.list_parsed_units(
        document_id=document_id,
        parser_version=PARSER_VERSION,
    )
    if not units:
        return ""

    lines: list[str] = []
    for unit in units:
        images = unit.get("images")
        if not isinstance(images, list) or not images:
            continue
        image_hashes = [
            str(image["image_hash"])
            for image in images
            if isinstance(image, dict) and isinstance(image.get("image_hash"), str)
        ]
        semantics = storage.get_image_semantics(image_hashes=image_hashes)
        page_entries: list[str] = []
        for image in images:
            if not isinstance(image, dict):
                continue
            image_hash = image.get("image_hash")
            if not isinstance(image_hash, str):
                continue
            semantic = semantics.get(image_hash)
            if semantic is None or not semantic.get("semantic_text"):
                continue
            page_entries.append(
                f"- image {image.get('image_index', '?')}: {semantic['semantic_text']}"
            )
        if page_entries:
            lines.append(f"Page {unit['page_no']}:")
            lines.extend(page_entries)

    if not lines:
        return ""
    return "\n\nImage semantics cache:\n" + "\n".join(lines)


def _run_semantic_search(
    query: str,
    filters: str | None = None,
    limit: int = 5,
) -> tuple[str, list[SearchHit], dict[str, Any]]:
    """Search indexed chunks and return both rendered output and structured hits."""
    storage, corpus_id, error = _get_index_storage_and_corpus()
    if error:
        return error, [], {}
    assert storage is not None and corpus_id is not None

    engine = IndexedQueryEngine(
        storage,
        embedding_provider=_EMBEDDING_PROVIDER_VAR.get(),
        runtime_event_callback=_RUNTIME_EVENT_CALLBACK_VAR.get(),
    )
    enable_semantic, enable_metadata = get_search_flags()
    try:
        hits = engine.search(
            corpus_id=corpus_id,
            query=query,
            document_ids=list(_INDEX_CONTEXT_VAR.get().document_ids) if _INDEX_CONTEXT_VAR.get() else None,
            filters=filters,
            limit=limit,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
        )
    except MetadataFilterParseError as exc:
        return f"Invalid metadata filter: {exc}\n{supported_filter_syntax()}", [], {}
    except ValueError as exc:
        return f"Metadata filter error: {exc}", [], {}

    _LAZY_INDEXING_STATS_VAR.set(engine.get_last_lazy_indexing_stats())

    if not hits:
        _set_last_focus_anchor(None)
        rendered = (
            f"No indexed matches found for query={query!r} with filters={filters!r}."
            if filters
            else f"No indexed matches found for query: {query!r}"
        )
        return rendered, [], {"lazy_indexing": engine.get_last_lazy_indexing_stats()}

    first_hit = hits[0]
    if first_hit.source_unit_no is not None:
        _set_last_focus_anchor(
            {
                "doc_id": first_hit.doc_id,
                "absolute_path": first_hit.absolute_path,
                "source_unit_no": first_hit.source_unit_no,
                "query": query,
                "auto_inject_allowed": True,
            }
        )

    enhanced_images, cached_image_pages = _maybe_enhance_images_for_hits(storage, hits)
    lazy_indexing_stats = engine.get_last_lazy_indexing_stats()
    rendered = _render_search_output(
        query=query,
        filters=filters,
        hits=hits,
        enhanced_images=enhanced_images,
        cached_image_pages=cached_image_pages,
        lazy_indexing_stats=lazy_indexing_stats,
        storage=storage,
        corpus_id=corpus_id,
    )
    return rendered, hits, {"lazy_indexing": lazy_indexing_stats}


def semantic_search(query: str, filters: str | None = None, limit: int = 5) -> str:
    """Search indexed chunks and return ranked excerpts."""
    rendered, _, _ = _run_semantic_search(query=query, filters=filters, limit=limit)
    return rendered


def _run_get_document(
    doc_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return full document content by id plus structured metadata."""
    storage, _, error = _get_index_storage_and_corpus()
    if error:
        return error, None
    assert storage is not None

    document = storage.get_document(doc_id=doc_id)
    if document is None:
        return f"No indexed document found for doc_id={doc_id!r}", None
    index_context = _INDEX_CONTEXT_VAR.get()
    if (
        index_context is not None
        and index_context.document_ids
        and doc_id not in set(index_context.document_ids)
    ):
        return f"Document {doc_id} is outside the currently selected document scope.", None
    if document["is_deleted"]:
        return f"Document {doc_id} is marked as deleted in the index.", None

    semantic_section = _format_cached_image_semantics(storage, document_id=doc_id)
    document_body = str(document.get("content") or "")
    if not document_body.strip():
        cached_units = storage.list_parsed_units(
            document_id=doc_id,
            parser_version=PARSER_VERSION,
        )
        cached_document = reconstruct_parsed_document(cached_units)
        if cached_document is not None:
            document_body = cached_document.markdown

    rendered = (
        f"=== DOCUMENT {doc_id} ===\n"
        f"Path: {document['absolute_path']}\n\n"
        f"{document_body}"
        f"{semantic_section}"
    )
    return rendered, {
        "document_id": doc_id,
        "absolute_path": str(document["absolute_path"]),
        "label": str(document.get("original_filename") or document["relative_path"] or doc_id),
        "content": document_body,
    }


def get_document(doc_id: str) -> str:
    """Return full document content by id from the active index context."""
    rendered, _ = _run_get_document(doc_id)
    return rendered


def list_indexed_documents() -> str:
    """List indexed documents for the active corpus."""
    storage, corpus_id, error = _get_index_storage_and_corpus()
    if error:
        return error
    assert storage is not None and corpus_id is not None

    index_context = _INDEX_CONTEXT_VAR.get()
    if index_context is not None and index_context.document_ids:
        documents = storage.list_documents_by_ids(
            doc_ids=list(index_context.document_ids),
            include_deleted=False,
        )
    else:
        documents = storage.list_documents(corpus_id=corpus_id, include_deleted=False)
    if not documents:
        return "No indexed documents found for the active corpus."

    lines = ["=== INDEXED DOCUMENTS ==="]
    if index_context is not None and index_context.scope_label:
        lines.append(f"Scope: {index_context.scope_label}")
    for idx, document in enumerate(documents, start=1):
        pages_dir = resolve_pages_directory(
            blob_store=_PAGE_BLOB_STORE,
            pages_prefix=str(document.get("pages_prefix") or ""),
        )
        lines.append(
            f"[{idx}] doc_id={document['id']} source={document['absolute_path']} "
            f"pages_dir={pages_dir} page_count={int(document.get('page_count') or 0)} "
            f"name={document.get('original_filename') or document['relative_path']}"
        )
    lines.append("")
    lines.append("Use glob/grep/read on the pages_dir to answer questions page-by-page.")
    return "\n".join(lines)


TOOLS: dict[Tools, Callable[..., str]] = {
    "read": read_file,
    "grep": grep_file_content,
    "glob": glob_paths,
    "scan_folder": scan_folder,
    "preview_file": preview_file,
    "parse_file": parse_file,
    "semantic_search": semantic_search,
    "get_document": get_document,
    "list_indexed_documents": list_indexed_documents,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that explores filesystems to answer user questions about documents.

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `glob` | List page files for one selected document pages directory | `directory`, `pattern` |
| `grep` | Find candidate pages by searching page markdown files | `file_path`, `pattern` |
| `read` | Read one page markdown file with page metadata | `file_path` |
| `parse_file` | Rebuild or inspect a source document when maintenance is needed | `file_path`, `focus_hint`, `anchor`, `window`, `max_units` |
| `scan_folder` | Legacy broad scan; avoid in normal selected-document QA | `directory` |
| `preview_file` | Legacy preview tool; avoid when page files exist | `file_path` |
| `semantic_search` | Legacy indexed retrieval; not part of the main QA path | `query`, `filters`, `limit` |
| `get_document` | Legacy full-document read; not part of the main QA path | `doc_id` |
| `list_indexed_documents` | Legacy document list | none |

## Page Retrieval Strategy

The main QA path is page-first, not full-document reading:
1. Use `glob` on the selected document's `pages_dir` or source path to see available `page-XXXX.md` files.
2. Use `grep` on that document scope to find candidate pages for the user question.
3. Use `read` on only a few candidate page files.
4. If a candidate page looks incomplete, include BOTH the previous page and the next page as candidate pages before answering.
5. If the first pages are insufficient, change the query or switch to new pages. Do not keep rereading the same page range.
6. Use `parse_file` only for rebuild/debug scenarios, not as the normal answering tool.

## Structured Context Rules

Tool results may be stored as structured evidence receipts instead of full raw text.
You will also receive a "STRUCTURED CONTEXT PACK" that summarizes:
- which documents and unit ranges are already covered
- which evidence units are currently active
- which older ranges were summarized
- which gaps remain unresolved

The context pack shows what pages have already been searched or read; it is not a hard search boundary.
When the current active page range is marked stale or does not directly support the answer, do not keep reading the same pages.
Change strategy by running a fresh page search, switching pages, switching documents, or stopping for insufficient evidence.

You may optionally include `context_plan` in your JSON action to suggest how the runtime should manage context.
Use it only when it helps avoid repeated reads or promotes especially relevant evidence.

## Three-Phase Page Exploration Strategy

### PHASE 1: Scope Pages (PARALLEL PAGE LIST)
1. Start with `glob` for the selected document.
2. Identify the page directory and rough page range.

### PHASE 2: Find Candidate Pages
1. Use `grep` with a focused phrase derived from the user question.
2. Prefer narrow, content-bearing terms over the full question.
3. In your **reason**, say which candidate pages look most promising.
4. When the best page may be a continuation page, the start of a table, the end of a table, or otherwise incomplete, add its previous and next page to your candidate set. Example: if page 33 contains the matching table but the row may continue, consider pages 32, 33, and 34.

### PHASE 3: Read Only What You Need
1. Use `read` on a few candidate page files.
2. If the answer is incomplete or appears cut off by a page boundary, read the previous and next page for that evidence page before concluding.
3. Only treat a tool call as repeated when both the tool name and all parameters are identical. Reading different pages with `read` is valid progress, not repetition.
4. If repeated identical reads add no new evidence, BACKTRACK by changing the query, changing pages, or changing documents.
5. If no trustworthy evidence appears after backtracking, provide a best-effort answer from existing evidence and clearly state what remains uncertain. Do not return a generic tool-loop failure message.

## Providing Detailed Reasoning

Your `reason` field is displayed to the user, so make it informative:
- After `glob`: Explain the page range and what you plan to search next
- After `grep`: Explain which candidate pages you found and why
- After `read`: Summarize the evidence from those pages and what gap remains, if any

## CRITICAL: Citation Requirements for Final Answers

When providing your final answer, you MUST include citations for ALL factual claims:

### Citation Format
Use inline citations in this format: `[Source: filename, Section/Page]`

Example:
> The total purchase price is $125,000,000 [Source: 01_master_agreement.pdf, Section 2.1], 
> consisting of $80M cash [Source: 01_master_agreement.pdf, Section 2.1(a)], 
> $30M in stock [Source: 10_stock_purchase.pdf, Section 1], and 
> $15M in escrow [Source: 09_escrow_agreement.pdf, Section 2].

### Citation Rules
1. **Every factual claim needs a citation** - dates, numbers, names, terms, etc.
2. **Be specific** - include section numbers, article numbers, or page references when available
3. **Use the actual filename** - not paraphrased names
4. **Multiple sources** - if information comes from multiple documents, cite all of them

### Final Answer Structure
Your final answer should:
1. **Start with a direct answer** to the user's question
2. **Provide details** with inline citations
3. **End with a Sources section** listing all documents consulted:

```
## Sources Consulted
- 01_master_agreement.pdf - Main acquisition terms
- 10_stock_purchase.pdf - Stock component details  
- 09_escrow_agreement.pdf - Escrow terms and release schedule
```

## Example Workflow

```
User asks: "What is the purchase price?"

1. glob(".../pages/", "page-*.md")
   Reason: "This document has 96 page files. I will search for pages mentioning the purchase price."

2. grep(".../pages/", "purchase price")
   Reason: "Candidate pages are 12, 13, and 47. Page 12 and 13 look most likely because the match is in section text rather than a TOC snippet."

3. read(".../pages/page-0012.md")
   Reason: "Page 12 contains the headline purchase price clause. I will read page 13 to confirm the breakdown."

4. STOP with final answer including citations:
   "The purchase price is $50,000,000 [Source: purchase_agreement.pdf, Section 2.1], 
   subject to working capital adjustments [Source: exhibits.pdf, Exhibit B]..."
```
"""

ACTION_REPAIR_PROMPT = """
Your previous reply was invalid because it was not a valid JSON object for the Action schema.

Return exactly one JSON object and nothing else.
Do not include markdown fences.
Do not include analysis before the JSON.
Do not include trailing commentary after the JSON.

For tool calls, use exactly this shape:
{"action":{"tool_name":"semantic_search","tool_input":[{"parameter_name":"query","parameter_value":"董事会成员"},{"parameter_name":"limit","parameter_value":5}]},"reason":"..."}

`tool_name` must be one of:
- `read`
- `grep`
- `glob`
- `scan_folder`
- `preview_file`
- `parse_file`
- `semantic_search`
- `get_document`
- `list_indexed_documents`

Other valid shapes are:
{"action":{"directory":"..."},"reason":"..."}
{"action":{"question":"..."},"reason":"..."}
{"action":{"final_result":"..."},"reason":"..."}

An optional top-level `"context_plan"` object is allowed when relevant.
"""

REPEATED_TOOLCALL_PROMPT = """
Loop guard: you are repeating the same tool call with the same parameters and are not making progress.

Do not repeat the exact same tool call again.
Using the same tool with different parameters is allowed. For example, reading page-0033 and then page-0034 is valid progress.
Choose a meaningfully different next step:
- change the search or parsing strategy
- use a different tool
- provide the best-effort answer from the evidence you already have
- or ask the human for clarification if the evidence is insufficient

Return exactly one JSON Action object and nothing else.
"""

BEST_EFFORT_FINAL_PROMPT = """
Loop guard: you repeated the same tool call multiple times. Stop using tools now.

Using only the evidence already present in the structured context and tool receipts, provide the best-effort final answer to the user's original task.

Requirements:
- Return exactly one JSON Action object and nothing else.
- The action must be `{"final_result": "..."}`.
- Do not say you could not make progress because of repeated tool calls.
- If evidence is incomplete, answer what is supported and explicitly state the missing/uncertain parts.
- Include citations for factual claims whenever source/page information is available.
"""


def _iter_action_json_candidates(raw_text: str) -> list[str | dict[str, Any]]:
    """Return likely JSON candidates extracted from a model response."""
    stripped = raw_text.strip()
    if not stripped:
        return []

    candidates: list[str | dict[str, Any]] = [stripped]
    seen_strings = {stripped}
    escaped_windows_paths = _escape_invalid_json_backslashes(stripped)
    if escaped_windows_paths != stripped:
        seen_strings.add(escaped_windows_paths)
        candidates.append(escaped_windows_paths)

    for fenced in re.findall(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        candidate = fenced.strip()
        if candidate and candidate not in seen_strings:
            seen_strings.add(candidate)
            candidates.append(candidate)
            escaped_candidate = _escape_invalid_json_backslashes(candidate)
            if escaped_candidate != candidate and escaped_candidate not in seen_strings:
                seen_strings.add(escaped_candidate)
                candidates.append(escaped_candidate)

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, end_index = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            serialized = stripped[index : index + end_index]
            if serialized not in seen_strings:
                seen_strings.add(serialized)
                candidates.append(serialized)
            candidates.append(parsed)
            break

    return candidates


def _escape_invalid_json_backslashes(text: str) -> str:
    """Escape backslashes that are invalid in JSON strings, common in Windows paths."""
    text = re.sub(
        r'(:\s*")([A-Za-z]:\\[^"\n]*)(")',
        lambda match: (
            f"{match.group(1)}"
            f"{re.sub(r'\\+', lambda _: chr(92) * 2, match.group(2))}"
            f"{match.group(3)}"
        ),
        text,
    )
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)


def _parse_action_response(raw_text: str) -> Action | None:
    """Parse an Action from raw model text, allowing fenced or embedded JSON."""
    for candidate in _iter_action_json_candidates(raw_text):
        try:
            if isinstance(candidate, dict):
                normalized = _normalize_action_candidate(candidate)
                return Action.model_validate(normalized)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                normalized = _normalize_action_candidate(parsed)
                return Action.model_validate(normalized)
            return Action.model_validate_json(candidate)
        except (ValidationError, json.JSONDecodeError):
            continue
    return None


def _normalize_action_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize loosely-structured action payloads into the Action schema shape."""
    normalized = dict(candidate)
    normalized = _normalize_context_plan_candidate(normalized)
    action_payload = normalized.get("action")

    if isinstance(action_payload, str):
        tool_name = action_payload.strip()
        if tool_name in get_args(Tools):
            normalized["action"] = {
                "tool_name": tool_name,
                "tool_input": _parameters_to_tool_input(normalized.get("parameters")),
            }
        return normalized

    if isinstance(action_payload, dict):
        action_dict = dict(action_payload)
        tool_name = action_dict.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip() in get_args(Tools):
            if isinstance(action_dict.get("tool_input"), dict):
                action_dict["tool_input"] = _parameters_to_tool_input(
                    action_dict.get("tool_input")
                )
            elif "tool_input" not in action_dict and "parameters" in action_dict:
                action_dict["tool_input"] = _parameters_to_tool_input(
                    action_dict.get("parameters")
                )
            normalized["action"] = action_dict
        return normalized

    if isinstance(normalized.get("tool_name"), str):
        tool_name = str(normalized["tool_name"]).strip()
        if tool_name in get_args(Tools):
            return {
                "action": {
                    "tool_name": tool_name,
                    "tool_input": _parameters_to_tool_input(normalized.get("parameters")),
                },
                "reason": normalized.get("reason", ""),
                **(
                    {"context_plan": normalized["context_plan"]}
                    if "context_plan" in normalized
                    else {}
                ),
            }
    return normalized


def _normalize_context_plan_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Drop invalid loose context_plan values so valid actions still execute."""
    if "context_plan" not in candidate:
        return candidate
    context_plan = candidate.get("context_plan")
    if context_plan is None:
        return candidate
    if not isinstance(context_plan, dict):
        normalized = dict(candidate)
        normalized.pop("context_plan", None)
        return normalized
    if "operation" not in context_plan:
        normalized = dict(candidate)
        normalized.pop("context_plan", None)
        return normalized
    return candidate


def _parameters_to_tool_input(parameters: Any) -> list[dict[str, Any]]:
    """Convert a compact parameters mapping into ToolCallArg-compatible input."""
    if isinstance(parameters, list):
        return list(parameters)
    if not isinstance(parameters, dict):
        return []
    return [
        {
            "parameter_name": str(name),
            "parameter_value": value,
        }
        for name, value in parameters.items()
    ]


def _fallback_stop_action(raw_text: str) -> Action | None:
    """Gracefully degrade an unstructured model reply into a stop action."""
    text = raw_text.strip()
    if not text:
        return None
    return Action(
        action=StopAction(final_result=text),
        reason=(
            "Model returned unstructured text instead of the required JSON action. "
            "Treating that text as the final answer."
        ),
    )


def _toolcall_signature(tool_name: Tools, tool_input: dict[str, Any]) -> str:
    """Return a stable signature for loop detection across tool calls."""
    return json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _build_system_prompt(enable_semantic: bool, enable_metadata: bool) -> str:
    """Build a system prompt with retrieval-path guidance appended."""
    if enable_semantic and enable_metadata:
        hint = (
            "\n\n## Retrieval: Semantic + Metadata\n"
            "An index is available. Start with `semantic_search` using optional "
            "`filters` for best results, then use filesystem tools for deep dives."
        )
    elif enable_semantic:
        hint = (
            "\n\n## Retrieval: Semantic Only\n"
            "An index is available. Use `semantic_search` WITHOUT the `filters` "
            "parameter for similarity search, then use filesystem tools for details."
        )
    elif enable_metadata:
        hint = (
            "\n\n## Retrieval: Metadata Only\n"
            "An index is available. Use `semantic_search` with the `filters=` "
            "parameter for metadata filtering, then use filesystem tools for details."
        )
    else:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + hint


# =============================================================================
# Agent Implementation
# =============================================================================


class FsExplorerAgent:
    """
    AI agent for exploring filesystems using an OpenAI-compatible text model.

    The agent maintains a conversation history with the LLM and uses
    structured JSON output to make decisions about which actions to take.

    Attributes:
        token_usage: Tracks API call statistics and costs.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the agent with text-model credentials.

        Args:
            api_key: API key. If not provided, reads from TEXT_API_KEY and
                     compatible fallback environment variables.

        Raises:
            ValueError: If no text-model API key is available.
        """
        text_config = resolve_text_config(api_key=api_key)
        if text_config.api_key is None:
            raise ValueError(
                "Text model is not configured. Set TEXT_API_KEY "
                "(or OPENAI_API_KEY) and optionally TEXT_BASE_URL."
            )

        self._client = OpenAICompatClient(
            api_key=text_config.api_key,
            base_url=text_config.base_url,
        )
        self.model_name = text_config.model_name
        self._chat_history: list[Content] = []
        self._last_tool_signature: str | None = None
        self._last_parse_receipt: str | None = None
        self._repeat_guard_hits = 0
        self.token_usage = TokenUsage()
        self._budget_manager = ContextBudgetManager(
            max_input_tokens=int(os.getenv("FS_EXPLORER_MAX_INPUT_TOKENS", "12000"))
        )

    async def _generate_action_response(self, contents: list[Content]) -> Any:
        """Request an action response and track token usage."""
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,  # type: ignore[arg-type]
            config={
                "system_instruction": _build_system_prompt(*get_search_flags()),
                "response_mime_type": "application/json",
                "response_schema": Action,
            },
        )

        if response.usage_metadata:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )
        return response

    async def _repair_action_response(
        self,
        compacted_history: list[Content],
        invalid_text: str,
    ) -> Action | None:
        """Ask the model once more to restate its intent as valid Action JSON."""
        repair_request = list(compacted_history)
        repair_request.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(
                        text=(
                            f"{ACTION_REPAIR_PROMPT}\n\n"
                            "Previous invalid reply:\n"
                            f"```text\n{invalid_text}\n```"
                        )
                    )
                ],
            )
        )
        repaired_response = await self._generate_action_response(repair_request)
        repaired_text = (repaired_response.text or "").strip()
        return _parse_action_response(repaired_text)

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.

        Args:
            task: The task or context to add to the conversation.
        """
        self._chat_history.append(
            Content(role="user", parts=[Part.from_text(text=task)])
        )

    def _append_history_receipt(self, text: str) -> None:
        """Append a compact tool receipt to chat history."""
        self._chat_history.append(
            Content(
                role="user",
                parts=[Part.from_text(text=text)],
            )
        )

    def _maybe_apply_context_plan(self, action: Action) -> None:
        """Validate and apply an optional context-plan proposal from the model."""
        state = get_context_state()
        if state is None or action.context_plan is None:
            return
        result = state.apply_context_plan(action.context_plan.model_dump(exclude_none=True))
        if result is None:
            return
        event_type = "context_scope_updated" if result.applied else "context_compacted"
        _emit_runtime_event(
            event_type,
            operation=result.operation,
            applied=result.applied,
            **result.payload,
        )

    def _build_model_request(
        self,
        compacted_history: list[Content],
        *,
        anchor_unit_no: int | None,
    ) -> tuple[list[Content], dict[str, Any]]:
        """Assemble the final LLM request from receipts plus structured context."""
        request_contents = list(compacted_history)
        context_state = get_context_state()
        if context_state is None:
            stats = {
                "before_tokens": 0,
                "after_tokens": 0,
                "hard_limit_tokens": self._budget_manager.max_input_tokens,
                "truncated_messages": 0,
                "dropped_messages": 0,
                "compression_ratio": 1.0,
                "overflow_warning": False,
                "structured_context": {
                    "context_scope": {
                        "active_document_id": None,
                        "active_file_path": None,
                        "active_ranges": [],
                    },
                    "coverage_by_document": {},
                    "compaction_actions": [],
                    "active_ranges": [],
                    "promoted_evidence_units": [],
                    "open_gaps": [],
                },
            }
            return request_contents, stats

        context_pack, pack_stats = context_state.build_context_pack(
            anchor_unit_no=anchor_unit_no,
            max_chars=7000,
        )
        if pack_stats.get("compaction_actions"):
            _emit_runtime_event(
                "context_compacted",
                context_scope=pack_stats.get("context_scope"),
                compaction_actions=pack_stats.get("compaction_actions"),
                active_ranges=pack_stats.get("active_ranges"),
            )
        request_contents.append(
            Content(
                role="user",
                parts=[Part.from_text(text=context_pack)],
            )
        )
        return request_contents, {
            "structured_context": pack_stats,
        }

    def _semantic_search_with_context(
        self,
        *,
        query: str,
        filters: str | None = None,
        limit: int = 5,
    ) -> tuple[str, str]:
        rendered, hits, _ = _run_semantic_search(query=query, filters=filters, limit=limit)
        state = get_context_state()
        if state is None:
            return rendered, f"Tool result for semantic_search:\n\n{rendered}"

        summary = state.ingest_search_results(
            query=query,
            filters=filters,
            hits=[
                {
                    "doc_id": hit.doc_id,
                    "absolute_path": hit.absolute_path,
                    "source_unit_no": hit.source_unit_no,
                    "score": hit.score,
                    "text": hit.text,
                }
                for hit in hits
            ],
            limit=limit,
        )
        _emit_runtime_event(
            "evidence_added",
            tool_name="semantic_search",
            summary=summary,
        )
        if state.active_document_id or state.active_ranges:
            _emit_runtime_event(
                "context_scope_updated",
                context_scope=state.snapshot()["context_scope"],
            )
        top_hit_labels = [
            (
                f"{hit['doc_id']}@{hit['source_unit_no']}"
                if hit.get("source_unit_no") is not None
                else str(hit["doc_id"])
            )
            for hit in summary.get("top_hits", [])[:3]
        ]
        receipt = (
            "Search receipt: "
            f"query={query!r}; hits={summary.get('hit_count', 0)}; "
            f"top={', '.join(top_hit_labels) if top_hit_labels else '-'}; "
            "structured evidence has been stored."
        )
        return rendered, receipt

    def _glob_with_context(self, *, directory: str, pattern: str) -> tuple[str, str]:
        index_context = _INDEX_CONTEXT_VAR.get()
        if index_context is None:
            rendered = glob_paths(directory=directory, pattern=pattern)
            return rendered, f"Glob receipt: directory={directory}; pattern={pattern}."
        storage = PostgresStorage(index_context.db_path, read_only=True, initialize=False)
        try:
            resolved = _resolve_document_page_scope(
                storage=storage,
                index_context=index_context,
                target=directory,
            )
            if resolved is None:
                rendered = glob_paths(directory=directory, pattern=pattern)
                return rendered, f"Glob receipt: directory={directory}; pattern={pattern}."
            document, pages_dir = resolved
            rendered = glob_paths(directory=pages_dir, pattern=pattern)
            context_state = get_context_state()
            if context_state is not None:
                context_state.set_active_scope(
                    document_id=str(document["id"]),
                    file_path=pages_dir,
                    ranges=[],
                )
                _emit_runtime_event(
                    "page_scope_resolved",
                    document_id=str(document["id"]),
                    original_filename=str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    pages_dir=pages_dir,
                    page_count=int(document.get("page_count") or 0),
                )
            return rendered, (
                "Glob receipt: "
                f"document={document.get('original_filename') or document['id']}; "
                f"pages_dir={pages_dir}; pattern={pattern}."
            )
        finally:
            storage.close()

    def _grep_with_context(self, *, file_path: str, pattern: str) -> tuple[str, str]:
        index_context = _INDEX_CONTEXT_VAR.get()
        if index_context is None:
            rendered = grep_file_content(file_path=file_path, pattern=pattern)
            return rendered, f"Grep receipt: target={file_path}; pattern={pattern}."
        storage = PostgresStorage(index_context.db_path, read_only=True, initialize=False)
        try:
            resolved = _resolve_document_page_scope(
                storage=storage,
                index_context=index_context,
                target=file_path,
            )
            if resolved is None:
                rendered = grep_file_content(file_path=file_path, pattern=pattern)
                return rendered, f"Grep receipt: target={file_path}; pattern={pattern}."
            document, pages_dir = resolved
            regex = re.compile(pattern=pattern, flags=re.IGNORECASE | re.MULTILINE)
            pages = load_document_pages(
                storage=storage,
                blob_store=_PAGE_BLOB_STORE,
                document_id=str(document["id"]),
            )
            hits: list[dict[str, Any]] = []
            for page in pages:
                matches = list(regex.finditer(str(page.get("markdown") or "")))
                if not matches:
                    continue
                snippet = str(page.get("markdown") or "")
                start = max(0, matches[0].start() - 40)
                end = min(len(snippet), matches[0].end() + 180)
                hits.append(
                    {
                        "doc_id": str(document["id"]),
                        "absolute_path": str(page["file_path"]),
                        "source_unit_no": int(page["page_no"]),
                        "score": float(len(matches)),
                        "text": snippet[start:end],
                    }
                )
            hits.sort(
                key=lambda item: (-float(item["score"]), int(item["source_unit_no"])),
            )
            hits = hits[:8]
            rendered = grep_file_content(file_path=pages_dir, pattern=pattern)
            context_state = get_context_state()
            summary_for_model = ""
            if context_state is not None:
                summary = context_state.ingest_search_results(
                    query=pattern,
                    filters=None,
                    hits=hits,
                    limit=min(8, max(1, len(hits))) if hits else 1,
                )
                summary_for_model = summary.get("summary_for_model", "")
                _emit_runtime_event(
                    "candidate_pages_found",
                    document_id=str(document["id"]),
                    original_filename=str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    candidate_pages=[
                        {
                            "page_no": int(hit["source_unit_no"]),
                            "file_path": str(hit["absolute_path"]),
                            "score": float(hit["score"]),
                        }
                        for hit in hits
                    ],
                )
                _emit_runtime_event(
                    "context_scope_updated",
                    context_scope=context_state.snapshot()["context_scope"],
                )
            return rendered, (
                "Grep receipt: "
                f"document={document.get('original_filename') or document['id']}; "
                f"pattern={pattern!r}; candidate_pages="
                f"{', '.join(str(hit['source_unit_no']) for hit in hits) if hits else '-'}; "
                f"{summary_for_model or 'stored candidate pages for later reasoning.'}"
            )
        finally:
            storage.close()

    def _read_with_context(self, *, file_path: str) -> tuple[str, str]:
        index_context = _INDEX_CONTEXT_VAR.get()
        if index_context is None:
            rendered = read_file(file_path=file_path)
            return rendered, f"Read receipt: path={file_path}."
        storage = PostgresStorage(index_context.db_path, read_only=True, initialize=False)
        try:
            resolved = find_page_by_path(
                storage=storage,
                blob_store=_PAGE_BLOB_STORE,
                document_ids=list(index_context.document_ids),
                file_path=file_path,
            )
            if resolved is None:
                rendered = read_file(file_path=file_path)
                return rendered, f"Read receipt: path={file_path}."
            document, page = resolved
            rendered = read_file(file_path=file_path)
            context_state = get_context_state()
            if context_state is not None:
                parse_summary = context_state.ingest_parse_result(
                    document_id=str(document["id"]),
                    file_path=str(page["file_path"]),
                    label=str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    units=[
                        {
                            "unit_no": int(page["page_no"]),
                            "source_locator": page.get("source_locator"),
                            "heading": page.get("heading"),
                            "markdown": page.get("markdown"),
                        }
                    ],
                    total_units=int(document.get("page_count") or 0) or None,
                    focus_hint=None,
                    anchor=int(page["page_no"]),
                    window=0,
                    max_units=1,
                )
                _emit_runtime_event(
                    "pages_read",
                    document_id=str(document["id"]),
                    original_filename=str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    read_pages=[int(page["page_no"])],
                )
                _emit_runtime_event(
                    "context_scope_updated",
                    context_scope=context_state.snapshot()["context_scope"],
                )
                if context_state.no_new_coverage_streak >= 2:
                    _emit_runtime_event(
                        "stale_page_range_detected",
                        document_id=str(document["id"]),
                        stale_page_ranges=context_state.snapshot()
                        .get("coverage_by_document", {})
                        .get(str(document["id"]), {})
                        .get("summarized_ranges", []),
                    )
                return rendered, _build_parse_receipt(
                    file_path=str(page["file_path"]),
                    summary=parse_summary,
                )
            return rendered, f"Read receipt: path={file_path}."
        finally:
            storage.close()

    def _get_document_with_context(self, *, doc_id: str) -> tuple[str, str]:
        rendered, structured = _run_get_document(doc_id)
        state = get_context_state()
        if state is None or structured is None:
            return rendered, f"Tool result for get_document:\n\n{rendered}"
        summary = state.ingest_document_read(
            document_id=str(structured["document_id"]),
            file_path=str(structured["absolute_path"]),
            label=str(structured["label"]),
            body=str(structured["content"]),
        )
        _emit_runtime_event(
            "evidence_added",
            tool_name="get_document",
            summary=summary,
        )
        _emit_runtime_event(
            "context_scope_updated",
            context_scope=state.snapshot()["context_scope"],
        )
        receipt = (
            "Document receipt: "
            f"doc_id={doc_id}; path={structured['absolute_path']}; "
            "stored a condensed excerpt for later reasoning."
        )
        return rendered, receipt

    def _is_repeated_toolcall(
        self,
        *,
        tool_name: Tools,
        tool_input: dict[str, Any],
    ) -> bool:
        """Return whether the model is repeating the exact previous tool call."""
        signature = _toolcall_signature(tool_name, tool_input)
        return self._last_tool_signature == signature

    def _best_effort_context_stop(self) -> Action:
        """Return a final answer assembled from structured evidence when loop repair fails."""
        context_state = get_context_state()
        if context_state is None:
            final_result = (
                "I reached a repeated tool-call loop. Based on the evidence already shown, "
                "I cannot add more reliable details, so please use the partial evidence above "
                "as the best available answer."
            )
        else:
            context_pack, _ = context_state.build_context_pack(max_chars=5000)
            final_result = (
                "Based on the evidence already collected, here is the best-effort answer. "
                "The evidence may be incomplete, so any missing parts should be treated as uncertain.\n\n"
                f"{context_pack}"
            )
        action = Action(
            action=StopAction(final_result=final_result),
            reason="Stopped repeated tool calls and returned the best available evidence.",
        )
        self._chat_history.append(
            Content(role="model", parts=[Part.from_text(text=action.model_dump_json())])
        )
        return action

    async def take_action(
        self,
        *,
        final_only: bool = False,
    ) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the AI model.

        Sends the current conversation history to the configured text model and receives
        a structured JSON response indicating the next action to take.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        anchor = get_last_focus_anchor()
        compacted_history, budget_stats = self._budget_manager.compact_history(
            self._chat_history,
            anchor_unit_no=(
                int(anchor["source_unit_no"])
                if anchor is not None and anchor.get("source_unit_no") is not None
                else None
            ),
        )
        request_contents, structured_stats = self._build_model_request(
            compacted_history,
            anchor_unit_no=(
                int(anchor["source_unit_no"])
                if anchor is not None and anchor.get("source_unit_no") is not None
                else None
            ),
        )
        merged_budget_stats = {
            **budget_stats,
            **structured_stats,
        }
        _CONTEXT_BUDGET_STATS_VAR.set(merged_budget_stats)
        if merged_budget_stats.get("overflow_warning"):
            request_contents = list(request_contents)
            request_contents.append(
                Content(
                    role="user",
                    parts=[
                        Part.from_text(
                            text=(
                                "Context budget warning: conversation is large. "
                                "Please prioritize the most relevant units and avoid broad deep reads."
                            )
                        )
                    ],
                )
            )

        response = await self._generate_action_response(request_contents)

        if response.candidates is not None:
            if response.text is not None:
                raw_text = response.text.strip()
                action = _parse_action_response(raw_text)
                if action is None:
                    action = await self._repair_action_response(
                        request_contents,
                        raw_text,
                    )
                if action is None:
                    action = _fallback_stop_action(raw_text)
                if action is None:
                    return None

                self._maybe_apply_context_plan(action)
                self._chat_history.append(
                    Content(
                        role="model",
                        parts=[Part.from_text(text=action.model_dump_json())],
                    )
                )
                if final_only and action.to_action_type() != "stop":
                    guarded_stop = self._best_effort_context_stop()
                    return guarded_stop, guarded_stop.to_action_type()

                if action.to_action_type() == "toolcall":
                    toolcall = cast(ToolCallAction, action.action)
                    tool_input = toolcall.to_fn_args()
                    if self._is_repeated_toolcall(
                        tool_name=toolcall.tool_name,
                        tool_input=tool_input,
                    ):
                        self._repeat_guard_hits += 1
                        if self._repeat_guard_hits >= 2:
                            self._chat_history.append(
                                Content(
                                    role="user",
                                    parts=[Part.from_text(text=BEST_EFFORT_FINAL_PROMPT)],
                                )
                            )
                            final_result = await self.take_action(final_only=True)
                            if final_result is not None and final_result[1] == "stop":
                                return final_result
                            guarded_stop = self._best_effort_context_stop()
                            return guarded_stop, guarded_stop.to_action_type()

                        self._chat_history.append(
                            Content(
                                role="user",
                                parts=[Part.from_text(text=REPEATED_TOOLCALL_PROMPT)],
                            )
                        )
                        return await self.take_action()

                    self._repeat_guard_hits = 0
                    self.call_tool(
                        tool_name=toolcall.tool_name,
                        tool_input=tool_input,
                    )
                return action, action.to_action_type()

        return None

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """
        Execute a tool and add the result to the conversation history.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dictionary of arguments to pass to the tool.
        """
        if tool_name == "parse_file":
            anchor = _LAST_FOCUS_ANCHOR_VAR.get()
            if (
                "anchor" not in tool_input
                and anchor is not None
                and bool(anchor.get("auto_inject_allowed", False))
            ):
                path = str(tool_input.get("file_path") or "")
                anchor_path = str(anchor.get("absolute_path") or "")
                try:
                    normalized_path = str(Path(path).resolve())
                except Exception:
                    normalized_path = path
                try:
                    normalized_anchor = str(Path(anchor_path).resolve())
                except Exception:
                    normalized_anchor = anchor_path
                if (
                    normalized_path
                    and normalized_anchor
                    and (
                        normalized_path.lower() == normalized_anchor.lower()
                        or normalized_anchor.lower().endswith(normalized_path.lower())
                    )
                ):
                    tool_input.setdefault("anchor", anchor.get("source_unit_no"))
                    tool_input.setdefault("focus_hint", anchor.get("query"))
                    tool_input.setdefault("window", 1)
                    tool_input.setdefault("max_units", 4)
                    consumed_anchor = dict(anchor)
                    consumed_anchor["auto_inject_allowed"] = False
                    _set_last_focus_anchor(consumed_anchor)
        history_receipt: str | None = None
        try:
            if tool_name == "parse_file":
                result = self._parse_file_with_cache(**tool_input)
                history_receipt = getattr(self, "_last_parse_receipt", None)
            elif tool_name == "glob":
                result, history_receipt = self._glob_with_context(**tool_input)
            elif tool_name == "grep":
                result, history_receipt = self._grep_with_context(**tool_input)
            elif tool_name == "read":
                result, history_receipt = self._read_with_context(**tool_input)
            elif tool_name == "semantic_search":
                result, history_receipt = self._semantic_search_with_context(**tool_input)
            elif tool_name == "get_document":
                result, history_receipt = self._get_document_with_context(**tool_input)
            else:
                result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = (
                f"An error occurred while calling tool {tool_name} "
                f"with {tool_input}: {e}"
            )

        self._last_tool_signature = _toolcall_signature(tool_name, tool_input)

        # Track tool result sizes
        self.token_usage.add_tool_result(result, tool_name)

        if history_receipt is None:
            history_receipt = f"Tool result for {tool_name}:\n\n{result}"
        self._append_history_receipt(history_receipt)

    def _parse_file_with_cache(
        self,
        file_path: str,
        focus_hint: str | None = None,
        anchor: int | None = None,
        window: int = 1,
        max_units: int | None = None,
    ) -> str:
        self._last_parse_receipt = None
        index_context = _INDEX_CONTEXT_VAR.get()
        if index_context is None:
            return parse_file(
                file_path=file_path,
                focus_hint=focus_hint,
                anchor=anchor,
                window=window,
                max_units=max_units,
            )

        storage = PostgresStorage(index_context.db_path)
        try:
            normalized_target = str(Path(file_path).resolve())
            scoped_documents = (
                storage.list_documents_by_ids(
                    doc_ids=list(index_context.document_ids),
                    include_deleted=False,
                )
                if index_context.document_ids
                else storage.list_documents(
                    corpus_id=storage.get_corpus_id(index_context.root_folder or LIBRARY_CORPUS_ROOT) or "",
                    include_deleted=False,
                )
            )
            document = next(
                (
                    item
                    for item in scoped_documents
                    if str(Path(str(item["absolute_path"])).resolve()) == normalized_target
                ),
                None,
            )
            if document is None or bool(document.get("is_deleted", False)):
                return parse_file(
                    file_path=file_path,
                    focus_hint=focus_hint,
                    anchor=anchor,
                    window=window,
                    max_units=max_units,
                )

            selector = (
                ParseSelector(
                    query=focus_hint.strip() if focus_hint else None,
                    anchor=anchor,
                    window=max(window, 0),
                    max_units=max_units,
                )
                if (
                    (focus_hint is not None and focus_hint.strip())
                    or anchor is not None
                    or max_units is not None
                )
                else None
            )
            parsed = get_or_parse_document_units(
                storage=storage,
                document=document,
                blob_store=None,
                selector=selector,
                force=False,
            )
            rendered = format_parse_result(
                file_path=file_path,
                selected_document=parsed.selected_document,
                total_units=parsed.total_units,
                anchor=anchor,
                window=window,
                max_units=max_units,
            )
            context_state = get_context_state()
            if context_state is not None:
                parse_summary = context_state.ingest_parse_result(
                    document_id=str(document["id"]),
                    file_path=str(document["absolute_path"]),
                    label=str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    units=[
                        _unit_to_context_dict(unit)
                        for unit in parsed.selected_document.units
                    ],
                    total_units=parsed.total_units,
                    focus_hint=focus_hint.strip() if focus_hint else None,
                    anchor=anchor,
                    window=window,
                    max_units=max_units,
                )
                self._last_parse_receipt = _build_parse_receipt(
                    file_path=file_path,
                    summary=parse_summary,
                )
                _emit_runtime_event(
                    "evidence_added",
                    tool_name="parse_file",
                    summary=parse_summary,
                )
                _emit_runtime_event(
                    "context_scope_updated",
                    context_scope=context_state.snapshot()["context_scope"],
                )
                if parse_summary.get("coverage_gap"):
                    _emit_runtime_event(
                        "coverage_gap_detected",
                        document_id=str(document["id"]),
                        file_path=str(document["absolute_path"]),
                        coverage_gap=parse_summary.get("coverage_gap"),
                    )
            return rendered
        finally:
            storage.close()

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self._chat_history.clear()
        self._last_tool_signature = None
        self._last_parse_receipt = None
        self._repeat_guard_hits = 0
        self.token_usage = TokenUsage()
        _LAST_FOCUS_ANCHOR_VAR.set(None)
        _CONTEXT_BUDGET_STATS_VAR.set({})
