"""
FsExplorer Agent for filesystem exploration using Google Gemini.

This module contains the agent that interacts with the Gemini AI model
to make decisions about filesystem exploration actions.
"""

import contextvars
import os
import re
from pathlib import Path
from typing import Callable, Any, cast
from dataclasses import dataclass

from dotenv import load_dotenv
from google.genai.types import Content, HttpOptions, Part
from google.genai import Client as GenAIClient

from .models import Action, ActionType, ToolCallAction, Tools
from .fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)
from .embeddings import EmbeddingProvider
from .document_parsing import PARSER_VERSION, enhance_page_image_semantics
from .image_semantics import build_image_semantic_enhancer
from .index_config import resolve_db_path
from .search import (
    IndexedQueryEngine,
    MetadataFilterParseError,
    supported_filter_syntax,
)
from .storage import PostgresStorage

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# =============================================================================
# Token Usage Tracking
# =============================================================================

# Gemini Flash pricing (per million tokens)
GEMINI_FLASH_INPUT_COST_PER_MILLION = 0.075
GEMINI_FLASH_OUTPUT_COST_PER_MILLION = 0.30


@dataclass
class TokenUsage:
    """
    Track token usage and costs across the session.

    Maintains running totals of API calls, token counts, and provides
    cost estimates based on Gemini Flash pricing.
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
        """Calculate estimated costs based on Gemini Flash pricing."""
        input_cost = (
            self.prompt_tokens / 1_000_000
        ) * GEMINI_FLASH_INPUT_COST_PER_MILLION
        output_cost = (
            self.completion_tokens / 1_000_000
        ) * GEMINI_FLASH_OUTPUT_COST_PER_MILLION
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
  Est. Cost (Gemini Flash):
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

    root_folder: str
    db_path: str


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


def set_index_context(folder: str, db_path: str | None = None) -> None:
    """Enable indexed tools for a specific folder corpus."""
    _INDEX_CONTEXT_VAR.set(
        IndexContext(
        root_folder=str(Path(folder).resolve()),
        db_path=resolve_db_path(db_path),
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
    _FIELD_CATALOG_SHOWN_VAR.set(False)
    _ENABLE_SEMANTIC_VAR.set(False)
    _ENABLE_METADATA_VAR.set(False)


def _get_index_storage_and_corpus() -> tuple[
    PostgresStorage | None, str | None, str | None
]:
    index_context = _INDEX_CONTEXT_VAR.get()
    if index_context is None:
        return None, None, "Index context is not configured. Re-run with `--use-index`."

    storage = PostgresStorage(index_context.db_path)
    corpus_id = storage.get_corpus_id(index_context.root_folder)
    if corpus_id is None:
        return (
            None,
            None,
            f"No index found for folder {index_context.root_folder}. "
            "Run `explore index <folder>` first.",
        )
    return storage, corpus_id, None


def _clean_excerpt(text: str, max_chars: int = 320) -> str:
    squashed = re.sub(r"\s+", " ", text).strip()
    if len(squashed) <= max_chars:
        return squashed
    return f"{squashed[:max_chars]}..."


def _normalize_lookup_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _emit_runtime_event(event_type: str, **data: Any) -> None:
    """Fan out an optional runtime event if a callback is configured."""
    callback = _RUNTIME_EVENT_CALLBACK_VAR.get()
    if callback is not None:
        callback(event_type, data)


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


def semantic_search(query: str, filters: str | None = None, limit: int = 5) -> str:
    """Search indexed chunks and return ranked excerpts."""
    storage, corpus_id, error = _get_index_storage_and_corpus()
    if error:
        return error
    assert storage is not None and corpus_id is not None

    engine = IndexedQueryEngine(
        storage,
        embedding_provider=_EMBEDDING_PROVIDER_VAR.get(),
    )
    enable_semantic, enable_metadata = get_search_flags()
    try:
        hits = engine.search(
            corpus_id=corpus_id,
            query=query,
            filters=filters,
            limit=limit,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
        )
    except MetadataFilterParseError as exc:
        return f"Invalid metadata filter: {exc}\n{supported_filter_syntax()}"
    except ValueError as exc:
        return f"Metadata filter error: {exc}"

    if not hits:
        if filters:
            return f"No indexed matches found for query={query!r} with filters={filters!r}."
        return f"No indexed matches found for query: {query!r}"

    enhanced_images, cached_image_pages = _maybe_enhance_images_for_hits(storage, hits)

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

    # Include a rich field catalog on the first search so the agent can
    # construct effective metadata filters.
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
                        for v in vals:
                            try:
                                nums.append(float(v))
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


def get_document(doc_id: str) -> str:
    """Return full document content by id from the active index context."""
    storage, _, error = _get_index_storage_and_corpus()
    if error:
        return error
    assert storage is not None

    document = storage.get_document(doc_id=doc_id)
    if document is None:
        return f"No indexed document found for doc_id={doc_id!r}"
    if document["is_deleted"]:
        return f"Document {doc_id} is marked as deleted in the index."

    semantic_section = _format_cached_image_semantics(storage, document_id=doc_id)

    return (
        f"=== DOCUMENT {doc_id} ===\n"
        f"Path: {document['absolute_path']}\n\n"
        f"{document['content']}"
        f"{semantic_section}"
    )


def list_indexed_documents() -> str:
    """List indexed documents for the active corpus."""
    storage, corpus_id, error = _get_index_storage_and_corpus()
    if error:
        return error
    assert storage is not None and corpus_id is not None

    documents = storage.list_documents(corpus_id=corpus_id, include_deleted=False)
    if not documents:
        return "No indexed documents found for the active corpus."

    lines = ["=== INDEXED DOCUMENTS ==="]
    for idx, document in enumerate(documents, start=1):
        lines.append(
            f"[{idx}] doc_id={document['id']} path={document['absolute_path']}"
        )
    lines.append("")
    lines.append("Use semantic_search(...) to find relevant doc_ids.")
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
| `scan_folder` | **PARALLEL SCAN** - Scan ALL documents in a folder at once | `directory` |
| `preview_file` | Quick preview of a single document (~first page) | `file_path` |
| `parse_file` | **DEEP READ** - Full content of a document | `file_path` |
| `read` | Read a plain text file | `file_path` |
| `grep` | Search for a pattern in a file | `file_path`, `pattern` |
| `glob` | Find files matching a pattern | `directory`, `pattern` |
| `semantic_search` | Search indexed chunks and metadata-filtered docs, then union/rank results | `query`, `filters`, `limit` |
| `get_document` | Read full indexed document by document id | `doc_id` |
| `list_indexed_documents` | List indexed documents for active corpus | none |

## Indexed Retrieval Strategy

When indexed tools are available:
1. Start with `semantic_search` to quickly find relevant documents.
2. Use `get_document` for the top candidate doc IDs.
3. If indexed tools report index is unavailable, fall back to filesystem tools (`scan_folder`, `parse_file`, etc.).

Filter syntax for `semantic_search(filters=...)`:
- `field=value`
- `field!=value`
- `field>=number`, `field<=number`, `field>number`, `field<number`
- `field in (a, b, c)`
- `field~substring`
- combine conditions with comma or `and`

## Three-Phase Document Exploration Strategy

### PHASE 1: Parallel Scan (Use `scan_folder`)
When you encounter a folder with documents:
1. Use `scan_folder` to scan ALL documents in parallel
2. This gives you a quick preview of every document at once
3. In your **reason**, explicitly list your document categorization:
   - **RELEVANT**: Documents clearly related to the query (list them)
   - **MAYBE**: Documents that might be relevant (list them)
   - **SKIP**: Documents not relevant (list them)

### PHASE 2: Deep Dive (Use `parse_file`)
1. Use `parse_file` on documents marked RELEVANT
2. In your **reason**, explain what key information you found
3. **WATCH FOR CROSS-REFERENCES** - look for mentions like:
   - "See Exhibit A/B/C..."
   - "As stated in the [Document Name]..."
   - "Refer to [filename]..."
   - Document numbers, exhibit labels, or file names
4. In your **reason**, note any cross-references you discovered

### PHASE 3: Backtracking (Revisit if Cross-Referenced)
**CRITICAL**: If a document you're reading references another document that you SKIPPED:
1. In your **reason**, explain: "Found cross-reference to [document] - need to backtrack"
2. Use `preview_file` or `parse_file` to read the referenced document
3. Continue this until all relevant cross-references are resolved

## Providing Detailed Reasoning

Your `reason` field is displayed to the user, so make it informative:
- After scanning: List which documents you're categorizing as RELEVANT/MAYBE/SKIP and why
- After parsing: Summarize key findings and any cross-references discovered
- When backtracking: Explain which reference led you back to a skipped document

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

1. scan_folder("./documents/")
   Reason: "Scanned 10 documents. Categorizing:
   - RELEVANT: purchase_agreement.pdf (mentions 'Purchase Price' in preview)
   - RELEVANT: financial_terms.pdf (contains pricing tables)
   - MAYBE: exhibits.pdf (referenced by other docs)
   - SKIP: employee_handbook.pdf, hr_policies.pdf (unrelated to pricing)"

2. parse_file("purchase_agreement.pdf")
   Reason: "Found purchase price of $50M in Section 2.1. Document references 
   'Exhibit B for price adjustments' - need to check exhibits.pdf next."

3. parse_file("exhibits.pdf")  [BACKTRACKING]
   Reason: "Backtracking to exhibits.pdf because purchase_agreement.pdf 
   referenced it for adjustment details. Found working capital adjustment 
   formula in Exhibit B."

4. STOP with final answer including citations:
   "The purchase price is $50,000,000 [Source: purchase_agreement.pdf, Section 2.1], 
   subject to working capital adjustments [Source: exhibits.pdf, Exhibit B]..."
```
"""

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
    AI agent for exploring filesystems using Google Gemini.

    The agent maintains a conversation history with the LLM and uses
    structured JSON output to make decisions about which actions to take.

    Attributes:
        token_usage: Tracks API call statistics and costs.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the agent with Google API credentials.

        Args:
            api_key: Google API key. If not provided, reads from
                     GOOGLE_API_KEY environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY not found within the current environment: "
                "please export it or provide it to the class constructor."
            )

        self._client = GenAIClient(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1beta"),
        )
        self._chat_history: list[Content] = []
        self.token_usage = TokenUsage()

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.

        Args:
            task: The task or context to add to the conversation.
        """
        self._chat_history.append(
            Content(role="user", parts=[Part.from_text(text=task)])
        )

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the AI model.

        Sends the current conversation history to Gemini and receives
        a structured JSON response indicating the next action to take.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        response = await self._client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=self._chat_history,  # type: ignore
            config={
                "system_instruction": _build_system_prompt(*get_search_flags()),
                "response_mime_type": "application/json",
                "response_schema": Action,
            },
        )

        # Track token usage from response metadata
        if response.usage_metadata:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        if response.candidates is not None:
            if response.candidates[0].content is not None:
                self._chat_history.append(response.candidates[0].content)
            if response.text is not None:
                action = Action.model_validate_json(response.text)
                if action.to_action_type() == "toolcall":
                    toolcall = cast(ToolCallAction, action.action)
                    self.call_tool(
                        tool_name=toolcall.tool_name,
                        tool_input=toolcall.to_fn_args(),
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
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = (
                f"An error occurred while calling tool {tool_name} "
                f"with {tool_input}: {e}"
            )

        # Track tool result sizes
        self.token_usage.add_tool_result(result, tool_name)

        self._chat_history.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(text=f"Tool result for {tool_name}:\n\n{result}")
                ],
            )
        )

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self._chat_history.clear()
        self.token_usage = TokenUsage()
