"""
Reference: legacy/python/src/fs_explorer/document_parsing.py

Document parsing adapters and cache-oriented parsing helpers.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pymupdf4llm
except ImportError:  # pragma: no cover - optional dependency
    pymupdf4llm = None

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None

PARSER_VERSION = "m2-v2"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
)

LOGICAL_UNIT_TARGET_CHARS = 1600
LOGICAL_UNIT_MIN_CHARS = 450
DEFAULT_PDF_CONTENT_MARGINS = (0.0, 36.0, 0.0, 36.0)


class DocumentParseError(RuntimeError):
    """Raised when a document cannot be parsed into normalized markdown."""

    def __init__(self, *, file_path: str, code: str, message: str) -> None:
        self.file_path = file_path
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ParseSelector:
    """Optional selector used to fetch focused parsed units."""

    unit_nos: tuple[int, ...] | None = None
    query: str | None = None
    anchor: int | None = None
    window: int = 1
    max_units: int | None = None


@dataclass(frozen=True)
class ParsedImage:
    """Minimal metadata captured for an extracted page image."""

    image_hash: str
    page_no: int
    image_index: int
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ParsedUnit:
    """A normalized parsed unit (page for PDF, logical block for other formats)."""

    unit_no: int
    markdown: str
    content_hash: str
    heading: str | None = None
    source_locator: str | None = None
    images: tuple[ParsedImage, ...] = ()

    @property
    def image_hashes(self) -> tuple[str, ...]:
        return tuple(image.image_hash for image in self.images)

    # Compatibility aliases for existing page-oriented code.
    @property
    def page_no(self) -> int:
        return self.unit_no

    @property
    def text(self) -> str:
        return self.markdown


@dataclass(frozen=True)
class ParsedPage:
    """Backward-compatible page-shaped parsed unit."""

    page_no: int
    markdown: str
    content_hash: str
    heading: str | None = None
    source_locator: str | None = None
    images: tuple[ParsedImage, ...] = ()

    @property
    def unit_no(self) -> int:
        return self.page_no

    @property
    def text(self) -> str:
        return self.markdown

    @property
    def image_hashes(self) -> tuple[str, ...]:
        return tuple(image.image_hash for image in self.images)


@dataclass(frozen=True, init=False)
class ParsedDocument:
    """A parsed document broken into units."""

    parser_name: str
    parser_version: str
    units: tuple[ParsedUnit, ...]

    def __init__(
        self,
        parser_name: str,
        parser_version: str,
        units: tuple[ParsedUnit | ParsedPage, ...] | None = None,
        pages: tuple[ParsedUnit | ParsedPage, ...] | None = None,
    ) -> None:
        raw_units = units if units is not None else pages
        coerced = tuple(_coerce_unit(item) for item in (raw_units or ()))
        object.__setattr__(self, "parser_name", parser_name)
        object.__setattr__(self, "parser_version", parser_version)
        object.__setattr__(self, "units", coerced)

    @property
    def pages(self) -> tuple[ParsedUnit, ...]:
        return self.units

    @property
    def markdown(self) -> str:
        return "\n\n".join(unit.markdown for unit in self.units if unit.markdown.strip())


@dataclass(frozen=True)
class ParseCacheHit:
    """A cached parsed document reconstructed from persistent units."""

    parser_name: str
    parser_version: str
    units: tuple[ParsedUnit, ...]

    @property
    def pages(self) -> tuple[ParsedUnit, ...]:
        return self.units

    @property
    def markdown(self) -> str:
        return "\n\n".join(unit.markdown for unit in self.units if unit.markdown.strip())


def _coerce_unit(item: ParsedUnit | ParsedPage) -> ParsedUnit:
    if isinstance(item, ParsedUnit):
        return item
    return ParsedUnit(
        unit_no=int(item.page_no),
        markdown=str(item.markdown),
        content_hash=str(item.content_hash),
        heading=item.heading,
        source_locator=item.source_locator,
        images=item.images,
    )


class ImageSemanticEnhancer(Protocol):
    """Application-provided image semantic enhancer."""

    def describe_image(
        self,
        *,
        file_path: str,
        page_no: int,
        image_index: int,
        image_bytes: bytes,
        mime_type: str | None,
    ) -> tuple[str, str | None]:
        """Return a semantic description and an optional model name."""


def compute_file_sha256(file_path: str) -> str:
    """Compute a stable content hash for a source file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_document(
    file_path: str,
    selector: ParseSelector | dict[str, object] | None = None,
) -> ParsedDocument:
    """Parse a document into normalized units, optionally selecting a focused subset."""
    resolved = str(Path(file_path).resolve())
    effective_selector = _coerce_selector(selector)
    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        raise DocumentParseError(
            file_path=resolved,
            code="missing_file",
            message=f"No such file: {resolved}",
        )

    ext = Path(resolved).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(
            file_path=resolved,
            code="unsupported_extension",
            message=(
                f"Unsupported file extension: {ext}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    if ext == ".pdf":
        parsed = _parse_pdf(resolved, selector=effective_selector)
    elif ext == ".md":
        parsed = _single_markdown_document(
            parser_name="markdown",
            markdown=_read_text_file(resolved),
        )
    elif ext == ".html":
        parsed = _single_markdown_document(
            parser_name="html",
            markdown=_html_to_markdown(_read_text_file(resolved)),
        )
    elif ext == ".doc":
        converted = _convert_doc_to_docx(resolved)
        try:
            parsed = _parse_with_docling(converted, parser_name="libreoffice+docling")
        finally:
            try:
                Path(converted).unlink(missing_ok=True)
            except OSError:
                pass
    else:
        parsed = _parse_with_docling(resolved, parser_name="docling")

    return select_parsed_document(parsed, effective_selector)


def select_parsed_document(
    parsed_document: ParsedDocument,
    selector: ParseSelector | dict[str, object] | None,
) -> ParsedDocument:
    """Apply an optional selector to a parsed document."""
    effective_selector = _coerce_selector(selector)
    if effective_selector is None:
        return parsed_document

    units = list(parsed_document.units)
    if not units:
        return parsed_document

    score_by_unit = _query_scores(units, effective_selector.query)
    selected_numbers: set[int] = set()

    if effective_selector.unit_nos:
        selected_numbers.update(
            unit_no
            for unit_no in effective_selector.unit_nos
            if any(unit.unit_no == unit_no for unit in units)
        )

    if effective_selector.anchor is not None:
        window = max(int(effective_selector.window), 0)
        selected_numbers.update(
            unit.unit_no
            for unit in units
            if abs(unit.unit_no - int(effective_selector.anchor)) <= window
        )

    if effective_selector.query and score_by_unit:
        ranked = sorted(
            score_by_unit.items(),
            key=lambda item: (-item[1], item[0]),
        )
        query_pick = max(1, min(4, len(ranked)))
        selected_numbers.update(unit_no for unit_no, _ in ranked[:query_pick])

    if not selected_numbers:
        selected_numbers = {unit.unit_no for unit in units}

    selected = [unit for unit in units if unit.unit_no in selected_numbers]
    if not selected:
        selected = list(units)

    if effective_selector.max_units is not None and effective_selector.max_units > 0:
        anchor = effective_selector.anchor
        selected.sort(
            key=lambda unit: (
                abs(unit.unit_no - anchor) if anchor is not None else 0,
                -score_by_unit.get(unit.unit_no, 0),
                unit.unit_no,
            )
        )
        selected = selected[: int(effective_selector.max_units)]

    selected.sort(key=lambda unit: unit.unit_no)
    return ParsedDocument(
        parser_name=parsed_document.parser_name,
        parser_version=parsed_document.parser_version,
        units=tuple(selected),
    )


def reconstruct_parsed_document(units: list[dict[str, object]]) -> ParseCacheHit | None:
    """Rebuild a parsed document from persisted units."""
    if not units:
        return None

    ordered = sorted(units, key=lambda item: int(item["page_no"]))
    parser_name = str(ordered[0]["parser_name"])
    parser_version = str(ordered[0]["parser_version"])
    rebuilt_units = tuple(
        ParsedUnit(
            unit_no=int(unit["page_no"]),
            markdown=str(unit["markdown"]),
            content_hash=str(unit["content_hash"]),
            heading=_optional_str(unit.get("heading")) or _derive_heading(str(unit["markdown"])),
            source_locator=_optional_str(unit.get("source_locator"))
            or f"unit-{int(unit['page_no'])}",
            images=tuple(
                ParsedImage(
                    image_hash=str(image["image_hash"]),
                    page_no=int(unit["page_no"]),
                    image_index=int(image["image_index"]),
                    mime_type=_optional_str(image.get("mime_type")),
                    width=_optional_int(image.get("width")),
                    height=_optional_int(image.get("height")),
                )
                for image in _coerce_image_list(unit.get("images"))
            ),
        )
        for unit in ordered
    )
    return ParseCacheHit(
        parser_name=parser_name,
        parser_version=parser_version,
        units=rebuilt_units,
    )


def enhance_page_image_semantics(
    *,
    storage,
    document_id: str,
    page_no: int,
    enhancer: ImageSemanticEnhancer,
    parser_version: str = PARSER_VERSION,
) -> int:
    """Lazy-fill missing image semantics for one parsed page."""
    _ = parser_version
    images = storage.list_image_semantics_for_document(
        document_id=document_id,
        page_nos=[page_no],
    )
    if not images:
        return 0

    missing_hashes = [
        str(image["image_hash"])
        for image in images
        if not image.get("semantic_text")
    ]
    if not missing_hashes:
        return 0

    document = storage.get_document(doc_id=document_id)
    if document is None:
        return 0
    file_path = str(document["absolute_path"])

    page_assets = _extract_pdf_images(file_path, page_no=page_no, include_bytes=True)
    asset_by_hash = {asset["image_hash"]: asset for asset in page_assets}
    enhanced = 0

    for image in images:
        image_hash = str(image["image_hash"])
        if image_hash not in missing_hashes:
            continue
        asset = asset_by_hash.get(image_hash)
        if asset is None:
            continue
        semantic_text, semantic_model = enhancer.describe_image(
            file_path=file_path,
            page_no=page_no,
            image_index=int(image["source_image_index"]),
            image_bytes=asset["image_bytes"],
            mime_type=_optional_str(image.get("mime_type")),
        )
        storage.update_image_semantic(
            image_hash=image_hash,
            semantic_text=semantic_text.strip(),
            semantic_model=semantic_model,
        )
        enhanced += 1

    return enhanced


def _coerce_selector(
    selector: ParseSelector | dict[str, object] | None,
) -> ParseSelector | None:
    if selector is None:
        return None
    if isinstance(selector, ParseSelector):
        return selector
    if not isinstance(selector, dict):
        return None

    raw_unit_nos = selector.get("unit_nos")
    unit_nos: tuple[int, ...] | None = None
    if isinstance(raw_unit_nos, list):
        cleaned = [int(v) for v in raw_unit_nos if _optional_int(v) is not None]
        if cleaned:
            unit_nos = tuple(sorted(set(cleaned)))

    return ParseSelector(
        unit_nos=unit_nos,
        query=_optional_str(selector.get("query")),
        anchor=_optional_int(selector.get("anchor")),
        window=max(_optional_int(selector.get("window")) or 1, 0),
        max_units=_optional_int(selector.get("max_units")),
    )


def resolve_requested_unit_nos(
    file_path: str,
    selector: ParseSelector | dict[str, object] | None,
) -> list[int] | None:
    """Resolve the concrete 1-based unit numbers needed for a focused parse."""
    effective_selector = _coerce_selector(selector)
    if effective_selector is None:
        return None

    explicit = _explicit_requested_unit_nos(effective_selector)
    if explicit:
        return explicit

    if not effective_selector.query:
        return None

    if Path(file_path).suffix.lower() != ".pdf" or fitz is None:
        return None

    max_candidates = max(1, min(int(effective_selector.max_units or 4), 8))
    try:
        with fitz.open(file_path) as document:
            return _query_candidate_pdf_unit_nos(
                document,
                query=str(effective_selector.query),
                max_candidates=max_candidates,
            )
    except Exception:
        return None


def _explicit_requested_unit_nos(selector: ParseSelector) -> list[int] | None:
    requested: set[int] = set()
    if selector.unit_nos:
        requested.update(int(unit_no) for unit_no in selector.unit_nos if int(unit_no) > 0)
    if selector.anchor is not None:
        window = max(int(selector.window), 0)
        anchor = int(selector.anchor)
        requested.update(
            unit_no for unit_no in range(anchor - window, anchor + window + 1) if unit_no > 0
        )
    if not requested:
        return None
    return sorted(requested)


def _query_terms(query: str | None) -> list[str]:
    if not query:
        return []
    lowered = query.strip().lower()
    if not lowered:
        return []
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}", lowered)
    ordered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _query_scores(units: list[ParsedUnit], query: str | None) -> dict[int, int]:
    if not query:
        return {}
    lowered = query.strip().lower()
    if not lowered:
        return {}
    terms = _query_terms(lowered)
    scores: dict[int, int] = {}
    for unit in units:
        haystack = unit.markdown.lower()
        score = 0
        if lowered in haystack:
            score += 100
        for term in terms:
            if term in haystack:
                score += max(1, min(len(term), 8))
        if score > 0:
            scores[unit.unit_no] = score
    return scores


def _single_markdown_document(*, parser_name: str, markdown: str) -> ParsedDocument:
    units = _logical_units_from_markdown(markdown)
    if not units:
        normalized = _normalize_markdown(markdown)
        units = [
            ParsedUnit(
                unit_no=1,
                markdown=normalized,
                content_hash=_text_sha256(normalized),
                heading=_derive_heading(normalized),
                source_locator="unit-1",
            )
        ]
    return ParsedDocument(
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
        units=tuple(units),
    )


def _parse_with_docling(file_path: str, *, parser_name: str) -> ParsedDocument:
    if DocumentConverter is None:
        raise DocumentParseError(
            file_path=file_path,
            code="docling_missing",
            message="Install `docling` to parse this document format.",
        )
    try:
        result = DocumentConverter().convert(file_path)
        markdown = result.document.export_to_markdown()
    except Exception as exc:  # pragma: no cover - exercised through error message
        raise DocumentParseError(
            file_path=file_path,
            code="docling_parse_failed",
            message=str(exc),
        ) from exc
    return _single_markdown_document(parser_name=parser_name, markdown=markdown)


def _parse_pdf(
    file_path: str,
    *,
    selector: ParseSelector | None = None,
) -> ParsedDocument:
    chunks: list[tuple[int, str]] = []
    parser_name = "pymupdf4llm"
    requested_pages = _requested_pdf_pages(selector)
    document = None
    diagnostics: list[str] = []
    try:
        if fitz is not None:
            try:
                document = fitz.open(file_path)
            except Exception as exc:
                diagnostics.append(f"PyMuPDF open failed: {type(exc).__name__}: {exc}")
                document = None
        else:
            diagnostics.append("PyMuPDF import is unavailable.")
        if document is not None and requested_pages is None and selector is not None and selector.query:
            candidate_unit_nos = _query_candidate_pdf_unit_nos(
                document,
                query=str(selector.query),
                max_candidates=max(1, min(int(selector.max_units or 4), 8)),
            )
            if candidate_unit_nos is not None:
                requested_pages = [int(unit_no) - 1 for unit_no in candidate_unit_nos]
        if document is not None and requested_pages is not None:
            requested_pages = [
                page_no
                for page_no in requested_pages
                if 0 <= page_no < int(document.page_count)
            ]
        if requested_pages == []:
            return ParsedDocument(
                parser_name=parser_name,
                parser_version=PARSER_VERSION,
                units=(),
            )

        if pymupdf4llm is not None:
            sources = [document] if document is not None else []
            sources.append(file_path)
            seen_sources: set[int | str] = set()
            for source in sources:
                source_key: int | str = id(source) if source is not None and source != file_path else str(source)
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                try:
                    payload = _pymupdf4llm_to_markdown(
                        source,
                        pages=requested_pages,
                        page_chunks=True,
                    )
                    chunks = _extract_pdf_markdown_chunks(payload)
                    if chunks:
                        break
                except Exception as exc:
                    diagnostics.append(
                        f"pymupdf4llm page_chunks failed for {type(source).__name__}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if not chunks:
                try:
                    payload = _pymupdf4llm_to_markdown(
                        file_path,
                        pages=requested_pages,
                        page_chunks=False,
                    )
                    chunks = _extract_pdf_markdown_chunks(payload)
                except Exception as exc:
                    diagnostics.append(
                        f"pymupdf4llm full-document fallback failed: {type(exc).__name__}: {exc}"
                    )
        else:
            diagnostics.append("pymupdf4llm import is unavailable.")

        if not chunks and document is not None:
            parser_name = "pymupdf"
            try:
                page_indices = (
                    requested_pages
                    if requested_pages is not None
                    else list(range(int(document.page_count)))
                )
                chunks = []
                for page_index in page_indices:
                    page = document.load_page(int(page_index))
                    try:
                        text = page.get_text("markdown")
                    except Exception as exc:
                        diagnostics.append(
                            f"PyMuPDF markdown extraction failed on page {int(page_index) + 1}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        text = page.get_text("text")
                    chunks.append((int(page_index) + 1, text))
            except Exception as exc:  # pragma: no cover - exercised through error path
                raise DocumentParseError(
                    file_path=file_path,
                    code="pdf_parse_failed",
                    message=str(exc),
                ) from exc
    finally:
        if document is not None:
            document.close()

    if not chunks:
        modules_missing = fitz is None or pymupdf4llm is None
        detail = " ".join(diagnostics).strip()
        if not detail:
            detail = "No page chunks were produced by pymupdf4llm or PyMuPDF."
        raise DocumentParseError(
            file_path=file_path,
            code="pdf_parser_unavailable" if modules_missing else "pdf_parse_failed",
            message=(
                "Install pymupdf4llm (and PyMuPDF) to parse PDF files."
                if modules_missing
                else f"PDF parser produced no pages. {detail}"
            ),
        )

    cleaned_chunks = _strip_pdf_headers_and_footers(chunks)
    units: list[ParsedUnit] = []
    for page_no, markdown in cleaned_chunks:
        # Reference: keep PDF parsing close to the legacy chain, but scrub
        # synthetic PyMuPDF table placeholder columns before final normalization.
        normalized = _normalize_markdown(_normalize_pdf_table_markdown(markdown))
        images = tuple(
            ParsedImage(
                image_hash=str(image["image_hash"]),
                page_no=page_no,
                image_index=int(image["image_index"]),
                mime_type=_optional_str(image.get("mime_type")),
                width=_optional_int(image.get("width")),
                height=_optional_int(image.get("height")),
            )
            for image in _extract_pdf_images(file_path, page_no=page_no)
        )
        units.append(
            ParsedUnit(
                unit_no=page_no,
                markdown=normalized,
                content_hash=_text_sha256(normalized),
                heading=_derive_heading(normalized),
                source_locator=f"page-{page_no}",
                images=images,
            )
        )

    return ParsedDocument(
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
        units=tuple(units),
    )


def _pymupdf4llm_to_markdown(
    source: object,
    *,
    pages: list[int] | None,
    page_chunks: bool,
) -> object:
    """Call pymupdf4llm across supported version-specific signatures."""
    if pymupdf4llm is None:
        raise RuntimeError("pymupdf4llm is unavailable")
    try:
        return pymupdf4llm.to_markdown(
            source,
            pages=pages,
            page_chunks=page_chunks,
            header=False,
            footer=False,
            margins=_pdf_content_margins(),
        )
    except TypeError:
        return pymupdf4llm.to_markdown(
            source,
            pages=pages,
            page_chunks=page_chunks,
            margins=_pdf_content_margins(),
        )


def _requested_pdf_pages(selector: ParseSelector | None) -> list[int] | None:
    """Translate a focused selector into 0-based PDF page numbers."""
    if selector is None:
        return None
    requested_unit_nos = _explicit_requested_unit_nos(selector)
    if not requested_unit_nos:
        return None
    return sorted(int(unit_no) - 1 for unit_no in requested_unit_nos if int(unit_no) > 0)


def _query_candidate_pdf_unit_nos(
    document,
    *,
    query: str,
    max_candidates: int,
) -> list[int] | None:
    lowered = query.strip().lower()
    if not lowered:
        return None

    terms = _query_terms(lowered)
    scored_pages: list[tuple[int, int]] = []
    for page_index in range(int(document.page_count)):
        try:
            haystack = document.load_page(page_index).get_text("text").lower()
        except Exception:
            continue
        score = 0
        if lowered in haystack:
            score += 100
        for term in terms:
            if term in haystack:
                score += max(1, min(len(term), 8))
        if score > 0:
            scored_pages.append((page_index + 1, score))

    if not scored_pages:
        return None

    scored_pages.sort(key=lambda item: (-item[1], item[0]))
    return sorted(page_no for page_no, _ in scored_pages[: max_candidates])


def _logical_units_from_markdown(markdown: str) -> list[ParsedUnit]:
    normalized = _normalize_markdown(markdown)
    if not normalized:
        return []

    blocks = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
    if not blocks:
        return []

    units: list[ParsedUnit] = []
    current_blocks: list[str] = []
    current_heading: str | None = None
    current_len = 0
    unit_no = 1

    def flush() -> None:
        nonlocal unit_no, current_blocks, current_heading, current_len
        if not current_blocks:
            return
        text = _normalize_markdown("\n\n".join(current_blocks))
        units.append(
            ParsedUnit(
                unit_no=unit_no,
                markdown=text,
                content_hash=_text_sha256(text),
                heading=current_heading or _derive_heading(text),
                source_locator=f"unit-{unit_no}",
            )
        )
        unit_no += 1
        current_blocks = []
        current_heading = None
        current_len = 0

    for block in blocks:
        block_heading = _derive_heading(block)
        block_is_heading = block.lstrip().startswith("#")
        block_len = len(block)

        force_heading_boundary = block_is_heading and current_blocks
        force_size_boundary = (
            current_blocks
            and current_len + block_len + 2 > LOGICAL_UNIT_TARGET_CHARS
            and current_len >= LOGICAL_UNIT_MIN_CHARS
        )
        if force_heading_boundary or force_size_boundary:
            flush()

        if not current_blocks:
            current_heading = block_heading
            current_len = 0
        elif current_heading is None and block_heading is not None:
            current_heading = block_heading

        current_blocks.append(block)
        current_len += block_len + (2 if current_len > 0 else 0)

    flush()
    return units


def _extract_pdf_markdown_chunks(payload: object) -> list[tuple[int, str]]:
    if isinstance(payload, list):
        raw_pages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        raw_pages = payload["pages"]
    else:
        raw_pages = [payload]

    chunks: list[tuple[int, str]] = []
    for index, item in enumerate(raw_pages, start=1):
        if isinstance(item, dict):
            page_no = _optional_int(
                item.get("page")
                or item.get("page_no")
                or item.get("page_number")
                or index
            ) or index
            text = (
                item.get("text")
                or item.get("markdown")
                or item.get("content")
                or item.get("md")
                or ""
            )
            chunks.append((page_no, str(text)))
        else:
            chunks.append((index, str(item)))
    return chunks


def _pdf_content_margins() -> tuple[float, float, float, float]:
    """Return page margins used to drop PDF header/footer bands for legacy parsing."""
    raw = os.getenv("FS_EXPLORER_PDF_CONTENT_MARGINS", "").strip()
    if not raw:
        return DEFAULT_PDF_CONTENT_MARGINS

    try:
        values = tuple(
            float(part.strip())
            for part in re.split(r"[,; ]+", raw)
            if part.strip()
        )
    except ValueError:
        return DEFAULT_PDF_CONTENT_MARGINS

    if len(values) == 1:
        value = values[0]
        return (value, value, value, value)
    if len(values) == 2:
        top, bottom = values
        return (0.0, top, 0.0, bottom)
    if len(values) == 4:
        left, top, right, bottom = values
        return (left, top, right, bottom)
    return DEFAULT_PDF_CONTENT_MARGINS


def _strip_pdf_headers_and_footers(
    chunks: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    first_lines = Counter()
    last_lines = Counter()
    tokenized: list[tuple[int, list[str]]] = []

    for page_no, markdown in chunks:
        lines = [line.rstrip() for line in markdown.splitlines()]
        tokenized.append((page_no, lines))
        first = next((line for line in lines if line.strip()), "")
        last = next((line for line in reversed(lines) if line.strip()), "")
        if _is_repeatable_header_footer(first):
            first_lines[first] += 1
        if _is_repeatable_header_footer(last):
            last_lines[last] += 1

    repeated_headers = {line for line, count in first_lines.items() if count >= 2}
    repeated_footers = {line for line, count in last_lines.items() if count >= 2}
    cleaned: list[tuple[int, str]] = []

    for page_no, lines in tokenized:
        trimmed = list(lines)
        if trimmed:
            first_index = next(
                (index for index, line in enumerate(trimmed) if line.strip()),
                None,
            )
            if first_index is not None and trimmed[first_index] in repeated_headers:
                trimmed[first_index] = ""
            last_index = next(
                (
                    index
                    for index in range(len(trimmed) - 1, -1, -1)
                    if trimmed[index].strip()
                ),
                None,
            )
            if last_index is not None and trimmed[last_index] in repeated_footers:
                trimmed[last_index] = ""
            last_index = next(
                (
                    index
                    for index in range(len(trimmed) - 1, -1, -1)
                    if trimmed[index].strip()
                ),
                None,
            )
            if last_index is not None and _is_page_number_footer(
                trimmed[last_index],
                page_no=page_no,
            ):
                trimmed[last_index] = ""
        cleaned.append((page_no, "\n".join(trimmed)))

    return cleaned


def _is_repeatable_header_footer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    return True


def _is_page_number_footer(text: str, *, page_no: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    compact = re.sub(r"\s+", "", stripped)
    escaped_page_no = re.escape(str(int(page_no)))
    patterns = (
        rf"^{escaped_page_no}$",
        rf"^[-_~·•]*{escaped_page_no}[-_~·•]*$",
        rf"^第{escaped_page_no}页$",
        rf"^{escaped_page_no}/\d+$",
        rf"^\d+/{escaped_page_no}$",
        rf"^{escaped_page_no}-\d+$",
        rf"^\d+-{escaped_page_no}$",
    )
    return any(re.fullmatch(pattern, compact) for pattern in patterns)


def _extract_pdf_images(
    file_path: str,
    *,
    page_no: int,
    include_bytes: bool = False,
) -> list[dict[str, object]]:
    if fitz is None:
        return []

    try:
        with fitz.open(file_path) as document:
            page = document[page_no - 1]
            images: list[dict[str, object]] = []
            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = int(image_info[0])
                extracted = document.extract_image(xref)
                image_bytes = extracted.get("image", b"")
                if not image_bytes:
                    continue
                item: dict[str, object] = {
                    "image_hash": hashlib.sha256(image_bytes).hexdigest(),
                    "image_index": image_index,
                    "mime_type": _image_mime_type(extracted.get("ext")),
                    "width": extracted.get("width"),
                    "height": extracted.get("height"),
                }
                if include_bytes:
                    item["image_bytes"] = image_bytes
                images.append(item)
            return images
    except Exception:
        return []


def _convert_doc_to_docx(file_path: str) -> str:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise DocumentParseError(
            file_path=file_path,
            code="libreoffice_missing",
            message="LibreOffice headless is required for .doc conversion. Install `soffice` and retry.",
        )

    output_dir = tempfile.mkdtemp(prefix="fs-explorer-doc-")
    try:
        completed = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                output_dir,
                file_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover - depends on local LibreOffice
        raise DocumentParseError(
            file_path=file_path,
            code="doc_conversion_failed",
            message=str(exc),
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise DocumentParseError(
            file_path=file_path,
            code="doc_conversion_failed",
            message=stderr,
        )

    converted = Path(output_dir) / f"{Path(file_path).stem}.docx"
    if not converted.exists():
        raise DocumentParseError(
            file_path=file_path,
            code="doc_conversion_failed",
            message="LibreOffice did not produce a DOCX output file.",
        )
    return str(converted)


def _read_text_file(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def _html_to_markdown(value: str) -> str:
    stripped = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", value)
    stripped = re.sub(
        r"(?i)</?(h[1-6]|p|div|li|br|tr|td|th|table|ul|ol|section|article)>",
        "\n",
        stripped,
    )
    stripped = re.sub(r"(?s)<[^>]+>", "", stripped)
    return html.unescape(stripped)


def _normalize_markdown(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_pdf_table_markdown(value: str) -> str:
    """Normalize unstable PyMuPDF markdown tables into a model-friendly shape.

    Reference: keep table titles, flatten multiline headers, remove synthetic
    placeholder columns, and merge obviously fragmented header columns so the
    downstream model does not invent hidden fields from malformed table grids.
    """

    placeholder_values = {"-", "--", "—", "N/A", "n/a", "null", "None"}

    def normalize_inline_text(text: str) -> str:
        normalized = re.sub(r"(?i)<br\s*/?>", "", str(text or ""))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
        normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])", "", normalized)
        normalized = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", normalized)
        return normalized

    def smart_join_text(parts: list[str]) -> str:
        normalized_parts = [normalize_inline_text(part) for part in parts]
        normalized_parts = [part for part in normalized_parts if part]
        if not normalized_parts:
            return ""
        joined = "".join(normalized_parts)
        joined = re.sub(r"\s+", " ", joined).strip()
        joined = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", joined)
        return joined

    def is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3

    def is_separator_line(cells: list[str]) -> bool:
        return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

    def parse_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip()[1:-1].split("|")]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    def is_placeholder_cell(value: str) -> bool:
        stripped = normalize_inline_text(value)
        return (
            not stripped
            or stripped in placeholder_values
            or bool(re.fullmatch(r":?-{3,}:?", stripped))
            or bool(re.fullmatch(r"Col\d+", stripped, flags=re.IGNORECASE))
        )

    def meaningful_cell(value: str) -> bool:
        return not is_placeholder_cell(value)

    def merge_cells(left: str, right: str) -> str:
        left = normalize_inline_text(left)
        right = normalize_inline_text(right)
        if not left:
            return right
        if not right:
            return left
        if right in left:
            return left
        if left in right:
            return right
        return smart_join_text([left, right])

    def maybe_extract_caption(rows: list[list[str]]) -> tuple[str | None, list[list[str]]]:
        if len(rows) < 3 or not is_separator_line(rows[1]):
            return None, rows
        meaningful = [normalize_inline_text(cell) for cell in rows[0] if meaningful_cell(cell)]
        if len(meaningful) != 1:
            return None, rows
        return meaningful[0], rows[2:]

    def merge_fragmented_header_columns(
        header: list[str],
        body_rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        width = len(header)
        keep_indexes = list(range(width))

        def header_is_fragment(text: str) -> bool:
            stripped = normalize_inline_text(text)
            return bool(stripped) and len(stripped) <= 10 and not re.search(r"\d", stripped)

        def body_group_is_sparse(start: int, end: int) -> bool:
            if end <= start:
                return False
            for row in body_rows:
                meaningful_count = sum(1 for index in range(start, end + 1) if meaningful_cell(row[index]))
                if meaningful_count > 1:
                    return False
            return True

        index = 0
        while index < width - 1:
            if not header_is_fragment(header[index]):
                index += 1
                continue
            group_end = index
            while (
                group_end + 1 < width
                and header_is_fragment(header[group_end + 1])
                and body_group_is_sparse(index, group_end + 1)
            ):
                group_end += 1
            if group_end == index:
                index += 1
                continue
            header[index] = smart_join_text(header[index : group_end + 1])
            for row in body_rows:
                row[index] = smart_join_text(row[index : group_end + 1])
            for drop_index in range(index + 1, group_end + 1):
                if drop_index in keep_indexes:
                    keep_indexes.remove(drop_index)
            index = group_end + 1

        normalized_header = [header[index] for index in keep_indexes]
        normalized_body = [[row[index] for index in keep_indexes] for row in body_rows]
        return normalized_header, normalized_body

    def normalize_table_block(lines: list[str]) -> list[str]:
        if len(lines) < 2:
            return lines
        rows = [parse_row(line) for line in lines]
        width = max(len(row) for row in rows)
        if width < 2:
            return lines
        rows = [row + [""] * (width - len(row)) for row in rows]
        caption, rows = maybe_extract_caption(rows)
        if caption:
            if len(rows) < 2:
                return lines
            header = rows[0]
            body_rows = rows[1:]
        else:
            if len(rows) < 2 or not is_separator_line(rows[1]):
                return lines
            header = rows[0]
            body_rows = rows[2:]
        placeholder_indexes = [
            index
            for index, cell in enumerate(header)
            if re.fullmatch(r"Col\d+", normalize_inline_text(cell), flags=re.IGNORECASE)
        ]
        keep_indexes = list(range(width))
        for index in placeholder_indexes:
            body_values = [normalize_inline_text(row[index]) for row in body_rows]
            if not any(meaningful_cell(value) for value in body_values):
                if index in keep_indexes:
                    keep_indexes.remove(index)
                continue

            target_index = next(
                (
                    candidate
                    for candidate in reversed(range(index))
                    if candidate in keep_indexes and candidate not in placeholder_indexes
                ),
                None,
            )
            if target_index is None:
                target_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, width)
                        if candidate in keep_indexes and candidate not in placeholder_indexes
                    ),
                    None,
                )
            if target_index is None:
                continue

            for row in body_rows:
                row[target_index] = merge_cells(row[target_index], row[index])
            if index in keep_indexes:
                keep_indexes.remove(index)

        header = [normalize_inline_text(header[index]) for index in keep_indexes]
        body_rows = [[normalize_inline_text(row[index]) for index in keep_indexes] for row in body_rows]
        if len(header) < 2:
            return lines

        header, body_rows = merge_fragmented_header_columns(header, body_rows)
        if len(header) < 2:
            return lines
        if any(is_placeholder_cell(cell) for cell in header):
            return lines

        rendered_lines: list[str] = []
        if caption:
            rendered_lines.extend([f"表格标题：{caption}", ""])
        rendered_lines.append(render_row(header))
        rendered_lines.append(render_row(["---"] * len(header)))
        rendered_lines.extend(render_row(row) for row in body_rows)
        return rendered_lines

    output_lines: list[str] = []
    pending_table: list[str] = []
    in_code_fence = False

    def flush_pending_table() -> None:
        nonlocal pending_table
        if pending_table:
            output_lines.extend(normalize_table_block(pending_table))
            pending_table = []

    for line in value.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_pending_table()
            in_code_fence = not in_code_fence
            output_lines.append(line)
            continue
        if not in_code_fence and is_table_line(line):
            pending_table.append(line)
            continue
        flush_pending_table()
        output_lines.append(line)

    flush_pending_table()
    return "\n".join(output_lines)


def _derive_heading(value: str) -> str | None:
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    first = value.splitlines()[0].strip() if value.splitlines() else ""
    if first and len(first) <= 90:
        return first
    return None


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coerce_image_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            items.append(item)
    return items


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _image_mime_type(value: object) -> str | None:
    ext = _optional_str(value)
    if ext is None:
        return None
    if "/" in ext:
        return ext
    return f"image/{ext.lower()}"
