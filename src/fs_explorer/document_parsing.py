"""
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

PARSER_VERSION = "m2-v1"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
)


class DocumentParseError(RuntimeError):
    """Raised when a document cannot be parsed into normalized markdown."""

    def __init__(self, *, file_path: str, code: str, message: str) -> None:
        self.file_path = file_path
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


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
class ParsedPage:
    """A normalized parsed page unit."""

    page_no: int
    markdown: str
    content_hash: str
    images: tuple[ParsedImage, ...] = ()

    @property
    def image_hashes(self) -> tuple[str, ...]:
        return tuple(image.image_hash for image in self.images)


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed document broken into page-like units."""

    parser_name: str
    parser_version: str
    pages: tuple[ParsedPage, ...]

    @property
    def markdown(self) -> str:
        return "\n\n".join(page.markdown for page in self.pages if page.markdown.strip())


@dataclass(frozen=True)
class ParseCacheHit:
    """A cached parsed document reconstructed from persistent page units."""

    parser_name: str
    parser_version: str
    pages: tuple[ParsedPage, ...]

    @property
    def markdown(self) -> str:
        return "\n\n".join(page.markdown for page in self.pages if page.markdown.strip())


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


def parse_document(file_path: str) -> ParsedDocument:
    """Parse a document into normalized page units."""
    resolved = str(Path(file_path).resolve())
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
        return _parse_pdf(resolved)
    if ext == ".md":
        return _single_page_document(
            parser_name="markdown",
            markdown=_read_text_file(resolved),
        )
    if ext == ".html":
        return _single_page_document(
            parser_name="html",
            markdown=_html_to_markdown(_read_text_file(resolved)),
        )
    if ext == ".doc":
        converted = _convert_doc_to_docx(resolved)
        try:
            return _parse_with_docling(converted, parser_name="libreoffice+docling")
        finally:
            try:
                Path(converted).unlink(missing_ok=True)
            except OSError:
                pass
    return _parse_with_docling(resolved, parser_name="docling")


def reconstruct_parsed_document(units: list[dict[str, object]]) -> ParseCacheHit | None:
    """Rebuild a parsed document from persisted page units."""
    if not units:
        return None

    ordered = sorted(units, key=lambda item: int(item["page_no"]))
    parser_name = str(ordered[0]["parser_name"])
    parser_version = str(ordered[0]["parser_version"])
    pages = tuple(
        ParsedPage(
            page_no=int(unit["page_no"]),
            markdown=str(unit["markdown"]),
            content_hash=str(unit["content_hash"]),
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
        pages=pages,
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
    units = storage.list_parsed_units(
        document_id=document_id,
        parser_version=parser_version,
    )
    target = next((unit for unit in units if int(unit["page_no"]) == page_no), None)
    if target is None:
        return 0

    images = _coerce_image_list(target.get("images"))
    if not images:
        return 0

    hashes = [str(image["image_hash"]) for image in images]
    semantics = storage.get_image_semantics(image_hashes=hashes)
    missing_hashes = [
        image_hash
        for image_hash in hashes
        if image_hash not in semantics
        or not semantics[image_hash].get("semantic_text")
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
            image_index=int(image["image_index"]),
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


def _single_page_document(*, parser_name: str, markdown: str) -> ParsedDocument:
    normalized = _normalize_markdown(markdown)
    page = ParsedPage(
        page_no=1,
        markdown=normalized,
        content_hash=_text_sha256(normalized),
    )
    return ParsedDocument(
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
        pages=(page,),
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
    return _single_page_document(parser_name=parser_name, markdown=markdown)


def _parse_pdf(file_path: str) -> ParsedDocument:
    chunks: list[tuple[int, str]] = []
    parser_name = "pymupdf4llm"
    if pymupdf4llm is not None:
        try:
            payload = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            chunks = _extract_pdf_markdown_chunks(payload)
        except Exception:
            chunks = []

    if not chunks and fitz is not None:
        parser_name = "pymupdf"
        try:
            with fitz.open(file_path) as document:
                chunks = [
                    (page.number + 1, page.get_text("markdown"))
                    for page in document
                ]
        except Exception as exc:  # pragma: no cover - exercised through error path
            raise DocumentParseError(
                file_path=file_path,
                code="pdf_parse_failed",
                message=str(exc),
            ) from exc

    if not chunks:
        raise DocumentParseError(
            file_path=file_path,
            code="pdf_parser_unavailable",
            message="Install pymupdf4llm (and PyMuPDF) to parse PDF files.",
        )

    cleaned_chunks = _strip_pdf_headers_and_footers(chunks)
    pages = []
    for page_no, markdown in cleaned_chunks:
        normalized = _normalize_markdown(markdown)
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
        pages.append(
            ParsedPage(
                page_no=page_no,
                markdown=normalized,
                content_hash=_text_sha256(normalized),
                images=images,
            )
        )

    return ParsedDocument(
        parser_name=parser_name,
        parser_version=PARSER_VERSION,
        pages=tuple(pages),
    )


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
                (index for index in range(len(trimmed) - 1, -1, -1) if trimmed[index].strip()),
                None,
            )
            if last_index is not None and trimmed[last_index] in repeated_footers:
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
