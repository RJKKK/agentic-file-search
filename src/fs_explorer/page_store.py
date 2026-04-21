"""Helpers for storing page-scoped document content in object storage."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .blob_store import BlobStore
from .document_parsing import ParsedDocument, ParsedUnit

_INVALID_STORAGE_FILENAME_CHARS = set('\\/:*?"<>|')


@dataclass(frozen=True)
class StoredPage:
    """One stored page/unit written to object storage."""

    page_no: int
    object_key: str
    heading: str | None
    source_locator: str | None
    content_hash: str
    char_count: int
    is_synthetic_page: bool


def validate_storage_filename(filename: str) -> str:
    """Validate that one uploaded filename can be used as an exact directory name."""
    base = Path(filename or "").name
    if not base or base in {".", ".."}:
        raise ValueError("Uploaded file must include a valid filename.")
    if any(char in _INVALID_STORAGE_FILENAME_CHARS for char in base):
        raise ValueError(
            "Filename contains characters that cannot be used as an exact storage directory name."
        )
    return base


def build_document_prefix(filename: str) -> str:
    return f"documents/{validate_storage_filename(filename)}"


def build_document_source_key(filename: str) -> str:
    validated = validate_storage_filename(filename)
    return f"{build_document_prefix(validated)}/source/{validated}"


def build_document_pages_prefix(filename: str) -> str:
    return f"{build_document_prefix(filename)}/pages"


def build_document_page_key(filename: str, page_no: int) -> str:
    return f"{build_document_pages_prefix(filename)}/page-{int(page_no):04d}.md"


def render_page_markdown(
    *,
    document_id: str,
    original_filename: str,
    page_no: int,
    page_label: str,
    content_type: str | None,
    source_locator: str | None,
    heading: str | None,
    body: str,
) -> str:
    """Render one stored page with deterministic front matter."""
    front_matter = [
        "---",
        f'document_id: "{document_id}"',
        f'original_filename: "{original_filename}"',
        f"page_no: {int(page_no)}",
        f'page_label: "{page_label}"',
        f'content_type: "{content_type or ""}"',
        f'source_locator: "{source_locator or ""}"',
        f'heading: "{heading or ""}"',
        "---",
        "",
    ]
    normalized_body = str(body or "").strip()
    return "\n".join(front_matter) + normalized_body + ("\n" if normalized_body else "")


def strip_page_front_matter(markdown: str) -> str:
    """Remove the lightweight front matter from one stored page."""
    text = str(markdown or "")
    if not text.startswith("---\n"):
        return text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return text
    return text[marker + 5 :].lstrip("\n")


def parse_page_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    """Parse simple key/value front matter plus body."""
    text = str(markdown or "")
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return {}, text
    raw_header = text[4:marker]
    body = text[marker + 5 :].lstrip("\n")
    header: dict[str, str] = {}
    for line in raw_header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned = value.strip().strip('"')
        header[key.strip()] = cleaned
    return header, body


def persist_document_pages(
    *,
    blob_store: BlobStore,
    document_id: str,
    original_filename: str,
    content_type: str | None,
    parsed_document: ParsedDocument,
    synthetic_pages: bool,
) -> list[StoredPage]:
    """Persist a parsed document as page-scoped markdown blobs."""
    pages_prefix = build_document_pages_prefix(original_filename)
    blob_store.delete_prefix(prefix=pages_prefix)
    stored_pages: list[StoredPage] = []
    ordered_units = sorted(parsed_document.units, key=lambda item: int(item.unit_no))
    for index, unit in enumerate(ordered_units, start=1):
        stored_pages.append(
            _persist_page(
                blob_store=blob_store,
                document_id=document_id,
                original_filename=original_filename,
                content_type=content_type,
                unit=unit,
                page_no=index,
                synthetic_pages=synthetic_pages,
            )
        )
    return stored_pages


def _persist_page(
    *,
    blob_store: BlobStore,
    document_id: str,
    original_filename: str,
    content_type: str | None,
    unit: ParsedUnit,
    page_no: int,
    synthetic_pages: bool,
) -> StoredPage:
    object_key = build_document_page_key(original_filename, page_no)
    page_label = str(page_no) if not synthetic_pages else f"synthetic-{page_no}"
    payload = render_page_markdown(
        document_id=document_id,
        original_filename=original_filename,
        page_no=page_no,
        page_label=page_label,
        content_type=content_type,
        source_locator=unit.source_locator,
        heading=unit.heading,
        body=unit.markdown,
    )
    blob_store.put(object_key=object_key, data=io.BytesIO(payload.encode("utf-8")))
    return StoredPage(
        page_no=page_no,
        object_key=object_key,
        heading=unit.heading,
        source_locator=unit.source_locator,
        content_hash=unit.content_hash,
        char_count=len(unit.markdown),
        is_synthetic_page=synthetic_pages,
    )
