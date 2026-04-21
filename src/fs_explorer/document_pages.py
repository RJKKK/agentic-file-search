"""Helpers for page manifests backed by object-store markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .blob_store import BlobStore, LocalBlobStore, resolve_object_store_dir
from .page_store import parse_page_front_matter
from .storage import DocumentPageRecord, PostgresStorage


def page_record_from_manifest(item: Any, *, document_id: str | None = None) -> DocumentPageRecord:
    """Coerce one manifest row into the storage dataclass."""
    resolved_document_id = document_id or (
        getattr(item, "document_id", None)
        if not isinstance(item, dict)
        else item.get("document_id")
    )
    if resolved_document_id is None:
        raise ValueError("document_id is required to build a page record.")
    return DocumentPageRecord(
        document_id=str(resolved_document_id),
        page_no=int(item.page_no if hasattr(item, "page_no") else item["page_no"]),
        object_key=str(item.object_key if hasattr(item, "object_key") else item["object_key"]),
        heading=(
            item.heading if hasattr(item, "heading") else item.get("heading")
        ),
        source_locator=(
            item.source_locator
            if hasattr(item, "source_locator")
            else item.get("source_locator")
        ),
        content_hash=str(
            item.content_hash if hasattr(item, "content_hash") else item["content_hash"]
        ),
        char_count=int(item.char_count if hasattr(item, "char_count") else item["char_count"]),
        is_synthetic_page=bool(
            item.is_synthetic_page
            if hasattr(item, "is_synthetic_page")
            else item.get("is_synthetic_page", False)
        ),
    )


def resolve_pages_directory(
    *,
    blob_store: BlobStore,
    pages_prefix: str,
) -> str:
    """Return the local directory path for one document pages prefix."""
    prefix = str(pages_prefix or "").strip().rstrip("/")
    if not prefix:
        return ""
    if isinstance(blob_store, LocalBlobStore):
        return str((blob_store.root_dir / prefix).resolve())
    return str((resolve_object_store_dir() / prefix).resolve())


def read_page_content(
    *,
    blob_store: BlobStore,
    page: dict[str, Any],
) -> dict[str, Any]:
    """Load one page blob and merge front matter with manifest metadata."""
    raw = blob_store.get(object_key=str(page["object_key"])).decode("utf-8")
    header, body = parse_page_front_matter(raw)
    return {
        "page_no": int(page["page_no"]),
        "unit_no": int(page["page_no"]),
        "object_key": str(page["object_key"]),
        "heading": page.get("heading") or header.get("heading") or None,
        "source_locator": page.get("source_locator") or header.get("source_locator") or None,
        "content_hash": str(page["content_hash"]),
        "char_count": int(page.get("char_count") or len(body)),
        "is_synthetic_page": bool(page.get("is_synthetic_page", False)),
        "page_label": header.get("page_label") or str(page["page_no"]),
        "markdown": body,
        "file_path": blob_store.materialize(object_key=str(page["object_key"])),
    }


def load_document_pages(
    *,
    storage: PostgresStorage,
    blob_store: BlobStore,
    document_id: str,
    page_nos: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Load manifest rows plus markdown content for one document."""
    pages = storage.list_document_pages(document_id=document_id, page_nos=page_nos)
    return [read_page_content(blob_store=blob_store, page=page) for page in pages]


def find_page_by_path(
    *,
    storage: PostgresStorage,
    blob_store: BlobStore,
    document_ids: list[str],
    file_path: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Resolve one page file path back to (document, page)."""
    normalized = str(Path(file_path).resolve()).lower()
    documents = storage.list_documents_by_ids(doc_ids=document_ids, include_deleted=False)
    for document in documents:
        pages = storage.list_document_pages(document_id=str(document["id"]))
        for page in pages:
            local_path = blob_store.materialize(object_key=str(page["object_key"]))
            if str(Path(local_path).resolve()).lower() == normalized:
                return document, read_page_content(blob_store=blob_store, page=page)
    return None
