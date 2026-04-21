"""Helpers for the single shared uploaded-document library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .blob_store import BlobStore, sanitize_filename
from .storage import CollectionRecord, PostgresStorage

LIBRARY_CORPUS_ROOT = "blob://library/default"


@dataclass(frozen=True)
class DocumentScope:
    """Resolved document selection for search and exploration."""

    corpus_id: str
    document_ids: list[str]
    documents: list[dict[str, Any]]
    collection: CollectionRecord | None = None

    @property
    def is_empty(self) -> bool:
        return not self.document_ids


def ensure_library_corpus(storage: PostgresStorage) -> str:
    """Return the single internal corpus id used by the document library."""
    return storage.get_or_create_corpus(LIBRARY_CORPUS_ROOT)


def get_library_corpus_id(
    storage: PostgresStorage,
    *,
    create_if_missing: bool = False,
) -> str | None:
    """Return the library corpus id, optionally creating it on write paths."""
    if create_if_missing:
        return ensure_library_corpus(storage)
    return storage.get_corpus_id(LIBRARY_CORPUS_ROOT)


def materialize_document(
    *,
    storage: PostgresStorage,
    blob_store: BlobStore,
    document: dict[str, Any],
) -> dict[str, Any]:
    """Ensure the document has a local readable path and persist it when changed."""
    object_key = str(document.get("object_key") or "").strip()
    if not object_key:
        return document
    materialized_path = blob_store.materialize(object_key=object_key)
    if str(document.get("absolute_path") or "") != materialized_path:
        updated = storage.update_document_absolute_path(
            doc_id=str(document["id"]),
            absolute_path=materialized_path,
        )
        if updated is not None:
            return updated
        refreshed = dict(document)
        refreshed["absolute_path"] = materialized_path
        return refreshed
    return document


def build_document_object_key(doc_id: str, filename: str) -> str:
    """Build the canonical object key for one uploaded document."""
    return f"documents/{doc_id}/{sanitize_filename(filename)}"


def resolve_document_scope(
    *,
    storage: PostgresStorage,
    document_ids: list[str] | None = None,
    collection_id: str | None = None,
) -> DocumentScope:
    """Resolve `document_ids ∪ collection.documents` and dedupe ids."""
    corpus_id = get_library_corpus_id(
        storage,
        create_if_missing=not bool(getattr(storage, "read_only", False)),
    ) or ""
    resolved_ids: list[str] = []
    collection: CollectionRecord | None = None

    if collection_id:
        collection = storage.get_collection(collection_id=collection_id)
        if collection is None or collection.is_deleted:
            raise ValueError("Collection not found.")
        for document in storage.list_collection_documents(
            collection_id=collection_id,
            include_deleted=False,
        ):
            doc_id = str(document["id"])
            if doc_id not in resolved_ids:
                resolved_ids.append(doc_id)

    for doc_id in document_ids or []:
        normalized = str(doc_id).strip()
        if normalized and normalized not in resolved_ids:
            resolved_ids.append(normalized)

    documents = storage.list_documents_by_ids(
        doc_ids=resolved_ids,
        include_deleted=False,
    )
    live_ids = [str(document["id"]) for document in documents]

    return DocumentScope(
        corpus_id=corpus_id,
        document_ids=live_ids,
        documents=documents,
        collection=collection,
    )
