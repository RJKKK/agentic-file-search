"""Indexed query helpers for agent tools."""

from __future__ import annotations

import os
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from ..embeddings import EmbeddingProvider
from ..indexing import IndexingPipeline
from ..storage import PostgresStorage, StorageBackend
from .filters import MetadataFilter, parse_metadata_filters
from .ranker import RankedDocument, rank_documents

_LAZY_INDEX_LOCKS: dict[str, threading.Lock] = {}


def _supports_document_ids(method: Callable[..., Any]) -> bool:
    """Return whether one storage method accepts a document_ids keyword."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "document_ids" in signature.parameters


def _call_storage_method(
    method: Callable[..., Any],
    *,
    corpus_id: str,
    document_ids: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call one storage method, passing document_ids only when supported."""
    call_kwargs = {"corpus_id": corpus_id, **kwargs}
    if document_ids is not None and _supports_document_ids(method):
        call_kwargs["document_ids"] = document_ids
    return method(**call_kwargs)


@dataclass(frozen=True)
class LazyIndexingStats:
    """Observable stats for on-demand indexing triggered by search."""

    triggered: bool = False
    indexed_documents: int = 0
    chunks_written: int = 0
    embeddings_written: int = 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "triggered": self.triggered,
            "indexed_documents": self.indexed_documents,
            "chunks_written": self.chunks_written,
            "embeddings_written": self.embeddings_written,
        }


@dataclass(frozen=True)
class SearchHit:
    """Ranked document hit from indexed retrieval."""

    doc_id: str
    relative_path: str
    absolute_path: str
    position: int | None
    source_unit_no: int | None
    text: str
    semantic_score: float
    metadata_score: int
    score: float
    matched_by: str


class IndexedQueryEngine:
    """Parallel retrieval engine for semantic + metadata query paths."""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider | None = None,
        runtime_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.runtime_event_callback = runtime_event_callback
        self._last_lazy_indexing_stats = LazyIndexingStats()

    def search(
        self,
        *,
        corpus_id: str,
        query: str,
        document_ids: list[str] | None = None,
        filters: str | None = None,
        limit: int = 5,
        enable_semantic: bool = True,
        enable_metadata: bool = True,
    ) -> list[SearchHit]:
        self._last_lazy_indexing_stats = LazyIndexingStats()
        self._ensure_lazy_index_ready(
            corpus_id=corpus_id,
            document_ids=document_ids,
            filters=filters,
        )
        normalized_limit = max(limit, 1)
        parsed_filters = self._parse_filters(corpus_id=corpus_id, filters=filters)
        semantic_limit = max(normalized_limit * 4, normalized_limit)
        metadata_limit = max(normalized_limit * 4, normalized_limit)

        run_semantic = enable_semantic
        run_metadata = enable_metadata and bool(parsed_filters)

        semantic_rows: list[dict[str, Any]]
        metadata_rows: list[dict[str, Any]]
        if run_semantic and run_metadata:
            semantic_rows, metadata_rows = self._search_parallel(
                corpus_id=corpus_id,
                query=query,
                document_ids=document_ids,
                metadata_filters=parsed_filters,
                semantic_limit=semantic_limit,
                metadata_limit=metadata_limit,
            )
        elif run_semantic:
            semantic_rows = self._semantic_query(
                corpus_id=corpus_id,
                query=query,
                document_ids=document_ids,
                limit=semantic_limit,
            )
            metadata_rows = []
        elif run_metadata:
            semantic_rows = []
            metadata_rows = self._metadata_query(
                corpus_id=corpus_id,
                document_ids=document_ids,
                metadata_filters=parsed_filters,
                limit=metadata_limit,
            )
        else:
            semantic_rows, metadata_rows = [], []

        ranked = self._merge_and_rank(
            semantic_rows=semantic_rows,
            metadata_rows=metadata_rows,
            limit=normalized_limit,
        )
        return [
            SearchHit(
                doc_id=doc.doc_id,
                relative_path=doc.relative_path,
                absolute_path=doc.absolute_path,
                position=doc.position,
                source_unit_no=doc.source_unit_no,
                text=doc.text,
                semantic_score=doc.semantic_score,
                metadata_score=doc.metadata_score,
                score=doc.combined_score,
                matched_by=doc.matched_by,
            )
            for doc in ranked
        ]

    def get_last_lazy_indexing_stats(self) -> dict[str, int | bool]:
        """Return lazy-indexing stats for the most recent search call."""
        return self._last_lazy_indexing_stats.as_dict()

    def _parse_filters(
        self, *, corpus_id: str, filters: str | None
    ) -> list[MetadataFilter]:
        if filters is None or not filters.strip():
            return []
        allowed_fields = self._allowed_filter_fields(corpus_id=corpus_id)
        return parse_metadata_filters(filters, allowed_fields=allowed_fields)

    def _allowed_filter_fields(self, *, corpus_id: str) -> set[str] | None:
        active_schema = self.storage.get_active_schema(corpus_id=corpus_id)
        if active_schema is None:
            return None
        fields = active_schema.schema_def.get("fields")
        if not isinstance(fields, list):
            return None
        allowed: set[str] = set()
        for field in fields:
            if isinstance(field, dict):
                name = field.get("name")
                if isinstance(name, str):
                    allowed.add(name)
        return allowed if allowed else None

    def _search_parallel(
        self,
        *,
        corpus_id: str,
        query: str,
        document_ids: list[str] | None,
        metadata_filters: list[MetadataFilter],
        semantic_limit: int,
        metadata_limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = executor.submit(
                self._semantic_query,
                corpus_id=corpus_id,
                query=query,
                document_ids=document_ids,
                limit=semantic_limit,
            )
            metadata_future = executor.submit(
                self._metadata_query,
                corpus_id=corpus_id,
                document_ids=document_ids,
                metadata_filters=metadata_filters,
                limit=metadata_limit,
            )
            semantic_rows = semantic_future.result()
            metadata_rows = metadata_future.result()
        return semantic_rows, metadata_rows

    def _semantic_query(
        self,
        *,
        corpus_id: str,
        query: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        scoped_storage, cleanup = self._acquire_query_storage()
        try:
            if self.embedding_provider is not None and scoped_storage.has_embeddings(
                corpus_id=corpus_id
            ):
                query_embedding = self.embedding_provider.embed_query(query)
                semantic_rows = _call_storage_method(
                    scoped_storage.search_chunks_semantic,
                    corpus_id=corpus_id,
                    query_embedding=query_embedding,
                    limit=limit,
                    document_ids=document_ids,
                )
                if semantic_rows:
                    return semantic_rows
            return _call_storage_method(
                scoped_storage.search_chunks,
                corpus_id=corpus_id,
                query=query,
                limit=limit,
                document_ids=document_ids,
            )
        finally:
            cleanup()

    def _metadata_query(
        self,
        *,
        corpus_id: str,
        document_ids: list[str] | None,
        metadata_filters: list[MetadataFilter],
        limit: int,
    ) -> list[dict[str, Any]]:
        scoped_storage, cleanup = self._acquire_query_storage()
        try:
            return _call_storage_method(
                scoped_storage.search_documents_by_metadata,
                corpus_id=corpus_id,
                filters=[flt.to_storage_dict() for flt in metadata_filters],
                limit=limit,
                document_ids=document_ids,
            )
        finally:
            cleanup()

    def _ensure_lazy_index_ready(
        self,
        *,
        corpus_id: str,
        document_ids: list[str] | None,
        filters: str | None,
    ) -> None:
        if not isinstance(self.storage, PostgresStorage):
            return

        active_documents = (
            self.storage.list_documents_by_ids(
                doc_ids=document_ids or [],
                include_deleted=False,
            )
            if document_ids
            else self.storage.list_documents(
                corpus_id=corpus_id,
                include_deleted=False,
            )
        )
        if not active_documents:
            return

        needs_refresh = any(self._document_needs_refresh(document) for document in active_documents)
        if not needs_refresh:
            return

        lock_key = ":".join(sorted(str(document["id"]) for document in active_documents))
        lock = _LAZY_INDEX_LOCKS.setdefault(lock_key, threading.Lock())
        with lock:
            writable_storage = PostgresStorage(self.storage.db_path)
            try:
                refreshed_docs = writable_storage.list_documents_by_ids(
                    doc_ids=[str(document["id"]) for document in active_documents],
                    include_deleted=False,
                )
                refreshed_needs_refresh = any(
                    self._document_needs_refresh(document) for document in refreshed_docs
                )
                if not refreshed_needs_refresh:
                    return

                active_schema = writable_storage.get_active_schema(corpus_id=corpus_id)
                with_metadata = bool((filters or "").strip())
                if (
                    active_schema is not None
                    and active_schema.schema_def.get("metadata_profile") is not None
                ):
                    with_metadata = True

                if self.runtime_event_callback is not None:
                    self.runtime_event_callback(
                        "lazy_indexing_started",
                        {
                            "corpus_id": corpus_id,
                            "document_ids": [str(document["id"]) for document in refreshed_docs],
                            "triggered": True,
                            "pending_documents": len(refreshed_docs),
                        },
                    )
                pipeline = IndexingPipeline(
                    storage=writable_storage,
                    embedding_provider=self.embedding_provider,
                )
                result = pipeline.index_documents(
                    refreshed_docs,
                    corpus_id=corpus_id,
                    discover_schema=with_metadata and active_schema is None,
                    with_metadata=with_metadata,
                    schema_name=active_schema.name if active_schema is not None else None,
                )
                self._last_lazy_indexing_stats = LazyIndexingStats(
                    triggered=True,
                    indexed_documents=int(result.indexed_files),
                    chunks_written=int(result.chunks_written),
                    embeddings_written=int(result.embeddings_written),
                )
                if self.runtime_event_callback is not None:
                    self.runtime_event_callback(
                        "lazy_indexing_done",
                        {
                            "corpus_id": corpus_id,
                            "document_ids": [str(document["id"]) for document in refreshed_docs],
                            **self._last_lazy_indexing_stats.as_dict(),
                        },
                    )
            finally:
                writable_storage.close()

    @staticmethod
    def _document_needs_refresh(document: dict[str, Any]) -> bool:
        indexed_hash = str(document.get("content_sha256", "") or "").strip()
        if not indexed_hash:
            return True
        absolute_path = str(document.get("absolute_path", "") or "")
        if not absolute_path or not os.path.exists(absolute_path):
            # Already-indexed documents can still be searched from stored chunks even if
            # the source file is currently unavailable or not materialized locally.
            return False
        try:
            stat = os.stat(absolute_path)
        except OSError:
            return False
        stored_mtime = float(document.get("file_mtime", 0.0) or 0.0)
        stored_size = int(document.get("file_size", 0) or 0)
        return (
            abs(float(stat.st_mtime) - stored_mtime) > 1e-6
            or int(stat.st_size) != stored_size
        )

    def _acquire_query_storage(self) -> tuple[StorageBackend, Callable[[], None]]:
        if isinstance(self.storage, PostgresStorage):
            clone = PostgresStorage(
                self.storage.db_path,
                read_only=self.storage.read_only,
                initialize=False,
                embedding_dim=self.storage.embedding_dim,
            )
            return clone, clone.close
        return self.storage, lambda: None

    @staticmethod
    def _merge_and_rank(
        *,
        semantic_rows: list[dict[str, Any]],
        metadata_rows: list[dict[str, Any]],
        limit: int,
    ) -> list[RankedDocument]:
        merged: dict[str, dict[str, Any]] = {}

        for row in semantic_rows:
            doc_id = str(row["doc_id"])
            score = float(row["score"])
            position = int(row["position"])
            entry = merged.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "relative_path": str(row["relative_path"]),
                    "absolute_path": str(row["absolute_path"]),
                    "position": position,
                    "source_unit_no": row.get("source_unit_no"),
                    "text": str(row["text"]),
                    "semantic_score": 0.0,
                    "metadata_score": 0,
                },
            )
            if score > float(entry["semantic_score"]):
                entry["semantic_score"] = score
                entry["position"] = position
                entry["source_unit_no"] = row.get("source_unit_no")
                entry["text"] = str(row["text"])

        for row in metadata_rows:
            doc_id = str(row["doc_id"])
            entry = merged.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "relative_path": str(row["relative_path"]),
                    "absolute_path": str(row["absolute_path"]),
                    "position": None,
                    "source_unit_no": None,
                    "text": str(row.get("preview_text", "")),
                    "semantic_score": 0.0,
                    "metadata_score": 0,
                },
            )
            entry["metadata_score"] = max(
                int(entry["metadata_score"]),
                int(row.get("metadata_score", 1)),
            )
            if not entry["text"]:
                entry["text"] = str(row.get("preview_text", ""))

        documents = [
            RankedDocument(
                doc_id=str(entry["doc_id"]),
                relative_path=str(entry["relative_path"]),
                absolute_path=str(entry["absolute_path"]),
                position=int(entry["position"])
                if entry["position"] is not None
                else None,
                text=str(entry["text"]),
                semantic_score=float(entry["semantic_score"]),
                metadata_score=int(entry["metadata_score"]),
                source_unit_no=int(entry["source_unit_no"])
                if entry["source_unit_no"] is not None
                else None,
            )
            for entry in merged.values()
        ]
        return rank_documents(documents, limit=limit)
