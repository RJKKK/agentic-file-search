"""
PostgreSQL storage backend for index persistence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency during lightweight tests
    psycopg = None

from .base import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    ImageSemanticRecord,
    ParsedUnitRecord,
    SchemaRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class _LocalStorageState:
    corpora: dict[str, dict[str, Any]] = field(default_factory=dict)
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    collections: dict[str, dict[str, Any]] = field(default_factory=dict)
    collection_documents: set[tuple[str, str]] = field(default_factory=set)
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    parsed_units: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    image_semantics: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunk_embeddings: dict[str, dict[str, Any]] = field(default_factory=dict)


_LOCAL_DB_REGISTRY: dict[str, _LocalStorageState] = {}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _query_terms(query: str, max_terms: int = 8) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())
    unique_terms: list[str] = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
        if len(unique_terms) >= max_terms:
            break
    if unique_terms:
        return unique_terms
    fallback = query.strip().lower()
    return [fallback] if fallback else []


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_metadata_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if not raw:
        return {}
    return json.loads(raw)


def _stringify_metadata(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", value.strip()):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class PostgresStorage:
    """PostgreSQL-backed persistence for corpora, documents, chunks, and schemas."""

    def __init__(
        self,
        db_path: str,
        *,
        read_only: bool = False,
        initialize: bool = True,
        embedding_dim: int = 768,
    ) -> None:
        self._use_local = "://" not in db_path
        self.db_path = self._normalize_dsn(db_path)
        self.read_only = read_only
        self.embedding_dim = embedding_dim
        self._vector_enabled = self._use_local
        self._conn = None
        self._state: _LocalStorageState | None = None

        if self._use_local:
            self._state = _LOCAL_DB_REGISTRY.setdefault(
                self.db_path,
                _LocalStorageState(),
            )
            if initialize and not read_only:
                self.initialize()
            return

        if psycopg is None:
            raise RuntimeError(
                "psycopg is not installed. Run `python -m pip install -e .` first."
            )

        connect_kwargs: dict[str, Any] = {}
        if read_only:
            connect_kwargs["options"] = "-c default_transaction_read_only=on"
        self._conn = psycopg.connect(self.db_path, autocommit=True, **connect_kwargs)

        if initialize and not read_only:
            self.initialize()
        else:
            self._vector_enabled = self._detect_vector_embedding_column()

    @staticmethod
    def _normalize_dsn(db_path: str) -> str:
        """
        Accept PostgreSQL DSN directly.

        For backward compatibility, if callers still pass a file-like path (e.g. *.duckdb),
        keep a local lightweight storage namespace keyed by the absolute path.
        """
        if "://" in db_path:
            return db_path
        return str(os.path.abspath(db_path))

    def close(self) -> None:
        """Close the underlying PostgreSQL connection."""
        if self._use_local:
            return
        assert self._conn is not None
        self._conn.close()

    def initialize(self) -> None:
        if self._use_local:
            return
        assert self._conn is not None
        with self._conn.cursor() as cur:
            vector_extension_available = self._ensure_vector_extension()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corpora (
                    id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    corpus_id TEXT NOT NULL REFERENCES corpora(id),
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL DEFAULT '',
                    object_key TEXT NOT NULL DEFAULT '',
                    storage_uri TEXT NOT NULL DEFAULT '',
                    content_type TEXT,
                    upload_status TEXT NOT NULL DEFAULT 'indexed',
                    content TEXT NOT NULL,
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    file_mtime DOUBLE PRECISION NOT NULL,
                    file_size BIGINT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    parsed_content_sha256 TEXT,
                    parsed_is_complete BOOLEAN NOT NULL DEFAULT FALSE,
                    last_indexed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    is_deleted BOOLEAN DEFAULT FALSE,
                    UNIQUE(corpus_id, relative_path)
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS parsed_content_sha256 TEXT
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS parsed_is_complete BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS original_filename TEXT NOT NULL DEFAULT ''
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS object_key TEXT NOT NULL DEFAULT ''
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS storage_uri TEXT NOT NULL DEFAULT ''
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS content_type TEXT
                """
            )
            cur.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS upload_status TEXT NOT NULL DEFAULT 'indexed'
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id),
                    text TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    source_unit_no INTEGER,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS source_unit_no INTEGER
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_units (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    parser_name TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    page_no INTEGER NOT NULL,
                    markdown TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    heading TEXT,
                    source_locator TEXT,
                    images_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, page_no, parser_version)
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE parsed_units
                ADD COLUMN IF NOT EXISTS heading TEXT
                """
            )
            cur.execute(
                """
                ALTER TABLE parsed_units
                ADD COLUMN IF NOT EXISTS source_locator TEXT
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS image_semantics (
                    image_hash TEXT PRIMARY KEY,
                    source_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_page_no INTEGER NOT NULL,
                    source_image_index INTEGER NOT NULL,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    semantic_text TEXT,
                    semantic_model TEXT,
                    last_enhanced_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schemas (
                    id TEXT PRIMARY KEY,
                    corpus_id TEXT NOT NULL REFERENCES corpora(id),
                    name TEXT NOT NULL,
                    schema_def JSONB NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(corpus_id, name)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_documents (
                    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (collection_id, document_id)
                )
                """
            )
            if vector_extension_available:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
                        corpus_id TEXT NOT NULL,
                        embedding VECTOR({self.embedding_dim}) NOT NULL
                    )
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
                        corpus_id TEXT NOT NULL,
                        embedding_json JSONB NOT NULL
                    )
                    """
                )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_corpus_active
                ON documents (corpus_id, is_deleted)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_object_key
                ON documents (object_key)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_position
                ON chunks (doc_id, position)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_unit
                ON chunks (doc_id, source_unit_no)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parsed_units_document_version
                ON parsed_units (document_id, parser_version, page_no)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_semantics_source_page
                ON image_semantics (source_document_id, source_page_no)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_corpus
                ON chunk_embeddings (corpus_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection_documents_collection
                ON collection_documents (collection_id, document_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
                ON chunks USING GIN (to_tsvector('simple', text))
                """
            )

        self._vector_enabled = self._detect_vector_embedding_column()

    def _ensure_vector_extension(self) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            return True
        except Exception as exc:
            logger.warning(
                "pgvector extension is not available in PostgreSQL (%s). "
                "Semantic vector search will be disabled.",
                exc,
            )
            return False

    def _detect_vector_embedding_column(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chunk_embeddings'
                      AND column_name = 'embedding'
                      AND udt_name = 'vector'
                )
                """
            )
            row = cur.fetchone()
        return bool(row and row[0])

    @staticmethod
    def _to_vector_literal(embedding: list[float]) -> str:
        values = ",".join(f"{float(v):.12g}" for v in embedding)
        return f"[{values}]"

    def _local(self) -> _LocalStorageState:
        if self._state is None:
            raise RuntimeError("Local storage state is not initialized.")
        return self._state

    def _local_documents_for_corpus(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        docs = [
            copy.deepcopy(document)
            for document in self._local().documents.values()
            if document["corpus_id"] == corpus_id
            and (include_deleted or not bool(document["is_deleted"]))
        ]
        docs.sort(key=lambda item: str(item["relative_path"]))
        return docs

    def _local_documents_by_ids(
        self,
        *,
        doc_ids: list[str],
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = {str(doc_id) for doc_id in doc_ids}
        docs = [
            copy.deepcopy(document)
            for document in self._local().documents.values()
            if str(document["id"]) in wanted
            and (include_deleted or not bool(document["is_deleted"]))
        ]
        docs.sort(key=lambda item: str(item["relative_path"]))
        return docs

    @staticmethod
    def _collection_record_from_dict(record: dict[str, Any]) -> CollectionRecord:
        return CollectionRecord(
            id=str(record["id"]),
            name=str(record["name"]),
            is_deleted=bool(record.get("is_deleted", False)),
            created_at=str(record["created_at"]),
            updated_at=str(record["updated_at"]),
        )

    @classmethod
    def _local_matches_metadata_filter(
        cls,
        metadata: dict[str, Any],
        flt: dict[str, Any],
    ) -> bool:
        field = str(flt["field"])
        operator = str(flt["operator"])
        expected = flt["value"]
        actual = metadata.get(field)

        if operator == "eq":
            if isinstance(expected, str):
                return str(actual).lower() == expected.lower()
            return actual == expected
        if operator == "ne":
            if isinstance(expected, str):
                return str(actual).lower() != expected.lower()
            return actual != expected
        if operator in {"gt", "gte", "lt", "lte"}:
            left = _numeric_value(actual)
            right = _numeric_value(expected)
            if left is None or right is None:
                return False
            if operator == "gt":
                return left > right
            if operator == "gte":
                return left >= right
            if operator == "lt":
                return left < right
            return left <= right
        if operator == "contains":
            return str(expected).lower() in str(actual or "").lower()
        if operator == "in":
            if not isinstance(expected, list):
                return False
            if isinstance(actual, str):
                return actual.lower() in {str(item).lower() for item in expected}
            return actual in expected
        raise ValueError(f"Unsupported metadata operator: {operator!r}")

    def get_or_create_corpus(self, root_path: str) -> str:
        if self._use_local:
            corpus_id = _stable_id("corpus", root_path)
            state = self._local()
            state.corpora.setdefault(
                root_path,
                {
                    "id": corpus_id,
                    "root_path": root_path,
                    "created_at": _utcnow_iso(),
                },
            )
            return str(state.corpora[root_path]["id"])
        corpus_id = _stable_id("corpus", root_path)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpora (id, root_path)
                VALUES (%s, %s)
                ON CONFLICT(root_path) DO NOTHING
                """,
                (corpus_id, root_path),
            )
            cur.execute(
                "SELECT id FROM corpora WHERE root_path = %s",
                (root_path,),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to create corpus for path: {root_path}")
        return str(row[0])

    def get_corpus_id(self, root_path: str) -> str | None:
        if self._use_local:
            record = self._local().corpora.get(root_path)
            return None if record is None else str(record["id"])
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM corpora WHERE root_path = %s",
                (root_path,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    def get_corpus_root_path(self, *, corpus_id: str) -> str | None:
        if self._use_local:
            for record in self._local().corpora.values():
                if str(record["id"]) == corpus_id:
                    return str(record["root_path"])
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT root_path FROM corpora WHERE id = %s",
                (corpus_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    def upsert_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        if self._use_local:
            state = self._local()
            state.documents[document.id] = {
                "id": document.id,
                "corpus_id": document.corpus_id,
                "relative_path": document.relative_path,
                "absolute_path": document.absolute_path,
                "original_filename": document.original_filename or document.relative_path,
                "object_key": document.object_key,
                "storage_uri": document.storage_uri,
                "content_type": document.content_type,
                "upload_status": document.upload_status or "indexed",
                "content": document.content,
                "metadata_json": document.metadata_json,
                "file_mtime": document.file_mtime,
                "file_size": document.file_size,
                "content_sha256": document.content_sha256,
                "parsed_content_sha256": document.content_sha256,
                "parsed_is_complete": True,
                "last_indexed_at": _utcnow_iso(),
                "is_deleted": False,
            }
            chunk_ids_to_delete = [
                chunk_id
                for chunk_id, item in state.chunks.items()
                if item["doc_id"] == document.id
            ]
            for chunk_id in chunk_ids_to_delete:
                state.chunks.pop(chunk_id, None)
                state.chunk_embeddings.pop(chunk_id, None)
            for chunk in chunks:
                state.chunks[chunk.id] = {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "position": chunk.position,
                    "source_unit_no": chunk.source_unit_no,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = %s)
                """,
                (document.id,),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (document.id,))

            cur.execute(
                """
                INSERT INTO documents (
                    id, corpus_id, relative_path, absolute_path, original_filename,
                    object_key, storage_uri, content_type, upload_status, content, metadata_json,
                    file_mtime, file_size, content_sha256,
                    parsed_content_sha256, parsed_is_complete, is_deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, TRUE, FALSE)
                ON CONFLICT(id) DO UPDATE SET
                    corpus_id = EXCLUDED.corpus_id,
                    relative_path = EXCLUDED.relative_path,
                    absolute_path = EXCLUDED.absolute_path,
                    original_filename = EXCLUDED.original_filename,
                    object_key = EXCLUDED.object_key,
                    storage_uri = EXCLUDED.storage_uri,
                    content_type = EXCLUDED.content_type,
                    upload_status = EXCLUDED.upload_status,
                    content = EXCLUDED.content,
                    metadata_json = EXCLUDED.metadata_json,
                    file_mtime = EXCLUDED.file_mtime,
                    file_size = EXCLUDED.file_size,
                    content_sha256 = EXCLUDED.content_sha256,
                    parsed_content_sha256 = EXCLUDED.parsed_content_sha256,
                    parsed_is_complete = TRUE,
                    last_indexed_at = CURRENT_TIMESTAMP,
                    is_deleted = FALSE
                """,
                (
                    document.id,
                    document.corpus_id,
                    document.relative_path,
                    document.absolute_path,
                    document.original_filename or document.relative_path,
                    document.object_key,
                    document.storage_uri,
                    document.content_type,
                    document.upload_status or "indexed",
                    document.content,
                    document.metadata_json,
                    document.file_mtime,
                    document.file_size,
                    document.content_sha256,
                    document.content_sha256,
                ),
            )

            if chunks:
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        id, doc_id, text, position, source_unit_no, start_char, end_char
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.doc_id,
                            chunk.text,
                            chunk.position,
                            chunk.source_unit_no,
                            chunk.start_char,
                            chunk.end_char,
                        )
                        for chunk in chunks
                    ],
                )

    def upsert_document_stub(self, document: DocumentRecord) -> None:
        if self._use_local:
            state = self._local()
            state.documents[document.id] = {
                "id": document.id,
                "corpus_id": document.corpus_id,
                "relative_path": document.relative_path,
                "absolute_path": document.absolute_path,
                "original_filename": document.original_filename or document.relative_path,
                "object_key": document.object_key,
                "storage_uri": document.storage_uri,
                "content_type": document.content_type,
                "upload_status": document.upload_status or "uploaded",
                "content": "",
                "metadata_json": document.metadata_json,
                "file_mtime": document.file_mtime,
                "file_size": document.file_size,
                "content_sha256": "",
                "parsed_content_sha256": None,
                "parsed_is_complete": False,
                "last_indexed_at": _utcnow_iso(),
                "is_deleted": False,
            }
            chunk_ids_to_delete = [
                chunk_id
                for chunk_id, item in state.chunks.items()
                if item["doc_id"] == document.id
            ]
            for chunk_id in chunk_ids_to_delete:
                state.chunks.pop(chunk_id, None)
                state.chunk_embeddings.pop(chunk_id, None)
            stale_parsed_keys = [
                key for key in list(state.parsed_units) if key[0] == document.id
            ]
            for key in stale_parsed_keys:
                state.parsed_units.pop(key, None)
            stale_image_hashes = [
                image_hash
                for image_hash, item in list(state.image_semantics.items())
                if item["source_document_id"] == document.id
            ]
            for image_hash in stale_image_hashes:
                state.image_semantics.pop(image_hash, None)
            return

        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = %s)
                """,
                (document.id,),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (document.id,))
            cur.execute("DELETE FROM parsed_units WHERE document_id = %s", (document.id,))
            cur.execute(
                "DELETE FROM image_semantics WHERE source_document_id = %s",
                (document.id,),
            )
            cur.execute(
                """
                INSERT INTO documents (
                    id, corpus_id, relative_path, absolute_path, original_filename,
                    object_key, storage_uri, content_type, upload_status, content, metadata_json,
                    file_mtime, file_size, content_sha256,
                    parsed_content_sha256, parsed_is_complete, is_deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, '', %s::jsonb, %s, %s, '', NULL, FALSE, FALSE)
                ON CONFLICT(id) DO UPDATE SET
                    corpus_id = EXCLUDED.corpus_id,
                    relative_path = EXCLUDED.relative_path,
                    absolute_path = EXCLUDED.absolute_path,
                    original_filename = EXCLUDED.original_filename,
                    object_key = EXCLUDED.object_key,
                    storage_uri = EXCLUDED.storage_uri,
                    content_type = EXCLUDED.content_type,
                    upload_status = EXCLUDED.upload_status,
                    content = '',
                    metadata_json = EXCLUDED.metadata_json,
                    file_mtime = EXCLUDED.file_mtime,
                    file_size = EXCLUDED.file_size,
                    content_sha256 = '',
                    parsed_content_sha256 = NULL,
                    parsed_is_complete = FALSE,
                    last_indexed_at = CURRENT_TIMESTAMP,
                    is_deleted = FALSE
                """,
                (
                    document.id,
                    document.corpus_id,
                    document.relative_path,
                    document.absolute_path,
                    document.original_filename or document.relative_path,
                    document.object_key,
                    document.storage_uri,
                    document.content_type,
                    document.upload_status or "uploaded",
                    document.metadata_json,
                    document.file_mtime,
                    document.file_size,
                ),
            )

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
        if self._use_local:
            deleted = 0
            for document in self._local().documents.values():
                if document["corpus_id"] != corpus_id:
                    continue
                should_delete = document["relative_path"] not in active_relative_paths
                if should_delete and not bool(document["is_deleted"]):
                    document["is_deleted"] = True
                    deleted += 1
                if not should_delete:
                    document["is_deleted"] = False
            return deleted
        with self._conn.cursor() as cur:
            if not active_relative_paths:
                cur.execute(
                    """
                    UPDATE documents
                    SET is_deleted = TRUE
                    WHERE corpus_id = %s AND is_deleted = FALSE
                    """,
                    (corpus_id,),
                )
            else:
                placeholders = ", ".join(["%s"] * len(active_relative_paths))
                cur.execute(
                    f"""
                    UPDATE documents
                    SET is_deleted = TRUE
                    WHERE corpus_id = %s
                      AND is_deleted = FALSE
                      AND relative_path NOT IN ({placeholders})
                    """,
                    (corpus_id, *sorted(active_relative_paths)),
                )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE corpus_id = %s AND is_deleted = TRUE
                """,
                (corpus_id,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_documents(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        if self._use_local:
            return [
                {
                    "id": str(document["id"]),
                    "corpus_id": str(document["corpus_id"]),
                    "relative_path": str(document["relative_path"]),
                    "absolute_path": str(document["absolute_path"]),
                    "original_filename": str(document.get("original_filename") or document["relative_path"]),
                    "object_key": str(document.get("object_key") or ""),
                    "storage_uri": str(document.get("storage_uri") or ""),
                    "content_type": document.get("content_type"),
                    "upload_status": str(document.get("upload_status") or "uploaded"),
                    "file_size": int(document["file_size"]),
                    "file_mtime": float(document["file_mtime"]),
                    "metadata_json": str(document["metadata_json"]),
                    "content_sha256": str(document["content_sha256"]),
                    "parsed_content_sha256": document.get("parsed_content_sha256"),
                    "parsed_is_complete": bool(document.get("parsed_is_complete", False)),
                    "last_indexed_at": str(document["last_indexed_at"]),
                    "is_deleted": bool(document["is_deleted"]),
                }
                for document in self._local_documents_for_corpus(
                    corpus_id=corpus_id,
                    include_deleted=include_deleted,
                )
            ]
        sql = """
            SELECT
                id,
                corpus_id,
                relative_path,
                absolute_path,
                original_filename,
                object_key,
                storage_uri,
                content_type,
                upload_status,
                file_size,
                file_mtime,
                metadata_json,
                content_sha256,
                parsed_content_sha256,
                parsed_is_complete,
                last_indexed_at,
                is_deleted
            FROM documents
            WHERE corpus_id = %s
        """
        params: list[Any] = [corpus_id]
        if not include_deleted:
            sql += " AND is_deleted = FALSE"
        sql += " ORDER BY relative_path"

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata_json_obj = row[11]
            if isinstance(metadata_json_obj, str):
                metadata_json = metadata_json_obj
            else:
                metadata_json = json.dumps(metadata_json_obj, sort_keys=True)
            results.append(
                {
                    "id": str(row[0]),
                    "corpus_id": str(row[1]),
                    "relative_path": str(row[2]),
                    "absolute_path": str(row[3]),
                    "original_filename": str(row[4] or row[2]),
                    "object_key": str(row[5] or ""),
                    "storage_uri": str(row[6] or ""),
                    "content_type": row[7],
                    "upload_status": str(row[8] or "uploaded"),
                    "file_size": int(row[9]),
                    "file_mtime": float(row[10]),
                    "metadata_json": metadata_json,
                    "content_sha256": str(row[12]),
                    "parsed_content_sha256": row[13],
                    "parsed_is_complete": bool(row[14]),
                    "last_indexed_at": row[15].isoformat()
                    if hasattr(row[15], "isoformat")
                    else str(row[15]),
                    "is_deleted": bool(row[16]),
                }
            )
        return results

    def list_documents_by_ids(
        self,
        *,
        doc_ids: list[str],
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        if self._use_local:
            return self._local_documents_by_ids(
                doc_ids=doc_ids,
                include_deleted=include_deleted,
            )

        placeholders = ", ".join(["%s"] * len(doc_ids))
        sql = f"""
            SELECT
                id,
                corpus_id,
                relative_path,
                absolute_path,
                original_filename,
                object_key,
                storage_uri,
                content_type,
                upload_status,
                file_size,
                file_mtime,
                metadata_json,
                content_sha256,
                parsed_content_sha256,
                parsed_is_complete,
                last_indexed_at,
                is_deleted
            FROM documents
            WHERE id IN ({placeholders})
        """
        params: list[Any] = [*doc_ids]
        if not include_deleted:
            sql += " AND is_deleted = FALSE"
        sql += " ORDER BY relative_path"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata_json_obj = row[11]
            if isinstance(metadata_json_obj, str):
                metadata_json = metadata_json_obj
            else:
                metadata_json = json.dumps(metadata_json_obj, sort_keys=True)
            results.append(
                {
                    "id": str(row[0]),
                    "corpus_id": str(row[1]),
                    "relative_path": str(row[2]),
                    "absolute_path": str(row[3]),
                    "original_filename": str(row[4] or row[2]),
                    "object_key": str(row[5] or ""),
                    "storage_uri": str(row[6] or ""),
                    "content_type": row[7],
                    "upload_status": str(row[8] or "uploaded"),
                    "file_size": int(row[9]),
                    "file_mtime": float(row[10]),
                    "metadata_json": metadata_json,
                    "content_sha256": str(row[12]),
                    "parsed_content_sha256": row[13],
                    "parsed_is_complete": bool(row[14]),
                    "last_indexed_at": row[15].isoformat()
                    if hasattr(row[15], "isoformat")
                    else str(row[15]),
                    "is_deleted": bool(row[16]),
                }
            )
        return results

    def count_chunks(self, *, corpus_id: str) -> int:
        if self._use_local:
            active_doc_ids = {
                document["id"]
                for document in self._local_documents_for_corpus(corpus_id=corpus_id)
            }
            return sum(
                1
                for chunk in self._local().chunks.values()
                if chunk["doc_id"] in active_doc_ids
            )
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.corpus_id = %s AND d.is_deleted = FALSE
                """,
                (corpus_id,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def search_chunks(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._use_local:
            terms = _query_terms(query)
            if not terms:
                return []
            allowed_doc_ids = {str(doc_id) for doc_id in document_ids or []}
            documents = {
                document["id"]: document
                for document in self._local_documents_for_corpus(corpus_id=corpus_id)
                if not allowed_doc_ids or str(document["id"]) in allowed_doc_ids
            }
            hits: list[dict[str, Any]] = []
            for chunk in self._local().chunks.values():
                document = documents.get(chunk["doc_id"])
                if document is None:
                    continue
                text = str(chunk["text"]).lower()
                score = sum(1 for term in terms if term in text)
                if score <= 0:
                    continue
                hits.append(
                    {
                        "doc_id": str(document["id"]),
                        "relative_path": str(document["relative_path"]),
                        "absolute_path": str(document["absolute_path"]),
                        "position": int(chunk["position"]),
                        "source_unit_no": _optional_int(chunk.get("source_unit_no")),
                        "text": str(chunk["text"]),
                        "score": int(score),
                    }
                )
            hits.sort(key=lambda item: (-int(item["score"]), item["relative_path"], int(item["position"])))
            return hits[:limit]
        terms = _query_terms(query)
        if not terms:
            return []

        score_expr = " + ".join(
            ["CASE WHEN lower(c.text) LIKE '%%' || %s || '%%' THEN 1 ELSE 0 END"]
            * len(terms)
        )
        sql = f"""
            SELECT * FROM (
                SELECT
                    d.id AS doc_id,
                    d.relative_path,
                    d.absolute_path,
                    c.position,
                    c.source_unit_no,
                    c.text,
                    ({score_expr}) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.corpus_id = %s
                  AND d.is_deleted = FALSE
            ) ranked
            WHERE score > 0
            ORDER BY score DESC, relative_path ASC, position ASC
            LIMIT %s
        """
        params: list[Any] = [*terms, corpus_id]
        if document_ids:
            placeholders = ", ".join(["%s"] * len(document_ids))
            sql = sql.replace(
                "AND d.is_deleted = FALSE",
                f"AND d.is_deleted = FALSE AND d.id IN ({placeholders})",
            )
            params.extend(document_ids)
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "doc_id": str(row[0]),
                    "relative_path": str(row[1]),
                    "absolute_path": str(row[2]),
                    "position": int(row[3]),
                    "source_unit_no": _optional_int(row[4]),
                    "text": str(row[5]),
                    "score": int(row[6]),
                }
            )
        return results

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, Any]],
        limit: int = 20,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._use_local:
            if not filters:
                return []
            allowed_doc_ids = {str(doc_id) for doc_id in document_ids or []}
            matches: list[dict[str, Any]] = []
            for document in self._local_documents_for_corpus(corpus_id=corpus_id):
                if allowed_doc_ids and str(document["id"]) not in allowed_doc_ids:
                    continue
                metadata = _parse_metadata_json(str(document["metadata_json"]))
                if not all(
                    self._local_matches_metadata_filter(metadata, flt)
                    for flt in filters
                ):
                    continue
                matches.append(
                    {
                        "doc_id": str(document["id"]),
                        "relative_path": str(document["relative_path"]),
                        "absolute_path": str(document["absolute_path"]),
                        "preview_text": str(document["content"])[:320],
                        "metadata_score": len(filters),
                    }
                )
            matches.sort(key=lambda item: item["relative_path"])
            return matches[:limit]
        if not filters:
            return []

        sql = """
            SELECT
                d.id,
                d.relative_path,
                d.absolute_path,
                substring(d.content, 1, 320) AS preview_text
            FROM documents d
            WHERE d.corpus_id = %s
              AND d.is_deleted = FALSE
        """
        params: list[Any] = [corpus_id]
        if document_ids:
            placeholders = ", ".join(["%s"] * len(document_ids))
            sql += f"\n  AND d.id IN ({placeholders})"
            params.extend(document_ids)

        for flt in filters:
            field = str(flt["field"])
            operator = str(flt["operator"])
            value = flt["value"]
            clause, clause_params = self._metadata_clause(
                field=field,
                operator=operator,
                value=value,
            )
            sql += f"\n  AND {clause}"
            params.extend(clause_params)

        sql += "\nORDER BY d.relative_path ASC\nLIMIT %s"
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        metadata_score = len(filters)
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "doc_id": str(row[0]),
                    "relative_path": str(row[1]),
                    "absolute_path": str(row[2]),
                    "preview_text": str(row[3]),
                    "metadata_score": metadata_score,
                }
            )
        return results

    def get_document(self, *, doc_id: str) -> dict[str, Any] | None:
        if self._use_local:
            document = self._local().documents.get(doc_id)
            return None if document is None else copy.deepcopy(document)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, corpus_id, relative_path, absolute_path, original_filename,
                    object_key, storage_uri, content_type, upload_status, content, metadata_json,
                    file_size, file_mtime, content_sha256,
                    parsed_content_sha256, parsed_is_complete, last_indexed_at, is_deleted
                FROM documents
                WHERE id = %s
                LIMIT 1
                """,
                (doc_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        metadata_json_obj = row[10]
        if isinstance(metadata_json_obj, str):
            metadata_json = metadata_json_obj
        else:
            metadata_json = json.dumps(metadata_json_obj, sort_keys=True)
        return {
            "id": str(row[0]),
            "corpus_id": str(row[1]),
            "relative_path": str(row[2]),
            "absolute_path": str(row[3]),
            "original_filename": str(row[4] or row[2]),
            "object_key": str(row[5] or ""),
            "storage_uri": str(row[6] or ""),
            "content_type": row[7],
            "upload_status": str(row[8] or "uploaded"),
            "content": str(row[9]),
            "metadata_json": metadata_json,
            "file_size": int(row[11]),
            "file_mtime": float(row[12]),
            "content_sha256": str(row[13]),
            "parsed_content_sha256": row[14],
            "parsed_is_complete": bool(row[15]),
            "last_indexed_at": row[16].isoformat()
            if hasattr(row[16], "isoformat")
            else str(row[16]),
            "is_deleted": bool(row[17]),
        }

    def update_document_metadata(
        self,
        *,
        doc_id: str,
        metadata_json: str,
    ) -> dict[str, Any] | None:
        parsed_metadata = _parse_metadata_json(metadata_json)
        if self._use_local:
            document = self._local().documents.get(doc_id)
            if document is None:
                return None
            document["metadata_json"] = _stringify_metadata(parsed_metadata)
            document["last_indexed_at"] = _utcnow_iso()
            return copy.deepcopy(document)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET metadata_json = %s::jsonb,
                    last_indexed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (_stringify_metadata(parsed_metadata), doc_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_document(doc_id=doc_id)

    def set_document_deleted(
        self,
        *,
        doc_id: str,
        is_deleted: bool,
    ) -> dict[str, Any] | None:
        if self._use_local:
            document = self._local().documents.get(doc_id)
            if document is None:
                return None
            document["is_deleted"] = bool(is_deleted)
            document["last_indexed_at"] = _utcnow_iso()
            return copy.deepcopy(document)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET is_deleted = %s,
                    last_indexed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (is_deleted, doc_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_document(doc_id=doc_id)

    def update_document_absolute_path(
        self,
        *,
        doc_id: str,
        absolute_path: str,
    ) -> dict[str, Any] | None:
        if self._use_local:
            document = self._local().documents.get(doc_id)
            if document is None:
                return None
            document["absolute_path"] = absolute_path
            return copy.deepcopy(document)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET absolute_path = %s
                WHERE id = %s
                """,
                (absolute_path, doc_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_document(doc_id=doc_id)

    def update_document_parse_state(
        self,
        *,
        doc_id: str,
        parsed_content_sha256: str | None,
        parsed_is_complete: bool,
    ) -> dict[str, Any] | None:
        if self._use_local:
            document = self._local().documents.get(doc_id)
            if document is None:
                return None
            document["parsed_content_sha256"] = parsed_content_sha256
            if parsed_is_complete:
                document["parsed_is_complete"] = True
            else:
                document["parsed_is_complete"] = bool(document.get("parsed_is_complete", False))
            document["last_indexed_at"] = _utcnow_iso()
            return copy.deepcopy(document)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET parsed_content_sha256 = %s,
                    parsed_is_complete = CASE
                        WHEN %s THEN TRUE
                        ELSE parsed_is_complete
                    END,
                    last_indexed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (parsed_content_sha256, parsed_is_complete, doc_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_document(doc_id=doc_id)

    def list_parsed_units(
        self,
        *,
        document_id: str,
        parser_version: str | None = None,
        unit_nos: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if unit_nos is not None and len(unit_nos) == 0:
            return []
        if self._use_local:
            allowed_units = set(unit_nos) if unit_nos is not None else None
            result = []
            for (doc_id, version, page_no), item in self._local().parsed_units.items():
                if doc_id != document_id:
                    continue
                if parser_version is not None and version != parser_version:
                    continue
                if allowed_units is not None and page_no not in allowed_units:
                    continue
                result.append(copy.deepcopy(item))
            result.sort(key=lambda item: int(item["page_no"]))
            return result
        sql = """
            SELECT
                document_id, parser_name, parser_version, page_no, markdown, content_hash,
                heading, source_locator, images_json
            FROM parsed_units
            WHERE document_id = %s
        """
        params: list[Any] = [document_id]
        if parser_version is not None:
            sql += " AND parser_version = %s"
            params.append(parser_version)
        if unit_nos:
            placeholders = ", ".join(["%s"] * len(unit_nos))
            sql += f" AND page_no IN ({placeholders})"
            params.extend(sorted(set(unit_nos)))
        sql += " ORDER BY page_no ASC"

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            images_obj = row[8]
            if isinstance(images_obj, list):
                images = images_obj
            else:
                images = json.loads(str(images_obj))
            results.append(
                {
                    "document_id": str(row[0]),
                    "parser_name": str(row[1]),
                    "parser_version": str(row[2]),
                    "page_no": int(row[3]),
                    "markdown": str(row[4]),
                    "content_hash": str(row[5]),
                    "heading": row[6],
                    "source_locator": row[7],
                    "images": images,
                }
            )
        return results

    def sync_parsed_units(
        self,
        *,
        document_id: str,
        parser_name: str,
        parser_version: str,
        units: list[ParsedUnitRecord],
    ) -> dict[str, int]:
        if self._use_local:
            state = self._local()
            existing = {
                int(item["page_no"]): item
                for item in self.list_parsed_units(
                    document_id=document_id,
                    parser_version=parser_version,
                )
            }
            upserted = 0
            untouched = 0
            desired_pages = {unit.page_no for unit in units}
            for unit in units:
                current = {
                    "document_id": document_id,
                    "parser_name": parser_name,
                    "parser_version": parser_version,
                    "page_no": unit.page_no,
                    "markdown": unit.markdown,
                    "content_hash": unit.content_hash,
                    "heading": unit.heading,
                    "source_locator": unit.source_locator,
                    "images": json.loads(unit.images_json),
                }
                previous = existing.get(unit.page_no)
                if previous == current:
                    untouched += 1
                else:
                    upserted += 1
                state.parsed_units[(document_id, parser_version, unit.page_no)] = current

            deleted = 0
            stale_keys = [
                key
                for key in list(state.parsed_units)
                if key[0] == document_id
                and key[1] == parser_version
                and key[2] not in desired_pages
            ]
            for key in stale_keys:
                state.parsed_units.pop(key, None)
                deleted += 1
            return {"upserted": upserted, "untouched": untouched, "deleted": deleted}
        existing = {
            int(unit["page_no"]): unit
            for unit in self.list_parsed_units(
                document_id=document_id,
                parser_version=parser_version,
            )
        }
        upserted = 0
        untouched = 0
        desired_pages = {unit.page_no for unit in units}

        with self._conn.cursor() as cur:
            for unit in units:
                previous = existing.get(unit.page_no)
                previous_images = previous.get("images") if previous is not None else []
                current_images = json.loads(unit.images_json)
                if (
                    previous is not None
                    and previous["content_hash"] == unit.content_hash
                    and previous["markdown"] == unit.markdown
                    and previous.get("heading") == unit.heading
                    and previous.get("source_locator") == unit.source_locator
                    and previous_images == current_images
                ):
                    untouched += 1
                    continue

                cur.execute(
                    """
                    INSERT INTO parsed_units (
                        document_id, parser_name, parser_version, page_no, markdown,
                        content_hash, heading, source_locator, images_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT(document_id, page_no, parser_version) DO UPDATE SET
                        parser_name = EXCLUDED.parser_name,
                        markdown = EXCLUDED.markdown,
                        content_hash = EXCLUDED.content_hash,
                        heading = EXCLUDED.heading,
                        source_locator = EXCLUDED.source_locator,
                        images_json = EXCLUDED.images_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        document_id,
                        parser_name,
                        parser_version,
                        unit.page_no,
                        unit.markdown,
                        unit.content_hash,
                        unit.heading,
                        unit.source_locator,
                        unit.images_json,
                    ),
                )
                upserted += 1

            if desired_pages:
                placeholders = ", ".join(["%s"] * len(desired_pages))
                cur.execute(
                    f"""
                    DELETE FROM parsed_units
                    WHERE document_id = %s
                      AND parser_version = %s
                      AND page_no NOT IN ({placeholders})
                    """,
                    (document_id, parser_version, *sorted(desired_pages)),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM parsed_units
                    WHERE document_id = %s AND parser_version = %s
                    """,
                    (document_id, parser_version),
                )
            deleted = cur.rowcount if cur.rowcount is not None else 0

        return {"upserted": upserted, "untouched": untouched, "deleted": int(deleted)}

    def upsert_parsed_units(
        self,
        *,
        document_id: str,
        parser_name: str,
        parser_version: str,
        units: list[ParsedUnitRecord],
    ) -> dict[str, int]:
        if self._use_local:
            state = self._local()
            existing = {
                int(item["page_no"]): item
                for item in self.list_parsed_units(
                    document_id=document_id,
                    parser_version=parser_version,
                )
            }
            upserted = 0
            untouched = 0
            for unit in units:
                current = {
                    "document_id": document_id,
                    "parser_name": parser_name,
                    "parser_version": parser_version,
                    "page_no": unit.page_no,
                    "markdown": unit.markdown,
                    "content_hash": unit.content_hash,
                    "heading": unit.heading,
                    "source_locator": unit.source_locator,
                    "images": json.loads(unit.images_json),
                }
                previous = existing.get(unit.page_no)
                if previous == current:
                    untouched += 1
                else:
                    upserted += 1
                state.parsed_units[(document_id, parser_version, unit.page_no)] = current
            return {"upserted": upserted, "untouched": untouched, "deleted": 0}

        existing = {
            int(unit["page_no"]): unit
            for unit in self.list_parsed_units(
                document_id=document_id,
                parser_version=parser_version,
            )
        }
        upserted = 0
        untouched = 0
        with self._conn.cursor() as cur:
            for unit in units:
                previous = existing.get(unit.page_no)
                previous_images = previous.get("images") if previous is not None else []
                current_images = json.loads(unit.images_json)
                if (
                    previous is not None
                    and previous["content_hash"] == unit.content_hash
                    and previous["markdown"] == unit.markdown
                    and previous.get("heading") == unit.heading
                    and previous.get("source_locator") == unit.source_locator
                    and previous_images == current_images
                ):
                    untouched += 1
                    continue
                cur.execute(
                    """
                    INSERT INTO parsed_units (
                        document_id, parser_name, parser_version, page_no, markdown,
                        content_hash, heading, source_locator, images_json, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT(document_id, page_no, parser_version) DO UPDATE SET
                        parser_name = EXCLUDED.parser_name,
                        markdown = EXCLUDED.markdown,
                        content_hash = EXCLUDED.content_hash,
                        heading = EXCLUDED.heading,
                        source_locator = EXCLUDED.source_locator,
                        images_json = EXCLUDED.images_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        document_id,
                        parser_name,
                        parser_version,
                        unit.page_no,
                        unit.markdown,
                        unit.content_hash,
                        unit.heading,
                        unit.source_locator,
                        unit.images_json,
                    ),
                )
                upserted += 1
        return {"upserted": upserted, "untouched": untouched, "deleted": 0}

    def upsert_image_semantics(
        self,
        *,
        images: list[ImageSemanticRecord],
    ) -> int:
        if self._use_local:
            if not images:
                return 0
            state = self._local()
            for image in images:
                existing = state.image_semantics.get(image.image_hash)
                if existing is None:
                    state.image_semantics[image.image_hash] = {
                        "image_hash": image.image_hash,
                        "source_document_id": image.source_document_id,
                        "source_page_no": image.source_page_no,
                        "source_image_index": image.source_image_index,
                        "mime_type": image.mime_type,
                        "width": image.width,
                        "height": image.height,
                        "semantic_text": image.semantic_text,
                        "semantic_model": image.semantic_model,
                    }
                else:
                    existing["mime_type"] = existing["mime_type"] or image.mime_type
                    existing["width"] = existing["width"] or image.width
                    existing["height"] = existing["height"] or image.height
            return len(images)
        if not images:
            return 0
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO image_semantics (
                    image_hash, source_document_id, source_page_no, source_image_index,
                    mime_type, width, height, semantic_text, semantic_model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(image_hash) DO UPDATE SET
                    mime_type = COALESCE(image_semantics.mime_type, EXCLUDED.mime_type),
                    width = COALESCE(image_semantics.width, EXCLUDED.width),
                    height = COALESCE(image_semantics.height, EXCLUDED.height)
                """,
                [
                    (
                        image.image_hash,
                        image.source_document_id,
                        image.source_page_no,
                        image.source_image_index,
                        image.mime_type,
                        image.width,
                        image.height,
                        image.semantic_text,
                        image.semantic_model,
                    )
                    for image in images
                ],
            )
        return len(images)

    def get_image_semantics(
        self,
        *,
        image_hashes: list[str],
    ) -> dict[str, dict[str, Any]]:
        if self._use_local:
            return {
                image_hash: copy.deepcopy(self._local().image_semantics[image_hash])
                for image_hash in image_hashes
                if image_hash in self._local().image_semantics
            }
        if not image_hashes:
            return {}
        placeholders = ", ".join(["%s"] * len(image_hashes))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    image_hash, source_document_id, source_page_no, source_image_index,
                    mime_type, width, height, semantic_text, semantic_model
                FROM image_semantics
                WHERE image_hash IN ({placeholders})
                """,
                tuple(image_hashes),
            )
            rows = cur.fetchall()

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[str(row[0])] = {
                "image_hash": str(row[0]),
                "source_document_id": str(row[1]),
                "source_page_no": int(row[2]),
                "source_image_index": int(row[3]),
                "mime_type": row[4],
                "width": row[5],
                "height": row[6],
                "semantic_text": row[7],
                "semantic_model": row[8],
            }
        return result

    def update_image_semantic(
        self,
        *,
        image_hash: str,
        semantic_text: str,
        semantic_model: str | None = None,
    ) -> None:
        if self._use_local:
            record = self._local().image_semantics.get(image_hash)
            if record is not None:
                record["semantic_text"] = semantic_text
                record["semantic_model"] = semantic_model
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE image_semantics
                SET semantic_text = %s,
                    semantic_model = %s,
                    last_enhanced_at = CURRENT_TIMESTAMP
                WHERE image_hash = %s
                """,
                (semantic_text, semantic_model, image_hash),
            )

    def save_schema(
        self,
        *,
        corpus_id: str,
        name: str,
        schema_def: dict[str, Any],
        is_active: bool = True,
    ) -> str:
        if self._use_local:
            state = self._local()
            schema_id = _stable_id("schema", f"{corpus_id}:{name}")
            if is_active:
                for schema in state.schemas.values():
                    if schema["corpus_id"] == corpus_id:
                        schema["is_active"] = False
            existing = next(
                (
                    schema
                    for schema in state.schemas.values()
                    if schema["corpus_id"] == corpus_id and schema["name"] == name
                ),
                None,
            )
            created_at = existing["created_at"] if existing is not None else _utcnow_iso()
            state.schemas[schema_id] = {
                "id": schema_id,
                "corpus_id": corpus_id,
                "name": name,
                "schema_def": copy.deepcopy(schema_def),
                "is_active": is_active,
                "created_at": created_at,
            }
            return schema_id
        schema_id = _stable_id("schema", f"{corpus_id}:{name}")
        with self._conn.cursor() as cur:
            if is_active:
                cur.execute(
                    "UPDATE schemas SET is_active = FALSE WHERE corpus_id = %s",
                    (corpus_id,),
                )

            cur.execute(
                """
                INSERT INTO schemas (id, corpus_id, name, schema_def, is_active)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                ON CONFLICT(corpus_id, name) DO UPDATE SET
                    schema_def = EXCLUDED.schema_def,
                    is_active = EXCLUDED.is_active
                """,
                (
                    schema_id,
                    corpus_id,
                    name,
                    json.dumps(schema_def, sort_keys=True),
                    is_active,
                ),
            )
        return schema_id

    def list_schemas(self, *, corpus_id: str) -> list[SchemaRecord]:
        if self._use_local:
            rows = [
                schema
                for schema in self._local().schemas.values()
                if schema["corpus_id"] == corpus_id
            ]
            rows.sort(key=lambda item: (str(item["created_at"]), str(item["name"])), reverse=True)
            return [
                SchemaRecord(
                    id=str(row["id"]),
                    corpus_id=str(row["corpus_id"]),
                    name=str(row["name"]),
                    schema_def=copy.deepcopy(row["schema_def"]),
                    is_active=bool(row["is_active"]),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, corpus_id, name, schema_def, is_active, created_at
                FROM schemas
                WHERE corpus_id = %s
                ORDER BY created_at DESC, name ASC
                """,
                (corpus_id,),
            )
            rows = cur.fetchall()
        return [self._row_to_schema_record(row) for row in rows]

    def get_schema_by_name(self, *, corpus_id: str, name: str) -> SchemaRecord | None:
        if self._use_local:
            for schema in self._local().schemas.values():
                if schema["corpus_id"] == corpus_id and schema["name"] == name:
                    return SchemaRecord(
                        id=str(schema["id"]),
                        corpus_id=str(schema["corpus_id"]),
                        name=str(schema["name"]),
                        schema_def=copy.deepcopy(schema["schema_def"]),
                        is_active=bool(schema["is_active"]),
                        created_at=str(schema["created_at"]),
                    )
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, corpus_id, name, schema_def, is_active, created_at
                FROM schemas
                WHERE corpus_id = %s AND name = %s
                LIMIT 1
                """,
                (corpus_id, name),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_schema_record(row)

    def get_active_schema(self, *, corpus_id: str) -> SchemaRecord | None:
        if self._use_local:
            active = [
                schema
                for schema in self._local().schemas.values()
                if schema["corpus_id"] == corpus_id and bool(schema["is_active"])
            ]
            if not active:
                return None
            active.sort(key=lambda item: str(item["created_at"]), reverse=True)
            schema = active[0]
            return SchemaRecord(
                id=str(schema["id"]),
                corpus_id=str(schema["corpus_id"]),
                name=str(schema["name"]),
                schema_def=copy.deepcopy(schema["schema_def"]),
                is_active=bool(schema["is_active"]),
                created_at=str(schema["created_at"]),
            )
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, corpus_id, name, schema_def, is_active, created_at
                FROM schemas
                WHERE corpus_id = %s AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (corpus_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_schema_record(row)

    @staticmethod
    def make_document_id(corpus_id: str, relative_path: str) -> str:
        return _stable_id("doc", f"{corpus_id}:{relative_path}")

    @staticmethod
    def make_chunk_id(
        doc_id: str, position: int, start_char: int, end_char: int
    ) -> str:
        return _stable_id("chunk", f"{doc_id}:{position}:{start_char}:{end_char}")

    @staticmethod
    def _row_to_schema_record(row: tuple[Any, ...]) -> SchemaRecord:
        schema_def_obj = row[3]
        if isinstance(schema_def_obj, dict):
            schema_def = schema_def_obj
        else:
            schema_def = json.loads(str(schema_def_obj))
        return SchemaRecord(
            id=str(row[0]),
            corpus_id=str(row[1]),
            name=str(row[2]),
            schema_def=schema_def,
            is_active=bool(row[4]),
            created_at=str(row[5]),
        )

    def store_chunk_embeddings(
        self,
        *,
        corpus_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
    ) -> int:
        if self._use_local:
            if not chunk_embeddings:
                return 0
            state = self._local()
            for chunk_id, embedding in chunk_embeddings:
                state.chunk_embeddings[chunk_id] = {
                    "chunk_id": chunk_id,
                    "corpus_id": corpus_id,
                    "embedding": list(embedding),
                }
            return len(chunk_embeddings)
        if not chunk_embeddings:
            return 0
        with self._conn.cursor() as cur:
            if self._vector_enabled:
                cur.executemany(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, corpus_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        corpus_id = EXCLUDED.corpus_id,
                        embedding = EXCLUDED.embedding
                    """,
                    [
                        (cid, corpus_id, self._to_vector_literal(emb))
                        for cid, emb in chunk_embeddings
                    ],
                )
            else:
                cur.executemany(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, corpus_id, embedding_json)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        corpus_id = EXCLUDED.corpus_id,
                        embedding_json = EXCLUDED.embedding_json
                    """,
                    [
                        (cid, corpus_id, json.dumps(emb))
                        for cid, emb in chunk_embeddings
                    ],
                )
        return len(chunk_embeddings)

    def search_chunks_semantic(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._use_local:
            if not self._vector_enabled:
                return []
            allowed_doc_ids = {str(doc_id) for doc_id in document_ids or []}
            docs = {
                document["id"]: document
                for document in self._local_documents_for_corpus(corpus_id=corpus_id)
                if not allowed_doc_ids or str(document["id"]) in allowed_doc_ids
            }
            scored: list[dict[str, Any]] = []
            for chunk_id, stored in self._local().chunk_embeddings.items():
                if stored["corpus_id"] != corpus_id:
                    continue
                chunk = self._local().chunks.get(chunk_id)
                if chunk is None:
                    continue
                document = docs.get(chunk["doc_id"])
                if document is None:
                    continue
                score = _cosine_similarity(list(stored["embedding"]), query_embedding)
                scored.append(
                    {
                        "doc_id": str(document["id"]),
                        "relative_path": str(document["relative_path"]),
                        "absolute_path": str(document["absolute_path"]),
                        "position": int(chunk["position"]),
                        "source_unit_no": _optional_int(chunk.get("source_unit_no")),
                        "text": str(chunk["text"]),
                        "score": float(score),
                    }
                )
            scored.sort(key=lambda item: (-float(item["score"]), item["relative_path"], int(item["position"])))
            return scored[:limit]
        if not self._vector_enabled:
            return []

        query_vector = self._to_vector_literal(query_embedding)
        sql = """
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.position,
                c.source_unit_no,
                c.text,
                1 - (ce.embedding <=> %s::vector) AS score
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE ce.corpus_id = %s
              AND d.is_deleted = FALSE
            ORDER BY ce.embedding <=> %s::vector ASC
            LIMIT %s
        """
        params: list[Any] = [query_vector, corpus_id]
        if document_ids:
            placeholders = ", ".join(["%s"] * len(document_ids))
            sql = sql.replace(
                "AND d.is_deleted = FALSE",
                f"AND d.is_deleted = FALSE AND d.id IN ({placeholders})",
            )
            params.extend(document_ids)
        params.extend([query_vector, limit])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "doc_id": str(row[0]),
                    "relative_path": str(row[1]),
                    "absolute_path": str(row[2]),
                    "position": int(row[3]),
                    "source_unit_no": _optional_int(row[4]),
                    "text": str(row[5]),
                    "score": float(row[6]),
                }
            )
        return results

    def get_metadata_field_values(
        self,
        *,
        corpus_id: str,
        field_names: list[str],
        max_distinct: int = 10,
    ) -> dict[str, list[str]]:
        if self._use_local:
            result: dict[str, list[str]] = {}
            documents = self._local_documents_for_corpus(corpus_id=corpus_id)
            for field in field_names:
                seen: list[str] = []
                for document in documents:
                    metadata = _parse_metadata_json(str(document["metadata_json"]))
                    value = metadata.get(field)
                    if value is None:
                        continue
                    text = str(value)
                    if not text or text in seen:
                        continue
                    seen.append(text)
                    if len(seen) >= max_distinct:
                        break
                result[field] = seen
            return result
        result: dict[str, list[str]] = {}
        with self._conn.cursor() as cur:
            for field in field_names:
                cur.execute(
                    """
                    SELECT DISTINCT d.metadata_json ->> %s AS val
                    FROM documents d
                    WHERE d.corpus_id = %s
                      AND d.is_deleted = FALSE
                      AND (d.metadata_json ->> %s) IS NOT NULL
                      AND (d.metadata_json ->> %s) != ''
                    LIMIT %s
                    """,
                    (field, corpus_id, field, field, max_distinct),
                )
                rows = cur.fetchall()
                result[field] = [str(row[0]) for row in rows]
        return result

    def has_embeddings(self, *, corpus_id: str) -> bool:
        if self._use_local:
            return any(
                embedding["corpus_id"] == corpus_id
                for embedding in self._local().chunk_embeddings.values()
            )
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE corpus_id = %s",
                (corpus_id,),
            )
            row = cur.fetchone()
        return bool(row and int(row[0]) > 0)

    def create_collection(self, *, name: str) -> CollectionRecord:
        collection_id = _stable_id("collection", f"{name}:{_utcnow_iso()}")
        created_at = _utcnow_iso()
        record = {
            "id": collection_id,
            "name": name,
            "is_deleted": False,
            "created_at": created_at,
            "updated_at": created_at,
        }
        if self._use_local:
            self._local().collections[collection_id] = record
            return self._collection_record_from_dict(record)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO collections (id, name, is_deleted, created_at, updated_at)
                VALUES (%s, %s, FALSE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (collection_id, name),
            )
        collection = self.get_collection(collection_id=collection_id)
        if collection is None:
            raise RuntimeError("Failed to create collection.")
        return collection

    def list_collections(self, *, include_deleted: bool = False) -> list[CollectionRecord]:
        if self._use_local:
            records = [
                self._collection_record_from_dict(record)
                for record in self._local().collections.values()
                if include_deleted or not bool(record.get("is_deleted", False))
            ]
            records.sort(key=lambda item: item.name.lower())
            return records
        sql = """
            SELECT id, name, is_deleted, created_at, updated_at
            FROM collections
        """
        if not include_deleted:
            sql += " WHERE is_deleted = FALSE"
        sql += " ORDER BY lower(name), created_at"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            CollectionRecord(
                id=str(row[0]),
                name=str(row[1]),
                is_deleted=bool(row[2]),
                created_at=row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                updated_at=row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
            )
            for row in rows
        ]

    def get_collection(self, *, collection_id: str) -> CollectionRecord | None:
        if self._use_local:
            record = self._local().collections.get(collection_id)
            return None if record is None else self._collection_record_from_dict(record)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, is_deleted, created_at, updated_at
                FROM collections
                WHERE id = %s
                LIMIT 1
                """,
                (collection_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return CollectionRecord(
            id=str(row[0]),
            name=str(row[1]),
            is_deleted=bool(row[2]),
            created_at=row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
            updated_at=row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
        )

    def update_collection(
        self,
        *,
        collection_id: str,
        name: str,
    ) -> CollectionRecord | None:
        if self._use_local:
            record = self._local().collections.get(collection_id)
            if record is None:
                return None
            record["name"] = name
            record["updated_at"] = _utcnow_iso()
            return self._collection_record_from_dict(record)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE collections
                SET name = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (name, collection_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_collection(collection_id=collection_id)

    def set_collection_deleted(
        self,
        *,
        collection_id: str,
        is_deleted: bool,
    ) -> CollectionRecord | None:
        if self._use_local:
            record = self._local().collections.get(collection_id)
            if record is None:
                return None
            record["is_deleted"] = bool(is_deleted)
            record["updated_at"] = _utcnow_iso()
            return self._collection_record_from_dict(record)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE collections
                SET is_deleted = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (is_deleted, collection_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_collection(collection_id=collection_id)

    def list_collection_documents(
        self,
        *,
        collection_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        if self._use_local:
            doc_ids = [
                doc_id
                for coll_id, doc_id in self._local().collection_documents
                if coll_id == collection_id
            ]
            return self._local_documents_by_ids(
                doc_ids=doc_ids,
                include_deleted=include_deleted,
            )
        sql = """
            SELECT d.id
            FROM collection_documents cd
            JOIN documents d ON d.id = cd.document_id
            WHERE cd.collection_id = %s
        """
        params: list[Any] = [collection_id]
        if not include_deleted:
            sql += " AND d.is_deleted = FALSE"
        sql += " ORDER BY d.relative_path"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return self.list_documents_by_ids(
            doc_ids=[str(row[0]) for row in rows],
            include_deleted=include_deleted,
        )

    def attach_documents_to_collection(
        self,
        *,
        collection_id: str,
        document_ids: list[str],
    ) -> int:
        if not document_ids:
            return 0
        unique_ids = sorted({str(doc_id) for doc_id in document_ids if str(doc_id).strip()})
        if self._use_local:
            before = len(self._local().collection_documents)
            for doc_id in unique_ids:
                self._local().collection_documents.add((collection_id, doc_id))
            return len(self._local().collection_documents) - before
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO collection_documents (collection_id, document_id)
                VALUES (%s, %s)
                ON CONFLICT(collection_id, document_id) DO NOTHING
                """,
                [(collection_id, doc_id) for doc_id in unique_ids],
            )
        return len(unique_ids)

    def detach_document_from_collection(
        self,
        *,
        collection_id: str,
        doc_id: str,
    ) -> bool:
        if self._use_local:
            key = (collection_id, doc_id)
            existed = key in self._local().collection_documents
            self._local().collection_documents.discard(key)
            return existed
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM collection_documents
                WHERE collection_id = %s AND document_id = %s
                """,
                (collection_id, doc_id),
            )
            return bool(cur.rowcount)

    def remove_document_from_all_collections(self, *, doc_id: str) -> int:
        if self._use_local:
            matches = [key for key in self._local().collection_documents if key[1] == doc_id]
            for key in matches:
                self._local().collection_documents.discard(key)
            return len(matches)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM collection_documents
                WHERE document_id = %s
                """,
                (doc_id,),
            )
            return int(cur.rowcount or 0)

    def create_hnsw_index(self, *, corpus_id: str) -> bool:
        if not self._vector_enabled:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
                    ON chunk_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            return True
        except Exception:
            return False

    @staticmethod
    def _numeric_json_expr() -> str:
        return (
            "(CASE WHEN (d.metadata_json ->> %s) ~ E'^-?[0-9]+(\\.[0-9]+)?$' "
            "THEN (d.metadata_json ->> %s)::double precision ELSE NULL END)"
        )

    @classmethod
    def _metadata_clause(
        cls,
        *,
        field: str,
        operator: str,
        value: Any,
    ) -> tuple[str, list[Any]]:
        json_expr = "d.metadata_json ->> %s"

        if operator in {"eq", "ne"}:
            comparator = "=" if operator == "eq" else "<>"
            if isinstance(value, bool):
                return (
                    f"lower(coalesce({json_expr}, '')) {comparator} %s",
                    [field, "true" if value else "false"],
                )
            if isinstance(value, (int, float)):
                return (
                    f"{cls._numeric_json_expr()} {comparator} %s",
                    [field, field, float(value)],
                )
            return (
                f"lower(coalesce({json_expr}, '')) {comparator} lower(%s)",
                [field, str(value)],
            )

        if operator in {"gt", "gte", "lt", "lte"}:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Metadata operator {operator!r} requires numeric value for field {field!r}."
                )
            comparator_map = {
                "gt": ">",
                "gte": ">=",
                "lt": "<",
                "lte": "<=",
            }
            comparator = comparator_map[operator]
            return (
                f"{cls._numeric_json_expr()} {comparator} %s",
                [field, field, float(value)],
            )

        if operator == "contains":
            return (
                f"lower(coalesce({json_expr}, '')) LIKE '%%' || lower(%s) || '%%'",
                [field, str(value)],
            )

        if operator == "in":
            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"Metadata `in` filter for field {field!r} has no values."
                )

            if all(isinstance(item, bool) for item in value):
                placeholders = ", ".join(["%s"] * len(value))
                return (
                    f"lower(coalesce({json_expr}, '')) IN ({placeholders})",
                    [field, *["true" if bool(item) else "false" for item in value]],
                )

            if all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                placeholders = ", ".join(["%s"] * len(value))
                return (
                    f"{cls._numeric_json_expr()} IN ({placeholders})",
                    [field, field, *[float(item) for item in value]],
                )

            placeholders = ", ".join(["%s"] * len(value))
            return (
                f"lower(coalesce({json_expr}, '')) IN ({placeholders})",
                [field, *[str(item).lower() for item in value]],
            )

        raise ValueError(f"Unsupported metadata operator: {operator!r}")
