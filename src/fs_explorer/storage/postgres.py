"""
PostgreSQL storage backend for index persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any

import psycopg

from .base import ChunkRecord, DocumentRecord, SchemaRecord

logger = logging.getLogger(__name__)


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
        self.db_path = self._normalize_dsn(db_path)
        self.read_only = read_only
        self.embedding_dim = embedding_dim
        self._vector_enabled = False

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
        fall back to environment DSN defaults.
        """
        if "://" in db_path:
            return db_path
        return (
            os.getenv("FS_EXPLORER_DB_DSN")
            or os.getenv("FS_EXPLORER_DB_PATH")
            or "postgresql://fs_explorer:devpassword@127.0.0.1:5432/fs_explorer"
        )

    def close(self) -> None:
        """Close the underlying PostgreSQL connection."""
        self._conn.close()

    def initialize(self) -> None:
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
                    content TEXT NOT NULL,
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    file_mtime DOUBLE PRECISION NOT NULL,
                    file_size BIGINT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    last_indexed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    is_deleted BOOLEAN DEFAULT FALSE,
                    UNIQUE(corpus_id, relative_path)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id),
                    text TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL
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
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_position
                ON chunks (doc_id, position)
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

    def get_or_create_corpus(self, root_path: str) -> str:
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
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM corpora WHERE root_path = %s",
                (root_path,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    def upsert_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
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
                    id, corpus_id, relative_path, absolute_path, content, metadata_json,
                    file_mtime, file_size, content_sha256, is_deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, FALSE)
                ON CONFLICT(id) DO UPDATE SET
                    corpus_id = EXCLUDED.corpus_id,
                    relative_path = EXCLUDED.relative_path,
                    absolute_path = EXCLUDED.absolute_path,
                    content = EXCLUDED.content,
                    metadata_json = EXCLUDED.metadata_json,
                    file_mtime = EXCLUDED.file_mtime,
                    file_size = EXCLUDED.file_size,
                    content_sha256 = EXCLUDED.content_sha256,
                    last_indexed_at = CURRENT_TIMESTAMP,
                    is_deleted = FALSE
                """,
                (
                    document.id,
                    document.corpus_id,
                    document.relative_path,
                    document.absolute_path,
                    document.content,
                    document.metadata_json,
                    document.file_mtime,
                    document.file_size,
                    document.content_sha256,
                ),
            )

            if chunks:
                cur.executemany(
                    """
                    INSERT INTO chunks (id, doc_id, text, position, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.doc_id,
                            chunk.text,
                            chunk.position,
                            chunk.start_char,
                            chunk.end_char,
                        )
                        for chunk in chunks
                    ],
                )

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
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
        sql = """
            SELECT id, relative_path, absolute_path, file_size, file_mtime, is_deleted
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
            results.append(
                {
                    "id": str(row[0]),
                    "relative_path": str(row[1]),
                    "absolute_path": str(row[2]),
                    "file_size": int(row[3]),
                    "file_mtime": float(row[4]),
                    "is_deleted": bool(row[5]),
                }
            )
        return results

    def count_chunks(self, *, corpus_id: str) -> int:
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
    ) -> list[dict[str, Any]]:
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
        params: list[Any] = [*terms, corpus_id, limit]
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
                    "text": str(row[4]),
                    "score": int(row[5]),
                }
            )
        return results

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
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
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, corpus_id, relative_path, absolute_path, content, metadata_json, is_deleted
                FROM documents
                WHERE id = %s
                LIMIT 1
                """,
                (doc_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        metadata_json_obj = row[5]
        if isinstance(metadata_json_obj, str):
            metadata_json = metadata_json_obj
        else:
            metadata_json = json.dumps(metadata_json_obj, sort_keys=True)
        return {
            "id": str(row[0]),
            "corpus_id": str(row[1]),
            "relative_path": str(row[2]),
            "absolute_path": str(row[3]),
            "content": str(row[4]),
            "metadata_json": metadata_json,
            "is_deleted": bool(row[6]),
        }

    def save_schema(
        self,
        *,
        corpus_id: str,
        name: str,
        schema_def: dict[str, Any],
        is_active: bool = True,
    ) -> str:
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
    ) -> list[dict[str, Any]]:
        if not self._vector_enabled:
            return []

        query_vector = self._to_vector_literal(query_embedding)
        sql = """
            SELECT
                d.id AS doc_id,
                d.relative_path,
                d.absolute_path,
                c.position,
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
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_vector, corpus_id, query_vector, limit))
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "doc_id": str(row[0]),
                    "relative_path": str(row[1]),
                    "absolute_path": str(row[2]),
                    "position": int(row[3]),
                    "text": str(row[4]),
                    "score": float(row[5]),
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
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE corpus_id = %s",
                (corpus_id,),
            )
            row = cur.fetchone()
        return bool(row and int(row[0]) > 0)

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
