"""
Storage interfaces and data models for index persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk stored for a document."""

    id: str
    doc_id: str
    text: str
    position: int
    start_char: int
    end_char: int
    embedding: list[float] | None = None


@dataclass(frozen=True)
class DocumentRecord:
    """A normalized document record for indexing."""

    id: str
    corpus_id: str
    relative_path: str
    absolute_path: str
    content: str
    metadata_json: str
    file_mtime: float
    file_size: int
    content_sha256: str


@dataclass(frozen=True)
class SchemaRecord:
    """A stored schema entry."""

    id: str
    corpus_id: str
    name: str
    schema_def: dict[str, Any]
    is_active: bool
    created_at: str


@dataclass(frozen=True)
class ParsedUnitRecord:
    """A parsed page-sized unit stored for incremental parse cache reuse."""

    document_id: str
    parser_name: str
    parser_version: str
    page_no: int
    markdown: str
    content_hash: str
    images_json: str


@dataclass(frozen=True)
class ImageSemanticRecord:
    """A cached image semantic placeholder or enriched description."""

    image_hash: str
    source_document_id: str
    source_page_no: int
    source_image_index: int
    mime_type: str | None
    width: int | None
    height: int | None
    semantic_text: str | None = None
    semantic_model: str | None = None


class StorageBackend(Protocol):
    """Protocol for persistence operations used by indexing and schema workflows."""

    def initialize(self) -> None:
        """Initialize required tables/indexes."""

    def get_or_create_corpus(self, root_path: str) -> str:
        """Return corpus id for a root path, creating if needed."""

    def get_corpus_id(self, root_path: str) -> str | None:
        """Return corpus id for a root path if present."""

    def upsert_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        """Insert or update a document and replace its chunks."""

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
        """Mark documents deleted when not present in the latest index run."""

    def list_documents(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List documents for a corpus."""

    def count_chunks(self, *, corpus_id: str) -> int:
        """Count chunks for active documents in a corpus."""

    def search_chunks(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks and return ranked matches."""

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search indexed documents by metadata filters."""

    def get_document(self, *, doc_id: str) -> dict[str, Any] | None:
        """Get a document by id."""

    def list_parsed_units(
        self,
        *,
        document_id: str,
        parser_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return cached parsed units for a document."""

    def sync_parsed_units(
        self,
        *,
        document_id: str,
        parser_name: str,
        parser_version: str,
        units: list[ParsedUnitRecord],
    ) -> dict[str, int]:
        """Upsert parsed units and delete stale pages for the same parser version."""

    def upsert_image_semantics(
        self,
        *,
        images: list[ImageSemanticRecord],
    ) -> int:
        """Insert image semantic placeholders for extracted images."""

    def get_image_semantics(
        self,
        *,
        image_hashes: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch cached image semantics keyed by image hash."""

    def update_image_semantic(
        self,
        *,
        image_hash: str,
        semantic_text: str,
        semantic_model: str | None = None,
    ) -> None:
        """Persist a lazy semantic enhancement result for an image hash."""

    def save_schema(
        self,
        *,
        corpus_id: str,
        name: str,
        schema_def: dict[str, Any],
        is_active: bool = True,
    ) -> str:
        """Create or update a schema entry."""

    def list_schemas(self, *, corpus_id: str) -> list[SchemaRecord]:
        """List all schemas for a corpus."""

    def get_schema_by_name(self, *, corpus_id: str, name: str) -> SchemaRecord | None:
        """Fetch a schema by name."""

    def get_active_schema(self, *, corpus_id: str) -> SchemaRecord | None:
        """Fetch active schema for a corpus if present."""

    def store_chunk_embeddings(
        self,
        *,
        corpus_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
    ) -> int:
        """Bulk-store (chunk_id, embedding) pairs. Return count written."""

    def search_chunks_semantic(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search chunks by cosine similarity against a query embedding."""

    def get_metadata_field_values(
        self,
        *,
        corpus_id: str,
        field_names: list[str],
        max_distinct: int = 10,
    ) -> dict[str, list[str]]:
        """Return up to *max_distinct* distinct non-empty values per metadata field."""

    def has_embeddings(self, *, corpus_id: str) -> bool:
        """Return True if the corpus has stored embeddings."""
