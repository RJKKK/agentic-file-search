"""Tests for M2 page-level parsing cache and lazy image semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import fs_explorer.indexing.pipeline as pipeline_module
from fs_explorer.document_parsing import (
    PARSER_VERSION,
    ParsedDocument,
    ParsedPage,
    enhance_page_image_semantics,
    parse_document,
)
from fs_explorer.indexing.pipeline import IndexingPipeline
from fs_explorer.storage.base import ChunkRecord, DocumentRecord, ParsedUnitRecord


class MemoryStorage:
    """Small in-memory storage double for indexing pipeline tests."""

    def __init__(self) -> None:
        self.corpora: dict[str, str] = {}
        self.documents: dict[str, dict[str, object]] = {}
        self.parsed_units: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.image_semantics: dict[str, dict[str, object]] = {}
        self.chunks: dict[str, list[ChunkRecord]] = {}

    def initialize(self) -> None:  # pragma: no cover - protocol compatibility
        return None

    def get_or_create_corpus(self, root_path: str) -> str:
        if root_path not in self.corpora:
            self.corpora[root_path] = f"corpus:{root_path}"
        return self.corpora[root_path]

    def get_corpus_id(self, root_path: str) -> str | None:
        return self.corpora.get(root_path)

    def upsert_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
    ) -> None:
        self.documents[document.id] = {
            "id": document.id,
            "corpus_id": document.corpus_id,
            "relative_path": document.relative_path,
            "absolute_path": document.absolute_path,
            "content": document.content,
            "metadata_json": document.metadata_json,
            "content_sha256": document.content_sha256,
            "file_mtime": document.file_mtime,
            "file_size": document.file_size,
            "is_deleted": False,
        }
        self.chunks[document.id] = chunks

    def mark_deleted_missing_documents(
        self,
        *,
        corpus_id: str,
        active_relative_paths: set[str],
    ) -> int:
        deleted = 0
        for document in self.documents.values():
            if document["corpus_id"] != corpus_id:
                continue
            if document["relative_path"] not in active_relative_paths:
                if not bool(document["is_deleted"]):
                    document["is_deleted"] = True
                    deleted += 1
            else:
                document["is_deleted"] = False
        return deleted

    def list_documents(
        self,
        *,
        corpus_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, object]]:
        results = []
        for document in self.documents.values():
            if document["corpus_id"] != corpus_id:
                continue
            if not include_deleted and bool(document["is_deleted"]):
                continue
            results.append(document.copy())
        results.sort(key=lambda item: str(item["relative_path"]))
        return results

    def count_chunks(self, *, corpus_id: str) -> int:
        return sum(
            len(chunks)
            for doc_id, chunks in self.chunks.items()
            if self.documents[doc_id]["corpus_id"] == corpus_id
            and not bool(self.documents[doc_id]["is_deleted"])
        )

    def search_chunks(self, *, corpus_id: str, query: str, limit: int = 5) -> list[dict[str, object]]:
        return []

    def search_documents_by_metadata(
        self,
        *,
        corpus_id: str,
        filters: list[dict[str, object]],
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return []

    def get_document(self, *, doc_id: str) -> dict[str, object] | None:
        document = self.documents.get(doc_id)
        return document.copy() if document is not None else None

    def list_parsed_units(
        self,
        *,
        document_id: str,
        parser_version: str | None = None,
    ) -> list[dict[str, object]]:
        if parser_version is None:
            items: list[dict[str, object]] = []
            for (doc_id, _version), values in self.parsed_units.items():
                if doc_id == document_id:
                    items.extend(unit.copy() for unit in values)
            return sorted(items, key=lambda unit: int(unit["page_no"]))
        return [
            unit.copy()
            for unit in self.parsed_units.get((document_id, parser_version), [])
        ]

    def sync_parsed_units(
        self,
        *,
        document_id: str,
        parser_name: str,
        parser_version: str,
        units: list[ParsedUnitRecord],
    ) -> dict[str, int]:
        existing = {
            int(unit["page_no"]): unit
            for unit in self.parsed_units.get((document_id, parser_version), [])
        }
        updated = 0
        untouched = 0
        current: list[dict[str, object]] = []
        for unit in units:
            payload = {
                "document_id": document_id,
                "parser_name": parser_name,
                "parser_version": parser_version,
                "page_no": unit.page_no,
                "markdown": unit.markdown,
                "content_hash": unit.content_hash,
                "images": json.loads(unit.images_json),
            }
            previous = existing.get(unit.page_no)
            if previous == payload:
                untouched += 1
            else:
                updated += 1
            current.append(payload)
        deleted = max(len(existing) - len(current), 0)
        self.parsed_units[(document_id, parser_version)] = current
        return {"upserted": updated, "untouched": untouched, "deleted": deleted}

    def upsert_image_semantics(self, *, images: list) -> int:
        written = 0
        for image in images:
            if image.image_hash not in self.image_semantics:
                self.image_semantics[image.image_hash] = {
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
                written += 1
        return written

    def get_image_semantics(
        self,
        *,
        image_hashes: list[str],
    ) -> dict[str, dict[str, object]]:
        return {
            image_hash: self.image_semantics[image_hash].copy()
            for image_hash in image_hashes
            if image_hash in self.image_semantics
        }

    def update_image_semantic(
        self,
        *,
        image_hash: str,
        semantic_text: str,
        semantic_model: str | None = None,
    ) -> None:
        self.image_semantics[image_hash]["semantic_text"] = semantic_text
        self.image_semantics[image_hash]["semantic_model"] = semantic_model

    def save_schema(self, *, corpus_id: str, name: str, schema_def: dict, is_active: bool = True) -> str:
        return f"schema:{corpus_id}:{name}"

    def list_schemas(self, *, corpus_id: str) -> list:
        return []

    def get_schema_by_name(self, *, corpus_id: str, name: str):
        return None

    def get_active_schema(self, *, corpus_id: str):
        return None

    def store_chunk_embeddings(
        self,
        *,
        corpus_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
    ) -> int:
        return 0

    def search_chunks_semantic(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, object]]:
        return []

    def get_metadata_field_values(
        self,
        *,
        corpus_id: str,
        field_names: list[str],
        max_distinct: int = 10,
    ) -> dict[str, list[str]]:
        return {}

    def has_embeddings(self, *, corpus_id: str) -> bool:
        return False


def test_parse_document_markdown_returns_single_page(tmp_path: Path) -> None:
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("# Title\n\nBody text", encoding="utf-8")

    parsed = parse_document(str(doc_path))

    assert parsed.parser_name == "markdown"
    assert parsed.parser_version == PARSER_VERSION
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_no == 1
    assert "Body text" in parsed.markdown


def test_indexing_pipeline_reuses_page_cache(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc_path = corpus / "cached.md"
    doc_path.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    parse_calls = {"count": 0}

    def fake_parse_document(file_path: str) -> ParsedDocument:
        parse_calls["count"] += 1
        text = Path(file_path).read_text(encoding="utf-8")
        page = ParsedPage(
            page_no=1,
            markdown=text,
            content_hash=f"hash:{text}",
        )
        return ParsedDocument(
            parser_name="markdown",
            parser_version=PARSER_VERSION,
            pages=(page,),
        )

    monkeypatch.setattr(pipeline_module, "parse_document", fake_parse_document)
    monkeypatch.setattr(
        pipeline_module,
        "extract_metadata",
        lambda **kwargs: {"path": os.path.basename(str(kwargs["file_path"]))},
    )

    storage = MemoryStorage()
    pipeline = IndexingPipeline(storage=storage, max_workers=1)

    first = pipeline.index_folder(str(corpus))
    second = pipeline.index_folder(str(corpus))

    assert first.indexed_files == 1
    assert first.parsed_cache_hits == 0
    assert first.parsed_pages_updated == 1
    assert second.indexed_files == 1
    assert second.parsed_cache_hits == 1
    assert second.parsed_pages_updated == 0
    assert parse_calls["count"] == 1


def test_lazy_image_semantics_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    storage = MemoryStorage()
    document_id = "doc:test"
    storage.documents[document_id] = {
        "id": document_id,
        "corpus_id": "corpus:test",
        "relative_path": "sample.pdf",
        "absolute_path": str(tmp_path / "sample.pdf"),
        "content": "page",
        "metadata_json": "{}",
        "content_sha256": "sha",
        "file_mtime": 0.0,
        "file_size": 0,
        "is_deleted": False,
    }
    storage.parsed_units[(document_id, PARSER_VERSION)] = [
        {
            "document_id": document_id,
            "parser_name": "pymupdf4llm",
            "parser_version": PARSER_VERSION,
            "page_no": 1,
            "markdown": "page",
            "content_hash": "page-hash",
            "images": [
                {
                    "image_hash": "hash-1",
                    "image_index": 1,
                    "mime_type": "png",
                    "width": 100,
                    "height": 80,
                }
            ],
        }
    ]
    storage.image_semantics["hash-1"] = {
        "image_hash": "hash-1",
        "source_document_id": document_id,
        "source_page_no": 1,
        "source_image_index": 1,
        "mime_type": "png",
        "width": 100,
        "height": 80,
        "semantic_text": None,
        "semantic_model": None,
    }

    monkeypatch.setattr(
        "fs_explorer.document_parsing._extract_pdf_images",
        lambda file_path, page_no, include_bytes=False: [
            {
                "image_hash": "hash-1",
                "image_index": 1,
                "mime_type": "png",
                "width": 100,
                "height": 80,
                "image_bytes": b"pixel-data",
            }
        ],
    )

    class StubEnhancer:
        def __init__(self) -> None:
            self.calls = 0

        def describe_image(self, **kwargs) -> tuple[str, str | None]:
            self.calls += 1
            return ("chart with legend", "stub-vision")

    enhancer = StubEnhancer()

    first = enhance_page_image_semantics(
        storage=storage,
        document_id=document_id,
        page_no=1,
        enhancer=enhancer,
    )
    second = enhance_page_image_semantics(
        storage=storage,
        document_id=document_id,
        page_no=1,
        enhancer=enhancer,
    )

    assert first == 1
    assert second == 0
    assert enhancer.calls == 1
    assert storage.image_semantics["hash-1"]["semantic_text"] == "chart with legend"
