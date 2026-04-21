"""
Indexing pipeline orchestration.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import SmartChunker
from .metadata import (
    ensure_langextract_schema_fields,
    extract_metadata,
    langextract_field_names,
)
from .schema import SchemaDiscovery
from ..document_parsing import (
    DocumentParseError,
    ParsedDocument,
    ParsedUnit,
    PARSER_VERSION,
    compute_file_sha256,
    parse_document,
    reconstruct_parsed_document,
)
from ..embeddings import EmbeddingProvider
from ..fs import SUPPORTED_EXTENSIONS
from ..storage import (
    ChunkRecord,
    DocumentRecord,
    ImageSemanticRecord,
    ParsedUnitRecord,
    PostgresStorage,
    StorageBackend,
)

_PARSE_ERROR_PREFIXES: tuple[str, ...] = (
    "Error parsing ",
    "Unsupported file extension",
    "No such file:",
)


def parse_file(file_path: str) -> str:
    """Compatibility hook for older tests and callers that monkeypatch parsing."""
    return parse_document(file_path).markdown


@dataclass(frozen=True)
class IndexingResult:
    """Summary output for an indexing run."""

    corpus_id: str
    indexed_files: int
    skipped_files: int
    deleted_files: int
    chunks_written: int
    active_documents: int
    schema_used: str | None
    embeddings_written: int = 0
    parsed_cache_hits: int = 0
    parsed_pages_updated: int = 0
    image_placeholders_written: int = 0


class IndexingPipeline:
    """Build and update corpus indexes from filesystem documents."""

    def __init__(
        self,
        storage: StorageBackend,
        chunker: SmartChunker | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        max_workers: int = 4,
    ) -> None:
        self.storage = storage
        self.chunker = chunker or SmartChunker()
        self.embedding_provider = embedding_provider
        self._max_workers = max_workers

    def index_folder(
        self,
        folder: str,
        *,
        discover_schema: bool = False,
        schema_name: str | None = None,
        with_metadata: bool = False,
        metadata_profile: dict[str, Any] | None = None,
    ) -> IndexingResult:
        root = str(Path(folder).resolve())
        if not os.path.exists(root) or not os.path.isdir(root):
            raise ValueError(f"No such directory: {root}")

        effective_with_metadata = with_metadata or metadata_profile is not None
        corpus_id = self.storage.get_or_create_corpus(root)
        schema_def, selected_schema_name = self._resolve_schema(
            corpus_id=corpus_id,
            root=root,
            discover_schema=discover_schema,
            schema_name=schema_name,
            with_metadata=effective_with_metadata,
            metadata_profile=metadata_profile,
        )
        effective_profile = metadata_profile or self._schema_metadata_profile(
            schema_def
        )

        # Pass 1: Parse all documents or reuse page-level cache
        parsed_docs: list[dict[str, Any]] = []
        skipped_files = 0
        active_paths: set[str] = set()
        parsed_cache_hits = 0

        for file_path in self._iter_supported_files(root):
            relative_path = os.path.relpath(file_path, root)
            active_paths.add(relative_path)
            doc_id = PostgresStorage.make_document_id(corpus_id, relative_path)
            file_sha256 = compute_file_sha256(file_path)
            existing_doc = self.storage.get_document(doc_id=doc_id)

            if (
                existing_doc is not None
                and existing_doc.get("content_sha256") == file_sha256
            ):
                cached_units = self.storage.list_parsed_units(
                    document_id=doc_id,
                    parser_version=PARSER_VERSION,
                )
                cached_document = reconstruct_parsed_document(cached_units)
                if cached_document is not None:
                    full_content, unit_offsets = self._join_units_with_offsets(
                        cached_document.units
                    )
                    parsed_docs.append(
                        {
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "doc_id": doc_id,
                            "file_sha256": file_sha256,
                            "content": full_content,
                            "parsed_document": cached_document,
                            "from_cache": True,
                            "unit_offsets": unit_offsets,
                        }
                    )
                    parsed_cache_hits += 1
                    continue

            try:
                parsed_document = self._parse_document_for_indexing(file_path)
            except DocumentParseError:
                skipped_files += 1
                continue

            full_content, unit_offsets = self._join_units_with_offsets(
                parsed_document.units
            )

            parsed_docs.append(
                {
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "doc_id": doc_id,
                    "file_sha256": file_sha256,
                    "content": full_content,
                    "parsed_document": parsed_document,
                    "from_cache": False,
                    "unit_offsets": unit_offsets,
                }
            )

        # Parallel metadata extraction across documents
        metadata_map = self._extract_metadata_batch(
            parsed_docs=[
                (item["file_path"], item["relative_path"], item["content"])
                for item in parsed_docs
            ],
            root_path=root,
            schema_def=schema_def,
            with_langextract=effective_with_metadata,
            langextract_profile=effective_profile,
        )

        # Pass 2: Chunk + upsert (sequential, DB writes)
        indexed_files = 0
        chunks_written = 0
        all_chunk_records: list[ChunkRecord] = []
        parsed_pages_updated = 0
        image_placeholders_written = 0

        for item in parsed_docs:
            file_path = str(item["file_path"])
            relative_path = str(item["relative_path"])
            doc_id = str(item["doc_id"])
            content = str(item["content"])
            parsed_document = item["parsed_document"]
            unit_offsets = list(item.get("unit_offsets") or [])
            chunks = self.chunker.chunk_text(content)
            metadata = metadata_map[relative_path]
            metadata_json = json.dumps(metadata, sort_keys=True)

            stat = os.stat(file_path)
            doc_record = DocumentRecord(
                id=doc_id,
                corpus_id=corpus_id,
                relative_path=relative_path,
                absolute_path=str(Path(file_path).resolve()),
                content=content,
                metadata_json=metadata_json,
                file_mtime=float(stat.st_mtime),
                file_size=int(stat.st_size),
                content_sha256=str(item["file_sha256"]),
            )

            chunk_records: list[ChunkRecord] = []
            for chunk in chunks:
                chunk_records.append(
                    ChunkRecord(
                        id=PostgresStorage.make_chunk_id(
                            doc_id,
                            chunk.position,
                            chunk.start_char,
                            chunk.end_char,
                        ),
                        doc_id=doc_id,
                        text=chunk.text,
                        position=chunk.position,
                        source_unit_no=self._pick_chunk_unit_no(
                            chunk_start=chunk.start_char,
                            chunk_end=chunk.end_char,
                            unit_offsets=unit_offsets,
                        ),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                    )
                )

            self.storage.upsert_document(doc_record, chunk_records)

            if not bool(item["from_cache"]):
                unit_records = [
                    ParsedUnitRecord(
                        document_id=doc_id,
                        parser_name=parsed_document.parser_name,
                        parser_version=parsed_document.parser_version,
                        page_no=page.page_no,
                        markdown=page.markdown,
                        content_hash=page.content_hash,
                        heading=page.heading,
                        source_locator=page.source_locator,
                        images_json=json.dumps(
                            [
                                {
                                    "image_hash": image.image_hash,
                                    "image_index": image.image_index,
                                    "mime_type": image.mime_type,
                                    "width": image.width,
                                    "height": image.height,
                                }
                                for image in page.images
                            ],
                            sort_keys=True,
                        ),
                    )
                    for page in parsed_document.pages
                ]
                sync_stats = self.storage.sync_parsed_units(
                    document_id=doc_id,
                    parser_name=parsed_document.parser_name,
                    parser_version=parsed_document.parser_version,
                    units=unit_records,
                )
                parsed_pages_updated += int(sync_stats["upserted"])

                image_records = [
                    ImageSemanticRecord(
                        image_hash=image.image_hash,
                        source_document_id=doc_id,
                        source_page_no=image.page_no,
                        source_image_index=image.image_index,
                        mime_type=image.mime_type,
                        width=image.width,
                        height=image.height,
                    )
                    for page in parsed_document.pages
                    for image in page.images
                ]
                image_placeholders_written += self.storage.upsert_image_semantics(
                    images=image_records
                )

            all_chunk_records.extend(chunk_records)
            indexed_files += 1
            chunks_written += len(chunk_records)

        deleted_files = self.storage.mark_deleted_missing_documents(
            corpus_id=corpus_id,
            active_relative_paths=active_paths,
        )
        active_documents = len(
            self.storage.list_documents(corpus_id=corpus_id, include_deleted=False)
        )

        embeddings_written = self._generate_and_store_embeddings(
            corpus_id=corpus_id,
            all_chunk_records=all_chunk_records,
        )

        return IndexingResult(
            corpus_id=corpus_id,
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            deleted_files=deleted_files,
            chunks_written=chunks_written,
            active_documents=active_documents,
            schema_used=selected_schema_name,
            embeddings_written=embeddings_written,
            parsed_cache_hits=parsed_cache_hits,
            parsed_pages_updated=parsed_pages_updated,
            image_placeholders_written=image_placeholders_written,
        )

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        corpus_id: str,
        discover_schema: bool = False,
        schema_name: str | None = None,
        with_metadata: bool = False,
        metadata_profile: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """Index only the selected uploaded documents."""
        effective_with_metadata = with_metadata or metadata_profile is not None
        schema_def, selected_schema_name = self._resolve_schema(
            corpus_id=corpus_id,
            root=corpus_id,
            discover_schema=discover_schema,
            schema_name=schema_name,
            with_metadata=effective_with_metadata,
            metadata_profile=metadata_profile,
        )
        effective_profile = metadata_profile or self._schema_metadata_profile(
            schema_def
        )

        parsed_docs: list[dict[str, Any]] = []
        skipped_files = 0
        parsed_cache_hits = 0

        for document in documents:
            doc_id = str(document["id"])
            file_path = str(document["absolute_path"])
            relative_path = str(
                document.get("relative_path")
                or document.get("original_filename")
                or doc_id
            )
            file_sha256 = compute_file_sha256(file_path)
            existing_doc = self.storage.get_document(doc_id=doc_id)

            if (
                existing_doc is not None
                and existing_doc.get("content_sha256") == file_sha256
            ):
                cached_units = self.storage.list_parsed_units(
                    document_id=doc_id,
                    parser_version=PARSER_VERSION,
                )
                cached_document = reconstruct_parsed_document(cached_units)
                if cached_document is not None:
                    full_content, unit_offsets = self._join_units_with_offsets(
                        cached_document.units
                    )
                    parsed_docs.append(
                        {
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "doc_id": doc_id,
                            "file_sha256": file_sha256,
                            "content": full_content,
                            "parsed_document": cached_document,
                            "from_cache": True,
                            "unit_offsets": unit_offsets,
                            "document_row": document,
                        }
                    )
                    parsed_cache_hits += 1
                    continue

            try:
                parsed_document = self._parse_document_for_indexing(file_path)
            except DocumentParseError:
                skipped_files += 1
                continue

            full_content, unit_offsets = self._join_units_with_offsets(
                parsed_document.units
            )
            parsed_docs.append(
                {
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "doc_id": doc_id,
                    "file_sha256": file_sha256,
                    "content": full_content,
                    "parsed_document": parsed_document,
                    "from_cache": False,
                    "unit_offsets": unit_offsets,
                    "document_row": document,
                }
            )

        metadata_map = self._extract_metadata_batch(
            parsed_docs=[
                (item["file_path"], item["relative_path"], item["content"])
                for item in parsed_docs
            ],
            root_path=corpus_id,
            schema_def=schema_def,
            with_langextract=effective_with_metadata,
            langextract_profile=effective_profile,
        )

        indexed_files = 0
        chunks_written = 0
        all_chunk_records: list[ChunkRecord] = []
        parsed_pages_updated = 0
        image_placeholders_written = 0

        for item in parsed_docs:
            file_path = str(item["file_path"])
            relative_path = str(item["relative_path"])
            doc_id = str(item["doc_id"])
            content = str(item["content"])
            parsed_document = item["parsed_document"]
            unit_offsets = list(item.get("unit_offsets") or [])
            document_row = dict(item["document_row"])
            chunks = self.chunker.chunk_text(content)
            metadata = metadata_map.get(relative_path, {})
            metadata_json = json.dumps(metadata, sort_keys=True)

            stat = os.stat(file_path)
            doc_record = DocumentRecord(
                id=doc_id,
                corpus_id=corpus_id,
                relative_path=relative_path,
                absolute_path=str(Path(file_path).resolve()),
                content=content,
                metadata_json=metadata_json,
                file_mtime=float(stat.st_mtime),
                file_size=int(stat.st_size),
                content_sha256=str(item["file_sha256"]),
                original_filename=str(
                    document_row.get("original_filename") or relative_path
                ),
                object_key=str(document_row.get("object_key") or ""),
                storage_uri=str(document_row.get("storage_uri") or ""),
                content_type=document_row.get("content_type"),
                upload_status="indexed",
            )

            chunk_records: list[ChunkRecord] = []
            for chunk in chunks:
                chunk_records.append(
                    ChunkRecord(
                        id=PostgresStorage.make_chunk_id(
                            doc_id,
                            chunk.position,
                            chunk.start_char,
                            chunk.end_char,
                        ),
                        doc_id=doc_id,
                        text=chunk.text,
                        position=chunk.position,
                        source_unit_no=self._pick_chunk_unit_no(
                            chunk_start=chunk.start_char,
                            chunk_end=chunk.end_char,
                            unit_offsets=unit_offsets,
                        ),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                    )
                )

            self.storage.upsert_document(doc_record, chunk_records)

            if not bool(item["from_cache"]):
                unit_records = [
                    ParsedUnitRecord(
                        document_id=doc_id,
                        parser_name=parsed_document.parser_name,
                        parser_version=parsed_document.parser_version,
                        page_no=page.page_no,
                        markdown=page.markdown,
                        content_hash=page.content_hash,
                        heading=page.heading,
                        source_locator=page.source_locator,
                        images_json=json.dumps(
                            [
                                {
                                    "image_hash": image.image_hash,
                                    "image_index": image.image_index,
                                    "mime_type": image.mime_type,
                                    "width": image.width,
                                    "height": image.height,
                                }
                                for image in page.images
                            ],
                            sort_keys=True,
                        ),
                    )
                    for page in parsed_document.pages
                ]
                sync_stats = self.storage.sync_parsed_units(
                    document_id=doc_id,
                    parser_name=parsed_document.parser_name,
                    parser_version=parsed_document.parser_version,
                    units=unit_records,
                )
                parsed_pages_updated += int(sync_stats["upserted"])

                image_records = [
                    ImageSemanticRecord(
                        image_hash=image.image_hash,
                        source_document_id=doc_id,
                        source_page_no=image.page_no,
                        source_image_index=image.image_index,
                        mime_type=image.mime_type,
                        width=image.width,
                        height=image.height,
                    )
                    for page in parsed_document.pages
                    for image in page.images
                ]
                image_placeholders_written += self.storage.upsert_image_semantics(
                    images=image_records
                )

            all_chunk_records.extend(chunk_records)
            indexed_files += 1
            chunks_written += len(chunk_records)

        embeddings_written = self._generate_and_store_embeddings(
            corpus_id=corpus_id,
            all_chunk_records=all_chunk_records,
        )
        active_documents = len(
            self.storage.list_documents(corpus_id=corpus_id, include_deleted=False)
        )

        return IndexingResult(
            corpus_id=corpus_id,
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            deleted_files=0,
            chunks_written=chunks_written,
            active_documents=active_documents,
            schema_used=selected_schema_name,
            embeddings_written=embeddings_written,
            parsed_cache_hits=parsed_cache_hits,
            parsed_pages_updated=parsed_pages_updated,
            image_placeholders_written=image_placeholders_written,
        )

    def _extract_metadata_batch(
        self,
        *,
        parsed_docs: list[tuple[str, str, str]],
        root_path: str,
        schema_def: dict[str, Any] | None,
        with_langextract: bool,
        langextract_profile: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Extract metadata for all documents in parallel using a thread pool."""

        def _extract_one(item: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
            file_path, relative_path, content = item
            metadata = extract_metadata(
                file_path=file_path,
                root_path=root_path,
                content=content,
                schema_def=schema_def,
                with_langextract=with_langextract,
                langextract_profile=langextract_profile,
            )
            return relative_path, metadata

        result: dict[str, dict[str, Any]] = {}
        if not parsed_docs:
            return result

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for relative_path, metadata in executor.map(_extract_one, parsed_docs):
                result[relative_path] = metadata

        return result

    def _parse_document_for_indexing(self, file_path: str):
        compatibility_parse = globals()["parse_file"]
        if compatibility_parse is parse_file:
            return parse_document(file_path)

        content = compatibility_parse(file_path)
        if self._is_parse_error(content):
            raise DocumentParseError(
                file_path=file_path,
                code="compat_parse_failed",
                message=content,
            )
        normalized = str(content)
        return ParsedDocument(
            parser_name="compat_parse_file",
            parser_version=PARSER_VERSION,
            units=(
                ParsedUnit(
                    unit_no=1,
                    markdown=normalized,
                    content_hash=self._sha256(normalized),
                    source_locator="unit-1",
                ),
            ),
        )

    def _resolve_schema(
        self,
        *,
        corpus_id: str,
        root: str,
        discover_schema: bool,
        schema_name: str | None,
        with_metadata: bool,
        metadata_profile: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        root_exists = os.path.isdir(root)
        if not root_exists and (discover_schema or with_metadata):
            fallback_name = schema_name or f"auto_{corpus_id.split('_')[-1]}"
            if with_metadata:
                effective_profile = metadata_profile
                base_schema: dict[str, Any] = {
                    "name": fallback_name,
                    "fields": [],
                }
                augmented_schema, _ = ensure_langextract_schema_fields(
                    base_schema,
                    effective_profile,
                )
                if effective_profile is not None:
                    augmented_schema["metadata_profile"] = effective_profile
                self.storage.save_schema(
                    corpus_id=corpus_id,
                    name=fallback_name,
                    schema_def=augmented_schema,
                    is_active=True,
                )
                return augmented_schema, fallback_name
            return None, None

        if discover_schema:
            schema_def = SchemaDiscovery().discover_from_folder(
                root,
                with_langextract=with_metadata,
                metadata_profile=metadata_profile,
            )
            discovered_name = str(schema_def.get("name", f"auto_{Path(root).name}"))
            self.storage.save_schema(
                corpus_id=corpus_id,
                name=discovered_name,
                schema_def=schema_def,
                is_active=True,
            )
            return schema_def, discovered_name

        if schema_name:
            schema = self.storage.get_schema_by_name(
                corpus_id=corpus_id, name=schema_name
            )
            if schema is None:
                raise ValueError(f"Schema '{schema_name}' not found for corpus {root}")
            if with_metadata:
                return self._augment_schema_for_langextract(
                    corpus_id=corpus_id,
                    schema_name=schema.name,
                    schema_def=schema.schema_def,
                    metadata_profile=metadata_profile,
                )
            return schema.schema_def, schema.name

        active = self.storage.get_active_schema(corpus_id=corpus_id)
        if active is None:
            if with_metadata:
                schema_def = SchemaDiscovery().discover_from_folder(
                    root,
                    with_langextract=True,
                    metadata_profile=metadata_profile,
                )
                discovered_name = str(schema_def.get("name", f"auto_{Path(root).name}"))
                self.storage.save_schema(
                    corpus_id=corpus_id,
                    name=discovered_name,
                    schema_def=schema_def,
                    is_active=True,
                )
                return schema_def, discovered_name
            return None, None
        if with_metadata:
            return self._augment_schema_for_langextract(
                corpus_id=corpus_id,
                schema_name=active.name,
                schema_def=active.schema_def,
                metadata_profile=metadata_profile,
            )
        return active.schema_def, active.name

    def _augment_schema_for_langextract(
        self,
        *,
        corpus_id: str,
        schema_name: str,
        schema_def: dict[str, Any],
        metadata_profile: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str]:
        effective_profile = metadata_profile or self._schema_metadata_profile(
            schema_def
        )
        existing_field_names = self._schema_field_names(schema_def)
        required = langextract_field_names(effective_profile)
        if required.issubset(existing_field_names):
            if metadata_profile is None and (
                effective_profile is None
                or self._schema_metadata_profile(schema_def) is not None
            ):
                return schema_def, schema_name

            augmented_with_profile, changed = ensure_langextract_schema_fields(
                schema_def,
                effective_profile,
            )
            if not changed:
                return schema_def, schema_name
            self.storage.save_schema(
                corpus_id=corpus_id,
                name=schema_name,
                schema_def=augmented_with_profile,
                is_active=True,
            )
            return augmented_with_profile, schema_name

        augmented_schema, _ = ensure_langextract_schema_fields(
            schema_def,
            effective_profile,
        )
        self.storage.save_schema(
            corpus_id=corpus_id,
            name=schema_name,
            schema_def=augmented_schema,
            is_active=True,
        )
        return augmented_schema, schema_name

    @staticmethod
    def _schema_metadata_profile(
        schema_def: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not schema_def:
            return None
        profile = schema_def.get("metadata_profile")
        if isinstance(profile, dict):
            return profile
        return None

    @staticmethod
    def _schema_field_names(schema_def: dict[str, Any]) -> set[str]:
        fields = schema_def.get("fields")
        if not isinstance(fields, list):
            return set()
        names: set[str] = set()
        for field in fields:
            if isinstance(field, dict):
                name = field.get("name")
                if isinstance(name, str):
                    names.add(name)
        return names

    def _generate_and_store_embeddings(
        self,
        *,
        corpus_id: str,
        all_chunk_records: list[ChunkRecord],
    ) -> int:
        """Embed chunk texts and store in the database. Returns count written."""
        if self.embedding_provider is None or not all_chunk_records:
            return 0

        texts = [cr.text for cr in all_chunk_records]
        embeddings = self.embedding_provider.embed_texts(texts)

        pairs: list[tuple[str, list[float]]] = [
            (cr.id, emb) for cr, emb in zip(all_chunk_records, embeddings)
        ]
        written = self.storage.store_chunk_embeddings(
            corpus_id=corpus_id,
            chunk_embeddings=pairs,
        )

        if isinstance(self.storage, PostgresStorage):
            self.storage.create_hnsw_index(corpus_id=corpus_id)

        return written

    @staticmethod
    def _join_units_with_offsets(
        units: tuple[ParsedUnit, ...],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        chunks: list[str] = []
        offsets: list[tuple[int, int, int]] = []
        cursor = 0
        for unit in units:
            text = unit.markdown.strip()
            if not text:
                continue
            if chunks:
                cursor += 2
            start = cursor
            chunks.append(text)
            cursor += len(text)
            offsets.append((unit.unit_no, start, cursor))
        return "\n\n".join(chunks), offsets

    @staticmethod
    def _pick_chunk_unit_no(
        *,
        chunk_start: int,
        chunk_end: int,
        unit_offsets: list[tuple[int, int, int]],
    ) -> int | None:
        if not unit_offsets:
            return None

        best_unit: int | None = None
        best_overlap = -1
        best_distance = 10**9
        for unit_no, unit_start, unit_end in unit_offsets:
            overlap = max(0, min(chunk_end, unit_end) - max(chunk_start, unit_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_unit = unit_no
                midpoint = (chunk_start + chunk_end) // 2
                unit_mid = (unit_start + unit_end) // 2
                best_distance = abs(midpoint - unit_mid)
                continue
            if overlap == best_overlap:
                midpoint = (chunk_start + chunk_end) // 2
                unit_mid = (unit_start + unit_end) // 2
                distance = abs(midpoint - unit_mid)
                if distance < best_distance:
                    best_unit = unit_no
                    best_distance = distance
        return best_unit

    @staticmethod
    def _iter_supported_files(root: str) -> list[str]:
        files: list[str] = []
        for current_root, _, filenames in os.walk(root):
            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(str(Path(current_root) / filename))
        files.sort()
        return files

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_parse_error(content: str) -> bool:
        return content.startswith(_PARSE_ERROR_PREFIXES)
