"""
Helpers for lazy parsed-unit caching backed by persistent storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .blob_store import BlobStore
from .document_parsing import (
    PARSER_VERSION,
    ParseSelector,
    ParsedDocument,
    compute_file_sha256,
    parse_document,
    resolve_requested_unit_nos,
    reconstruct_parsed_document,
    select_parsed_document,
)
from .storage import ImageSemanticRecord, ParsedUnitRecord, PostgresStorage


@dataclass(frozen=True)
class CachedParseResult:
    selected_document: ParsedDocument
    total_units: int | None
    from_cache: bool
    cache_hits: int
    parsed_units_updated: int
    images_detected: int
    search_index_stale: bool
    parsed_is_complete: bool


def format_parse_result(
    *,
    file_path: str,
    selected_document: ParsedDocument,
    total_units: int | None,
    anchor: int | None = None,
    window: int = 1,
    max_units: int | None = None,
) -> str:
    """Render a selected parsed document using the existing tool-friendly format."""
    if total_units is not None and len(selected_document.units) == total_units:
        return selected_document.markdown

    total_display = total_units if total_units is not None else "?"
    lines: list[str] = [
        f"=== FOCUSED PARSE of {file_path} ===",
        (
            f"Units returned: {len(selected_document.units)} / {total_display} "
            f"(anchor={anchor}, window={window}, max_units={max_units})"
        ),
        "",
    ]
    for unit in selected_document.units:
        lines.append(
            f"[UNIT {unit.unit_no} | source={unit.source_locator or f'unit-{unit.unit_no}'}"
            f" | heading={unit.heading or '-'}]"
        )
        lines.append(unit.markdown)
        lines.append("")
    return "\n".join(lines).strip()


def _requested_unit_nos(file_path: str, selector: ParseSelector | None) -> list[int] | None:
    return resolve_requested_unit_nos(file_path, selector)


def _to_unit_records(document_id: str, parsed_document: ParsedDocument) -> list[ParsedUnitRecord]:
    return [
        ParsedUnitRecord(
            document_id=document_id,
            parser_name=parsed_document.parser_name,
            parser_version=parsed_document.parser_version,
            page_no=unit.page_no,
            markdown=unit.markdown,
            content_hash=unit.content_hash,
            heading=unit.heading,
            source_locator=unit.source_locator,
            images_json=json.dumps(
                [
                    {
                        "image_hash": image.image_hash,
                        "image_index": image.image_index,
                        "mime_type": image.mime_type,
                        "width": image.width,
                        "height": image.height,
                    }
                    for image in unit.images
                ],
                sort_keys=True,
            ),
        )
        for unit in parsed_document.units
    ]


def _to_image_records(document_id: str, parsed_document: ParsedDocument) -> list[ImageSemanticRecord]:
    return [
        ImageSemanticRecord(
            image_hash=image.image_hash,
            source_document_id=document_id,
            source_page_no=image.page_no,
            source_image_index=image.image_index,
            mime_type=image.mime_type,
            width=image.width,
            height=image.height,
        )
        for unit in parsed_document.units
        for image in unit.images
    ]


def get_or_parse_document_units(
    *,
    storage: PostgresStorage,
    document: dict[str, object],
    blob_store: BlobStore | None = None,
    selector: ParseSelector | None = None,
    force: bool = False,
) -> CachedParseResult:
    """Fetch requested units from persistent cache or parse and persist them lazily."""
    object_key = str(document.get("object_key") or "")
    if blob_store is not None and object_key:
        file_path = blob_store.materialize(object_key=object_key)
        if str(document.get("absolute_path") or "") != file_path:
            storage.update_document_absolute_path(
                doc_id=str(document["id"]),
                absolute_path=file_path,
            )
            document = dict(document)
            document["absolute_path"] = file_path
    else:
        file_path = str(document["absolute_path"])
    document_id = str(document["id"])
    file_sha256 = compute_file_sha256(file_path)
    parsed_hash = document.get("parsed_content_sha256")
    parsed_is_complete = bool(document.get("parsed_is_complete", False))
    requested_unit_nos = _requested_unit_nos(file_path, selector)

    if not force and parsed_hash == file_sha256:
        if parsed_is_complete:
            cached_units = storage.list_parsed_units(
                document_id=document_id,
                parser_version=PARSER_VERSION,
            )
            cached_document = reconstruct_parsed_document(cached_units)
            if cached_document is not None:
                selected = select_parsed_document(cached_document, selector)
                if selected.units:
                    return CachedParseResult(
                        selected_document=ParsedDocument(
                            parser_name=cached_document.parser_name,
                            parser_version=cached_document.parser_version,
                            units=selected.units,
                        ),
                        total_units=len(cached_document.units),
                        from_cache=True,
                        cache_hits=len(selected.units),
                        parsed_units_updated=0,
                        images_detected=sum(len(unit.images) for unit in selected.units),
                        search_index_stale=str(document.get("content_sha256", "")) != file_sha256,
                        parsed_is_complete=True,
                    )
        elif requested_unit_nos is not None:
            cached_units = storage.list_parsed_units(
                document_id=document_id,
                parser_version=PARSER_VERSION,
                unit_nos=requested_unit_nos,
            )
            cached_numbers = {int(unit["page_no"]) for unit in cached_units}
            if set(requested_unit_nos).issubset(cached_numbers):
                cached_document = reconstruct_parsed_document(cached_units)
                if cached_document is not None:
                    selected = select_parsed_document(cached_document, selector)
                    if selected.units:
                        return CachedParseResult(
                            selected_document=ParsedDocument(
                                parser_name=cached_document.parser_name,
                                parser_version=cached_document.parser_version,
                                units=selected.units,
                            ),
                            total_units=None,
                            from_cache=True,
                            cache_hits=len(selected.units),
                            parsed_units_updated=0,
                            images_detected=sum(len(unit.images) for unit in selected.units),
                            search_index_stale=str(document.get("content_sha256", "")) != file_sha256,
                            parsed_is_complete=False,
                        )

    parsed_document = parse_document(file_path, selector=selector)
    storage.upsert_parsed_units(
        document_id=document_id,
        parser_name=parsed_document.parser_name,
        parser_version=parsed_document.parser_version,
        units=_to_unit_records(document_id, parsed_document),
    )
    storage.upsert_image_semantics(images=_to_image_records(document_id, parsed_document))

    is_complete = selector is None
    storage.update_document_parse_state(
        doc_id=document_id,
        parsed_content_sha256=file_sha256,
        parsed_is_complete=is_complete,
    )

    total_units = len(parsed_document.units) if is_complete else None
    return CachedParseResult(
        selected_document=parsed_document,
        total_units=total_units,
        from_cache=False,
        cache_hits=0,
        parsed_units_updated=len(parsed_document.units),
        images_detected=sum(len(unit.images) for unit in parsed_document.units),
        search_index_stale=str(document.get("content_sha256", "")) != file_sha256,
        parsed_is_complete=is_complete,
    )


def resolve_document_by_path(
    *,
    storage: PostgresStorage,
    corpus_root: str,
    file_path: str,
) -> dict[str, object] | None:
    """Best-effort document lookup for a filesystem path within a corpus."""
    corpus_id = storage.get_corpus_id(str(Path(corpus_root).resolve()))
    if corpus_id is None:
        return None
    resolved_path = Path(file_path).resolve()
    try:
        relative_path = str(resolved_path.relative_to(Path(corpus_root).resolve())).replace("\\", "/")
    except Exception:
        return None
    doc_id = storage.make_document_id(corpus_id, relative_path)
    return storage.get_document(doc_id=doc_id)
