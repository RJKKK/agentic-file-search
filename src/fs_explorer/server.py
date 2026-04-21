"""
FastAPI server for FsExplorer web UI.

Provides REST APIs for indexing/search and SSE endpoints for real-time workflow
streaming.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import (
    clear_index_context,
    get_context_budget_stats,
    get_context_state_snapshot,
    get_last_focus_anchor,
    get_lazy_indexing_stats,
    initialize_context_state,
    set_index_context,
    set_runtime_event_callback,
    set_search_flags,
)
from .blob_store import LocalBlobStore
from .document_library import (
    build_document_object_key,
    ensure_library_corpus,
    get_library_corpus_id,
    materialize_document,
    resolve_document_scope,
)
from .document_parsing import PARSER_VERSION, ParseSelector
from .document_cache import get_or_parse_document_units
from .embeddings import EmbeddingProvider
from .exploration_trace import ExplorationTrace, extract_cited_sources
from .explore_sessions import (
    ExploreSession,
    ExploreSessionManager,
    encode_sse_event,
    utc_now,
)
from .index_config import resolve_db_path
from .indexing import IndexingPipeline
from .indexing.metadata import auto_discover_profile
from .search import IndexedQueryEngine
from .storage import CollectionRecord, DocumentRecord, PostgresStorage
from .workflow import (
    AskHumanEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ToolCallEvent,
    get_agent,
    reset_agent,
    workflow,
)

app = FastAPI(title="FsExplorer", description="AI-powered filesystem exploration")

_corpus_locks: dict[str, asyncio.Lock] = {}
_session_manager = ExploreSessionManager()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
_LEGACY_UI = Path(__file__).parent / "ui.html"
_blob_store = LocalBlobStore()


def _get_corpus_lock(folder: str) -> asyncio.Lock:
    """Return a per-folder asyncio lock, creating one if needed."""
    normalized = str(Path(folder).resolve())
    if normalized not in _corpus_locks:
        _corpus_locks[normalized] = asyncio.Lock()
    return _corpus_locks[normalized]


def _frontend_entry_path() -> Path | None:
    """Return the preferred frontend entry point for the current environment."""
    candidate = _FRONTEND_DIST / "index.html"
    if candidate.exists():
        return candidate
    if _LEGACY_UI.exists():
        return _LEGACY_UI
    return None


def _frontend_asset_path(relative_path: str) -> Path | None:
    """Resolve one requested frontend asset inside the built Vite dist folder."""
    if not relative_path:
        return None
    dist_root = _FRONTEND_DIST.resolve()
    candidate = (dist_root / relative_path).resolve()
    try:
        candidate.relative_to(dist_root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _serve_frontend_entry() -> HTMLResponse | FileResponse:
    """Serve the built frontend when available, otherwise fall back to legacy HTML."""
    entry_path = _frontend_entry_path()
    if entry_path is None:
        return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)
    return FileResponse(entry_path)


class ExploreSessionCreateRequest(BaseModel):
    """Request model for starting an exploration session."""

    task: str
    document_ids: list[str] | None = None
    collection_id: str | None = None
    db_path: str | None = None
    enable_semantic: bool = False
    enable_metadata: bool = False


class HumanReplyRequest(BaseModel):
    """Request model for answering an ask-human pause."""

    response: str


class IndexRequest(BaseModel):
    """Request model for index build/refresh."""

    folder: str = "."
    db_path: str | None = None
    discover_schema: bool = True
    schema_name: str | None = None
    with_metadata: bool = False
    metadata_profile: dict[str, Any] | None = None
    with_embeddings: bool = False


class AutoProfileRequest(BaseModel):
    """Request model for auto-profile generation."""

    folder: str = "."


class SearchRequest(BaseModel):
    """Request model for search queries."""

    query: str
    document_ids: list[str] | None = None
    collection_id: str | None = None
    filters: str | None = None
    limit: int = 5
    db_path: str | None = None


class DocumentUpdateRequest(BaseModel):
    """Patchable document metadata fields."""

    metadata: dict[str, Any]


class DocumentParseRequest(BaseModel):
    """Request model for explicit parsed-unit refreshes."""

    mode: str = "incremental"
    force: bool = False
    focus_hint: str | None = None
    anchor: int | None = None
    window: int = 1
    max_units: int | None = None


class CollectionCreateRequest(BaseModel):
    """Request model for creating a collection."""

    name: str


class CollectionUpdateRequest(BaseModel):
    """Request model for renaming a collection."""

    name: str


class CollectionDocumentAttachRequest(BaseModel):
    """Request model for attaching documents to a collection."""

    document_ids: list[str]


def _make_trace_id() -> str:
    return uuid4().hex


def _json_with_trace(
    payload: dict[str, Any],
    *,
    status_code: int = 200,
    trace_id: str | None = None,
) -> JSONResponse:
    resolved_trace_id = trace_id or _make_trace_id()
    response = JSONResponse(payload, status_code=status_code)
    response.headers["X-Trace-Id"] = resolved_trace_id
    return response


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
    }
    if details:
        payload["details"] = details
    return _json_with_trace(payload, status_code=status_code, trace_id=trace_id)


async def _set_session_state(
    session: ExploreSession,
    *,
    status: str | None = None,
    pending_question: str | None = None,
    final_result: str | None = None,
    error: str | None = None,
    last_focus_anchor: dict[str, Any] | None = None,
    context_budget_stats: dict[str, Any] | None = None,
    context_state_snapshot: dict[str, Any] | None = None,
    lazy_indexing_stats: dict[str, Any] | None = None,
) -> None:
    """Update session status metadata safely."""
    async with session._lock:
        if status is not None:
            session.status = status
        session.pending_question = pending_question
        session.final_result = final_result
        session.error = error
        if last_focus_anchor is not None:
            session.last_focus_anchor = last_focus_anchor
        if context_budget_stats is not None:
            session.context_budget_stats = context_budget_stats
        if context_state_snapshot is not None:
            session.context_state_snapshot = context_state_snapshot
        if lazy_indexing_stats is not None:
            session.lazy_indexing_stats = lazy_indexing_stats
        session.updated_at = utc_now()


def _resolve_folder_path(folder: str) -> Path:
    """Resolve and validate a folder path."""
    folder_path = Path(folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid folder: {folder}")
    return folder_path


def _ensure_index_exists(folder_path: Path, db_path: str | None) -> str:
    """Validate that an index exists for the requested folder."""
    resolved_db_path = resolve_db_path(db_path)
    try:
        storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="No index found for this folder.") from exc

    try:
        corpus_id = storage.get_corpus_id(str(folder_path))
    finally:
        storage.close()

    if corpus_id is None:
        raise HTTPException(status_code=404, detail="No index found for this folder.")
    return resolved_db_path


def _load_document_or_404(
    *,
    storage: PostgresStorage,
    doc_id: str,
) -> dict[str, Any]:
    document = storage.get_document(doc_id=doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _parse_metadata_json(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    return json.loads(raw)


def _serialize_document_summary(
    document: dict[str, Any],
    *,
    parsed_units_count: int | None = None,
) -> dict[str, Any]:
    metadata = _parse_metadata_json(document.get("metadata_json"))
    indexed_hash = str(document.get("content_sha256", "") or "")
    parsed_hash = document.get("parsed_content_sha256")
    parsed_complete = bool(document.get("parsed_is_complete", False))
    if bool(document.get("is_deleted", False)):
        status = "deleted"
    elif indexed_hash:
        status = "indexed"
    elif parsed_hash:
        status = "parsed_complete" if parsed_complete else "parsed_partial"
    else:
        status = str(document.get("upload_status") or "uploaded")
    payload = {
        "id": str(document["id"]),
        "corpus_id": str(document["corpus_id"]),
        "relative_path": str(document["relative_path"]),
        "absolute_path": str(document["absolute_path"]),
        "original_filename": str(
            document.get("original_filename") or document.get("relative_path") or ""
        ),
        "object_key": str(document.get("object_key") or ""),
        "storage_uri": str(document.get("storage_uri") or ""),
        "content_type": document.get("content_type"),
        "upload_status": str(document.get("upload_status") or "uploaded"),
        "file_size": int(document.get("file_size", 0) or 0),
        "file_mtime": float(document.get("file_mtime", 0.0) or 0.0),
        "content_sha256": indexed_hash,
        "parsed_content_sha256": parsed_hash,
        "parsed_is_complete": parsed_complete,
        "is_deleted": bool(document.get("is_deleted", False)),
        "status": status,
        "metadata": metadata,
    }
    if "last_indexed_at" in document:
        payload["last_indexed_at"] = document["last_indexed_at"]
    if parsed_units_count is not None:
        payload["parsed_units_count"] = int(parsed_units_count)
    return payload


def _serialize_collection(collection: CollectionRecord) -> dict[str, Any]:
    return {
        "id": collection.id,
        "name": collection.name,
        "is_deleted": collection.is_deleted,
        "created_at": collection.created_at,
        "updated_at": collection.updated_at,
    }


def _serialize_parsed_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_no": int(unit["page_no"]),
        "page_no": int(unit["page_no"]),
        "parser_name": str(unit["parser_name"]),
        "parser_version": str(unit["parser_version"]),
        "heading": unit.get("heading"),
        "source_locator": unit.get("source_locator"),
        "content_hash": str(unit["content_hash"]),
        "markdown": str(unit["markdown"]),
        "images": list(unit.get("images") or []),
    }


def _sync_document_parse(
    *,
    storage: PostgresStorage,
    document: dict[str, Any],
    selector: ParseSelector | None = None,
    force: bool,
) -> dict[str, Any]:
    parse_result = get_or_parse_document_units(
        storage=storage,
        document=document,
        blob_store=_blob_store,
        selector=selector,
        force=force,
    )
    refreshed_units = storage.list_parsed_units(
        document_id=str(document["id"]),
        parser_version=PARSER_VERSION,
    )
    return {
        "parsed_units": refreshed_units,
        "selected_units": parse_result.selected_document.units,
        "selected_total_units": parse_result.total_units,
        "cache_hits": parse_result.cache_hits,
        "parsed_units_updated": parse_result.parsed_units_updated,
        "images_detected": parse_result.images_detected,
        "search_index_stale": parse_result.search_index_stale,
        "from_cache": parse_result.from_cache,
        "parsed_is_complete": parse_result.parsed_is_complete,
    }


async def _run_exploration_session(session: ExploreSession) -> None:
    """Execute one workflow session and publish events into the SSE stream."""
    index_storage: PostgresStorage | None = None
    trace = ExplorationTrace(root_directory=str(_PROJECT_ROOT))
    step_number = 0
    loop = asyncio.get_running_loop()

    def runtime_event_callback(event_type: str, data: dict[str, Any]) -> None:
        loop.create_task(session.publish(event_type, data))

    try:
        clear_index_context()
        resolved_db_path = resolve_db_path(session.db_path)
        index_storage = PostgresStorage(resolved_db_path)
        scope = resolve_document_scope(
            storage=index_storage,
            document_ids=session.document_ids,
            collection_id=session.collection_id,
        )
        if scope.is_empty:
            raise ValueError("Question answering requires at least one selected document.")
        scoped_documents = [
            materialize_document(
                storage=index_storage,
                blob_store=_blob_store,
                document=document,
            )
            for document in scope.documents
        ]
        document_names = [
            str(document.get("original_filename") or document.get("relative_path") or document["id"])
            for document in scoped_documents
        ]
        initialize_context_state(
            task=session.task,
            documents=[
                {
                    "document_id": str(document["id"]),
                    "label": str(
                        document.get("original_filename")
                        or document.get("relative_path")
                        or document["id"]
                    ),
                    "file_path": str(document["absolute_path"]),
                }
                for document in scoped_documents
            ],
            collection_name=scope.collection.name if scope.collection is not None else None,
        )
        set_index_context(
            db_path=resolved_db_path,
            document_ids=scope.document_ids,
            scope_label=(
                scope.collection.name
                if scope.collection is not None
                else f"{len(scope.document_ids)} selected documents"
            ),
        )
        set_search_flags(
            enable_semantic=session.enable_semantic,
            enable_metadata=session.enable_metadata,
        )
        set_runtime_event_callback(runtime_event_callback)
        reset_agent()
        await _set_session_state(session, status="running")
        await session.publish(
            "start",
            {
                "task": session.task,
                "document_ids": list(scope.document_ids),
                "collection_id": scope.collection.id if scope.collection is not None else None,
                "document_names": document_names,
            },
        )
        await session.publish(
            "context_scope_updated",
            {
                "context_scope": get_context_state_snapshot().get("context_scope", {}),
            },
        )

        handler = workflow.run(
            start_event=InputEvent(
                task=session.task,
                document_ids=list(scope.document_ids),
                document_labels=document_names,
                collection_name=scope.collection.name if scope.collection is not None else None,
                enable_semantic=session.enable_semantic,
                enable_metadata=session.enable_metadata,
            )
        )
        async with session._lock:
            session.handler = handler
            session.updated_at = utc_now()

        async for event in handler.stream_events():
            if isinstance(event, ToolCallEvent):
                step_number += 1
                resolved_document_path: str | None = None
                if event.tool_name == "get_document":
                    doc_id = event.tool_input.get("doc_id")
                    if index_storage is not None and isinstance(doc_id, str) and doc_id:
                        document = index_storage.get_document(doc_id=doc_id)
                        if document and not document["is_deleted"]:
                            resolved_document_path = str(document["absolute_path"])
                trace.record_tool_call(
                    step_number=step_number,
                    tool_name=event.tool_name,
                    tool_input=event.tool_input,
                    resolved_document_path=resolved_document_path,
                )
                await session.publish(
                    "tool_call",
                    {
                        "step": step_number,
                        "tool_name": event.tool_name,
                        "tool_input": event.tool_input,
                        "reason": event.reason,
                        "context_plan": event.context_plan,
                    },
                )
            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                trace.record_go_deeper(step_number=step_number, directory=event.directory)
                await session.publish(
                    "go_deeper",
                    {
                        "step": step_number,
                        "directory": event.directory,
                        "reason": event.reason,
                        "context_plan": event.context_plan,
                    },
                )
            elif isinstance(event, AskHumanEvent):
                step_number += 1
                await _set_session_state(
                    session,
                    status="awaiting_human",
                    pending_question=event.question,
                )
                await session.publish(
                    "ask_human",
                    {
                        "step": step_number,
                        "question": event.question,
                        "reason": event.reason,
                        "context_plan": event.context_plan,
                    },
                )

        result = await handler
        cited_sources = extract_cited_sources(result.final_result)
        agent = get_agent()
        last_focus_anchor = get_last_focus_anchor()
        context_budget_stats = get_context_budget_stats()
        context_state_snapshot = get_context_state_snapshot()
        lazy_indexing_stats = get_lazy_indexing_stats()
        usage = agent.token_usage
        _, _, total_cost = usage._calculate_cost()

        if result.error:
            await _set_session_state(
                session,
                status="error",
                pending_question=None,
                final_result=result.final_result,
                error=result.error,
                last_focus_anchor=last_focus_anchor,
                context_budget_stats=context_budget_stats,
                context_state_snapshot=context_state_snapshot,
                lazy_indexing_stats=lazy_indexing_stats,
            )
            await session.publish("error", {"message": result.error})
            return

        await _set_session_state(
            session,
            status="completed",
            pending_question=None,
            final_result=result.final_result,
            error=None,
            last_focus_anchor=last_focus_anchor,
            context_budget_stats=context_budget_stats,
            context_state_snapshot=context_state_snapshot,
            lazy_indexing_stats=lazy_indexing_stats,
        )
        await session.publish(
            "complete",
            {
                "final_result": result.final_result,
                "error": result.error,
                "stats": {
                    "steps": step_number,
                    "api_calls": usage.api_calls,
                    "documents_scanned": usage.documents_scanned,
                    "documents_parsed": usage.documents_parsed,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "tool_result_chars": usage.tool_result_chars,
                    "estimated_cost": round(total_cost, 6),
                    "last_focus_anchor": last_focus_anchor,
                    "context_budget": context_budget_stats,
                    "context_scope": context_state_snapshot.get("context_scope", {}),
                    "lazy_indexing": lazy_indexing_stats,
                },
                "trace": {
                    "step_path": trace.step_path,
                    "referenced_documents": trace.sorted_documents(),
                    "cited_sources": cited_sources,
                    "context_scope": context_state_snapshot.get("context_scope", {}),
                    "coverage_by_document": context_state_snapshot.get("coverage_by_document", {}),
                    "compaction_actions": context_state_snapshot.get("compaction_actions", []),
                    "active_ranges": context_state_snapshot.get("context_scope", {}).get("active_ranges", []),
                    "promoted_evidence_units": context_state_snapshot.get("promoted_evidence_units", []),
                },
            },
        )
    except Exception as exc:
        await _set_session_state(
            session,
            status="error",
            pending_question=None,
            error=str(exc),
            context_state_snapshot=get_context_state_snapshot(),
        )
        await session.publish("error", {"message": str(exc)})
    finally:
        if index_storage is not None:
            index_storage.close()
        async with session._lock:
            session.handler = None
            session.updated_at = utc_now()
        reset_agent()
        clear_index_context()


@app.get("/", include_in_schema=False)
async def get_ui():
    """Serve the main frontend entry point."""
    return _serve_frontend_entry()


@app.get("/api/folders")
async def list_folders(path: str = "."):
    """
    List folders in the given path.
    Returns list of folder names and current path info.
    """
    try:
        base_path = Path(path).resolve()
        if not base_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not base_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        folders = sorted(
            [
                f.name
                for f in base_path.iterdir()
                if f.is_dir() and not f.name.startswith(".")
            ]
        )
        parent = str(base_path.parent) if base_path != base_path.parent else None

        return {
            "current": str(base_path),
            "parent": parent,
            "folders": folders,
            "files_count": len([f for f in base_path.iterdir() if f.is_file()]),
        }
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/index/status")
async def index_status(folder: str, db_path: str | None = None):
    """Check whether a folder has been indexed and return status details."""
    try:
        folder_path = Path(folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return {"indexed": False}

        resolved_db_path = resolve_db_path(db_path)
        try:
            storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
        except Exception:
            return {"indexed": False}

        try:
            corpus_id = storage.get_corpus_id(str(folder_path))
            if corpus_id is None:
                storage.close()
                return {"indexed": False}

            docs = storage.list_documents(corpus_id=corpus_id, include_deleted=False)
            active_schema = storage.get_active_schema(corpus_id=corpus_id)
            has_embeddings = storage.has_embeddings(corpus_id=corpus_id)

            schema_name: str | None = None
            has_metadata = False
            schema_fields: list[str] = []
            if active_schema is not None:
                schema_name = active_schema.name
                has_metadata = active_schema.schema_def.get("metadata_profile") is not None
                fields_def = active_schema.schema_def.get("fields")
                if isinstance(fields_def, list):
                    for field in fields_def:
                        if isinstance(field, dict) and isinstance(field.get("name"), str):
                            schema_fields.append(field["name"])

            storage.close()
            return {
                "indexed": True,
                "corpus_id": corpus_id,
                "document_count": len(docs),
                "schema_name": schema_name,
                "has_metadata": has_metadata,
                "has_embeddings": has_embeddings,
                "schema_fields": schema_fields,
            }
        except Exception:
            storage.close()
            return {"indexed": False}
    except Exception:
        return {"indexed": False}


@app.post("/api/index/auto-profile")
async def generate_auto_profile(request: AutoProfileRequest):
    """Generate an auto-discovered metadata profile for preview/editing."""
    try:
        folder_path = Path(request.folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse({"error": f"Invalid folder: {request.folder}"}, status_code=400)

        profile = await asyncio.to_thread(auto_discover_profile, str(folder_path))
        return {"profile": profile}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/index")
async def build_index(request: IndexRequest):
    """Build or refresh the index for a selected folder."""
    try:
        folder_path = Path(request.folder).resolve()
        if not folder_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not folder_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        lock = _get_corpus_lock(str(folder_path))
        async with lock:
            resolved_db_path = resolve_db_path(request.db_path)
            embedding_provider: EmbeddingProvider | None = None
            if request.with_embeddings:
                try:
                    embedding_provider = EmbeddingProvider()
                except ValueError:
                    embedding_provider = None
            pipeline = IndexingPipeline(
                storage=PostgresStorage(resolved_db_path),
                embedding_provider=embedding_provider,
            )
            effective_with_metadata = request.with_metadata or request.metadata_profile is not None
            discover_schema = request.discover_schema or effective_with_metadata
            result = pipeline.index_folder(
                str(folder_path),
                discover_schema=discover_schema,
                schema_name=request.schema_name,
                with_metadata=effective_with_metadata,
                metadata_profile=request.metadata_profile,
            )

        return {
            "db_path": resolved_db_path,
            "folder": str(folder_path),
            "corpus_id": result.corpus_id,
            "indexed_files": result.indexed_files,
            "skipped_files": result.skipped_files,
            "deleted_files": result.deleted_files,
            "chunks_written": result.chunks_written,
            "active_documents": result.active_documents,
            "schema_used": result.schema_used,
            "embeddings_written": result.embeddings_written,
            "parsed_cache_hits": result.parsed_cache_hits,
            "parsed_pages_updated": result.parsed_pages_updated,
            "image_placeholders_written": result.image_placeholders_written,
            "metadata_mode": "langextract" if effective_with_metadata else "heuristic",
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/search")
async def search_index(request: SearchRequest):
    """Search the selected document scope and return ranked hits."""
    try:
        resolved_db_path = resolve_db_path(request.db_path)
        storage = PostgresStorage(resolved_db_path)
        scope = resolve_document_scope(
            storage=storage,
            document_ids=request.document_ids,
            collection_id=request.collection_id,
        )
        if scope.is_empty:
            storage.close()
            return JSONResponse(
                {"error": "At least one document or one collection must be selected."},
                status_code=400,
            )
        materialized_scope = [
            materialize_document(
                storage=storage,
                blob_store=_blob_store,
                document=document,
            )
            for document in scope.documents
        ]

        embedding_provider: EmbeddingProvider | None = None
        if storage.has_embeddings(corpus_id=scope.corpus_id):
            try:
                embedding_provider = EmbeddingProvider()
            except ValueError:
                pass

        engine = IndexedQueryEngine(storage, embedding_provider=embedding_provider)
        hits = engine.search(
            corpus_id=scope.corpus_id,
            query=request.query,
            document_ids=scope.document_ids,
            filters=request.filters,
            limit=request.limit,
        )
        storage.close()
        document_names = {
            str(document["id"]): str(
                document.get("original_filename")
                or document.get("relative_path")
                or document["id"]
            )
            for document in materialized_scope
        }

        return {
            "query": request.query,
            "document_ids": list(scope.document_ids),
            "collection_id": request.collection_id,
            "lazy_indexing": engine.get_last_lazy_indexing_stats(),
            "hits": [
                {
                    "doc_id": hit.doc_id,
                    "original_filename": document_names.get(hit.doc_id, hit.relative_path),
                    "relative_path": hit.relative_path,
                    "absolute_path": hit.absolute_path,
                    "position": hit.position,
                    "source_unit_no": hit.source_unit_no,
                    "text": hit.text,
                    "semantic_score": hit.semantic_score,
                    "metadata_score": hit.metadata_score,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                }
                for hit in hits
            ],
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/documents")
async def list_documents_api(
    db_path: str | None = None,
    include_deleted: bool = False,
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """List documents in the shared uploaded document library."""
    trace_id = _make_trace_id()
    if page < 1 or page_size < 1:
        return _error_response(
            status_code=400,
            error_code="invalid_pagination",
            message="`page` and `page_size` must be positive integers.",
            trace_id=trace_id,
        )

    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
        try:
            corpus_id = get_library_corpus_id(storage, create_if_missing=False)
            documents = (
                storage.list_documents(
                    corpus_id=corpus_id,
                    include_deleted=include_deleted,
                )
                if corpus_id is not None
                else []
            )
        finally:
            storage.close()

        query = (q or "").strip().lower()
        if query:
            filtered_documents: list[dict[str, Any]] = []
            for document in documents:
                metadata_text = json.dumps(
                    _parse_metadata_json(document.get("metadata_json")),
                    ensure_ascii=False,
                    sort_keys=True,
                ).lower()
                haystack = " ".join(
                    [
                        str(document.get("original_filename", "")).lower(),
                        str(document.get("relative_path", "")).lower(),
                        str(document.get("absolute_path", "")).lower(),
                        metadata_text,
                    ]
                )
                if query in haystack:
                    filtered_documents.append(document)
            documents = filtered_documents

        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = documents[start:end]

        return _json_with_trace(
            {
                "library": "default",
                "corpus_id": corpus_id or "",
                "page": page,
                "page_size": page_size,
                "total": total,
                "items": [_serialize_document_summary(item) for item in page_items],
            },
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_list_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.post("/api/documents")
async def upload_document_api(
    file: UploadFile = File(...),
    db_path: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
    discover_schema: bool = Form(default=True),
    with_metadata: bool = Form(default=False),
    with_embeddings: bool = Form(default=False),
):
    """Upload a document into the shared library without eager parsing or indexing."""
    trace_id = _make_trace_id()
    _ = (overwrite, discover_schema, with_metadata, with_embeddings)
    filename = Path(file.filename or "").name
    if not filename:
        return _error_response(
            status_code=400,
            error_code="missing_filename",
            message="Uploaded file must include a filename.",
            trace_id=trace_id,
        )

    try:
        lock = _get_corpus_lock("blob://library/default")
        async with lock:
            resolved_db_path = resolve_db_path(db_path)
            storage = PostgresStorage(resolved_db_path)
            try:
                corpus_id = ensure_library_corpus(storage)
                doc_id = uuid4().hex
                object_key = build_document_object_key(doc_id, filename)
                blob_head = _blob_store.put(object_key=object_key, data=file.file)
                document_record = DocumentRecord(
                    id=doc_id,
                    corpus_id=corpus_id,
                    relative_path=object_key,
                    absolute_path=blob_head.absolute_path,
                    content="",
                    metadata_json="{}",
                    file_mtime=float(Path(blob_head.absolute_path).stat().st_mtime),
                    file_size=int(blob_head.size),
                    content_sha256="",
                    original_filename=filename,
                    object_key=object_key,
                    storage_uri=blob_head.storage_uri,
                    content_type=file.content_type,
                    upload_status="uploaded",
                )
                storage.upsert_document_stub(document_record)
                document = _load_document_or_404(storage=storage, doc_id=doc_id)
                parsed_units = storage.list_parsed_units(
                    document_id=doc_id,
                    parser_version=PARSER_VERSION,
                )
            finally:
                storage.close()

        return _json_with_trace(
            {
                "document": _serialize_document_summary(
                    document,
                    parsed_units_count=len(parsed_units),
                ),
                "upload_result": {
                    "corpus_id": document["corpus_id"],
                    "storage_uri": document.get("storage_uri"),
                    "parsed_on_upload": False,
                    "indexed_on_upload": False,
                },
            },
            status_code=201,
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_upload_failed",
            message=str(exc),
            trace_id=trace_id,
        )
    finally:
        await file.close()


@app.get("/api/documents/{doc_id}")
async def get_document_api(doc_id: str, db_path: str | None = None):
    """Return detail for one indexed document, including parse summary."""
    trace_id = _make_trace_id()
    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
        try:
            document = _load_document_or_404(storage=storage, doc_id=doc_id)
            parsed_units = storage.list_parsed_units(
                document_id=doc_id,
                parser_version=PARSER_VERSION,
            )
        finally:
            storage.close()

        return _json_with_trace(
            {
                "document": _serialize_document_summary(
                    document,
                    parsed_units_count=len(parsed_units),
                ),
                "parse_summary": {
                    "parser_version": PARSER_VERSION,
                    "parsed_units_count": len(parsed_units),
                    "latest_unit_no": (
                        max(int(unit["page_no"]) for unit in parsed_units)
                        if parsed_units
                        else 0
                    ),
                    "images_detected": sum(
                        len(unit.get("images") or []) for unit in parsed_units
                    ),
                },
            },
            trace_id=trace_id,
        )
    except HTTPException as exc:
        return _error_response(
            status_code=exc.status_code,
            error_code="document_not_found",
            message=str(exc.detail),
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_detail_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.patch("/api/documents/{doc_id}")
async def update_document_api(
    doc_id: str,
    request: DocumentUpdateRequest,
    db_path: str | None = None,
):
    """Replace document metadata JSON for one indexed document."""
    trace_id = _make_trace_id()
    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path)
        try:
            updated = storage.update_document_metadata(
                doc_id=doc_id,
                metadata_json=json.dumps(request.metadata, sort_keys=True),
            )
        finally:
            storage.close()

        if updated is None:
            return _error_response(
                status_code=404,
                error_code="document_not_found",
                message="Document not found",
                trace_id=trace_id,
            )

        return _json_with_trace(
            {"document": _serialize_document_summary(updated)},
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_update_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.delete("/api/documents/{doc_id}")
async def delete_document_api(doc_id: str, db_path: str | None = None):
    """Logically delete one indexed document."""
    trace_id = _make_trace_id()
    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path)
        try:
            document = _load_document_or_404(storage=storage, doc_id=doc_id)
            storage.remove_document_from_all_collections(doc_id=doc_id)
            updated = storage.set_document_deleted(doc_id=doc_id, is_deleted=True)
            object_key = str(document.get("object_key") or "")
            if object_key:
                _blob_store.delete(object_key=object_key)
        finally:
            storage.close()

        if updated is None:
            return _error_response(
                status_code=404,
                error_code="document_not_found",
                message="Document not found",
                trace_id=trace_id,
            )

        return _json_with_trace(
            {
                "document": _serialize_document_summary(updated),
                "deleted": True,
            },
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_delete_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.post("/api/documents/{doc_id}/parse")
async def parse_document_api(
    doc_id: str,
    request: DocumentParseRequest,
    db_path: str | None = None,
):
    """Refresh parsed-unit cache for a specific indexed document."""
    trace_id = _make_trace_id()
    if request.mode not in {"incremental", "full"}:
        return _error_response(
            status_code=400,
            error_code="invalid_parse_mode",
            message="`mode` must be either `incremental` or `full`.",
            trace_id=trace_id,
        )

    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path)
        try:
            document = _load_document_or_404(storage=storage, doc_id=doc_id)
            parse_result = _sync_document_parse(
                storage=storage,
                document=document,
                selector=ParseSelector(
                    query=request.focus_hint.strip() if request.focus_hint else None,
                    anchor=request.anchor,
                    window=max(request.window, 0),
                    max_units=request.max_units,
                )
                if (
                    request.focus_hint
                    or request.anchor is not None
                    or request.max_units is not None
                )
                else None,
                force=bool(request.force or request.mode == "full"),
            )
        finally:
            storage.close()

        return _json_with_trace(
            {
                "document_id": doc_id,
                "parser_version": PARSER_VERSION,
                "parsed_units": len(parse_result["parsed_units"]),
                "parsed_units_updated": parse_result["parsed_units_updated"],
                "cache_hits": parse_result["cache_hits"],
                "images_detected": parse_result["images_detected"],
                "search_index_stale": parse_result["search_index_stale"],
                "from_cache": parse_result["from_cache"],
                "parsed_is_complete": parse_result["parsed_is_complete"],
            },
            trace_id=trace_id,
        )
    except HTTPException as exc:
        return _error_response(
            status_code=exc.status_code,
            error_code="document_not_found",
            message=str(exc.detail),
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_parse_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.get("/api/documents/{doc_id}/pages")
async def list_document_pages_api(
    doc_id: str,
    db_path: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """Return paginated parsed units for a document."""
    trace_id = _make_trace_id()
    if page < 1 or page_size < 1:
        return _error_response(
            status_code=400,
            error_code="invalid_pagination",
            message="`page` and `page_size` must be positive integers.",
            trace_id=trace_id,
        )

    try:
        resolved_db_path = resolve_db_path(db_path)
        storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
        try:
            _load_document_or_404(storage=storage, doc_id=doc_id)
            parsed_units = storage.list_parsed_units(
                document_id=doc_id,
                parser_version=PARSER_VERSION,
            )
        finally:
            storage.close()

        total = len(parsed_units)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = parsed_units[start:end]
        return _json_with_trace(
            {
                "document_id": doc_id,
                "parser_version": PARSER_VERSION,
                "page": page,
                "page_size": page_size,
                "total": total,
                "items": [_serialize_parsed_unit(unit) for unit in page_items],
            },
            trace_id=trace_id,
        )
    except HTTPException as exc:
        return _error_response(
            status_code=exc.status_code,
            error_code="document_not_found",
            message=str(exc.detail),
            trace_id=trace_id,
        )
    except Exception as exc:
        return _error_response(
            status_code=500,
            error_code="document_pages_failed",
            message=str(exc),
            trace_id=trace_id,
        )


@app.get("/api/collections")
async def list_collections_api(db_path: str | None = None):
    """List saved document collections."""
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
    try:
        collections = storage.list_collections(include_deleted=False)
        return {"items": [_serialize_collection(collection) for collection in collections]}
    finally:
        storage.close()


@app.post("/api/collections")
async def create_collection_api(
    request: CollectionCreateRequest,
    db_path: str | None = None,
):
    """Create a new collection."""
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name is required.")
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path)
    try:
        collection = storage.create_collection(name=name)
        return {"collection": _serialize_collection(collection)}
    finally:
        storage.close()


@app.patch("/api/collections/{collection_id}")
async def update_collection_api(
    collection_id: str,
    request: CollectionUpdateRequest,
    db_path: str | None = None,
):
    """Rename a collection."""
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name is required.")
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path)
    try:
        collection = storage.update_collection(collection_id=collection_id, name=name)
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")
        return {"collection": _serialize_collection(collection)}
    finally:
        storage.close()


@app.delete("/api/collections/{collection_id}")
async def delete_collection_api(collection_id: str, db_path: str | None = None):
    """Delete a collection."""
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path)
    try:
        collection = storage.set_collection_deleted(
            collection_id=collection_id,
            is_deleted=True,
        )
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")
        return {"collection": _serialize_collection(collection), "deleted": True}
    finally:
        storage.close()


@app.get("/api/collections/{collection_id}/documents")
async def list_collection_documents_api(collection_id: str, db_path: str | None = None):
    """List documents inside one collection."""
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
    try:
        collection = storage.get_collection(collection_id=collection_id)
        if collection is None or collection.is_deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        documents = storage.list_collection_documents(
            collection_id=collection_id,
            include_deleted=False,
        )
        return {
            "collection": _serialize_collection(collection),
            "items": [_serialize_document_summary(document) for document in documents],
        }
    finally:
        storage.close()


@app.post("/api/collections/{collection_id}/documents")
async def attach_collection_documents_api(
    collection_id: str,
    request: CollectionDocumentAttachRequest,
    db_path: str | None = None,
):
    """Attach documents to a collection."""
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path)
    try:
        collection = storage.get_collection(collection_id=collection_id)
        if collection is None or collection.is_deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        attached = storage.attach_documents_to_collection(
            collection_id=collection_id,
            document_ids=request.document_ids,
        )
        documents = storage.list_collection_documents(
            collection_id=collection_id,
            include_deleted=False,
        )
        return {
            "collection": _serialize_collection(collection),
            "attached": attached,
            "items": [_serialize_document_summary(document) for document in documents],
        }
    finally:
        storage.close()


@app.delete("/api/collections/{collection_id}/documents/{doc_id}")
async def detach_collection_document_api(
    collection_id: str,
    doc_id: str,
    db_path: str | None = None,
):
    """Detach one document from a collection."""
    resolved_db_path = resolve_db_path(db_path)
    storage = PostgresStorage(resolved_db_path)
    try:
        collection = storage.get_collection(collection_id=collection_id)
        if collection is None or collection.is_deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        removed = storage.detach_document_from_collection(
            collection_id=collection_id,
            doc_id=doc_id,
        )
        return {"removed": removed}
    finally:
        storage.close()


@app.post("/api/explore/sessions")
async def create_explore_session(request: ExploreSessionCreateRequest):
    """Create and launch a new exploration session."""
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="No task provided")
    resolved_db_path = resolve_db_path(request.db_path)
    storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
    try:
        scope = resolve_document_scope(
            storage=storage,
            document_ids=request.document_ids,
            collection_id=request.collection_id,
        )
    finally:
        storage.close()
    if scope.is_empty:
        raise HTTPException(
            status_code=400,
            detail="At least one document or one collection must be selected.",
        )

    session = await _session_manager.create_session(
        task=task,
        document_ids=scope.document_ids,
        collection_id=request.collection_id,
        db_path=resolved_db_path,
        enable_semantic=bool(request.enable_semantic),
        enable_metadata=bool(request.enable_metadata),
    )
    session.workflow_task = asyncio.create_task(
        _run_exploration_session(session),
        name=f"explore-session-{session.session_id}",
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
    }


@app.get("/api/explore/sessions/{session_id}")
async def get_explore_session(session_id: str):
    """Return the current state of an exploration session."""
    session = await _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.snapshot()


@app.get("/api/explore/sessions/{session_id}/events")
async def stream_explore_session_events(session_id: str, request: Request):
    """Stream exploration events over SSE."""
    session, queue, history = await _session_manager.subscribe(session_id)
    if session is None or queue is None or history is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream():
        keepalive_ticks = 0
        try:
            for event in history:
                yield encode_sse_event(session.session_id, event)
                if event.type in {"complete", "error"}:
                    return

            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                    keepalive_ticks = 0
                except TimeoutError:
                    keepalive_ticks += 1
                    if keepalive_ticks >= 15:
                        keepalive_ticks = 0
                        yield ": keepalive\n\n"
                    continue

                if event is None:
                    return
                yield encode_sse_event(session.session_id, event)
                if event.type in {"complete", "error"}:
                    return
        finally:
            await _session_manager.unsubscribe(session, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/explore/sessions/{session_id}/reply")
async def reply_to_explore_session(session_id: str, request: HumanReplyRequest):
    """Resume an exploration session after an ask-human event."""
    session = await _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "awaiting_human":
        raise HTTPException(
            status_code=409,
            detail="Session is not waiting for human input",
        )

    response = request.response.strip()
    if not response:
        raise HTTPException(status_code=400, detail="Response cannot be empty")

    async with session._lock:
        handler = session.handler
        if handler is None or getattr(handler, "ctx", None) is None:
            raise HTTPException(status_code=409, detail="Session handler is not available")
        session.status = "running"
        session.pending_question = None
        session.updated_at = utc_now()

    handler.ctx.send_event(HumanAnswerEvent(response=response))
    return {
        "session_id": session.session_id,
        "status": "running",
    }


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """Serve built frontend assets and fall back to the SPA entry point."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    asset_path = _frontend_asset_path(full_path)
    if asset_path is not None:
        return FileResponse(asset_path)

    if Path(full_path).suffix:
        raise HTTPException(status_code=404, detail="Not found")

    return _serve_frontend_entry()


if __name__ == "__main__":
    run_server()
