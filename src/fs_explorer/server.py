"""
FastAPI server for FsExplorer web UI.

Provides REST APIs for indexing/search and SSE endpoints for real-time workflow
streaming.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import (
    clear_index_context,
    set_index_context,
    set_runtime_event_callback,
    set_search_flags,
)
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
from .storage import PostgresStorage
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


def _get_corpus_lock(folder: str) -> asyncio.Lock:
    """Return a per-folder asyncio lock, creating one if needed."""
    normalized = str(Path(folder).resolve())
    if normalized not in _corpus_locks:
        _corpus_locks[normalized] = asyncio.Lock()
    return _corpus_locks[normalized]


class ExploreSessionCreateRequest(BaseModel):
    """Request model for starting an exploration session."""

    task: str
    folder: str = "."
    use_index: bool = False
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

    corpus_folder: str
    query: str
    filters: str | None = None
    limit: int = 5
    db_path: str | None = None


async def _set_session_state(
    session: ExploreSession,
    *,
    status: str | None = None,
    pending_question: str | None = None,
    final_result: str | None = None,
    error: str | None = None,
) -> None:
    """Update session status metadata safely."""
    async with session._lock:
        if status is not None:
            session.status = status
        session.pending_question = pending_question
        session.final_result = final_result
        session.error = error
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


async def _run_exploration_session(session: ExploreSession) -> None:
    """Execute one workflow session and publish events into the SSE stream."""
    folder_path = Path(session.folder).resolve()
    index_storage: PostgresStorage | None = None
    trace = ExplorationTrace(root_directory=str(folder_path))
    step_number = 0
    loop = asyncio.get_running_loop()

    def runtime_event_callback(event_type: str, data: dict[str, Any]) -> None:
        loop.create_task(session.publish(event_type, data))

    try:
        clear_index_context()
        if session.use_index:
            resolved_db_path = resolve_db_path(session.db_path)
            index_storage = PostgresStorage(resolved_db_path)
            corpus_id = index_storage.get_corpus_id(str(folder_path))
            if corpus_id is None:
                raise ValueError(
                    "No index found for the selected folder. "
                    "Run `explore index <folder>` first."
                )
            set_index_context(str(folder_path), resolved_db_path)

        set_search_flags(
            enable_semantic=session.enable_semantic and session.use_index,
            enable_metadata=session.enable_metadata and session.use_index,
        )
        set_runtime_event_callback(runtime_event_callback)
        reset_agent()
        await _set_session_state(session, status="running")
        await session.publish(
            "start",
            {
                "task": session.task,
                "folder": str(folder_path),
                "use_index": session.use_index,
            },
        )

        handler = workflow.run(
            start_event=InputEvent(
                task=session.task,
                folder=str(folder_path),
                use_index=session.use_index,
                enable_semantic=session.enable_semantic and session.use_index,
                enable_metadata=session.enable_metadata and session.use_index,
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
                    },
                )

        result = await handler
        cited_sources = extract_cited_sources(result.final_result)
        agent = get_agent()
        usage = agent.token_usage
        _, _, total_cost = usage._calculate_cost()

        if result.error:
            await _set_session_state(
                session,
                status="error",
                pending_question=None,
                final_result=result.final_result,
                error=result.error,
            )
            await session.publish("error", {"message": result.error})
            return

        await _set_session_state(
            session,
            status="completed",
            pending_question=None,
            final_result=result.final_result,
            error=None,
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
                },
                "trace": {
                    "step_path": trace.step_path,
                    "referenced_documents": trace.sorted_documents(),
                    "cited_sources": cited_sources,
                },
            },
        )
    except Exception as exc:
        await _set_session_state(
            session,
            status="error",
            pending_question=None,
            error=str(exc),
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


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI HTML file."""
    html_path = Path(__file__).parent / "ui.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)


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
    """Search an indexed corpus and return ranked hits."""
    try:
        folder_path = Path(request.corpus_folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse(
                {"error": f"Invalid folder: {request.corpus_folder}"},
                status_code=400,
            )

        resolved_db_path = resolve_db_path(request.db_path)
        storage = PostgresStorage(resolved_db_path, read_only=True, initialize=False)
        corpus_id = storage.get_corpus_id(str(folder_path))
        if corpus_id is None:
            storage.close()
            return JSONResponse(
                {"error": "No index found for this folder."},
                status_code=404,
            )

        embedding_provider: EmbeddingProvider | None = None
        if storage.has_embeddings(corpus_id=corpus_id):
            try:
                embedding_provider = EmbeddingProvider()
            except ValueError:
                pass

        engine = IndexedQueryEngine(storage, embedding_provider=embedding_provider)
        hits = engine.search(
            corpus_id=corpus_id,
            query=request.query,
            filters=request.filters,
            limit=request.limit,
        )
        storage.close()

        return {
            "corpus_folder": str(folder_path),
            "query": request.query,
            "hits": [
                {
                    "doc_id": hit.doc_id,
                    "relative_path": hit.relative_path,
                    "absolute_path": hit.absolute_path,
                    "position": hit.position,
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


@app.post("/api/explore/sessions")
async def create_explore_session(request: ExploreSessionCreateRequest):
    """Create and launch a new exploration session."""
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="No task provided")

    folder_path = _resolve_folder_path(request.folder)
    use_index = bool(request.use_index)
    db_path = request.db_path
    if use_index:
        db_path = _ensure_index_exists(folder_path, db_path)

    session = await _session_manager.create_session(
        task=task,
        folder=str(folder_path),
        use_index=use_index,
        db_path=db_path,
        enable_semantic=bool(request.enable_semantic and use_index),
        enable_metadata=bool(request.enable_metadata and use_index),
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


if __name__ == "__main__":
    run_server()
