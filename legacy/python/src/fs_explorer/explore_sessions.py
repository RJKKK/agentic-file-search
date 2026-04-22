"""
In-memory session management for SSE-based exploration workflows.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    """Serialize timestamps in a stable UTC format."""
    return value.astimezone(timezone.utc).isoformat()


@dataclass(slots=True)
class ExploreStreamEvent:
    """A single event emitted to SSE subscribers."""

    sequence: int
    type: str
    timestamp: datetime
    data: dict[str, Any]

    def as_payload(self, session_id: str) -> dict[str, Any]:
        """Convert the event to the wire payload."""
        return {
            "session_id": session_id,
            "type": self.type,
            "sequence": self.sequence,
            "timestamp": isoformat_utc(self.timestamp),
            "data": self.data,
        }


@dataclass(slots=True)
class ExploreSession:
    """Runtime state for one exploration run."""

    session_id: str
    task: str
    document_ids: list[str]
    collection_id: str | None
    db_path: str | None
    enable_semantic: bool
    enable_metadata: bool
    status: str = "created"
    pending_question: str | None = None
    final_result: str | None = None
    error: str | None = None
    last_focus_anchor: dict[str, Any] | None = None
    context_budget_stats: dict[str, Any] = field(default_factory=dict)
    context_state_snapshot: dict[str, Any] = field(default_factory=dict)
    lazy_indexing_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "triggered": False,
            "indexed_documents": 0,
            "chunks_written": 0,
            "embeddings_written": 0,
        }
    )
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    history: list[ExploreStreamEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue[ExploreStreamEvent | None]] = field(default_factory=set)
    handler: Any | None = None
    workflow_task: asyncio.Task[None] | None = None
    _next_sequence: int = 1
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the session state."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "awaiting_human": self.status == "awaiting_human",
            "pending_question": self.pending_question,
            "final_result": self.final_result,
            "error": self.error,
            "last_focus_anchor": self.last_focus_anchor,
            "context_budget_stats": self.context_budget_stats,
            "context_state_snapshot": self.context_state_snapshot,
            "lazy_indexing_stats": self.lazy_indexing_stats,
            "task": self.task,
            "document_ids": list(self.document_ids),
            "collection_id": self.collection_id,
            "db_path": self.db_path,
            "enable_semantic": self.enable_semantic,
            "enable_metadata": self.enable_metadata,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }

    def is_terminal(self) -> bool:
        """Return whether the session is finished."""
        return self.status in {"completed", "error", "closed"}

    async def publish(self, event_type: str, data: dict[str, Any]) -> ExploreStreamEvent:
        """Record a new event and fan it out to current subscribers."""
        async with self._lock:
            event = ExploreStreamEvent(
                sequence=self._next_sequence,
                type=event_type,
                timestamp=utc_now(),
                data=data,
            )
            self._next_sequence += 1
            self.updated_at = event.timestamp
            self.history.append(event)
            subscribers = list(self.subscribers)

        for subscriber in subscribers:
            subscriber.put_nowait(event)
        return event


class ExploreSessionManager:
    """Registry for in-flight and recently completed exploration sessions."""

    def __init__(self, *, retention_minutes: int = 15) -> None:
        self._sessions: dict[str, ExploreSession] = {}
        self._retention = timedelta(minutes=retention_minutes)
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        *,
        task: str,
        document_ids: list[str],
        collection_id: str | None,
        db_path: str | None,
        enable_semantic: bool,
        enable_metadata: bool,
    ) -> ExploreSession:
        """Create and register a new session."""
        await self.cleanup()
        session = ExploreSession(
            session_id=uuid4().hex,
            task=task,
            document_ids=list(document_ids),
            collection_id=collection_id,
            db_path=db_path,
            enable_semantic=enable_semantic,
            enable_metadata=enable_metadata,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> ExploreSession | None:
        """Fetch a session by id."""
        await self.cleanup()
        async with self._lock:
            return self._sessions.get(session_id)

    async def subscribe(
        self,
        session_id: str,
    ) -> tuple[
        ExploreSession | None,
        asyncio.Queue[ExploreStreamEvent | None] | None,
        list[ExploreStreamEvent] | None,
    ]:
        """Register a live subscriber queue for a session."""
        session = await self.get_session(session_id)
        if session is None:
            return None, None, None

        queue: asyncio.Queue[ExploreStreamEvent | None] = asyncio.Queue()
        async with session._lock:
            history = list(session.history)
            session.subscribers.add(queue)
        return session, queue, history

    async def unsubscribe(
        self,
        session: ExploreSession,
        queue: asyncio.Queue[ExploreStreamEvent | None],
    ) -> None:
        """Remove a live subscriber queue."""
        async with session._lock:
            session.subscribers.discard(queue)

    async def cleanup(self) -> None:
        """Drop expired terminal sessions."""
        cutoff = utc_now() - self._retention
        async with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_terminal() and session.updated_at < cutoff
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)


def encode_sse_event(session_id: str, event: ExploreStreamEvent) -> str:
    """Serialize an event using the SSE wire format."""
    payload = json.dumps(event.as_payload(session_id), ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n"
