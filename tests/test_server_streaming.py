"""Tests for SSE-based exploration streaming APIs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import fs_explorer.server as server_module
from fs_explorer.blob_store import LocalBlobStore
from fs_explorer.explore_sessions import ExploreSessionManager, utc_now
from fs_explorer.server import app
from fs_explorer.workflow import GoDeeperEvent, ToolCallEvent


class FakeTokenUsage:
    api_calls = 1
    documents_scanned = 0
    documents_parsed = 0
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15
    tool_result_chars = 42

    def _calculate_cost(self) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0015


class FakeAgent:
    def __init__(self) -> None:
        self.token_usage = FakeTokenUsage()


class FakeContext:
    def __init__(self) -> None:
        self.sent_events: list[object] = []
        self.reply_event = threading.Event()

    def send_event(self, event) -> None:
        self.sent_events.append(event)
        self.reply_event.set()


class FakeHandler:
    def __init__(
        self,
        *,
        events_before_reply: list[object],
        events_after_reply: list[object] | None = None,
        final_result: str = "Done",
        error: str | None = None,
        wait_for_reply: bool = False,
    ) -> None:
        self.ctx = FakeContext()
        self._events_before_reply = list(events_before_reply)
        self._events_after_reply = list(events_after_reply or [])
        self._final_result = final_result
        self._error = error
        self._wait_for_reply = wait_for_reply

    async def stream_events(self):
        for event in self._events_before_reply:
            yield event
        if self._wait_for_reply:
            while not self.ctx.reply_event.is_set():
                await asyncio.sleep(0.01)
            for event in self._events_after_reply:
                yield event

    async def _result(self):
        if self._wait_for_reply:
            while not self.ctx.reply_event.is_set():
                await asyncio.sleep(0.01)
        return SimpleNamespace(final_result=self._final_result, error=self._error)

    def __await__(self):
        return self._result().__await__()


def _collect_sse_events(
    client: TestClient,
    session_id: str,
    *,
    stop_after: str,
) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_event: str | None = None

    with client.stream("GET", f"/api/explore/sessions/{session_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                assert current_event is not None
                events.append((current_event, payload))
                if current_event == stop_after:
                    break
    return events


def _wait_for_status(
    client: TestClient,
    session_id: str,
    expected_status: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    last_payload: dict | None = None
    while time.time() < deadline:
        response = client.get(f"/api/explore/sessions/{session_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] == expected_status:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(f"Session did not reach {expected_status!r}: {last_payload}")


@pytest.fixture(autouse=True)
def fresh_session_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "_session_manager", ExploreSessionManager())
    monkeypatch.setattr(
        server_module,
        "_blob_store",
        LocalBlobStore(tmp_path / "object_store"),
    )


def _upload(client: TestClient, *, db_path: str, name: str, content: str) -> dict:
    response = client.post(
        "/api/documents",
        data={"db_path": db_path},
        files={"file": (name, content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 201
    return response.json()["document"]


def test_create_session_and_stream_complete(monkeypatch, tmp_path: Path) -> None:
    db_path = str(tmp_path / "stream.duckdb")
    client = TestClient(app)
    document = _upload(
        client,
        db_path=db_path,
        name="board.md",
        content="# Board\n\nAll directors listed here.\n",
    )

    handler = FakeHandler(
        events_before_reply=[
            ToolCallEvent(
                tool_name="list_indexed_documents",
                tool_input={},
                reason="Start from the selected document scope",
            ),
            GoDeeperEvent(directory="selected-documents", reason="Inspect details"),
        ],
        final_result="Finished analysis",
    )

    monkeypatch.setattr(server_module.workflow, "run", lambda start_event: handler)
    monkeypatch.setattr(server_module, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(server_module, "reset_agent", lambda: None)

    create_response = client.post(
        "/api/explore/sessions",
        json={
            "task": "Summarize",
            "document_ids": [document["id"]],
            "db_path": db_path,
        },
    )
    assert create_response.status_code == 200
    session_id = create_response.json()["session_id"]

    events = _collect_sse_events(client, session_id, stop_after="complete")
    assert [event_type for event_type, _ in events] == [
        "start",
        "context_scope_updated",
        "tool_call",
        "go_deeper",
        "complete",
    ]
    assert events[0][1]["data"]["document_ids"] == [document["id"]]
    assert events[1][1]["data"]["context_scope"]["active_ranges"] == []
    assert events[-1][1]["data"]["final_result"] == "Finished analysis"
    assert "context_scope" in events[-1][1]["data"]["trace"]


def test_ask_human_session_can_resume_via_reply(tmp_path: Path) -> None:
    db_path = str(tmp_path / "reply.duckdb")
    client = TestClient(app)
    document = _upload(
        client,
        db_path=db_path,
        name="note.md",
        content="# Note\n\nUse note.txt.\n",
    )

    session = asyncio.run(
        server_module._session_manager.create_session(
            task="Need more detail",
            document_ids=[document["id"]],
            collection_id=None,
            db_path=db_path,
            enable_semantic=False,
            enable_metadata=False,
        )
    )

    class SyntheticContext:
        def __init__(self) -> None:
            self.sent_events: list[object] = []

        def send_event(self, event) -> None:
            self.sent_events.append(event)
            session.status = "completed"
            session.pending_question = None
            session.final_result = "Resumed successfully"
            session.updated_at = utc_now()
            loop = asyncio.get_running_loop()

            async def finalize() -> None:
                await session.publish(
                    "tool_call",
                    {
                        "step": 2,
                        "tool_name": "read",
                        "tool_input": {"file_path": "note.txt"},
                        "reason": "Continue after answer",
                    },
                )
                await session.publish(
                    "complete",
                    {
                        "final_result": "Resumed successfully",
                        "error": None,
                        "stats": {"steps": 2},
                        "trace": {"step_path": [], "referenced_documents": [], "cited_sources": []},
                    },
                )

            loop.create_task(finalize())

    handler = SimpleNamespace(ctx=SyntheticContext())
    session.handler = handler
    session.status = "awaiting_human"
    session.pending_question = "Need the target file?"
    asyncio.run(
        session.publish(
            "start",
            {
                "task": "Need more detail",
                "document_ids": [document["id"]],
                "collection_id": None,
            },
        )
    )
    asyncio.run(
        session.publish(
            "ask_human",
            {"step": 1, "question": "Need the target file?", "reason": "Missing context"},
        )
    )

    awaiting_payload = _wait_for_status(client, session.session_id, "awaiting_human")
    assert awaiting_payload["pending_question"] == "Need the target file?"

    reply_response = client.post(
        f"/api/explore/sessions/{session.session_id}/reply",
        json={"response": "Use note.txt"},
    )
    assert reply_response.status_code == 200
    assert handler.ctx.sent_events[0].response == "Use note.txt"

    completed_payload = _wait_for_status(client, session.session_id, "completed")
    assert completed_payload["final_result"] == "Resumed successfully"


def test_sse_stream_includes_runtime_cache_and_image_events(tmp_path: Path) -> None:
    db_path = str(tmp_path / "runtime.duckdb")
    client = TestClient(app)
    document = _upload(
        client,
        db_path=db_path,
        name="report.md",
        content="# Report\n\nCharts and image captions.\n",
    )

    session = asyncio.run(
        server_module._session_manager.create_session(
            task="Summarize charts",
            document_ids=[document["id"]],
            collection_id=None,
            db_path=db_path,
            enable_semantic=True,
            enable_metadata=False,
        )
    )
    asyncio.run(
        session.publish(
            "start",
            {"task": "Summarize charts", "document_ids": [document["id"]], "collection_id": None},
        )
    )
    asyncio.run(
        session.publish(
            "lazy_indexing_started",
            {
                "corpus_id": "corpus-1",
                "document_ids": [document["id"]],
                "triggered": True,
                "pending_documents": 1,
            },
        )
    )
    asyncio.run(
        session.publish(
            "lazy_indexing_done",
            {
                "corpus_id": "corpus-1",
                "document_ids": [document["id"]],
                "triggered": True,
                "indexed_documents": 1,
                "chunks_written": 8,
                "embeddings_written": 0,
            },
        )
    )
    asyncio.run(
        session.publish(
            "cache_hit",
            {
                "cache_kind": "image_semantics",
                "doc_id": document["id"],
                "absolute_path": "report.md",
                "page_no": 1,
                "cached_images": 1,
            },
        )
    )
    session.status = "completed"
    session.final_result = "Done"
    asyncio.run(
        session.publish(
            "complete",
            {
                "final_result": "Done",
                "error": None,
                "stats": {
                    "lazy_indexing": {
                        "triggered": True,
                        "indexed_documents": 1,
                        "chunks_written": 8,
                        "embeddings_written": 0,
                    }
                },
                "trace": {},
            },
        )
    )

    events = _collect_sse_events(client, session.session_id, stop_after="complete")
    assert [event_type for event_type, _ in events] == [
        "start",
        "lazy_indexing_started",
        "lazy_indexing_done",
        "cache_hit",
        "complete",
    ]
    assert events[1][1]["data"]["pending_documents"] == 1
    assert events[2][1]["data"]["chunks_written"] == 8
    assert events[3][1]["data"]["cache_kind"] == "image_semantics"


def test_frontend_uses_eventsource_not_websocket() -> None:
    ui_path = Path("src/fs_explorer/ui.html")
    content = ui_path.read_text(encoding="utf-8")

    assert "EventSource(" in content
    assert "cache_hit" in content
    assert "WebSocket" not in content
