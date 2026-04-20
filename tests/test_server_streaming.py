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
from fs_explorer.explore_sessions import ExploreSessionManager, utc_now
from fs_explorer.server import app
from fs_explorer.workflow import AskHumanEvent, GoDeeperEvent, ToolCallEvent


class FakeTokenUsage:
    """Minimal token usage surface for server completion payloads."""

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
    """Minimal fake agent."""

    def __init__(self) -> None:
        self.token_usage = FakeTokenUsage()


class FakeContext:
    """Workflow context that accepts human replies."""

    def __init__(self) -> None:
        self.sent_events: list[object] = []
        self.reply_event = threading.Event()
        self.reply_value: str | None = None

    def send_event(self, event) -> None:
        self.sent_events.append(event)
        self.reply_value = event.response
        self.reply_event.set()


class FakeHandler:
    """Awaitable workflow handler with a programmable event stream."""

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


def _collect_sse_events(client: TestClient, session_id: str, *, stop_after: str) -> list[tuple[str, dict]]:
    """Read SSE events until a specific event type is received."""
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
    """Poll the session state until it reaches the requested status."""
    deadline = time.time() + timeout_seconds
    last_payload: dict | None = None
    while time.time() < deadline:
        response = client.get(f"/api/explore/sessions/{session_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] == expected_status:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(
        f"Session {session_id} did not reach status={expected_status!r}; "
        f"last payload={last_payload}"
    )


@pytest.fixture(autouse=True)
def fresh_session_manager(monkeypatch) -> None:
    """Give each test an isolated in-memory session registry."""
    monkeypatch.setattr(server_module, "_session_manager", ExploreSessionManager())


def test_create_session_and_stream_complete(monkeypatch, tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()

    handler = FakeHandler(
        events_before_reply=[
            ToolCallEvent(
                tool_name="scan_folder",
                tool_input={"directory": str(folder)},
                reason="Start broad",
            ),
            GoDeeperEvent(directory=str(folder), reason="Inspect folder"),
        ],
        final_result="Finished analysis",
    )

    monkeypatch.setattr(server_module.workflow, "run", lambda start_event: handler)
    monkeypatch.setattr(server_module, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(server_module, "reset_agent", lambda: None)

    client = TestClient(app)
    create_response = client.post(
        "/api/explore/sessions",
        json={"task": "Summarize", "folder": str(folder)},
    )

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["status"] == "created"
    session_id = payload["session_id"]

    status_response = client.get(f"/api/explore/sessions/{session_id}")
    assert status_response.status_code == 200
    assert status_response.json()["session_id"] == session_id

    events = _collect_sse_events(client, session_id, stop_after="complete")
    assert [event_type for event_type, _ in events] == [
        "start",
        "tool_call",
        "go_deeper",
        "complete",
    ]

    complete_payload = events[-1][1]
    assert complete_payload["session_id"] == session_id
    assert complete_payload["type"] == "complete"
    assert complete_payload["sequence"] == 4
    assert "timestamp" in complete_payload
    assert complete_payload["data"]["final_result"] == "Finished analysis"


def test_ask_human_session_can_resume_via_reply(monkeypatch, tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()

    session = asyncio.run(
        server_module._session_manager.create_session(
            task="Need more detail",
            folder=str(folder),
            use_index=False,
            db_path=None,
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
                        "tool_input": {"file_path": str(folder / "note.txt")},
                        "reason": "Continue after answer",
                    },
                )
                await session.publish(
                    "complete",
                    {
                        "final_result": "Resumed successfully",
                        "error": None,
                        "stats": {
                            "steps": 2,
                            "api_calls": 1,
                            "documents_scanned": 0,
                            "documents_parsed": 0,
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                            "tool_result_chars": 42,
                            "estimated_cost": 0.0015,
                        },
                        "trace": {
                            "step_path": [],
                            "referenced_documents": [],
                            "cited_sources": [],
                        },
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
                "folder": str(folder),
                "use_index": False,
            },
        )
    )
    asyncio.run(
        session.publish(
            "ask_human",
            {
                "step": 1,
                "question": "Need the target file?",
                "reason": "Missing context",
            },
        )
    )

    client = TestClient(app)
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

    replayed_events = _collect_sse_events(client, session.session_id, stop_after="complete")
    assert [event_type for event_type, _ in replayed_events] == [
        "start",
        "ask_human",
        "tool_call",
        "complete",
    ]
    assert replayed_events[1][1]["data"]["question"] == "Need the target file?"


def test_reply_requires_existing_waiting_session(monkeypatch, tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()

    handler = FakeHandler(events_before_reply=[], final_result="done")
    monkeypatch.setattr(server_module.workflow, "run", lambda start_event: handler)
    monkeypatch.setattr(server_module, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(server_module, "reset_agent", lambda: None)

    client = TestClient(app)

    missing_response = client.post(
        "/api/explore/sessions/missing/reply",
        json={"response": "hi"},
    )
    assert missing_response.status_code == 404

    create_response = client.post(
        "/api/explore/sessions",
        json={"task": "Done fast", "folder": str(folder)},
    )
    session_id = create_response.json()["session_id"]
    _collect_sse_events(client, session_id, stop_after="complete")

    completed_reply = client.post(
        f"/api/explore/sessions/{session_id}/reply",
        json={"response": "too late"},
    )
    assert completed_reply.status_code == 409


def test_sse_stream_includes_runtime_cache_and_image_events(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()

    session = asyncio.run(
        server_module._session_manager.create_session(
            task="Summarize charts",
            folder=str(folder),
            use_index=True,
            db_path=str(tmp_path / "index.duckdb"),
            enable_semantic=True,
            enable_metadata=False,
        )
    )
    asyncio.run(
        session.publish(
            "start",
            {"task": "Summarize charts", "folder": str(folder), "use_index": True},
        )
    )
    asyncio.run(
        session.publish(
            "cache_hit",
            {
                "cache_kind": "image_semantics",
                "doc_id": "doc-1",
                "absolute_path": str(folder / "report.pdf"),
                "page_no": 1,
                "cached_images": 1,
            },
        )
    )
    asyncio.run(
        session.publish(
            "image_enhance_started",
            {
                "doc_id": "doc-1",
                "absolute_path": str(folder / "report.pdf"),
                "page_no": 2,
                "image_hashes": ["img-2"],
                "pending_images": 1,
            },
        )
    )
    asyncio.run(
        session.publish(
            "image_enhance_done",
            {
                "doc_id": "doc-1",
                "absolute_path": str(folder / "report.pdf"),
                "page_no": 2,
                "image_hashes": ["img-2"],
                "enhanced_images": 1,
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
                "stats": {},
                "trace": {},
            },
        )
    )

    client = TestClient(app)
    events = _collect_sse_events(client, session.session_id, stop_after="complete")

    assert [event_type for event_type, _ in events] == [
        "start",
        "cache_hit",
        "image_enhance_started",
        "image_enhance_done",
        "complete",
    ]
    assert events[1][1]["data"]["cache_kind"] == "image_semantics"
    assert events[2][1]["data"]["pending_images"] == 1
    assert events[3][1]["data"]["enhanced_images"] == 1


def test_frontend_uses_eventsource_not_websocket() -> None:
    ui_path = Path("src/fs_explorer/ui.html")
    content = ui_path.read_text(encoding="utf-8")

    assert "EventSource(" in content
    assert "cache_hit" in content
    assert "image_enhance_started" in content
    assert "image_enhance_done" in content
    assert "WebSocket" not in content
    assert "ws://" not in content
    assert "wss://" not in content
