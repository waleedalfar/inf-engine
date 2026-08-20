"""Tests for server.py's HTTP layer, in isolation from the real model/engine.

Patches server._manager with a fake PagedSessionManager-shaped object so these
tests exercise only the FastAPI routing / streaming / thread-hop contract —
the actual engine behavior is covered by tests/test_paged_session.py.

Run:
    wsl bash -c "cd /home/waleed/mlproj && .venv/bin/pytest tests/test_server.py -v"
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

import server


class _FakeManager:
    """Mimics PagedSessionManager.submit()'s contract: call emit(...) from a
    background thread, exactly like the real engine thread would."""

    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self._sessions: dict[int, object] = {}
        self.submitted: list[tuple[int, list[dict]]] = []

    def submit(self, session_id, messages, emit) -> None:
        self.submitted.append((session_id, messages))

        def run() -> None:
            for event in self._events:
                emit(event)

        threading.Thread(target=run, daemon=True).start()


def test_chat_streams_ndjson_events():
    fake = _FakeManager([
        {"type": "token", "text": "hi"},
        {"type": "done", "final_text": "hi", "turns": 1},
    ])
    server._manager = fake
    client = TestClient(server.app)

    with client.stream(
        "POST", "/v1/chat", json={"messages": [{"role": "user", "content": "hello"}]}
    ) as r:
        lines = [line for line in r.iter_lines() if line]

    assert len(lines) == 3
    assert '"type": "session"' in lines[0] or '"type":"session"' in lines[0]
    assert "token" in lines[1]
    assert "done" in lines[2]
    assert fake.submitted[0][1] == [{"role": "user", "content": "hello"}]


def test_chat_uses_provided_session_id():
    fake = _FakeManager([{"type": "done", "final_text": "ok", "turns": 1}])
    server._manager = fake
    client = TestClient(server.app)

    with client.stream(
        "POST", "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "session_id": 42},
    ) as r:
        lines = [line for line in r.iter_lines() if line]

    assert '"session_id": 42' in lines[0]
    assert fake.submitted[0][0] == 42


def test_chat_stream_stops_on_error_event():
    fake = _FakeManager([
        {"type": "error", "reason": "context_overflow", "detail": "too long"},
        {"type": "token", "text": "should not appear"},
    ])
    server._manager = fake
    client = TestClient(server.app)

    with client.stream(
        "POST", "/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    ) as r:
        lines = [line for line in r.iter_lines() if line]

    # session + error only — stream must stop at the terminal event.
    assert len(lines) == 2
    assert "error" in lines[1]


def test_chat_503_when_model_not_loaded():
    server._manager = None
    client = TestClient(server.app)
    r = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_healthz_reports_loading_then_ok():
    server._manager = None
    client = TestClient(server.app)
    assert client.get("/healthz").json() == {"status": "loading"}

    fake = _FakeManager([])
    fake._sessions = {1: object(), 2: object()}
    server._manager = fake
    assert client.get("/healthz").json() == {"status": "ok", "active_sessions": 2}
