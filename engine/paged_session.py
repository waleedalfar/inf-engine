"""Multi-session tool-calling driver on top of LlamaPagedEngine.

Bridges the single-sequence agent loop (engine/agent.py: generate → parse
<tool_call> → execute → inject → repeat) onto the continuous-batching
engine, so many independent conversations can share one loaded model and
have their decode steps batched together.

Ownership model: exactly one thread calls ``PagedSessionManager.run_forever``
and it is the *only* code allowed to touch ``self.engine`` (LlamaPagedEngine
is not thread-safe — see engine/llama_paged_engine.py). Any other thread
must go through ``submit()``, which only pushes onto a thread-safe queue.

Each session's turn is a fresh ``LlamaRequest``: the full conversation
history is reformatted and re-prefilled every turn, exactly like
``AgentLoop.run()`` does for the single-session CLI. This is O(n^2) in
token cost per session but sidesteps an entire class of incremental
position-tracking bugs, matching the deliberate correctness-over-efficiency
choice already made in agent.py.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.agent import Tool, dispatch_tool_calls
from engine.chat import format_messages
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.tool_parser import has_tool_call, strip_thinking


@dataclass
class Session:
    """One conversation being driven through the paged engine."""

    session_id: int
    messages: list[dict]
    emit: Callable[[dict], None]
    """Called (from the engine thread) with event dicts: {"type": "token"|"tool_exec"|"done"|"error", ...}."""
    status: str = "queued"  # queued | active | done | error
    turn: int = 0
    tool_calls_made: list = field(default_factory=list)
    req: LlamaRequest | None = None
    _emitted: int = 0  # len(req.generated) already streamed to `emit`


def _make_stop_check(tokenizer: Any) -> Callable[[list[int]], bool]:
    """Stop a decode as soon as the generated text contains a closed <tool_call> block.

    Re-decodes the full generated-token tail each call — cheap relative to a
    forward pass, and avoids tracking partial-tag state across steps.
    """

    def stop_check(generated: list[int]) -> bool:
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return has_tool_call(text)

    return stop_check


class PagedSessionManager:
    """Drives an LlamaPagedEngine across many concurrent tool-using sessions.

    Args:
        engine:          LlamaPagedEngine sharing one loaded model.
        tokenizer:       QwenTokenizer (encode/decode).
        tools:           Tools exposed to every session (shared, read-only).
        max_turns:       Max tool-call iterations per session before giving up.
        max_new_tokens:  Max tokens generated per turn.
        enable_thinking:  Passed through to format_messages.
        max_ctx:         If set, truncate history like AgentLoop does (not
                         yet implemented here — sessions are expected to stay
                         within context; see follow-up work).
    """

    def __init__(
        self,
        engine: LlamaPagedEngine,
        tokenizer: Any,
        tools: list[Tool] | None = None,
        max_turns: int = 8,
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
        max_ctx: int | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.tools = {t.name: t for t in (tools or [])}
        self.tool_list = list(self.tools.values())
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.max_ctx = max_ctx

        self._incoming: queue.SimpleQueue = queue.SimpleQueue()
        self._sessions: dict[int, Session] = {}       # session_id -> Session
        self._req_to_session: dict[int, int] = {}      # LlamaRequest.req_id -> session_id
        self._next_req_id = 0

    # ------------------------------------------------------------------
    # Thread-safe entry point — callable from any thread
    # ------------------------------------------------------------------

    def submit(self, session_id: int, messages: list[dict], emit: Callable[[dict], None]) -> None:
        """Enqueue a new user turn for *session_id*. Never blocks, never touches the engine."""
        self._incoming.put((session_id, messages, emit))

    # ------------------------------------------------------------------
    # Engine-thread loop — the ONLY code allowed to call self.engine.*
    # ------------------------------------------------------------------

    def run_forever(self, stop_event: threading.Event, idle_sleep: float = 0.001) -> None:
        while not stop_event.is_set():
            self.run_once()
            if not self.engine.has_work and self._incoming.empty():
                time.sleep(idle_sleep)

    def run_once(self) -> None:
        """One scheduling tick: drain new submissions, step the engine, stream
        deltas, and handle any turns that just stopped. Exposed separately from
        ``run_forever`` so tests can drive it synchronously without a thread."""
        self._drain_incoming()
        if self.engine.has_work:
            now = time.monotonic()
            completions = self.engine.step(now)
            self._stream_new_tokens()
            self._handle_turn_ends(completions)

    def _drain_incoming(self) -> None:
        while True:
            try:
                session_id, messages, emit = self._incoming.get_nowait()
            except queue.Empty:
                return
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id, messages=list(messages), emit=emit)
                self._sessions[session_id] = session
            else:
                session.messages = list(messages)
                session.emit = emit
                session.status = "queued"
            self._start_turn(session)

    def _start_turn(self, session: Session) -> None:
        prompt = format_messages(session.messages, self.tool_list or None, self.enable_thinking)
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if self.max_ctx is not None:
            hard_limit = self.max_ctx - self.max_new_tokens
            if len(ids) > hard_limit:
                session.emit(
                    {
                        "type": "error",
                        "reason": "context_overflow",
                        "detail": f"prompt {len(ids)} tokens exceeds {hard_limit}",
                    }
                )
                session.status = "error"
                self._sessions.pop(session.session_id, None)
                return

        req_id = self._next_req_id
        self._next_req_id += 1

        req = LlamaRequest(
            req_id=req_id,
            prompt_ids=ids,
            max_new_tokens=self.max_new_tokens,
            stop_check=_make_stop_check(self.tokenizer),
        )
        session.req = req
        session.status = "active"
        session._emitted = 0
        self._req_to_session[req_id] = session.session_id

        if not self.engine.submit(req):
            session.emit({"type": "error", "reason": "queue_full"})
            session.status = "error"
            self._sessions.pop(session.session_id, None)
            del self._req_to_session[req_id]

    def _stream_new_tokens(self) -> None:
        for session in self._sessions.values():
            req = session.req
            if req is None or session.status != "active":
                continue
            if len(req.generated) > session._emitted:
                new_ids = req.generated[session._emitted :]
                session._emitted = len(req.generated)
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                if text:
                    session.emit({"type": "token", "text": text})

    def _handle_turn_ends(self, completions: list[LlamaRequest]) -> None:
        for req in completions:
            session_id = self._req_to_session.pop(req.req_id, None)
            if session_id is None:
                continue
            session = self._sessions.get(session_id)
            if session is None:
                continue
            self._on_turn_end(session, req)

    def _on_turn_end(self, session: Session, req: LlamaRequest) -> None:
        # Drop the trailing EOS id before decoding, same as AgentLoop._decode —
        # otherwise it renders into raw_text and leaks into the visible response.
        gen_ids = req.generated
        if gen_ids and gen_ids[-1] == self.engine.eos:
            gen_ids = gen_ids[:-1]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        visible_text = strip_thinking(raw_text)

        if not has_tool_call(visible_text):
            session.messages = session.messages + [{"role": "assistant", "content": raw_text}]
            session.status = "done"
            session.emit({"type": "done", "final_text": visible_text, "turns": session.turn + 1})
            self._sessions.pop(session.session_id, None)
            return

        session.messages, calls = dispatch_tool_calls(
            visible_text, raw_text, session.messages, self.tools
        )
        session.tool_calls_made.extend(calls)
        for call in calls:
            session.emit({"type": "tool_exec", "name": call.name, "arguments": call.arguments})

        session.turn += 1
        if session.turn >= self.max_turns:
            last = session.messages[-1].get("content", "") if session.messages else ""
            session.status = "done"
            session.emit(
                {"type": "done", "final_text": strip_thinking(last), "turns": session.turn}
            )
            self._sessions.pop(session.session_id, None)
            return

        self._start_turn(session)
