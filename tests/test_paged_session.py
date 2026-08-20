"""Tests for engine/paged_session.py — multi-session tool-calling driver on
top of LlamaPagedEngine.

Uses a synthetic mini-LLaMA (same pattern as test_paged_cache.py) so no
weights file is required, and a character-level mock tokenizer (same
pattern as test_agent.py) so token ids are deterministic and easy to reason
about in assertions.

Run:
    wsl bash -c "cd /home/waleed/mlproj && .venv/bin/pytest tests/test_paged_session.py -v"
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import torch

from engine.agent import Tool
from engine.config import LlamaConfig
from engine.llama_paged_engine import LlamaPagedEngine
from engine.paged_session import PagedSessionManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures (self-contained, mirroring test_paged_cache.py / test_agent.py)
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Character-level tokenizer. Token 0 = <|im_end|>, everything else assigned
    a unique id on first use — deterministic and easy to reason about."""

    IM_END_ID = 0
    _SPECIAL = "<|im_end|>"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {self._SPECIAL: 0}
        self._rev: dict[int, str] = {0: self._SPECIAL}
        self._next = 1

    def _get_id(self, c: str) -> int:
        if c not in self._vocab:
            self._vocab[c] = self._next
            self._rev[self._next] = c
            self._next += 1
        return self._vocab[c]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        while text:
            if text.startswith(self._SPECIAL):
                ids.append(0)
                text = text[len(self._SPECIAL):]
            else:
                ids.append(self._get_id(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        for i in ids:
            if i == 0:
                if not skip_special_tokens:
                    parts.append(self._SPECIAL)
            else:
                parts.append(self._rev.get(i, "?"))
        return "".join(parts)

    @property
    def im_end_id(self) -> int:
        return self.IM_END_ID


def _mini_config() -> LlamaConfig:
    # n_ctx generous: formatted ChatML prompts (esp. with a tool schema injected
    # into the system message) run to 600+ characters under the char-level mock
    # tokenizer, well beyond a "real" tokenizer's token count for the same text.
    return LlamaConfig(
        name="test-mini",
        vocab_size=256,
        n_ctx=1024,
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
    )


@pytest.fixture()
def tok() -> _MockTokenizer:
    return _MockTokenizer()


def _script_tokens(tokenizer: _MockTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# A simpler scripted model: keyed by *submission order* (0 = first prompt
# admitted, 1 = second, ...), which is stable and easy for tests to reason
# about since LlamaPagedEngine assigns seq_id/req_id in submission order.
# ---------------------------------------------------------------------------


class _OrderScriptedModel:
    def __init__(self, config: LlamaConfig, scripts: list[list[int]]):
        from engine.llama_model import LlamaModel
        from engine.llama_weights import LlamaWeights

        d = config.d_model
        h = config.n_kv_heads * config.head_dim
        f = config.intermediate_size
        torch.manual_seed(7)

        def rand(*shape):
            return torch.randn(*shape, dtype=DTYPE, device=DEVICE)

        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": rand(config.vocab_size, d),
            "model.norm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
        }
        for i in range(config.n_layer):
            p = f"model.layers.{i}."
            tensors |= {
                p + "input_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
                p + "post_attention_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
                p + "self_attn.q_proj.weight": rand(d, d),
                p + "self_attn.k_proj.weight": rand(h, d),
                p + "self_attn.v_proj.weight": rand(h, d),
                p + "self_attn.o_proj.weight": rand(d, d),
                p + "mlp.gate_proj.weight": rand(f, d),
                p + "mlp.up_proj.weight": rand(f, d),
                p + "mlp.down_proj.weight": rand(d, f),
            }
        weights = LlamaWeights(tensors, config)
        self._real = LlamaModel(weights, config)
        self.w = self._real.w
        self.config = config
        self._scripts = scripts       # list indexed by submission order
        self._ptr = [0] * len(scripts)
        self._admitted = 0            # how many prompts have been prefilled so far
        # Real LlamaPagedEngine seq_id (assigned FCFS, mirrored here with a local
        # counter since prefills happen in the same order) -> script index. Looking
        # this up via cache.seq_lens (rather than a manually-tracked row list) stays
        # correct across evictions/turns — a stale, never-pruned row list previously
        # caused decode calls to read a *finished* script's leftover tokens (e.g. its
        # trailing EOS) once a session's row was reused by a later turn.
        self._seq_to_idx: dict[int, int] = {}
        self._next_engine_seq_id = 0

    def forward(self, input_ids, cache=None, start_pos=0, position_ids=None, attn_mask=None):
        logits = self._real.forward(
            input_ids, cache=cache, start_pos=start_pos, position_ids=position_ids, attn_mask=attn_mask
        )
        B = input_ids.shape[0]
        out = torch.full_like(logits, float("-inf"))
        # Prefill never passes attn_mask (LlamaPagedEngine._prefill omits it);
        # decode always does. Using B == 1 to detect prefill breaks down when
        # only one session is active, since decode calls are also B == 1 then.
        if attn_mask is None and self._admitted < len(self._scripts):
            # Prefill call for the next not-yet-admitted script (engine admits FCFS).
            idx = self._admitted
            self._admitted += 1
            seq_id = self._next_engine_seq_id
            self._next_engine_seq_id += 1
            self._seq_to_idx[seq_id] = idx
            tok = self._scripts[idx][self._ptr[idx]] if self._ptr[idx] < len(self._scripts[idx]) else 0
            self._ptr[idx] += 1
            out[0, -1, tok] = 0.0
        else:
            # Decode call: cache.seq_lens is maintained by the engine in the same
            # insertion/removal order as its active-sequence batch rows, so it's a
            # reliable row -> real seq_id mapping (unlike a manually-tracked list).
            seq_ids = list(cache.seq_lens.keys())
            for b in range(B):
                idx = self._seq_to_idx[seq_ids[b]]
                p = self._ptr[idx]
                tok = self._scripts[idx][p] if p < len(self._scripts[idx]) else 0
                self._ptr[idx] += 1
                out[b, -1, tok] = 0.0
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_session_no_tools_end_to_end(tok):
    """One session, no tool call: streams tokens then emits a single 'done'."""
    config = _mini_config()
    reply = "hi<|im_end|>"
    script = [_script_tokens(tok, reply)]
    model = _OrderScriptedModel(config, script)

    engine = LlamaPagedEngine(model, n_total_blocks=32, block_size=8, eos_token=tok.im_end_id)
    mgr = PagedSessionManager(engine, tok, tools=[], max_turns=4, max_new_tokens=16)

    events: list[dict] = []
    mgr.submit(1, [{"role": "user", "content": "hello"}], events.append)

    for _ in range(50):
        mgr.run_once()
        if not engine.has_work and mgr._incoming.empty():
            break

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["final_text"] == "hi"
    # Tokens streamed should reconstitute the same text (order preserved).
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "hi"


def test_single_session_with_tool_call(tok):
    """Session emits a tool call, gets a result injected, then finishes on turn 2."""
    config = _mini_config()
    call_text = '<tool_call>\n{"name": "echo", "arguments": {"msg": "hi"}}\n</tool_call><|im_end|>'
    final_text = "done<|im_end|>"
    script = [_script_tokens(tok, call_text), _script_tokens(tok, final_text)]

    # Submission order for turn 2 is still index 0 in a *new* LlamaRequest but the
    # scripted model tracks admission order globally, so the second turn's prefill
    # becomes script index 1 as long as no other session interleaves — true here.
    model = _OrderScriptedModel(config, script)
    engine = LlamaPagedEngine(model, n_total_blocks=128, block_size=8, eos_token=tok.im_end_id)

    calls_seen = []
    tools = [Tool(name="echo", description="echo", parameters={}, fn=lambda msg: calls_seen.append(msg) or msg)]
    # max_new_tokens must exceed the scripted call_text's token count (70 under
    # the char-level mock tokenizer) or stop_check never sees the closing tag.
    mgr = PagedSessionManager(engine, tok, tools=tools, max_turns=4, max_new_tokens=96)

    events: list[dict] = []
    mgr.submit(1, [{"role": "user", "content": "call echo"}], events.append)

    # 70-token call_text + turn-2 final_text needs well over 50 decode steps.
    for _ in range(200):
        mgr.run_once()
        if not engine.has_work and mgr._incoming.empty():
            break

    tool_events = [e for e in events if e["type"] == "tool_exec"]
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "echo"
    assert calls_seen == ["hi"]

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["final_text"] == "done"


def test_two_concurrent_sessions_do_not_interleave_output(tok):
    """Two sessions submitted together get correct, non-mixed streamed text."""
    config = _mini_config()
    reply_a = "AAA<|im_end|>"
    reply_b = "BB<|im_end|>"
    script = [_script_tokens(tok, reply_a), _script_tokens(tok, reply_b)]
    model = _OrderScriptedModel(config, script)

    engine = LlamaPagedEngine(model, n_total_blocks=64, block_size=8, eos_token=tok.im_end_id)
    mgr = PagedSessionManager(engine, tok, tools=[], max_turns=4, max_new_tokens=16)

    events_a: list[dict] = []
    events_b: list[dict] = []
    mgr.submit(1, [{"role": "user", "content": "one"}], events_a.append)
    mgr.submit(2, [{"role": "user", "content": "two"}], events_b.append)

    for _ in range(50):
        mgr.run_once()
        if not engine.has_work and mgr._incoming.empty():
            break

    text_a = "".join(e["text"] for e in events_a if e["type"] == "token")
    text_b = "".join(e["text"] for e in events_b if e["type"] == "token")
    assert text_a == "AAA"
    assert text_b == "BB"


def test_stop_check_none_preserves_eos_behavior():
    """Regression guard: requests without stop_check still stop only on EOS/max_tokens."""
    from engine.llama_paged_engine import LlamaRequest

    req = LlamaRequest(req_id=0, prompt_ids=[1, 2], max_new_tokens=5)
    assert req.stop_check is None


def test_engine_thread_ownership_smoke(tok):
    """run_forever on a background thread drives a session to completion."""
    config = _mini_config()
    script = [_script_tokens(tok, "ok<|im_end|>")]
    model = _OrderScriptedModel(config, script)
    engine = LlamaPagedEngine(model, n_total_blocks=32, block_size=8, eos_token=tok.im_end_id)
    mgr = PagedSessionManager(engine, tok, tools=[], max_turns=4, max_new_tokens=16)

    events: list[dict] = []
    done_event = threading.Event()

    def emit(e):
        events.append(e)
        if e["type"] == "done":
            done_event.set()

    stop_event = threading.Event()
    thread = threading.Thread(target=mgr.run_forever, args=(stop_event,), daemon=True)
    thread.start()
    try:
        mgr.submit(1, [{"role": "user", "content": "hi"}], emit)
        assert done_event.wait(timeout=10), "session never completed"
    finally:
        stop_event.set()
        thread.join(timeout=5)

    assert events[-1]["final_text"] == "ok"
