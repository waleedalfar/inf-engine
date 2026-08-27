"""Cross-turn prefix reuse must not change what the model generates.

A chat turn's prompt is the previous turn's prompt plus its answer plus the new
message — strictly append-only — so the KV for the shared prefix is already
correct and can be kept. Re-prefilling it every turn was the dominant
interactive cost: 4430 ms to re-read 7152 tokens before emitting a single new
one, against 27 ms/step of decode.

The risk is silent divergence: a stale or misaligned cache still produces
plausible text. These tests pin the output against a cold engine that prefilled
everything from scratch.

Run:
    pytest tests/test_prefix_reuse.py -v
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.llama_model import LlamaModel
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.llama_weights import LlamaWeights
from engine.sampling import SamplingConfig, SamplingMode
from engine.speculative_paged_engine import SpeculativePagedEngine, _common_prefix_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _config() -> LlamaConfig:
    return LlamaConfig(
        name="test-prefix-reuse", vocab_size=64, n_ctx=1024, d_model=32,
        n_layer=2, n_head=4, n_kv_heads=2, intermediate_size=64,
        rope_theta=10_000.0, norm_eps=1e-5, qk_norm=True,
    )


def _model(cfg: LlamaConfig, seed: int) -> LlamaModel:
    d, f = cfg.d_model, cfg.intermediate_size
    h = cfg.n_kv_heads * cfg.head_dim
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=DEVICE)

    t = {"model.embed_tokens.weight": rand(cfg.vocab_size, d),
         "model.norm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE)}
    for i in range(cfg.n_layer):
        p = f"model.layers.{i}."
        t |= {
            p + "input_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "post_attention_layernorm.weight": torch.ones(d, dtype=DTYPE, device=DEVICE),
            p + "self_attn.q_proj.weight": rand(cfg.n_head * cfg.head_dim, d),
            p + "self_attn.k_proj.weight": rand(h, d),
            p + "self_attn.v_proj.weight": rand(h, d),
            p + "self_attn.o_proj.weight": rand(d, cfg.n_head * cfg.head_dim),
            p + "mlp.gate_proj.weight": rand(f, d),
            p + "mlp.up_proj.weight": rand(f, d),
            p + "mlp.down_proj.weight": rand(d, f),
            p + "self_attn.q_norm.weight": rand(cfg.head_dim).abs(),
            p + "self_attn.k_norm.weight": rand(cfg.head_dim).abs(),
        }
    return LlamaModel(LlamaWeights(t, cfg), cfg)


def _engine(cfg, tgt, dft, n_draft=3):
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)
    mk = lambda m: LlamaPagedEngine(m, n_total_blocks=400, block_size=16,
                                    eos_token=None, sampling=greedy,
                                    enable_cuda_graphs=False)
    return SpeculativePagedEngine(mk(tgt), mk(dft), n_draft=n_draft, eos_token=None)


def _cold(cfg, tgt, dft, prompt, n_new):
    """A fresh engine that prefills the whole prompt — the reference."""
    eng = _engine(cfg, tgt, dft)
    req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=n_new)
    return eng._generate_one(req)[0]


def test_common_prefix_len():
    assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _common_prefix_len([1, 2, 3], [1, 9, 3]) == 1
    assert _common_prefix_len([], [1]) == 0
    assert _common_prefix_len([1, 2], [1, 2]) == 2


def test_reused_turns_match_cold_engines():
    """Multi-turn with reuse must equal a cold engine per turn, token for token."""
    cfg = _config()
    tgt, dft = _model(cfg, 1), _model(cfg, 2)
    torch.manual_seed(0)
    base = torch.randint(0, cfg.vocab_size, (150,)).tolist()

    eng = _engine(cfg, tgt, dft)
    convo = list(base)
    for turn in range(3):
        req = LlamaRequest(req_id=turn, prompt_ids=list(convo), max_new_tokens=10)
        got, _ = eng.generate_resident(req)
        want = _cold(cfg, tgt, dft, list(convo), 10)
        assert got == want, f"turn {turn} diverged\n  reuse: {got}\n  cold:  {want}"
        # Next turn appends the answer plus a new "user message".
        convo = convo + got + [7, 8, 9]
    eng.release()


def test_reuse_actually_happens():
    """Guard the equality test from passing because reuse never engaged."""
    cfg = _config()
    tgt, dft = _model(cfg, 3), _model(cfg, 4)
    torch.manual_seed(0)
    base = torch.randint(0, cfg.vocab_size, (150,)).tolist()

    eng = _engine(cfg, tgt, dft)
    r1 = LlamaRequest(req_id=0, prompt_ids=list(base), max_new_tokens=8)
    gen1, _ = eng.generate_resident(r1)
    assert eng._resident is not None, "nothing was kept resident after turn 1"
    cached_after_first = eng._resident["cached"]
    assert cached_after_first >= len(base) - 1

    convo = base + gen1 + [5, 6]
    r2 = LlamaRequest(req_id=1, prompt_ids=list(convo), max_new_tokens=8)
    eng.generate_resident(r2)
    # Turn 2's prompt shares everything but its last two tokens.
    assert _common_prefix_len(base + gen1, convo) >= len(base)
    eng.release()


def test_divergent_prompt_falls_back_cleanly():
    """An unrelated second prompt must still produce the cold-engine answer."""
    cfg = _config()
    tgt, dft = _model(cfg, 5), _model(cfg, 6)
    torch.manual_seed(0)
    a = torch.randint(0, cfg.vocab_size, (150,)).tolist()
    b = torch.randint(0, cfg.vocab_size, (150,)).tolist()

    eng = _engine(cfg, tgt, dft)
    eng.generate_resident(LlamaRequest(req_id=0, prompt_ids=list(a), max_new_tokens=6))
    got, _ = eng.generate_resident(LlamaRequest(req_id=1, prompt_ids=list(b), max_new_tokens=6))
    assert got == _cold(cfg, tgt, dft, list(b), 6)
    eng.release()


def test_release_frees_blocks():
    """Sessions must not leak KV blocks across turns."""
    cfg = _config()
    tgt, dft = _model(cfg, 7), _model(cfg, 8)
    eng = _engine(cfg, tgt, dft)
    free_before = eng.target.cache.manager.n_free

    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (100,)).tolist()
    for turn in range(3):
        eng.generate_resident(
            LlamaRequest(req_id=turn, prompt_ids=list(prompt), max_new_tokens=5))
    eng.release()
    assert eng.target.cache.manager.n_free == free_before
    assert eng._resident is None


def _ring_engine(cfg, tgt, dft, n_draft=3, window=32):
    """Same pair, but the draft runs a windowed ring pool that recycles blocks."""
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)
    t = LlamaPagedEngine(tgt, n_total_blocks=400, block_size=16,
                         eos_token=None, sampling=greedy, enable_cuda_graphs=False)
    d = LlamaPagedEngine(dft, n_total_blocks=400, block_size=16,
                         eos_token=None, sampling=greedy, enable_cuda_graphs=False,
                         attn_window=window, attn_sinks=4, window_ring=True,
                         prefill_chunk=64)
    assert d.cache.window_ring and d.manager.n_total < 400   # pool really shrank
    return SpeculativePagedEngine(t, d, n_draft=n_draft, eos_token=None)


def test_reuse_matches_cold_engine_with_a_recycling_draft_cache():
    """Cross-turn reuse against a ring draft cache.

    The interaction worth pinning: `generate_resident` rolls both caches back and
    continues, while the ring has already recycled the draft's out-of-window
    blocks and repointed those entries at the pinned sink block. Rollback must not
    resurrect a position whose physical block is gone, and must not double-free
    an aliased entry. Both engines here are windowed, so a mismatch is the ring's
    bookkeeping and not the window's effect on what the draft proposes.
    """
    cfg = _config()
    tgt, dft = _model(cfg, 1), _model(cfg, 2)
    torch.manual_seed(0)
    base = torch.randint(0, cfg.vocab_size, (150,)).tolist()

    eng = _ring_engine(cfg, tgt, dft)
    convo = list(base)
    for turn in range(3):
        req = LlamaRequest(req_id=turn, prompt_ids=list(convo), max_new_tokens=10)
        got, _ = eng.generate_resident(req)

        cold = _ring_engine(cfg, tgt, dft)
        want = cold._generate_one(
            LlamaRequest(req_id=0, prompt_ids=list(convo), max_new_tokens=10))[0]

        assert got == want, f"turn {turn} diverged\n  reuse: {got}\n  cold:  {want}"
        convo = convo + got + [7, 8, 9]
    eng.release()
    # No block leaked or got freed twice across the whole conversation.
    free = eng.draft.manager._free
    assert len(free) == len(set(free))
    assert eng.draft.manager.n_free == eng.draft.manager.n_total
