"""Correctness tests for SpeculativePagedEngine.

Key invariant: under greedy decoding, speculative decoding with any draft model
produces token-for-token identical output to standard greedy decode with the
target model alone.  This is a mathematical guarantee of the Leviathan et al.
accept/reject algorithm:
  - Draft token accepted iff target's greedy argmax agrees → prob 0 or 1
  - On rejection, correction = target's argmax = same as standard decode

Run:
    wsl bash -c "cd /home/waleed/mlproj && .venv/bin/pytest tests/test_speculative_paged.py -v"
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.sampling import SamplingConfig, SamplingMode
from engine.speculative import _get_probs
from engine.speculative_paged_engine import (
    SpeculativePagedEngine,
    _get_probs_batch,
    _RollingCtx,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _mini_config() -> LlamaConfig:
    return LlamaConfig(
        name="test-mini-spec-paged",
        vocab_size=64,
        n_ctx=128,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=64,
    )


def _mini_model(config: LlamaConfig, seed: int = 0):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(seed)

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
    return LlamaModel(weights, config)


def _make_paged_engine(model, n_blocks: int = 200, block_size: int = 16,
                       greedy: bool = True) -> LlamaPagedEngine:
    sampling = SamplingConfig(mode=SamplingMode.GREEDY if greedy else SamplingMode.TOP_K,
                              top_k=5, temperature=1.0)
    return LlamaPagedEngine(
        model,
        n_total_blocks=n_blocks,
        block_size=block_size,
        eos_token=None,
        sampling=sampling,
        enable_cuda_graphs=False,
    )


# ---------------------------------------------------------------------------
# Greedy identity: spec decode == standard decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draft", [1, 2, 4])
@pytest.mark.parametrize("prompt_len,max_new", [(3, 8), (5, 4), (1, 10)])
def test_greedy_spec_matches_standard(n_draft, prompt_len, max_new):
    """Core gate: greedy speculative paged == greedy standard paged, bit-for-bit."""
    cfg = _mini_config()
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (prompt_len,)).tolist()

    target_model = _mini_model(cfg, seed=42)
    draft_model  = _mini_model(cfg, seed=7)   # different weights → forced rejections

    # Standard decode
    std_engine = _make_paged_engine(target_model)
    std_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    std_results = std_engine.run_offline([std_req])
    std_tokens = std_results[0]

    # Speculative decode
    t_engine = _make_paged_engine(target_model)
    d_engine = _make_paged_engine(draft_model)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=n_draft, eos_token=None)
    spec_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    spec_results, stats = spec_engine.run_offline([spec_req])
    spec_tokens = spec_results[0]

    assert spec_tokens == std_tokens, (
        f"n_draft={n_draft}, prompt_len={prompt_len}, max_new={max_new}\n"
        f"  spec:     {spec_tokens}\n"
        f"  standard: {std_tokens}"
    )
    assert stats.n_steps > 0


# ---------------------------------------------------------------------------
# Phase 3: graphed verify step
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
def test_step_verify_graphed_matches_eager():
    """_step_verify_graphed must return logits identical to _step_verify_eager."""
    cfg = _mini_config()
    model = _mini_model(cfg, seed=42)
    prompt = [1, 2, 3, 4, 5]
    verify_ids = [10, 20, 30, 40, 50]  # q_len=5

    # Eager engine
    eager_engine = _make_paged_engine(model, n_blocks=200, block_size=16)
    req_e = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=20)
    eager_engine._prefill(req_e, 0.0)
    eager_logits = eager_engine._step_verify_eager(0, verify_ids)

    # Graphed engine (same model — shares weights but independent cache)
    graph_engine = LlamaPagedEngine(
        model, n_total_blocks=200, block_size=16,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
        enable_cuda_graphs=True,
    )
    req_g = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=20)
    graph_engine._prefill(req_g, 0.0)
    graph_logits = graph_engine._step_verify_graphed(0, verify_ids)

    assert eager_logits.shape == graph_logits.shape
    torch.testing.assert_close(eager_logits, graph_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("n_draft", [1, 2, 4])
@pytest.mark.parametrize("prompt_len,max_new", [(3, 8), (5, 4), (1, 10)])
def test_greedy_spec_graphed_verify_matches_standard(n_draft, prompt_len, max_new):
    """Graphed verify (Phase 3): spec decode with graphed target == standard greedy decode."""
    cfg = _mini_config()
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (prompt_len,)).tolist()

    target_model = _mini_model(cfg, seed=42)
    draft_model  = _mini_model(cfg, seed=7)

    # Standard decode
    std_engine = _make_paged_engine(target_model)
    std_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    std_results = std_engine.run_offline([std_req])

    # Spec decode with graphed verify on target
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)
    t_engine = LlamaPagedEngine(target_model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy, enable_cuda_graphs=True)
    d_engine = LlamaPagedEngine(draft_model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy, enable_cuda_graphs=True)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=n_draft, eos_token=None)
    spec_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    spec_results, stats = spec_engine.run_offline([spec_req])

    assert spec_results[0] == std_results[0], (
        f"n_draft={n_draft}, prompt_len={prompt_len}, max_new={max_new}\n"
        f"  spec (graphed verify): {spec_results[0]}\n"
        f"  standard:              {std_results[0]}"
    )
    assert stats.n_steps > 0


# ---------------------------------------------------------------------------
# max_new_tokens is always respected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draft,max_new", [(1, 5), (4, 7), (2, 3)])
def test_max_new_tokens_respected(n_draft, max_new):
    cfg = _mini_config()
    prompt = [1, 2, 3, 4]

    cfg2 = _mini_config()
    target_model = _mini_model(cfg2, seed=11)
    draft_model  = _mini_model(cfg2, seed=22)

    t_engine = _make_paged_engine(target_model)
    d_engine = _make_paged_engine(draft_model)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=n_draft, eos_token=None)
    req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    results, stats = spec_engine.run_offline([req])
    tokens = results[0]

    assert len(tokens) == max_new, (
        f"Expected {max_new} tokens, got {len(tokens)}: {tokens}"
    )


# ---------------------------------------------------------------------------
# Stats sanity
# ---------------------------------------------------------------------------

def test_stats_sanity():
    """SpecStats counters must be non-negative and sum correctly."""
    cfg = _mini_config()
    prompt = [1, 2, 3]
    max_new = 12
    n_draft = 4

    target_model = _mini_model(cfg, seed=55)
    draft_model  = _mini_model(cfg, seed=66)

    t_engine = _make_paged_engine(target_model)
    d_engine = _make_paged_engine(draft_model)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=n_draft)
    req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    _, stats = spec_engine.run_offline([req])

    assert stats.n_accepted >= 0
    assert stats.n_rejected >= 0
    assert stats.n_bonus >= 0
    assert stats.n_steps > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.tokens_per_step > 0.0
    # Prefill emits 1 token (not counted in stats).
    # The spec loop emits the rest. When exactly 1 token remains, a direct
    # target step fires (K<=0 path) — also not counted — so total can be
    # max_new-1 or max_new-2 depending on the step boundary.
    total = stats.n_accepted + stats.n_rejected + stats.n_bonus
    assert max_new - 2 <= total <= max_new - 1, (
        f"total={total} out of expected range [{max_new - 2}, {max_new - 1}]"
    )


# ---------------------------------------------------------------------------
# Cache cleanup: engines usable again after run_offline
# ---------------------------------------------------------------------------

def test_cache_cleanup_after_run():
    """Block pool must be fully restored after a completed request."""
    cfg = _mini_config()
    target_model = _mini_model(cfg, seed=1)
    draft_model  = _mini_model(cfg, seed=2)

    n_blocks = 100
    t_engine = _make_paged_engine(target_model, n_blocks=n_blocks)
    d_engine = _make_paged_engine(draft_model, n_blocks=n_blocks)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=2)

    req1 = LlamaRequest(req_id=0, prompt_ids=[1, 2, 3], max_new_tokens=8)
    spec_engine.run_offline([req1])

    # Both pools fully restored.
    assert t_engine.manager.n_free == n_blocks
    assert d_engine.manager.n_free == n_blocks

    # Engine is reusable for another request.
    req2 = LlamaRequest(req_id=1, prompt_ids=[4, 5], max_new_tokens=4)
    results2, _ = spec_engine.run_offline([req2])
    assert len(results2[1]) == 4


# ---------------------------------------------------------------------------
# Same-model test: exercises accept path, bonus token, and draft-sync step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draft", [1, 2])
def test_same_model_all_accept_and_bonus(n_draft):
    """With draft==target, every draft token is accepted and bonus fires every step.

    This is the canonical test for the all-accepted + bonus + draft-sync branch
    of the spec loop.  With identical models and the same KV history, draft and
    target compute identical logits at every position (both caches are kept in
    sync by construction), so under greedy acceptance is always 1.0.

    Note: the greedy identity property holds independently of acceptance rate,
    so we don't assert spec_tokens == std_tokens (both produce correct output
    regardless), but we DO assert that the bonus path executed.
    """
    cfg = _mini_config()
    prompt = [1, 2, 3, 4]
    max_new = 8

    model = _mini_model(cfg, seed=99)          # same model for draft and target
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    # Standard decode
    std_engine = LlamaPagedEngine(model, n_total_blocks=200, block_size=16,
                                  eos_token=None, sampling=greedy)
    std_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    std_results = std_engine.run_offline([std_req])

    # Spec decode (same model for draft and target)
    t_engine = LlamaPagedEngine(model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy)
    d_engine = LlamaPagedEngine(model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=n_draft, eos_token=None)
    spec_req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=max_new)
    spec_results, stats = spec_engine.run_offline([spec_req])
    spec_tokens = spec_results[0]

    # Greedy identity still holds
    assert spec_tokens == std_results[0], (
        f"n_draft={n_draft}\n  spec={spec_tokens}\n  std={std_results[0]}"
    )
    # The all-accepted + bonus path must have executed at least once
    assert stats.n_bonus > 0, (
        f"n_draft={n_draft}: expected bonus tokens but got n_bonus=0; "
        f"n_accepted={stats.n_accepted}, n_rejected={stats.n_rejected}"
    )
    # With same-model greedy, draft is never rejected (accept_prob always = 1.0)
    assert stats.n_rejected == 0, f"Unexpected rejections: n_rejected={stats.n_rejected}"
    assert stats.acceptance_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multiple requests processed sequentially
# ---------------------------------------------------------------------------

def test_multiple_requests():
    cfg = _mini_config()
    target_model = _mini_model(cfg, seed=3)
    draft_model  = _mini_model(cfg, seed=4)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    t_engine = LlamaPagedEngine(target_model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy)
    d_engine = LlamaPagedEngine(draft_model, n_total_blocks=200, block_size=16,
                                eos_token=None, sampling=greedy)
    spec_engine = SpeculativePagedEngine(t_engine, d_engine, n_draft=2)

    requests = [
        LlamaRequest(req_id=i, prompt_ids=[i + 1, i + 2], max_new_tokens=5)
        for i in range(3)
    ]
    results, stats = spec_engine.run_offline(requests)

    assert set(results.keys()) == {0, 1, 2}
    for req_id, tokens in results.items():
        assert len(tokens) == 5, f"req {req_id}: expected 5 tokens, got {len(tokens)}"

    assert stats.n_steps > 0


# ---------------------------------------------------------------------------
# Batched accept/reject helpers (Phase 4 — sync elimination)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(SamplingMode))
@pytest.mark.parametrize("rep_penalty", [1.0, 1.15])
def test_get_probs_batch_matches_row_wise(mode, rep_penalty):
    """_get_probs_batch must be row-for-row identical to _get_probs.

    The GPU-batched accept/reject path replaces K separate _get_probs calls
    with one batched call; any divergence here silently changes which draft
    tokens get accepted.
    """
    torch.manual_seed(7)
    logits = torch.randn(5, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randint(0, 64, (23,), device=DEVICE)
    cfg = SamplingConfig(mode=mode, temperature=0.8, top_k=8, top_p=0.9,
                         repetition_penalty=rep_penalty)

    batched = _get_probs_batch(logits, cfg, context)
    for i in range(logits.shape[0]):
        expected = _get_probs(logits[i], cfg, context)
        torch.testing.assert_close(batched[i], expected, rtol=1e-5, atol=1e-6)


def test_get_probs_batch_ignores_context_without_penalty():
    """With penalty 1.0 the context is unused, so passing None must be identical."""
    torch.manual_seed(8)
    logits = torch.randn(3, 64, device=DEVICE, dtype=DTYPE)
    cfg = SamplingConfig(mode=SamplingMode.TOP_P, top_p=0.9, repetition_penalty=1.0)
    context = torch.randint(0, 64, (11,), device=DEVICE)

    torch.testing.assert_close(
        _get_probs_batch(logits, cfg, context), _get_probs_batch(logits, cfg, None)
    )


def test_rolling_ctx_tracks_generated_tokens():
    """_RollingCtx must expose exactly what torch.tensor(req.generated) would."""
    ctx = _RollingCtx([5, 9], capacity=16, t_device=DEVICE, d_device=DEVICE)
    generated = [5, 9]
    for chunk in ([1, 2, 3], [], [4]):
        ctx.extend(chunk)
        generated += chunk
        expected = torch.tensor(generated, device=DEVICE)
        torch.testing.assert_close(ctx.target, expected)
        torch.testing.assert_close(ctx.draft, expected)


def test_rolling_ctx_empty_is_none():
    ctx = _RollingCtx([], capacity=8, t_device=DEVICE, d_device=DEVICE)
    assert ctx.target is None and ctx.draft is None


def test_sampling_spec_respects_repetition_penalty_context():
    """End-to-end run with a repetition penalty exercises the _RollingCtx path."""
    cfg = _mini_config()
    torch.manual_seed(11)
    sampling = SamplingConfig(mode=SamplingMode.TOP_K, top_k=8, temperature=0.9,
                              repetition_penalty=1.2,
                              generator=torch.Generator(device=DEVICE).manual_seed(11))
    t_engine = LlamaPagedEngine(_mini_model(cfg, seed=1), n_total_blocks=200,
                                block_size=16, eos_token=None, sampling=sampling)
    d_engine = LlamaPagedEngine(_mini_model(cfg, seed=2), n_total_blocks=200,
                                block_size=16, eos_token=None, sampling=sampling)
    spec = SpeculativePagedEngine(t_engine, d_engine, n_draft=3)

    req = LlamaRequest(req_id=0, prompt_ids=[1, 2, 3], max_new_tokens=20)
    generated, stats = spec._generate_one(req)

    assert len(generated) == 20
    assert all(0 <= t < cfg.vocab_size for t in generated)
    # Every spec step emits at least one token, and the prefill token plus the
    # final K<=0 target step are not counted in stats.
    emitted = stats.n_accepted + stats.n_rejected + stats.n_bonus
    assert stats.n_steps <= emitted <= len(generated)
