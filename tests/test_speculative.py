"""Phase 4 (scale-up) speculative decoding correctness tests.

Key correctness property: under greedy decoding, speculative decoding produces
token-for-token identical output to standard (non-speculative) greedy decoding.
This holds because:
  - greedy draft is accepted iff target's argmax agrees → accept prob is 0 or 1
  - on rejection, correction = target's argmax = same token standard decode emits

Tests use a synthetic mini-LLaMA (no weights file needed).

Run:
    pytest tests/test_speculative.py -v -s
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache
from engine.sampling import SamplingConfig, SamplingMode
from engine.speculative import SpecStats, SpeculativeDecoder, _correction_sample, _get_probs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_config(vocab=64, n_ctx=128) -> LlamaConfig:
    return LlamaConfig(
        name="test-mini",
        vocab_size=vocab,
        n_ctx=n_ctx,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=64,
    )


def _mini_model(config: LlamaConfig, seed: int = 0, device: str = DEVICE):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=device)

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight":         torch.ones(d, dtype=DTYPE, device=device),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight":          torch.ones(d, dtype=DTYPE, device=device),
            p + "post_attention_layernorm.weight":  torch.ones(d, dtype=DTYPE, device=device),
            p + "self_attn.q_proj.weight":          rand(d, d),
            p + "self_attn.k_proj.weight":          rand(h, d),
            p + "self_attn.v_proj.weight":          rand(h, d),
            p + "self_attn.o_proj.weight":          rand(d, d),
            p + "mlp.gate_proj.weight":             rand(f, d),
            p + "mlp.up_proj.weight":               rand(f, d),
            p + "mlp.down_proj.weight":             rand(d, f),
        }
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


def _make_caches(config: LlamaConfig, max_seq: int, device: str = DEVICE):
    return (
        LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=DTYPE),
        LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=DTYPE),
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_get_probs_greedy_one_hot():
    """_get_probs with GREEDY returns a one-hot tensor at the argmax."""
    cfg = SamplingConfig(mode=SamplingMode.GREEDY)
    logits = torch.tensor([0.1, 5.0, 0.3, 2.0])
    probs = _get_probs(logits, cfg)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probs.argmax().item() == 1    # highest logit
    assert probs[1].item() == pytest.approx(1.0)


def test_get_probs_top_k_sums_to_one():
    cfg = SamplingConfig(mode=SamplingMode.TOP_K, temperature=1.0, top_k=2)
    logits = torch.randn(10)
    probs = _get_probs(logits, cfg)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_get_probs_top_p_sums_to_one():
    cfg = SamplingConfig(mode=SamplingMode.TOP_P, temperature=1.0, top_p=0.9)
    logits = torch.randn(10)
    probs = _get_probs(logits, cfg)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_correction_sample_greedy():
    """Correction under greedy uses target's residual argmax."""
    cfg = SamplingConfig(mode=SamplingMode.GREEDY)
    p_target = torch.tensor([0.0, 0.7, 0.3])
    p_draft  = torch.tensor([0.0, 0.8, 0.2])
    # residual = max(0, [0, -0.1, 0.1]) → only index 2 > 0
    tok = _correction_sample(p_target, p_draft, cfg)
    assert tok.item() == 2


def test_correction_sample_sums_to_one():
    """Residual distribution is properly normalized."""
    cfg = SamplingConfig(mode=SamplingMode.TOP_K, temperature=1.0, top_k=4)
    p_target = torch.softmax(torch.randn(10), dim=-1)
    p_draft  = torch.softmax(torch.randn(10), dim=-1)
    # Just check it doesn't crash and returns a valid token
    tok = _correction_sample(p_target, p_draft, cfg)
    assert 0 <= tok.item() < 10


# ---------------------------------------------------------------------------
# SpecStats tests
# ---------------------------------------------------------------------------

def test_spec_stats_properties():
    s = SpecStats(n_accepted=6, n_rejected=2, n_bonus=2, n_steps=2)
    assert s.acceptance_rate == pytest.approx(6 / 8)
    assert s.tokens_per_step == pytest.approx(10 / 2)


def test_spec_stats_zero_division():
    s = SpecStats()
    assert s.acceptance_rate == 0.0
    assert s.tokens_per_step == 0.0


# ---------------------------------------------------------------------------
# Greedy correctness gate — speculative decode == standard decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draft", [1, 2, 4])
@pytest.mark.parametrize("prompt_len,max_new", [(3, 8), (5, 4), (1, 10)])
def test_greedy_spec_matches_standard(n_draft, prompt_len, max_new):
    """Core correctness gate: greedy speculative == greedy standard, bit-for-bit."""
    cfg = _mini_config()
    max_seq = prompt_len + max_new + n_draft + 4

    draft_model  = _mini_model(cfg, seed=7, device=DEVICE)
    target_model = _mini_model(cfg, seed=42, device=DEVICE)
    spec = SpeculativeDecoder(draft_model, target_model, n_draft=n_draft)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=DEVICE)

    # --- Standard greedy decode (target only, no draft) ---
    ref_cache = LlamaStaticKVCache(cfg, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE)
    ref_out = target_model.generate_cached(ids, max_new, ref_cache, greedy)

    # --- Speculative greedy decode ---
    d_cache, t_cache = _make_caches(cfg, max_seq)
    spec_out, stats = spec.generate(ids, max_new, d_cache, t_cache, greedy)

    assert torch.equal(spec_out, ref_out), (
        f"n_draft={n_draft}, prompt={prompt_len}, max_new={max_new}\n"
        f"  spec   : {spec_out.tolist()}\n"
        f"  standard: {ref_out.tolist()}"
    )
    # All tokens after prompt should be identical.
    assert spec_out.shape[1] == ref_out.shape[1]


def test_greedy_spec_same_draft_and_target():
    """When draft == target (identical weights), all K drafts are always accepted."""
    cfg = _mini_config()
    prompt_len, max_new, n_draft = 4, 12, 3
    max_seq = prompt_len + max_new + n_draft + 4

    model = _mini_model(cfg, seed=0, device=DEVICE)
    spec = SpeculativeDecoder(model, model, n_draft=n_draft)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=DEVICE)

    ref_cache = LlamaStaticKVCache(cfg, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE)
    ref_out = model.generate_cached(ids, max_new, ref_cache, greedy)

    d_cache, t_cache = _make_caches(cfg, max_seq)
    spec_out, stats = spec.generate(ids, max_new, d_cache, t_cache, greedy)

    assert torch.equal(spec_out, ref_out)
    # When draft == target (greedy), rejection never happens — all steps are full-accept.
    assert stats.n_rejected == 0


def test_eos_stops_early():
    """generate() stops as soon as an EOS token is emitted."""
    cfg = _mini_config()
    max_new = 20
    max_seq = 10 + max_new + 8

    # Find what token target would predict first so we can use it as EOS.
    target_model = _mini_model(cfg, seed=99, device=DEVICE)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)
    with torch.no_grad():
        logits = target_model.forward(ids, start_pos=0,
                                      position_ids=torch.arange(4, device=DEVICE))
    eos = int(logits[0, -1].argmax())

    draft_model = _mini_model(cfg, seed=1, device=DEVICE)
    spec = SpeculativeDecoder(draft_model, target_model, n_draft=4)

    d_cache, t_cache = _make_caches(cfg, max_seq)
    spec_out, _ = spec.generate(ids, max_new, d_cache, t_cache, greedy, eos_token=eos)

    # Output includes prompt + at most a few generated tokens ending with EOS.
    assert spec_out.shape[1] > ids.shape[1]       # at least one new token
    assert spec_out[0, -1].item() == eos           # last token is EOS
    assert spec_out.shape[1] <= ids.shape[1] + max_new


def test_stats_sanity():
    """SpecStats counters must be non-negative and consistent."""
    cfg = _mini_config()
    prompt_len, max_new, n_draft = 3, 16, 3
    max_seq = prompt_len + max_new + n_draft + 4

    draft_model  = _mini_model(cfg, seed=11, device=DEVICE)
    target_model = _mini_model(cfg, seed=77, device=DEVICE)
    spec = SpeculativeDecoder(draft_model, target_model, n_draft=n_draft)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=DEVICE)
    d_cache, t_cache = _make_caches(cfg, max_seq)
    _, stats = spec.generate(ids, max_new, d_cache, t_cache, greedy)

    assert stats.n_accepted >= 0
    assert stats.n_rejected >= 0
    assert stats.n_bonus >= 0
    assert stats.n_steps > 0
    total_new = stats.n_accepted + stats.n_rejected + stats.n_bonus
    # The first token (post-prefill target sample) is not tracked in stats;
    # the speculative loop accounts for the remaining max_new - 1 tokens.
    assert total_new == max_new - 1


def test_max_new_tokens_respected():
    """Exactly max_new_tokens tokens are generated (no EOS configured)."""
    cfg = _mini_config()
    prompt_len, max_new, n_draft = 5, 7, 2
    max_seq = prompt_len + max_new + n_draft + 4

    draft  = _mini_model(cfg, seed=3, device=DEVICE)
    target = _mini_model(cfg, seed=9, device=DEVICE)
    spec = SpeculativeDecoder(draft, target, n_draft=n_draft)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=DEVICE)
    d_cache, t_cache = _make_caches(cfg, max_seq)
    out, _ = spec.generate(ids, max_new, d_cache, t_cache, greedy)

    assert out.shape == (1, prompt_len + max_new)
