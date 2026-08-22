"""Correctness tests for Q/K/V and gate/up projection fusion.

Fusing is a pure concatenation along the output dimension, so a fused model must
produce *bit-identical* logits to an unfused one — not merely close. These tests
check that, and they also assert the fused path actually ran: an output-equality
test alone would pass just as happily if ``fuse_projections`` were a no-op and
both models took the unfused branch (see
``.claude/skills/device-string-and-noop-feature-audit``).

Run:
    pytest tests/test_fused_projections.py -v
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.fuse_weights import FUSED_GROUPS, fuse_projections
from engine.llama_model import LlamaModel
from engine.llama_weights import LlamaWeights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _config(qk_norm: bool = True) -> LlamaConfig:
    return LlamaConfig(
        name="test-fuse",
        vocab_size=64,
        n_ctx=128,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=64,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        qk_norm=qk_norm,
    )


def _model(config: LlamaConfig, seed: int = 0, dtype: torch.dtype = DTYPE) -> LlamaModel:
    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, device=DEVICE)

    tensors = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight": torch.ones(d, dtype=dtype, device=DEVICE),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight": torch.ones(d, dtype=dtype, device=DEVICE),
            p + "post_attention_layernorm.weight": torch.ones(d, dtype=dtype, device=DEVICE),
            p + "self_attn.q_proj.weight": rand(config.n_head * config.head_dim, d),
            p + "self_attn.k_proj.weight": rand(h, d),
            p + "self_attn.v_proj.weight": rand(h, d),
            p + "self_attn.o_proj.weight": rand(d, config.n_head * config.head_dim),
            p + "mlp.gate_proj.weight": rand(f, d),
            p + "mlp.up_proj.weight": rand(f, d),
            p + "mlp.down_proj.weight": rand(d, f),
        }
        if config.qk_norm:
            tensors[p + "self_attn.q_norm.weight"] = rand(config.head_dim).abs()
            tensors[p + "self_attn.k_norm.weight"] = rand(config.head_dim).abs()
    return LlamaModel(LlamaWeights(tensors, config), config)


# ---------------------------------------------------------------------------
# The fused path is real (not a silently-skipped no-op)
# ---------------------------------------------------------------------------

def test_fusion_replaces_source_weights():
    """After fusing, layers expose the fused key and no longer the sources."""
    model = fuse_projections(_model(_config()))
    for i in range(model.config.n_layer):
        layer = model.w.layer(i)
        for fused_key, sources in FUSED_GROUPS.items():
            assert fused_key in layer, f"layer {i} missing {fused_key}"
            for s in sources:
                assert s not in layer, f"layer {i} still carries unfused {s}"


def test_fused_weight_has_concatenated_width():
    cfg = _config()
    model = fuse_projections(_model(cfg))
    layer = model.w.layer(0)
    qkv = layer["self_attn.qkv_proj.weight"]
    expected = (cfg.n_head + 2 * cfg.n_kv_heads) * cfg.head_dim
    assert qkv.shape[0] == expected
    assert layer["mlp.gate_up_proj.weight"].shape[0] == 2 * cfg.intermediate_size


def test_unfused_model_still_has_separate_weights():
    """Guards the comparison in the equality tests from being a tautology."""
    layer = _model(_config()).w.layer(0)
    assert "self_attn.q_proj.weight" in layer
    assert "self_attn.qkv_proj.weight" not in layer


# ---------------------------------------------------------------------------
# Fusing does not change the math
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qk_norm", [True, False])
def test_fused_forward_bit_identical(qk_norm):
    cfg = _config(qk_norm=qk_norm)
    ids = torch.randint(0, cfg.vocab_size, (1, 7), device=DEVICE)

    plain = _model(cfg, seed=3).forward(ids)
    fused = fuse_projections(_model(cfg, seed=3)).forward(ids)

    torch.testing.assert_close(fused, plain, rtol=0, atol=0)


def test_fused_forward_matches_with_kv_cache():
    """Multi-token then single-token forward — the shapes attention splits differ."""
    from engine.kv_cache import LlamaStaticKVCache

    cfg = _config()
    ids = torch.randint(0, cfg.vocab_size, (1, 5), device=DEVICE)
    nxt = torch.randint(0, cfg.vocab_size, (1, 1), device=DEVICE)

    outs = []
    for fuse in (False, True):
        m = _model(cfg, seed=4)
        if fuse:
            fuse_projections(m)
        cache = LlamaStaticKVCache(cfg, batch=1, max_seq=32, device=DEVICE, dtype=DTYPE)
        m.forward(ids, cache=cache, start_pos=0)
        outs.append(m.forward(nxt, cache=cache, start_pos=5))

    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)


def test_fusion_is_idempotent():
    """Re-running fusion on an already-fused model must not corrupt it."""
    cfg = _config()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device=DEVICE)
    once = fuse_projections(_model(cfg, seed=5))
    twice = fuse_projections(fuse_projections(_model(cfg, seed=5)))
    torch.testing.assert_close(twice.forward(ids), once.forward(ids), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# INT4 path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE != "cuda", reason="INT4 fused kernel requires CUDA")
def test_int4_fused_matches_int4_unfused():
    """Fusing packed INT4 weights is a cat of nibbles — same values, same output."""
    from engine.quantize import quantize_llama

    # d_model must be a multiple of the INT4 group size (128) for quantization.
    cfg = LlamaConfig(**{**_config().__dict__, "d_model": 256,
                         "intermediate_size": 512, "n_head": 2, "n_kv_heads": 1})
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)

    bf16 = torch.bfloat16
    plain = quantize_llama(_model(cfg, seed=6, dtype=bf16))
    fused = fuse_projections(quantize_llama(_model(cfg, seed=6, dtype=bf16)))

    assert "self_attn.qkv_proj.weight" in fused.w.layer(0)
    assert "self_attn.q_proj.weight" in plain.w.layer(0)
    torch.testing.assert_close(fused.forward(ids), plain.forward(ids),
                               rtol=1e-3, atol=1e-3)
