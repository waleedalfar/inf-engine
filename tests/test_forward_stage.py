"""Phase 1 (distributed pipeline-parallel) tests: LlamaModel.forward_stage.

``forward_stage`` splits ``LlamaModel.forward`` at its two natural seams — the
embedding lookup and the final norm/lm_head — so a model can later run as a
chain of pipeline stages across a network boundary. These tests only cover
the in-process, single-device case: no networking, no KV-cache layer-range
scoping (that generalization is Phase 2). The bar is bit-identical output vs.
the existing ``forward()`` for a single stage, and logit-identical output
when a forward is split into two stages and stitched back together
in-process — both for an uncached (prefill) call and across cached
(decode) steps, since decode is the path most sensitive to start_pos/
position_ids bookkeeping.

Run:
    pytest tests/test_forward_stage.py -v
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Mini-model helpers (mirrors tests/test_paged_cache.py's pattern)
# ---------------------------------------------------------------------------

def _mini_config(n_layer: int = 4, qk_norm: bool = False) -> LlamaConfig:
    return LlamaConfig(
        name="test-mini-stage",
        vocab_size=256,
        n_ctx=128,
        d_model=64,
        n_layer=n_layer,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
        qk_norm=qk_norm,
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
            p + "mlp.down_proj.weight":              rand(d, f),
        }
        if config.qk_norm:
            tensors[p + "self_attn.q_norm.weight"] = torch.ones(
                config.head_dim, dtype=DTYPE, device=device
            )
            tensors[p + "self_attn.k_norm.weight"] = torch.ones(
                config.head_dim, dtype=DTYPE, device=device
            )
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


# ---------------------------------------------------------------------------
# forward() == forward_stage(0, n_layer, True, True, ...)
# ---------------------------------------------------------------------------

def test_forward_stage_single_stage_matches_forward():
    cfg = _mini_config(n_layer=4)
    model = _mini_model(cfg, seed=1)
    ids = torch.randint(0, cfg.vocab_size, (2, 6), device=DEVICE)
    pos = torch.arange(6, device=DEVICE)

    with torch.no_grad():
        expected = model.forward(ids, start_pos=0, position_ids=pos)
        actual = model.forward_stage(ids, 0, cfg.n_layer, True, True, start_pos=0, position_ids=pos)

    assert torch.equal(expected, actual)


def test_forward_stage_respects_n_ctx_like_forward():
    cfg = _mini_config(n_layer=2)
    model = _mini_model(cfg, seed=5)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)

    with pytest.raises(ValueError, match="exceeds n_ctx"):
        model.forward_stage(ids, 0, cfg.n_layer, True, True, start_pos=cfg.n_ctx)


# ---------------------------------------------------------------------------
# Splitting a forward into two in-process stages reproduces the single call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_layer,split,qk_norm", [
    (4, 1, False),
    (4, 2, False),
    (4, 3, False),
    (6, 3, False),
    (4, 2, True),
])
def test_forward_stage_split_matches_single_call(n_layer, split, qk_norm):
    cfg = _mini_config(n_layer=n_layer, qk_norm=qk_norm)
    model = _mini_model(cfg, seed=2)
    ids = torch.randint(0, cfg.vocab_size, (2, 5), device=DEVICE)
    pos = torch.arange(5, device=DEVICE)

    with torch.no_grad():
        expected = model.forward(ids, start_pos=0, position_ids=pos)
        hidden = model.forward_stage(ids, 0, split, True, False, start_pos=0, position_ids=pos)
        actual = model.forward_stage(hidden, split, n_layer, False, True, start_pos=0, position_ids=pos)

    assert torch.equal(expected, actual)


def test_forward_stage_non_last_returns_residual_not_logits():
    """Asserts the intermediate stage's own output shape directly — a stub
    that silently fell through to full-model behavior would still pass an
    output-equality test against forward(), so this checks the new path's
    own state, not just parity with the fallback."""
    cfg = _mini_config(n_layer=4)
    model = _mini_model(cfg, seed=3)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)

    with torch.no_grad():
        hidden = model.forward_stage(ids, 0, 2, True, False, start_pos=0)

    assert hidden.shape == (1, 4, cfg.d_model)
    assert hidden.shape[-1] != cfg.vocab_size


def test_forward_stage_last_only_returns_logits():
    cfg = _mini_config(n_layer=4)
    model = _mini_model(cfg, seed=3)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)

    with torch.no_grad():
        hidden = model.forward_stage(ids, 0, 2, True, False, start_pos=0)
        logits = model.forward_stage(hidden, 2, 4, False, True, start_pos=0)

    assert logits.shape == (1, 4, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Split stages stay correct across cached (decode) steps, not just prefill
# ---------------------------------------------------------------------------

def test_forward_stage_split_with_kv_cache_multistep_matches_single_call():
    cfg = _mini_config(n_layer=4)
    model_ref = _mini_model(cfg, seed=4)
    model_split = _mini_model(cfg, seed=4)  # identical weights (same seed)
    split_layer = 2

    prompt = torch.randint(0, cfg.vocab_size, (1, 3), device=DEVICE)
    cache_ref = LlamaStaticKVCache(cfg, batch=1, max_seq=8, device=DEVICE, dtype=DTYPE)
    cache_split = LlamaStaticKVCache(cfg, batch=1, max_seq=8, device=DEVICE, dtype=DTYPE)

    pos = torch.arange(3, device=DEVICE)
    with torch.no_grad():
        ref_logits = model_ref.forward(prompt, cache=cache_ref, start_pos=0, position_ids=pos)
        hidden = model_split.forward_stage(
            prompt, 0, split_layer, True, False, cache=cache_split, start_pos=0, position_ids=pos
        )
        split_logits = model_split.forward_stage(
            hidden, split_layer, cfg.n_layer, False, True, cache=cache_split, start_pos=0, position_ids=pos
        )

    assert torch.equal(ref_logits, split_logits)

    # Decode one more token through both paths (the trickiest case: start_pos > 0).
    next_tok = ref_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start_pos = 3
    pos_s = torch.tensor([start_pos], device=DEVICE)
    with torch.no_grad():
        ref_logits2 = model_ref.forward(next_tok, cache=cache_ref, start_pos=start_pos, position_ids=pos_s)
        hidden2 = model_split.forward_stage(
            next_tok, 0, split_layer, True, False, cache=cache_split, start_pos=start_pos, position_ids=pos_s
        )
        split_logits2 = model_split.forward_stage(
            hidden2, split_layer, cfg.n_layer, False, True, cache=cache_split, start_pos=start_pos, position_ids=pos_s
        )

    assert torch.equal(ref_logits2, split_logits2)
