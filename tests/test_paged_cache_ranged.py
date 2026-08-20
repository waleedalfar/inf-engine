"""Phase 2 (distributed pipeline-parallel) tests: PagedLlamaKVCache layer-range scoping.

``owned_layers``/``layer_offset`` let a ``PagedLlamaKVCache`` store only the
layers a pipeline stage actually runs, instead of every layer in the model.
These tests only cover the in-process case (two ranged caches driven by two
``LlamaModel.forward_stage`` calls in the same process, no networking — that's
Phase 3). The bar: splitting the KV cache by layer range must not change
generated output at all (token-for-token identical to one full-range cache
across multiple decode steps), and each ranged cache's own memory footprint
must shrink proportionally to the layers it owns — asserted directly on the
new path's state, not just inferred from output parity.

Run:
    pytest tests/test_paged_cache_ranged.py -v
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.paged_cache import BlockManager, PagedLlamaKVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _mini_config(n_layer: int = 4) -> LlamaConfig:
    return LlamaConfig(
        name="test-mini-ranged",
        vocab_size=256,
        n_ctx=128,
        d_model=64,
        n_layer=n_layer,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
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
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

def test_default_owned_layers_is_full_range():
    """No owned_layers passed → identical to pre-Phase-2 behavior."""
    cfg = _mini_config(n_layer=4)
    mgr = BlockManager(n_total=16, block_size=8)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)

    assert cache.owned_layers == range(4)
    assert cache.layer_offset == 0
    assert cache.k_pool.shape[0] == 4


def test_ranged_pool_sized_to_owned_layers_not_full_model():
    cfg = _mini_config(n_layer=8)
    mgr = BlockManager(n_total=16, block_size=8)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE, owned_layers=range(3, 6))

    assert cache.owned_layers == range(3, 6)
    assert cache.layer_offset == 3
    assert cache.k_pool.shape[0] == 3
    assert cache.v_pool.shape[0] == 3


def test_extend_rejects_layer_outside_owned_range():
    cfg = _mini_config(n_layer=8)
    mgr = BlockManager(n_total=16, block_size=8)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE, owned_layers=range(3, 6))
    cache.allocate_sequence(0, 1)
    cache.begin_step([0])

    n_kv, d = cfg.n_kv_heads, cfg.head_dim
    k = torch.randn(1, n_kv, 1, d, device=DEVICE)
    v = torch.randn(1, n_kv, 1, d, device=DEVICE)

    with pytest.raises(ValueError, match="not owned"):
        cache.extend(layer=6, k_new=k, v_new=v)
    with pytest.raises(ValueError, match="not owned"):
        cache.extend(layer=2, k_new=k, v_new=v)


# ---------------------------------------------------------------------------
# memory_bytes() shrinks proportionally — asserted directly, not just parity
# ---------------------------------------------------------------------------

def test_ranged_cache_memory_proportional_to_owned_layers():
    cfg = _mini_config(n_layer=8)
    mgr_full = BlockManager(n_total=32, block_size=8)
    mgr_ranged = BlockManager(n_total=32, block_size=8)
    full = PagedLlamaKVCache(cfg, mgr_full, DEVICE, DTYPE)
    ranged = PagedLlamaKVCache(cfg, mgr_ranged, DEVICE, DTYPE, owned_layers=range(2))  # 2 of 8 layers

    assert ranged.memory_bytes() == pytest.approx(full.memory_bytes() * (2 / 8))


# ---------------------------------------------------------------------------
# End-to-end: splitting the KV cache across two ranged caches (one per
# pipeline stage) must reproduce the same tokens as a single full-range cache,
# across both prefill and multiple cached decode steps.
# ---------------------------------------------------------------------------

def test_split_ranged_caches_match_single_full_cache_multistep():
    cfg = _mini_config(n_layer=4)
    split_layer = 2

    model_ref = _mini_model(cfg, seed=7)
    model_split = _mini_model(cfg, seed=7)  # identical weights (same seed)

    mgr_ref = BlockManager(n_total=32, block_size=4)
    cache_ref = PagedLlamaKVCache(cfg, mgr_ref, DEVICE, DTYPE)

    mgr_a = BlockManager(n_total=32, block_size=4)
    mgr_b = BlockManager(n_total=32, block_size=4)
    cache_a = PagedLlamaKVCache(cfg, mgr_a, DEVICE, DTYPE, owned_layers=range(0, split_layer))
    cache_b = PagedLlamaKVCache(cfg, mgr_b, DEVICE, DTYPE, owned_layers=range(split_layer, cfg.n_layer))

    T_p = 5
    ids = torch.randint(0, cfg.vocab_size, (1, T_p), device=DEVICE)
    pos = torch.arange(T_p, device=DEVICE)

    cache_ref.allocate_sequence(0, T_p)
    cache_ref.begin_step([0])
    cache_a.allocate_sequence(0, T_p)
    cache_a.begin_step([0])
    cache_b.allocate_sequence(0, T_p)
    cache_b.begin_step([0])

    with torch.no_grad():
        ref_logits = model_ref.forward(ids, cache=cache_ref, start_pos=0, position_ids=pos)
        hidden = model_split.forward_stage(
            ids, 0, split_layer, True, False, cache=cache_a, start_pos=0, position_ids=pos
        )
        split_logits = model_split.forward_stage(
            hidden, split_layer, cfg.n_layer, False, True, cache=cache_b, start_pos=0, position_ids=pos
        )

    assert torch.equal(ref_logits, split_logits)

    # Two more cached decode steps — the trickiest bookkeeping (start_pos > 0,
    # ensure_slot on both ranged caches independently).
    next_tok = ref_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start_pos = T_p
    for step in range(2):
        pos_s = torch.tensor([start_pos], device=DEVICE)

        cache_ref.ensure_slot(0)
        cache_ref.begin_step([0])
        cache_a.ensure_slot(0)
        cache_a.begin_step([0])
        cache_b.ensure_slot(0)
        cache_b.begin_step([0])

        with torch.no_grad():
            ref_logits = model_ref.forward(next_tok, cache=cache_ref, start_pos=start_pos, position_ids=pos_s)
            hidden = model_split.forward_stage(
                next_tok, 0, split_layer, True, False, cache=cache_a, start_pos=start_pos, position_ids=pos_s
            )
            split_logits = model_split.forward_stage(
                hidden, split_layer, cfg.n_layer, False, True, cache=cache_b, start_pos=start_pos, position_ids=pos_s
            )

        assert torch.equal(ref_logits, split_logits), f"diverged at step {step}"

        next_tok = ref_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start_pos += 1

    # Each ranged cache's seq_lens tracking is independent and both must have
    # advanced correctly (this is the state a Phase-4 coordinator will read
    # to decide when to ensure_slot / free_sequence per stage).
    assert cache_a.seq_lens[0] == cache_ref.seq_lens[0]
    assert cache_b.seq_lens[0] == cache_ref.seq_lens[0]
