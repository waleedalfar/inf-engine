"""Phase 3 (scale-up) paged KV cache correctness tests.

Tests BlockManager allocation/free, PagedLlamaKVCache write/gather equivalence
against LlamaStaticKVCache for single-sequence generation, block reuse after
sequence free, and LlamaPagedEngine offline batching.

No weights file required — all tests use a synthetic mini-LLaMA.

Run:
    pytest tests/test_paged_cache.py -v -s
"""

from __future__ import annotations

import pytest
import torch

from engine.config import LlamaConfig
from engine.paged_cache import BlockManager, PagedLlamaKVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_config() -> LlamaConfig:
    return LlamaConfig(
        name="test-mini",
        vocab_size=256,
        n_ctx=128,
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
    )


def _mini_model(config: LlamaConfig, device: str = DEVICE):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(99)

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


# ---------------------------------------------------------------------------
# BlockManager tests
# ---------------------------------------------------------------------------

def test_block_manager_allocate_and_free():
    mgr = BlockManager(n_total=8, block_size=16)
    assert mgr.n_free == 8
    ids = mgr.allocate(3)
    assert len(ids) == 3
    assert mgr.n_free == 5
    mgr.free(ids)
    assert mgr.n_free == 8


def test_block_manager_oom():
    mgr = BlockManager(n_total=4, block_size=16)
    with pytest.raises(RuntimeError, match="OOM"):
        mgr.allocate(5)


def test_block_manager_reuse():
    mgr = BlockManager(n_total=4, block_size=16)
    ids = mgr.allocate(4)
    assert mgr.n_free == 0
    mgr.free(ids[:2])
    assert mgr.n_free == 2
    new_ids = mgr.allocate(2)
    # freed blocks come back
    assert sorted(new_ids) == sorted(ids[:2])


def test_blocks_needed():
    mgr = BlockManager(n_total=100, block_size=16)
    assert mgr.blocks_needed(0) == 0
    assert mgr.blocks_needed(1) == 1
    assert mgr.blocks_needed(16) == 1
    assert mgr.blocks_needed(17) == 2
    assert mgr.blocks_needed(32) == 2
    assert mgr.blocks_needed(33) == 3


# ---------------------------------------------------------------------------
# PagedLlamaKVCache tests
# ---------------------------------------------------------------------------

def test_paged_single_token_write():
    """Writing one token and gathering returns the right value."""
    cfg = _mini_config()
    mgr = BlockManager(n_total=16, block_size=8)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)
    cache.allocate_sequence(0, 1)
    cache.begin_step([0])

    n_kv, d = cfg.n_kv_heads, cfg.head_dim
    k = torch.randn(1, n_kv, 1, d)
    v = torch.randn(1, n_kv, 1, d)

    # Only need to call extend for one layer to test write+gather
    k_out, v_out = cache.extend(layer=0, k_new=k, v_new=v)
    assert k_out.shape == (1, n_kv, 1, d)
    assert torch.allclose(k_out, k, atol=1e-6)
    assert torch.allclose(v_out, v, atol=1e-6)


def test_paged_multi_token_prefill_spanning_blocks():
    """Prefill spanning multiple blocks: gathered output equals original K/V."""
    cfg = _mini_config()
    mgr = BlockManager(n_total=32, block_size=4)   # small blocks to force spanning
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)

    T = 10  # 10 tokens → 3 blocks (4+4+2)
    cache.allocate_sequence(0, T)
    cache.begin_step([0])

    n_kv, d = cfg.n_kv_heads, cfg.head_dim
    k = torch.randn(1, n_kv, T, d)
    v = torch.randn(1, n_kv, T, d)

    k_out = v_out = None
    for layer in range(cfg.n_layer):
        k_out, v_out = cache.extend(layer, k, v)

    assert k_out.shape == (1, n_kv, T, d)
    assert torch.allclose(k_out, k, atol=1e-5), f"max diff: {(k_out - k).abs().max():.2e}"
    assert torch.allclose(v_out, v, atol=1e-5)


def test_paged_matches_static_greedy_decode():
    """Paged cache must produce token-for-token identical output to LlamaStaticKVCache."""
    from engine.kv_cache import LlamaStaticKVCache
    from engine.sampling import SamplingConfig, SamplingMode

    cfg = _mini_config()
    model = _mini_model(cfg)

    torch.manual_seed(5)
    T_p = 7
    max_new = 6
    ids = torch.randint(0, cfg.vocab_size, (1, T_p), device=DEVICE)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    # --- Static reference ---
    static = LlamaStaticKVCache(cfg, batch=1, max_seq=T_p + max_new + 1, device=DEVICE, dtype=DTYPE)
    ref = model.generate_cached(ids, max_new, static, greedy)

    # --- Paged: manually replicate generate_cached ---
    mgr = BlockManager(n_total=64, block_size=4)
    paged = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)
    paged.allocate_sequence(0, T_p)
    paged.begin_step([0])

    pos = torch.arange(T_p)
    logits = model.forward(ids, cache=paged, start_pos=0, position_ids=pos)
    next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
    out = torch.cat([ids, next_tok], dim=1)

    for step in range(1, max_new):
        start = T_p + step - 1
        paged.ensure_slot(0)
        paged.begin_step([0])
        pos_s = torch.tensor([start])
        logits = model.forward(next_tok, cache=paged, start_pos=start, position_ids=pos_s)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)

    assert torch.equal(out, ref), (
        f"Paged diverged from static:\n  paged : {out.tolist()}\n  static: {ref.tolist()}"
    )


def test_block_reuse_after_sequence_free():
    """After seq A finishes, seq B gets the same blocks — no OOM on a tight pool."""
    cfg = _mini_config()
    mgr = BlockManager(n_total=4, block_size=8)   # exactly 4 blocks = 32 token slots
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)

    # Seq A occupies all 4 blocks
    cache.allocate_sequence(0, 32)
    assert mgr.n_free == 0

    # Free A → pool is restored
    cache.free_sequence(0)
    assert mgr.n_free == 4

    # Seq B can now allocate
    cache.allocate_sequence(1, 16)
    assert mgr.n_free == 2


def test_duplicate_seq_id_raises():
    """Allocating the same seq_id twice without freeing must raise."""
    cfg = _mini_config()
    mgr = BlockManager(n_total=16, block_size=8)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)
    cache.allocate_sequence(0, 4)
    with pytest.raises(ValueError, match="already allocated"):
        cache.allocate_sequence(0, 4)


def test_ensure_slot_allocates_on_block_boundary():
    """ensure_slot must add a block exactly when seq_len is a multiple of block_size."""
    cfg = _mini_config()
    bs = 4
    mgr = BlockManager(n_total=16, block_size=bs)
    cache = PagedLlamaKVCache(cfg, mgr, DEVICE, DTYPE)

    cache.allocate_sequence(0, bs)   # allocates 1 block
    n_before = mgr.n_free

    # Simulate filling that block via seq_lens
    cache.seq_lens[0] = bs           # block is now full
    cache.ensure_slot(0)             # must allocate 1 new block
    assert mgr.n_free == n_before - 1

    cache.seq_lens[0] = bs + 1       # not a boundary
    n_mid = mgr.n_free
    cache.ensure_slot(0)             # should NOT allocate
    assert mgr.n_free == n_mid


# ---------------------------------------------------------------------------
# LlamaPagedEngine tests
# ---------------------------------------------------------------------------

def test_paged_engine_offline_lengths():
    """Engine processes N requests and returns the right number of generated tokens."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest

    cfg = _mini_config()
    model = _mini_model(cfg)
    engine = LlamaPagedEngine(model, n_total_blocks=64, block_size=8, eos_token=None)

    requests = [
        LlamaRequest(req_id=0, prompt_ids=[1, 2, 3],       max_new_tokens=5),
        LlamaRequest(req_id=1, prompt_ids=[4, 5],           max_new_tokens=3),
        LlamaRequest(req_id=2, prompt_ids=[6, 7, 8, 9],    max_new_tokens=4),
    ]
    results = engine.run_offline(requests)

    assert set(results.keys()) == {0, 1, 2}
    assert len(results[0]) == 5
    assert len(results[1]) == 3
    assert len(results[2]) == 4


def test_paged_engine_block_reclaim():
    """Blocks freed by finished sequences are reclaimed for new ones (tight pool)."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest

    cfg = _mini_config()
    model = _mini_model(cfg)
    # Pool only large enough for one request at a time (prompt 2 tokens → 1 block,
    # but we give a few extra to cover decode growth)
    engine = LlamaPagedEngine(model, n_total_blocks=12, block_size=8, eos_token=None)

    # Submit 4 requests sequentially; pool reuse means all complete despite small pool
    requests = [
        LlamaRequest(req_id=i, prompt_ids=[i + 1, i + 2], max_new_tokens=2)
        for i in range(4)
    ]
    results = engine.run_offline(requests)
    assert len(results) == 4


def test_paged_engine_eos_stops_early():
    """Engine respects eos_token and stops before max_new_tokens."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.sampling import SamplingConfig, SamplingMode

    cfg = _mini_config()
    model = _mini_model(cfg)

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)
    pos = torch.arange(4, device=DEVICE)
    with torch.no_grad():
        logits = model.forward(ids, start_pos=0, position_ids=pos)
    # Find what token the model would predict first (greedy)
    first_predicted = int(logits[0, -1, :].argmax())

    # Use that token as EOS — engine should stop after 1 token
    engine = LlamaPagedEngine(
        model, n_total_blocks=32, block_size=8,
        eos_token=first_predicted,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
    )
    req = LlamaRequest(req_id=0, prompt_ids=ids[0].tolist(), max_new_tokens=10)
    results = engine.run_offline([req])
    assert len(results[0]) == 1
    assert results[0][0] == first_predicted
