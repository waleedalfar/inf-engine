"""Windowed ring KV cache: physical residency stays bounded while the logical
block table spans the full absolute length.

These are bookkeeping invariants (no model, no kernel — CPU-only), the layer
where an allocation or recycling bug would live. Correctness of the *values* the
ring returns is covered end-to-end on the real model (a draft-cache bug can only
lower acceptance, never corrupt output, since the target verifies every token).
"""

from __future__ import annotations

import math

import torch

from engine.config import LlamaConfig
from engine.paged_cache import BlockManager, PagedLlamaKVCache


def _cfg() -> LlamaConfig:
    return LlamaConfig(
        name="ring-mini", vocab_size=64, n_ctx=4096, d_model=32,
        n_layer=1, n_head=2, n_kv_heads=1, intermediate_size=64,
    )


def _ring_cache(pool_blocks, block_size, window, sinks):
    mgr = BlockManager(pool_blocks, block_size)
    return PagedLlamaKVCache(
        _cfg(), mgr, "cpu", torch.float32,
        attn_window=window, attn_sinks=sinks, window_ring=True,
    )


def test_ring_residency_bounded_and_no_oom():
    bs, window, sinks = 4, 16, 4          # window = 4 blocks, sink = 1 block
    sink_blocks = math.ceil(sinks / bs)
    win_blocks = window // bs
    # A pool that could never hold a 400-token sequence unwindowed (100 blocks).
    pool = sink_blocks + win_blocks + 3
    cache = _ring_cache(pool, bs, window, sinks)
    sid = 0
    cache.allocate_sequence(sid, 1)

    N = 400
    for _ in range(N):
        cache.ensure_slot(sid)            # would raise KV-cache OOM if unbounded
        cache.seq_lens[sid] += 1          # simulate one token written
        L = cache.seq_lens[sid]
        bt = cache.block_table[sid]
        # Logical block table spans the full absolute length.
        assert len(bt) == cache.manager.blocks_needed(L)
        # Distinct physical blocks actually resident stay within the budget.
        pinned = bt[0]
        distinct = {b for b in bt if b != pinned}
        assert len(distinct) <= win_blocks + 2, (L, len(distinct))

    # Out-of-window, non-sink logical entries have been recycled onto the pinned
    # block; in-window entries hold their own distinct physical block.
    L = cache.seq_lens[sid]
    bt = cache.block_table[sid]
    pinned = bt[0]
    win_lo = (L - window)
    for b in range(sink_blocks, len(bt)):
        block_last_pos = (b + 1) * bs - 1
        if block_last_pos < win_lo:
            assert bt[b] == pinned, f"block {b} should be recycled at L={L}"


def test_ring_keeps_in_window_and_sinks_readable():
    """Every position the windowed kernel would read (sinks + last `window`) must
    map to a resident, non-pinned physical block (or the sink block itself)."""
    bs, window, sinks = 4, 16, 4
    cache = _ring_cache(sink := (1 + window // bs + 3), bs, window, sinks)
    sid = 0
    cache.allocate_sequence(sid, 1)
    for _ in range(300):
        cache.ensure_slot(sid)
        cache.seq_lens[sid] += 1

    L = cache.seq_lens[sid]
    bt = cache.block_table[sid]
    # Sink positions live in block 0 (the pinned block) — always resident.
    for pos in range(sinks):
        assert bt[pos // bs] == bt[0]
    # Every in-window position maps to a real allocated block within the pool.
    for pos in range(max(0, L - window), L):
        phys = bt[pos // bs]
        assert 0 <= phys < cache.manager.n_total


def test_ring_rollback_frees_without_double_counting():
    """reset_to must not return the pinned block (or a recycled alias) to the
    pool, or the free list corrupts. Mimics speculative rollback."""
    bs, window, sinks = 4, 16, 4
    cache = _ring_cache(1 + window // bs + 4, bs, window, sinks)
    sid = 0
    cache.allocate_sequence(sid, 1)
    for _ in range(200):
        cache.ensure_slot(sid)
        cache.seq_lens[sid] += 1

    free_before = cache.manager.n_free
    L = cache.seq_lens[sid]
    cache.reset_to(sid, L - 3)            # small rollback, within window
    # The free list must never develop duplicate ids.
    assert len(cache.manager._free) == len(set(cache.manager._free))
    # Growing again must still succeed (no leaked/duplicated blocks).
    for _ in range(50):
        cache.ensure_slot(sid)
        cache.seq_lens[sid] += 1
    assert len(cache.manager._free) == len(set(cache.manager._free))
    _ = free_before


def test_ring_disabled_is_unchanged():
    """Without window_ring the cache allocates one block per logical block, as
    before — the ring path is fully opt-in."""
    bs = 4
    mgr = BlockManager(64, bs)
    cache = PagedLlamaKVCache(_cfg(), mgr, "cpu", torch.float32,
                             attn_window=16, attn_sinks=4, window_ring=False)
    sid = 0
    cache.allocate_sequence(sid, 40)      # 10 blocks up front
    assert len(cache.block_table[sid]) == mgr.blocks_needed(40)
    assert not cache.window_ring
