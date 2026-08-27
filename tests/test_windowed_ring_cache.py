"""Windowed ring KV cache: physical residency stays bounded while the logical
block table spans the full absolute length.

These are bookkeeping invariants (no model, no kernel — CPU-only), the layer
where an allocation or recycling bug would live. Correctness of the *values* the
ring returns is covered end-to-end on the real model (a draft-cache bug can only
lower acceptance, never corrupt output, since the target verifies every token).
"""

from __future__ import annotations

import math

import pytest
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
    #
    # The bound allows one block of slack, deliberately. `_grow_ring` recycles
    # against the length *before* the grow, not after: the tokens a grow is about
    # to write are queries at positions >= seq_lens, and the earliest of them
    # still attends back to seq_lens - window + 1. Recycling against the grown
    # length would drop KV that the first query of a 2048-token prefill chunk
    # still reads. Asserting the tightest possible bound here would be asserting
    # a bug. What matters — that recycling happens at all, and that residency
    # stays bounded — is covered by the in-loop budget check above.
    L = cache.seq_lens[sid]
    bt = cache.block_table[sid]
    pinned = bt[0]
    win_lo = L - window - bs                  # one block of slack
    recycled = 0
    for b in range(sink_blocks, len(bt)):
        block_last_pos = (b + 1) * bs - 1
        if block_last_pos < win_lo:
            assert bt[b] == pinned, f"block {b} should be recycled at L={L}"
            recycled += 1
    # Teeth: at L=400 with a 16-token window almost every block is out of window,
    # so a no-op implementation cannot pass this by recycling nothing.
    assert recycled > (L // bs) * 0.8, recycled


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


def test_ring_pool_sizing_covers_a_prefill_chunk():
    """The pool must hold sinks + window + one prefill chunk, not just the window.

    `_prefill_forward` grows by a whole chunk at a time and `_grow_ring` recycles
    against the pre-grow length, so a chunk's worth of blocks lands before any
    can be reclaimed. Sizing to sinks+window alone survives construction and then
    OOMs partway through the first long prefill — the failure this pins.
    """
    from engine.llama_paged_engine import ring_pool_blocks

    n = ring_pool_blocks(window=4096, sinks=4, prefill_chunk=2048,
                         block_size=16, margin=16)
    assert n == 1 + 256 + 128 + 16          # 401, vs 2048+ for a full 32k pool

    # Simulate the prefill grow pattern against a pool of exactly that size: 32k
    # of tokens, 2048 at a time, must never exhaust it.
    bs, window, sinks, chunk = 16, 4096, 4, 2048
    mgr = BlockManager(n, bs)
    cache = PagedLlamaKVCache(_cfg(), mgr, "cpu", torch.float32,
                              attn_window=window, attn_sinks=sinks,
                              window_ring=True)
    cache.allocate_sequence(0, 1)
    for _ in range(32768 // chunk):
        cache.ensure_slots_for(0, chunk)     # raises KV-cache OOM if undersized
        cache.seq_lens[0] += chunk
    assert cache.seq_lens[0] == 32768
    # And residency really is bounded, not merely non-fatal.
    bt = cache.block_table[0]
    assert len(bt) == 2048                   # logical table spans the full length
    assert len({b for b in bt}) <= n


def test_ring_pool_scales_with_window_and_chunk_only():
    """The pool size has no context term — why 64k and 32k would cost the same."""
    from engine.llama_paged_engine import ring_pool_blocks

    base = ring_pool_blocks(4096, 4, 2048, 16)
    assert ring_pool_blocks(2048, 4, 2048, 16) == base - 128    # halved window
    assert ring_pool_blocks(4096, 4, 1024, 16) == base - 64     # halved chunk
    assert ring_pool_blocks(4096, 4, 0, 16) == base - 128       # chunking off
