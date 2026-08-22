"""Paged flash-decoding attention: GQA-native, split-KV, reads the block table directly.

Why this exists
---------------
At decode and speculative-verify time the query is 1-5 rows. PyTorch SDPA is
~25x off the memory-bandwidth floor at those shapes, for three compounding
reasons, none of which a different SDPA call fixes:

1. The flash backend rejects arbitrary ``attn_mask``, so any offset-causal or
   continuous-batching mask lands on the mem-efficient backend.
2. ``enable_gqa`` is unusable together with ``attn_mask`` on that backend, so
   K/V must be expanded from ``n_kv`` heads to ``n_head`` first — a 4x
   materialization of the single largest tensor in the step.
3. A 1-5 row query occupies a handful of CTAs. The GPU sits mostly idle.

This kernel answers all three:

* **GQA-native.** One program owns one KV head and *all* ``n_rep`` query heads
  that share it, so K/V are read once and reused across the group. No expansion
  ever exists.
* **Split over the KV axis** (flash-decoding). The KV history is partitioned
  into ``n_splits`` chunks that run concurrently and are merged by a
  log-sum-exp combine. This is what lets a 5-row query saturate 70 SMs, and it
  is the property the whole design rests on.
* **Paged.** K/V are gathered from the physical pool through the block table
  inside the kernel. Nothing is materialized into a contiguous ``(1, n_kv,
  len_bucket, D)`` tensor first, so the gather traffic disappears and — because
  splits past the true length exit immediately — CUDA-graph bucket padding
  costs nothing rather than up to 2x.

Layout
------
Query rows within a program are ``(rep, token)`` flattened::

    row r  ->  rep = r // q_len,  tok = r % q_len
    query head = kv_head * n_rep + rep
    absolute position of that query = (kv_len - q_len) + tok

Masking
-------
Key ``j`` is admitted iff ``j <= (kv_len - q_len) + tok``. Because
``tok <= q_len - 1``, that bound is always ``< kv_len``, so the causal bound
subsumes the true-length bound and padded positions are excluded automatically.
Getting this wrong is the dominant failure mode for kernels of this shape — see
``.claude/skills/attention-causal-mask-audit`` before editing any of it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# A split whose every position is masked contributes nothing. It must still
# produce a well-defined (0 numerator, -inf lse) partial rather than NaN.
# constexpr so the @triton.jit kernels can reference it.
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _paged_flash_decode_kernel(
    Q, KPool, VPool, BlockTable, KVLens,
    OutAcc, OutLse,
    scale,
    stride_qb, stride_qh, stride_qt,
    stride_kp_blk, stride_kp_h, stride_kp_pos,
    stride_bt_b,
    stride_oa_b, stride_oa_h, stride_oa_s, stride_oa_m,
    stride_ol_b, stride_ol_h, stride_ol_s,
    q_len, n_rep, split_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE: tl.constexpr,
):
    """One program = (batch, kv_head, kv_split). Emits a normalized partial + lse."""
    b = tl.program_id(0)
    h_kv = tl.program_id(1)
    s = tl.program_id(2)

    kv_len = tl.load(KVLens + b)
    start = s * split_len
    end = tl.minimum(start + split_len, kv_len)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    rep = offs_m // q_len
    tok = offs_m % q_len
    row_valid = offs_m < q_len * n_rep
    # Absolute position of each query row; masked-off rows get -1 so every key
    # fails the causal test for them.
    q_pos = tl.where(row_valid, (kv_len - q_len) + tok, -1)

    # --- load this program's query rows: head = h_kv*n_rep + rep -------------
    q_ptr = (Q + b * stride_qb
             + (h_kv * n_rep + rep)[:, None] * stride_qh
             + tok[:, None] * stride_qt
             + offs_d[None, :])
    q = tl.load(q_ptr, mask=row_valid[:, None], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for p0 in range(start, end, BLOCK_N):
        pos = p0 + tl.arange(0, BLOCK_N)
        in_range = pos < end
        # Paged gather: position -> (physical block, offset within block).
        blk = tl.load(BlockTable + b * stride_bt_b + pos // PAGE,
                      mask=in_range, other=0)
        off = pos % PAGE
        kv_off = (blk[:, None] * stride_kp_blk
                  + h_kv * stride_kp_h
                  + off[:, None] * stride_kp_pos
                  + offs_d[None, :])
        k = tl.load(KPool + kv_off, mask=in_range[:, None], other=0.0).to(tl.float32)
        v = tl.load(VPool + kv_off, mask=in_range[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * scale                      # (BLOCK_M, BLOCK_N)
        # Offset-causal: key j admitted iff j <= q_pos. Subsumes j < kv_len.
        admit = in_range[None, :] & (pos[None, :] <= q_pos[:, None])
        qk = tl.where(admit, qk, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # An all-masked tile leaves m_new at -inf; exp(-inf - -inf) is NaN, so
        # rebase on a finite value. Both exps then underflow to 0, which is the
        # arithmetically correct contribution for a tile that admits nothing.
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Store the split's *normalized* output plus its log-sum-exp, so the combine
    # is a plain lse-weighted average.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = tl.where(l_i == 0.0, NEG_INF, m_i + tl.log(l_safe))

    oa = (OutAcc + b * stride_oa_b + h_kv * stride_oa_h + s * stride_oa_s
          + offs_m[:, None] * stride_oa_m + offs_d[None, :])
    tl.store(oa, acc, mask=row_valid[:, None])
    ol = OutLse + b * stride_ol_b + h_kv * stride_ol_h + s * stride_ol_s + offs_m
    tl.store(ol, lse, mask=row_valid)


@triton.jit
def _combine_splits_kernel(
    InAcc, InLse, Out,
    stride_ia_b, stride_ia_h, stride_ia_s, stride_ia_m,
    stride_il_b, stride_il_h, stride_il_s,
    stride_ob, stride_oh, stride_ot,
    q_len, n_rep,
    N_SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge per-split partials by log-sum-exp weighting."""
    b = tl.program_id(0)
    h_kv = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    # tl.arange needs a power-of-two extent, so round up and mask the tail —
    # split counts are not generally powers of two.
    offs_s = tl.arange(0, SPLITS_POW2)
    split_valid = offs_s < N_SPLITS
    rep = offs_m // q_len
    tok = offs_m % q_len
    row_valid = offs_m < q_len * n_rep

    lse_ptr = (InLse + b * stride_il_b + h_kv * stride_il_h
               + offs_s[None, :] * stride_il_s + offs_m[:, None])
    lse = tl.load(lse_ptr, mask=row_valid[:, None] & split_valid[None, :],
                  other=NEG_INF)                                     # (BLOCK_M, SPLITS_POW2)

    m = tl.max(lse, axis=1)
    m_safe = tl.where(m == NEG_INF, 0.0, m)                          # all-empty row guard
    w = tl.exp(lse - m_safe[:, None])                                # (BLOCK_M, N_SPLITS)
    denom = tl.sum(w, axis=1)
    denom = tl.where(denom == 0.0, 1.0, denom)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for s in range(N_SPLITS):
        ap = (InAcc + b * stride_ia_b + h_kv * stride_ia_h + s * stride_ia_s
              + offs_m[:, None] * stride_ia_m + offs_d[None, :])
        a = tl.load(ap, mask=row_valid[:, None], other=0.0)
        ws = tl.load(InLse + b * stride_il_b + h_kv * stride_il_h + s * stride_il_s + offs_m,
                     mask=row_valid, other=NEG_INF)
        acc += a * tl.exp(ws - m_safe)[:, None]
    acc = acc / denom[:, None]

    op = (Out + b * stride_ob + (h_kv * n_rep + rep)[:, None] * stride_oh
          + tok[:, None] * stride_ot + offs_d[None, :])
    tl.store(op, acc.to(Out.dtype.element_ty), mask=row_valid[:, None])


def _pick_splits(kv_len: int, n_ctas_per_split: int, target_ctas: int = 512) -> int:
    """Enough KV splits to fill the GPU, capped so each split still has real work."""
    if kv_len <= 0:
        return 1
    want = max(1, target_ctas // max(n_ctas_per_split, 1))
    # Never more splits than 128-position chunks — beyond that the combine
    # dominates and each split reads too little to amortize its launch.
    return max(1, min(want, triton.cdiv(kv_len, 128), 64))


def paged_flash_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int,
    n_splits: int | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """GQA-native paged attention over a KV history held in physical blocks.

    Args:
        q:           Queries, ``(B, n_head, q_len, D)``. Query row ``t`` is at
                     absolute position ``kv_lens[b] - q_len + t`` — i.e. the new
                     tokens' K/V must already be written into the pool.
        k_pool:      ``(n_blocks, n_kv, page_size, D)`` for this layer.
        v_pool:      Same shape as ``k_pool``.
        block_table: ``(B, max_blocks)`` physical block id per logical block.
        kv_lens:     ``(B,)`` valid KV positions per sequence, *including* the
                     ``q_len`` just written.
        page_size:   Tokens per physical block.
        n_splits:    KV splits. Must be a Python int (constant per CUDA-graph
                     capture); defaults to a heuristic from ``kv_lens.max()``.
        block_n:     KV positions processed per inner iteration.

    Returns:
        ``(B, n_head, q_len, D)``, same dtype as ``q``.
    """
    B, n_head, q_len, D = q.shape
    n_blocks, n_kv, page, dk = k_pool.shape
    assert dk == D, f"head_dim mismatch: q={D} pool={dk}"
    assert page == page_size, f"page_size={page_size} != pool page {page}"
    assert n_head % n_kv == 0, f"n_head={n_head} not divisible by n_kv={n_kv}"
    n_rep = n_head // n_kv

    rows = n_rep * q_len
    BLOCK_M = max(16, triton.next_power_of_2(rows))
    BLOCK_D = triton.next_power_of_2(D)

    if n_splits is None:
        n_splits = _pick_splits(int(kv_lens.max().item()), B * n_kv)
    max_len = block_table.shape[1] * page_size
    split_len = triton.cdiv(triton.cdiv(max_len, n_splits), block_n) * block_n

    acc = torch.empty((B, n_kv, n_splits, BLOCK_M, D), device=q.device, dtype=torch.float32)
    lse = torch.empty((B, n_kv, n_splits, BLOCK_M), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    _paged_flash_decode_kernel[(B, n_kv, n_splits)](
        q, k_pool, v_pool, block_table, kv_lens,
        acc, lse,
        D ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        block_table.stride(0),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        q_len, n_rep, split_len,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=BLOCK_D, PAGE=page_size,
        num_warps=4, num_stages=2,
    )
    _combine_splits_kernel[(B, n_kv)](
        acc, lse, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        q_len, n_rep,
        N_SPLITS=n_splits, SPLITS_POW2=triton.next_power_of_2(n_splits),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out
