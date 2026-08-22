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
    Q, KPool, VPool, KScale, VScale, BlockTable, KVLens,
    OutAcc, OutLse,
    scale,
    stride_qb, stride_qh, stride_qt,
    stride_kp_blk, stride_kp_h, stride_kp_pos,
    stride_ks_blk, stride_ks_h, stride_ks_pos,
    stride_bt_b,
    stride_oa_b, stride_oa_h, stride_oa_s, stride_oa_m,
    stride_ol_b, stride_ol_h, stride_ol_s,
    stride_o_b, stride_o_h, stride_o_t,
    q_len, n_rep, split_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE: tl.constexpr,
    SINGLE_SPLIT: tl.constexpr,
    QUANT_KV: tl.constexpr,
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
    # BLOCK_D is padded up to tl.dot's minimum contraction width (16), so the
    # tail columns are inactive rather than real channels.
    d_valid = offs_d < head_dim
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
    # Loaded in the pool's native dtype: tl.dot accumulates in fp32 regardless,
    # so bf16 operands hit bf16 tensor cores at full accumulator precision
    # instead of being widened to fp32 (which also silently selects TF32).
    q = tl.load(q_ptr, mask=row_valid[:, None] & d_valid[None, :], other=0.0)

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
        kv_mask = in_range[:, None] & d_valid[None, :]
        k = tl.load(KPool + kv_off, mask=kv_mask, other=0.0)
        v = tl.load(VPool + kv_off, mask=kv_mask, other=0.0)

        if QUANT_KV:
            # Per-(position, head) scales, so dequantization folds into work we
            # already do: K's scale is a per-key column factor on the scores,
            # and V's is a per-row factor applied before the second dot. Neither
            # needs a separate pass over the tile.
            s_off = (blk * stride_ks_blk + h_kv * stride_ks_h + off * stride_ks_pos)
            ks = tl.load(KScale + s_off, mask=in_range, other=0.0)
            vs = tl.load(VScale + s_off, mask=in_range, other=0.0)
            # Widen to the query dtype, not fp32: an 8-bit code fits bf16
            # exactly, and the dot stays on bf16 tensor cores. Going through
            # fp32 with allow_tf32=False drops off tensor cores entirely and
            # measured slower than not quantizing at all.
            k = k.to(q.dtype)
            v = (v.to(tl.float32) * vs[:, None]).to(q.dtype)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False).to(tl.float32)
            qk = qk * ks[None, :] * scale
        else:
            qk = tl.dot(q, tl.trans(k), allow_tf32=False).to(tl.float32) * scale                      # (BLOCK_M, BLOCK_N)
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
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=False).to(tl.float32)
        m_i = m_new

    # Store the split's *normalized* output plus its log-sum-exp, so the combine
    # is a plain lse-weighted average.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    if SINGLE_SPLIT:
        # Nothing to merge — write the final answer and skip the combine launch.
        # At short context that launch is a large share of total attention time.
        op = (OutAcc + b * stride_o_b + (h_kv * n_rep + rep)[:, None] * stride_o_h
              + tok[:, None] * stride_o_t + offs_d[None, :])
        tl.store(op, acc.to(OutAcc.dtype.element_ty),
                 mask=row_valid[:, None] & d_valid[None, :])
    else:
        lse = tl.where(l_i == 0.0, NEG_INF, m_i + tl.log(l_safe))
        oa = (OutAcc + b * stride_oa_b + h_kv * stride_oa_h + s * stride_oa_s
              + offs_m[:, None] * stride_oa_m + offs_d[None, :])
        tl.store(oa, acc, mask=row_valid[:, None] & d_valid[None, :])
        ol = OutLse + b * stride_ol_b + h_kv * stride_ol_h + s * stride_ol_s + offs_m
        tl.store(ol, lse, mask=row_valid)


@triton.jit
def _combine_splits_kernel(
    InAcc, InLse, Out,
    stride_ia_b, stride_ia_h, stride_ia_s, stride_ia_m,
    stride_il_b, stride_il_h, stride_il_s,
    stride_ob, stride_oh, stride_ot,
    q_len, n_rep, head_dim,
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
    d_valid = offs_d < head_dim
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
        a = tl.load(ap, mask=row_valid[:, None] & d_valid[None, :], other=0.0)
        ws = tl.load(InLse + b * stride_il_b + h_kv * stride_il_h + s * stride_il_s + offs_m,
                     mask=row_valid, other=NEG_INF)
        acc += a * tl.exp(ws - m_safe)[:, None]
    acc = acc / denom[:, None]

    op = (Out + b * stride_ob + (h_kv * n_rep + rep)[:, None] * stride_oh
          + tok[:, None] * stride_ot + offs_d[None, :])
    tl.store(op, acc.to(Out.dtype.element_ty), mask=row_valid[:, None] & d_valid[None, :])


def _pick_splits(kv_len: int, n_ctas_per_split: int, target_ctas: int = 512) -> int:
    """Enough KV splits to fill the GPU, capped so each split still has real work."""
    if kv_len <= 0:
        return 1
    want = max(1, target_ctas // max(n_ctas_per_split, 1))
    # Never more splits than 128-position chunks — beyond that the combine
    # dominates and each split reads too little to amortize its launch.
    return max(1, min(want, triton.cdiv(kv_len, 128), 64))


_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _scratch(key: tuple, acc_shape: tuple, lse_shape: tuple, device) -> tuple:
    """Reusable per-shape workspace for the split partials.

    Allocating these per call showed up as real cost at short context, where
    there is little KV to read. Holding them in a module-level cache is also
    what CUDA-graph capture wants: the pointers are baked into the graph at
    capture time, and a live reference here keeps them from being freed.
    """
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (torch.empty(acc_shape, device=device, dtype=torch.float32),
               torch.empty(lse_shape, device=device, dtype=torch.float32))
        _SCRATCH[key] = buf
    return buf


def paged_flash_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
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
        k_scale:     ``(n_blocks, n_kv, page_size)`` dequantization scales when
                     the pools hold 8-bit codes (INT8 or FP8). Required together
                     with ``v_scale``; omit both for a bf16/fp32 pool.
        v_scale:     Same shape as ``k_scale``.
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
    quant = k_scale is not None
    assert quant == (v_scale is not None), "k_scale and v_scale must both be given"
    if not quant:
        # Kernel arguments must still be valid pointers; reuse the pools.
        k_scale = v_scale = k_pool

    rows = n_rep * q_len
    BLOCK_M = max(16, triton.next_power_of_2(rows))
    # tl.dot requires a contraction dim of at least 16.
    BLOCK_D = max(16, triton.next_power_of_2(D))

    if n_splits is None:
        n_splits = _pick_splits(int(kv_lens.max().item()), B * n_kv)
    max_len = block_table.shape[1] * page_size
    split_len = triton.cdiv(triton.cdiv(max_len, n_splits), block_n) * block_n

    single = n_splits == 1
    acc, lse = _scratch(
        (q.device, B, n_kv, n_splits, BLOCK_M, BLOCK_D),
        (B, n_kv, n_splits, BLOCK_M, BLOCK_D), (B, n_kv, n_splits, BLOCK_M), q.device,
    )
    out = torch.empty_like(q)

    _paged_flash_decode_kernel[(B, n_kv, n_splits)](
        q, k_pool, v_pool, k_scale, v_scale, block_table, kv_lens,
        out if single else acc, lse,
        D ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        block_table.stride(0),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        q_len, n_rep, split_len, D,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_D=BLOCK_D, PAGE=page_size,
        SINGLE_SPLIT=single, QUANT_KV=quant, num_warps=4, num_stages=2,
    )
    if single:
        return out
    _combine_splits_kernel[(B, n_kv)](
        acc, lse, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        q_len, n_rep, D,
        N_SPLITS=n_splits, SPLITS_POW2=triton.next_power_of_2(n_splits),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Fused quantized KV write
# ---------------------------------------------------------------------------

@triton.jit
def _write_kv_quant_kernel(
    KNew, VNew, KPool, VPool, KScale, VScale, BlockTable, SeqLens,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_pb, stride_ph, stride_pp,
    stride_sb, stride_sh, stride_sp,
    stride_bt_b,
    head_dim,
    PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QMAX: tl.constexpr,
    IS_INT: tl.constexpr,
):
    """One program per (sequence, kv_head, new token): quantize and scatter.

    Doing this with torch ops costs ~10 kernels per layer (two amax reductions,
    two divides, two casts, and four scatters) and measured *slower* than the
    bf16 path it was meant to beat — the write overhead swamped the read saving.
    Here the whole thing is one launch and the scale never leaves registers
    until its single store.
    """
    a = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)

    pos = tl.load(SeqLens + a) + t
    blk = tl.load(BlockTable + a * stride_bt_b + pos // PAGE)
    off = pos % PAGE

    d = tl.arange(0, BLOCK_D)
    dm = d < head_dim
    # K and V are separate tensors with independently derived layouts — K goes
    # through QK-norm and RoPE, V does not — so they do NOT share strides.
    # Reusing K's for both silently corrupts V at every token past the first.
    k_src = a * stride_kb + h * stride_kh + t * stride_kt + d
    v_src = a * stride_vb + h * stride_vh + t * stride_vt + d
    dst = blk * stride_pb + h * stride_ph + off * stride_pp + d
    s_at = blk * stride_sb + h * stride_sh + off * stride_sp

    k = tl.load(KNew + k_src, mask=dm, other=0.0).to(tl.float32)
    v = tl.load(VNew + v_src, mask=dm, other=0.0).to(tl.float32)
    # amax/QMAX puts the largest magnitude exactly at the format's limit, so
    # nothing saturates and the full code range is in use.
    ks = tl.maximum(tl.max(tl.abs(k), axis=0) / QMAX, 1e-12)
    vs = tl.maximum(tl.max(tl.abs(v), axis=0) / QMAX, 1e-12)
    kq = k / ks
    vq = v / vs
    if IS_INT:
        # Triton's float->int cast truncates; round half away from zero, then
        # clamp so a boundary case cannot wrap.
        kq = tl.where(kq >= 0, tl.floor(kq + 0.5), tl.ceil(kq - 0.5))
        vq = tl.where(vq >= 0, tl.floor(vq + 0.5), tl.ceil(vq - 0.5))
        kq = tl.minimum(tl.maximum(kq, -QMAX), QMAX)
        vq = tl.minimum(tl.maximum(vq, -QMAX), QMAX)

    tl.store(KPool + dst, kq.to(KPool.dtype.element_ty), mask=dm)
    tl.store(VPool + dst, vq.to(VPool.dtype.element_ty), mask=dm)
    tl.store(KScale + s_at, ks)
    tl.store(VScale + s_at, vs)


def write_kv_quant(
    k_new: torch.Tensor, v_new: torch.Tensor,
    k_pool: torch.Tensor, v_pool: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor,
    block_table: torch.Tensor, seq_lens: torch.Tensor,
    *, page_size: int, qmax: float, is_int: bool,
) -> None:
    """Quantize ``k_new``/``v_new`` to 8-bit and scatter them into the pool.

    Args:
        k_new, v_new: (A, n_kv, q_len, head_dim) in the model dtype.
        k_pool, v_pool: (n_blocks, n_kv, page_size, head_dim) INT8/FP8, one layer.
        k_scale, v_scale: (n_blocks, n_kv, page_size) float32, for one layer.
        block_table: (A, n_blocks_cap) physical block ids.
        seq_lens: (A,) token count *before* this write.
        page_size: Tokens per physical block.
        qmax: Largest representable magnitude of the storage format.
        is_int: Round-to-nearest before storing (integer formats only).
    """
    A, n_kv, q_len, D = k_new.shape
    assert k_new.stride(3) == 1 and v_new.stride(3) == 1, \
        "write_kv_quant requires head_dim to be the contiguous axis"
    _write_kv_quant_kernel[(A, n_kv, q_len)](
        k_new, v_new, k_pool, v_pool, k_scale, v_scale, block_table, seq_lens,
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        block_table.stride(0),
        D,
        PAGE=page_size, BLOCK_D=max(16, triton.next_power_of_2(D)),
        QMAX=qmax, IS_INT=is_int, num_warps=4,
    )
