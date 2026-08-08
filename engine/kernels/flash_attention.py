"""Flash-Attention forward pass in Triton (Phase 5, Kernel 2).

## The problem with naive attention

Naive attention computes the full score matrix S = QKᵀ (shape N×N per head), softmaxes it,
then multiplies by V. Materializing S costs **O(N²) memory** and, worse, writes N² scores to
HBM and reads them back for the softmax and the SV product — N² HBM traffic that dominates at
long N. Attention is memory-bound, so those round-trips, not the flops, are the bottleneck.

## The tiling strategy (reconstructable from this description)

Flash-Attention never materializes S. It tiles the computation and keeps a running softmax:

* The query axis is split into blocks of BLOCK_M rows; one Triton program owns one query
  block and one (batch·head). It loads its Q block (BLOCK_M × d) into SRAM once.
* It then streams over the key/value axis in blocks of BLOCK_N. For each K/V block it:
  1. computes the partial scores  S_ij = Q_block · K_blockᵀ · scale   (BLOCK_M × BLOCK_N),
  2. applies the causal mask (a query may not see future keys),
  3. updates the **online softmax** running state per query row — the running max m, the
     running denominator l, and the output accumulator acc — using the standard rescaling:
        m_new   = max(m, rowmax(S_ij))
        alpha   = exp(m - m_new)                  # rescales old partial results
        p       = exp(S_ij - m_new)               # this block's softmax numerators
        l       = l * alpha + rowsum(p)
        acc     = acc * alpha + p · V_block
        m       = m_new
* After the last K/V block, acc / l is the exact softmax-weighted output. The only state held
  is the BLOCK_M×d accumulator plus two BLOCK_M vectors — **O(N) memory, never O(N²)**.

Because S is consumed block-by-block inside SRAM and never written to HBM, the HBM traffic is
just Q, K, V, and O (each O(N·d)) instead of the N² scores. That is the whole point.

Correctness against naive attention is verified in tests within tolerance.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, Out,
    scale,
    n_ctx,
    stride_qz, stride_qn, stride_qd,
    stride_kz, stride_kn, stride_kd,
    stride_vz, stride_vn, stride_vd,
    stride_oz, stride_on, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One program per (query block, batch*head). Streams K/V with an online softmax."""
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # (BLOCK_M,) query rows
    offs_n = tl.arange(0, BLOCK_N)                          # (BLOCK_N,) key cols within a block
    offs_d = tl.arange(0, D)                                # (D,) head dim

    q_ptrs = Q + pid_z * stride_qz + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)   # (BLOCK_M, D) in SRAM

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)     # running max per row
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                   # running softmax denom
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)                 # running output accumulator

    # Causal: a query block at rows [pid_m*BLOCK_M ..] never needs keys past its last row.
    end_n = (pid_m + 1) * BLOCK_M if CAUSAL else n_ctx
    for start_n in range(0, end_n, BLOCK_N):
        n_idx = start_n + offs_n                                   # (BLOCK_N,) absolute key indices
        k_ptrs = K + pid_z * stride_kz + (n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=n_idx[:, None] < n_ctx, other=0.0)   # (BLOCK_N, D)

        s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale          # (BLOCK_M, BLOCK_N)
        s = tl.where(n_idx[None, :] < n_ctx, s, -float("inf"))        # mask padding keys
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= n_idx[None, :], s, -float("inf"))

        m_ij = tl.max(s, axis=1)                                      # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])                               # (BLOCK_M, BLOCK_N)
        alpha = tl.exp(m_i - m_new)                                  # rescale prior results
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + pid_z * stride_vz + (n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_idx[:, None] < n_ctx, other=0.0)   # (BLOCK_N, D)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)   # (BLOCK_M, D)
        m_i = m_new

    acc = acc / l_i[:, None]                                          # finalize softmax
    o_ptrs = Out + pid_z * stride_oz + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < n_ctx)


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = True, block_m: int = 64, block_n: int = 64,
) -> torch.Tensor:
    """Flash-Attention forward.

    Args:
        q, k, v: Shape (batch, n_head, seq, head_dim), fp32.
        causal:  Apply the causal mask.
        block_m, block_n: Query/key tile sizes.

    Returns:
        Attention output. Shape: (batch, n_head, seq, head_dim).
    """
    batch, n_head, seq, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)
    z = batch * n_head
    q3, k3, v3 = (t.reshape(z, seq, head_dim).contiguous() for t in (q, k, v))
    out = torch.empty_like(q3)

    grid = (triton.cdiv(seq, block_m), z)
    _flash_fwd_kernel[grid](
        q3, k3, v3, out, scale, seq,
        q3.stride(0), q3.stride(1), q3.stride(2),
        k3.stride(0), k3.stride(1), k3.stride(2),
        v3.stride(0), v3.stride(1), v3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=block_m, BLOCK_N=block_n, D=head_dim, CAUSAL=causal,
    )
    return out.reshape(batch, n_head, seq, head_dim)


def naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Reference attention that materializes the full N×N score matrix (O(N²) memory)."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale                       # (B, H, N, N)
    if causal:
        n = q.shape[-2]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v                         # (B, H, N, d)
