"""Fused rotary positional embedding (RoPE) in Triton.

The torch formulation ``x * cos + rotate_half(x) * sin`` is cheap arithmetic
spread over many launches: two table gathers, then per tensor a negate, a
``torch.cat`` to build ``rotate_half``, two multiplies and an add. Applied to Q
and K in every layer that is ~12 kernels per layer — 432 for a 36-layer verify
step, and after the RMSNorm and projection fusions it was the largest remaining
non-matmul cost in the profile.

This kernel does one program per head-vector: load the two halves, look up the
angle for that token's position, rotate in registers, store once. Two launches
per layer (Q and K) instead of twelve, and no ``cat`` allocation at all.

The rotation pairs channel ``i`` with ``i + head_dim/2``:

    out[i]          = x[i]          * cos[i]          - x[i + half] * sin[i]
    out[i + half]   = x[i + half]   * cos[i + half]   + x[i]        * sin[i + half]

which is exactly ``rotate_half``'s ``cat([-x[half:], x[:half]])`` written out.

Arithmetic runs in float32; the torch path ran in the activation dtype, so on
bf16 this is slightly more accurate as well as faster.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    out_ptr, x_ptr, cos_ptr, sin_ptr, pos_ptr,
    stride_pb, stride_pt,
    T, H, half, table_stride,
    BLOCK: tl.constexpr,
):
    """One program per (batch, token, head) vector of length 2*half."""
    row = tl.program_id(0)          # b*T*H + t*H + h, over a contiguous (B,T,H,D)
    bt = row // H                   # b*T + t
    t = bt % T
    b = bt // T
    pos = tl.load(pos_ptr + b * stride_pb + t * stride_pt)

    off = tl.arange(0, BLOCK)
    mask = off < half
    base = row * 2 * half
    tbase = pos * table_stride

    x1 = tl.load(x_ptr + base + off,        mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + half + off, mask=mask, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + tbase + off,        mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + tbase + off,        mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + tbase + half + off, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + tbase + half + off, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2

    tl.store(out_ptr + base + off,
             o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + base + half + off,
             o2.to(out_ptr.dtype.element_ty), mask=mask)


def triton_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a token-major (B, T, H, head_dim) tensor.

    Args:
        x:            Query or key heads, contiguous. Shape: (B, T, H, head_dim)
        cos:          Cosine table. Shape: (max_seq, head_dim)
        sin:          Sine table.   Shape: (max_seq, head_dim)
        position_ids: Absolute positions. Shape: (T,) or (B, T).

    Returns:
        Rotated tensor, same shape and dtype as ``x``.
    """
    B, T, H, D = x.shape
    x = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x)

    if position_ids.dim() == 1:
        stride_pb, stride_pt = 0, position_ids.stride(0)
    else:
        stride_pb, stride_pt = position_ids.stride(0), position_ids.stride(1)

    half = D // 2
    block = triton.next_power_of_2(half)
    _rope_kernel[(B * T * H,)](
        out, x, cos, sin, position_ids,
        stride_pb, stride_pt,
        T, H, half, cos.stride(0),
        BLOCK=block, num_warps=4 if block <= 1024 else 8,
    )
    return out
