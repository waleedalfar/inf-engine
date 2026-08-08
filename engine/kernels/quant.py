"""Weight-only quantization kernels: INT8 (W8A16) and INT4 (W4A16).

Phase 5 (original): INT8 W8A16 — 4× weight compression, per-column symmetric.
Phase 2 scale-up:   INT4 W4A16 — 8× weight compression, per-group symmetric
                    (group_size=128).  Per-group is mandatory for INT4 because
                    only 16 discrete levels can't represent a whole column well.

INT4 storage: two nibbles packed per int8 byte (high nibble = even index,
low nibble = odd index).  Scales are stored as fp16 per (group, output_col).

Memory at decode (batch=1):
    bf16 7B  = 14 GB   INT8 7B  =  7 GB   INT4 7B  = 3.5 GB
    bf16 14B = 28 GB   INT8 14B = 14 GB   INT4 14B = 7 GB

"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-column symmetric INT8 quantization of a (d_in, d_out) weight.

    Args:
        w: Weight, shape (d_in, d_out), float.

    Returns:
        (wq, scale): wq int8 (d_in, d_out); scale float (d_out,) such that
        ``w ≈ wq.float() * scale[None, :]``.
    """
    scale = w.abs().amax(dim=0) / 127.0                     # (d_out,) per-column
    scale = torch.clamp(scale, min=1e-8)
    wq = torch.clamp(torch.round(w / scale[None, :]), -127, 127).to(torch.int8)
    return wq, scale


@triton.jit
def _int8_matmul_kernel(
    A, Wq, Scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = (A[M,K] @ Wq[K,N]) * Scale[N], with Wq loaded as int8 and dequantized."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a = tl.load(A + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)      # (BM, BK) fp
        w = tl.load(Wq + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0)         # (BK, BN) int8
        acc += tl.dot(a, w.to(a.dtype), allow_tf32=False)                            # dequant happens via scale below
    scale = tl.load(Scale + offs_n, mask=offs_n < N, other=0.0)                      # (BN,) per-column
    acc = acc * scale[None, :]
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# INT4 W4A16 — 8× weight compression, per-group symmetric
# ---------------------------------------------------------------------------

_INT4_MAX = 7          # symmetric: range [-7, 7], zero at 0
_INT4_PACK = 2         # two nibbles per byte


def quantize_weight_int4(
    w: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric INT4 quantization of an (d_in, d_out) weight.

    ``group_size`` contiguous rows share one scale per output column, so the
    scale tensor has shape (n_groups, d_out) where n_groups = d_in // group_size.
    Two int4 values are packed per int8 byte (high nibble first).

    Args:
        w:          Weight matrix, shape (d_in, d_out), float.
        group_size: Number of rows per quantization group (128 is standard).

    Returns:
        packed:  INT4 weights packed into int8, shape (d_in // 2, d_out).
        scale:   Per-group scale, shape (n_groups, d_out), same dtype as ``w``.
    """
    d_in, d_out = w.shape
    if d_in % group_size != 0:
        raise ValueError(f"d_in={d_in} not divisible by group_size={group_size}")
    if d_in % _INT4_PACK != 0:
        raise ValueError(f"d_in={d_in} must be even for nibble packing")

    n_groups = d_in // group_size
    wg = w.reshape(n_groups, group_size, d_out)               # (G, gs, d_out)
    scale = wg.abs().amax(dim=1) / _INT4_MAX                  # (G, d_out)
    scale = scale.clamp(min=1e-8)

    # Quantize each group; expand scale back for broadcasting.
    scale_exp = scale[:, None, :].expand_as(wg)               # (G, gs, d_out)
    wq = (wg / scale_exp).round().clamp(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    wq_flat = wq.reshape(d_in, d_out)                         # (d_in, d_out) int8 in [-7,7]

    # Pack pairs of rows: high nibble = row 2k, low nibble = row 2k+1.
    even = wq_flat[0::2] & 0x0F                               # (d_in//2, d_out) low bits
    odd  = wq_flat[1::2] & 0x0F                               # (d_in//2, d_out) low bits
    packed = (even.to(torch.uint8) << 4 | odd.to(torch.uint8)).to(torch.int8)
    return packed, scale


def dequantize_weight_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Unpack and dequantize INT4 weights back to float.

    Args:
        packed:     Packed int8 weights, shape (d_in // 2, d_out).
        scale:      Per-group scales, shape (n_groups, d_out).
        group_size: Must match the value used at quantization time.

    Returns:
        Reconstructed weight, shape (d_in, d_out), dtype same as ``scale``.
    """
    d_in_half, d_out = packed.shape
    d_in = d_in_half * _INT4_PACK

    # High nibble (even rows): arithmetic right shift on int8 sign-extends from bit 3.
    high = packed >> 4                                         # int8 SAR: result in [-8, 7]

    # Low nibble (odd rows): mask to 4 bits, sign-extend from 4 bits to 8 bits.
    # Trick: shift left 4 to put the nibble's sign bit into int8 bit 7, then SAR.
    low_u = (packed & 0x0F).to(torch.int8)                    # unsigned nibble in [0, 15]
    low = (low_u << 4) >> 4                                    # sign-extend: 9→-7, 5→5

    wq = torch.empty(d_in, d_out, dtype=torch.int8, device=packed.device)
    wq[0::2] = high
    wq[1::2] = low

    # Apply per-group scale.
    n_groups = d_in // group_size
    wq_groups = wq.reshape(n_groups, group_size, d_out).to(scale.dtype)
    w = wq_groups * scale[:, None, :]                         # (G, gs, d_out)
    return w.reshape(d_in, d_out)


def int4_matmul(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Compute ``a @ W`` where W is stored as packed INT4 with per-group scales.

    Dequantizes W to float on the fly (no fused kernel yet — the memory saving
    is the primary benefit at decode batch sizes; fused INT4 matmul is a future
    Triton kernel).

    Args:
        a:          Activations, shape (M, d_in), float.
        packed:     Packed int8, shape (d_in // 2, d_out).
        scale:      Per-group scales, shape (n_groups, d_out), same dtype as ``a``.
        group_size: Group size used at quantization time.

    Returns:
        Output (M, d_out) in ``a``'s dtype.
    """
    w = dequantize_weight_int4(packed, scale.to(a.dtype), group_size)  # (d_in, d_out)
    return a @ w                                               # (M, d_in) @ (d_in, d_out)


def int8_matmul(a: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ (wq.float() * scale)`` with weights streamed as int8.

    Args:
        a:     Activations, shape (M, K), fp16/fp32.
        wq:    INT8 weight, shape (K, N).
        scale: Per-column scale, shape (N,), same float dtype as ``a``.

    Returns:
        Output (M, N) in ``a``'s dtype.
    """
    m, k = a.shape
    k2, n = wq.shape
    assert k == k2, (a.shape, wq.shape)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _int8_matmul_kernel[grid](
        a, wq, scale.to(a.dtype), c, m, n, k,
        a.stride(0), a.stride(1), wq.stride(0), wq.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c
