"""Weight-only quantization kernels: INT8 (W8A16) and INT4 (W4A16).

INT8 W8A16: 2× weight compression, per-column symmetric scale.
INT4 W4A16: 4× weight compression, per-group symmetric scale (group_size=128).
            Per-group is mandatory for INT4 — only 16 discrete levels can't
            represent a whole column accurately.

INT4 storage: two nibbles packed per int8 byte (high nibble = even row,
low nibble = odd row).  Scales stored as float32 per (group, output_col).

INT4 fused matmul: single Triton kernel reads packed INT4, unpacks nibbles,
applies per-group scale, and accumulates directly into float32 — without
writing the intermediate bf16 weight matrix to HBM.  This halves memory
bandwidth vs the dequantize-then-matmul pattern.

Memory at decode (batch=1):
    bf16 8B = 16 GB   INT4 8B = 4 GB   (4× weight reduction)
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


@triton.jit
def _int4_matmul_kernel(
    A, Packed, Scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_pk, stride_pn,
    stride_sg, stride_sn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused W4A16 matmul: C[M,N] = A[M,K] @ W[K,N], W stored as packed INT4.

    Packed layout: packed[k//2, n] stores two int4 weights:
        bits 7-4 (high nibble): original weight row k   (even k)
        bits 3-0 (low  nibble): original weight row k+1 (odd  k)
    Reads packed INT4, unpacks nibbles, scales, and accumulates into float32
    without writing an intermediate bf16 weight tensor to HBM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # (BLOCK_M,)
    offs_n    = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # (BLOCK_N,)
    offs_half = tl.arange(0, GROUP_SIZE // 2)                 # (64,) — index within packed group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, GROUP_SIZE):
        # Stride-2 pointer arithmetic for even/odd A columns within this group.
        k_even = k0 + offs_half * 2          # (64,) column indices k0, k0+2, ..., k0+126
        k_odd  = k_even + 1                  # (64,) column indices k0+1, k0+3, ..., k0+127

        a_even = tl.load(
            A + offs_m[:, None] * stride_am + k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_even[None, :] < K),
            other=0.0,
        )  # (BLOCK_M, GROUP_SIZE//2) bfloat16

        a_odd = tl.load(
            A + offs_m[:, None] * stride_am + k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_odd[None, :] < K),
            other=0.0,
        )  # (BLOCK_M, GROUP_SIZE//2) bfloat16

        # Load packed weights for this group: rows [k0//2 .. k0//2 + GROUP_SIZE//2 - 1].
        pk = k0 // 2 + offs_half             # (64,) packed-row indices

        p = tl.load(
            Packed + pk[:, None] * stride_pk + offs_n[None, :] * stride_pn,
            mask=(pk[:, None] < K // 2) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)  # (GROUP_SIZE//2, BLOCK_N) int8

        # Unpack nibbles with sign extension.
        # High nibble (even rows): arithmetic right shift fills with sign bit.
        high = p >> 4                                          # (64, BLOCK_N) int8 in [-8, 7]
        # Low nibble (odd rows): isolate 4 bits, shift to top of byte, SAR back.
        low_u = (p & 0x0F).to(tl.int8)                       # (64, BLOCK_N) int8 in [0, 15]
        low   = (low_u << 4).to(tl.int8) >> 4                # (64, BLOCK_N) int8 in [-8, 7]

        # Two tl.dot calls avoid the unfused dequantize-then-matmul memory round-trip.
        partial = (
            tl.dot(a_even, high.to(tl.bfloat16), allow_tf32=False)   # (BLOCK_M, BLOCK_N)
            + tl.dot(a_odd, low.to(tl.bfloat16), allow_tf32=False)   # (BLOCK_M, BLOCK_N)
        )  # float32 accumulator

        # Per-group scale: one scalar per (group, output column), float32.
        s = tl.load(
            Scale + (k0 // GROUP_SIZE) * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )  # (BLOCK_N,) float32

        acc = acc + partial * s[None, :]

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def int4_matmul(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Fused W4A16 matmul: unpacks INT4 nibbles on-the-fly, no intermediate bf16 tensor.

    On CUDA, launches a Triton kernel that reads packed INT4, unpacks, scales, and
    accumulates directly into float32 without writing a decompressed weight matrix.
    Falls back to dequantize-then-matmul on CPU (for tests).

    Args:
        a:          Activations, shape (M, K), bfloat16.
        packed:     Packed INT4 weights, shape (K//2, N), int8.
        scale:      Per-group scales, shape (K//128, N), float32.
        group_size: Must be 128 (kernel is compiled for this group size).

    Returns:
        Output (M, N), bfloat16.
    """
    if not a.is_cuda:
        w = dequantize_weight_int4(packed, scale.to(a.dtype), group_size)
        return a @ w
    assert group_size == 128, f"fused INT4 kernel requires group_size=128, got {group_size}"
    m, k = a.shape
    _, n = packed.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 64))
    _int4_matmul_kernel[grid](
        a, packed, scale, c,
        m, n, k,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=128, BLOCK_M=16, BLOCK_N=64,
    )
    return c


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
