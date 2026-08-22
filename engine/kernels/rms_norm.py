"""Fused RMSNorm in Triton.

The torch expression ``x / sqrt(mean(x^2) + eps) * weight`` launches six
kernels — square, mean-reduce, add-eps, sqrt, divide, scale — and moves the row
through HBM on each one. At decode time every row is tiny (one token × 1024 or
4096 channels), so all six are latency-bound launches doing almost no work.

Qwen3 runs four RMSNorms per layer (input, post-attention, and per-head QK-norm
on Q and K), so a 28-layer draft model pays this 113 times per forward: profiling
a graphed 0.6B draft step showed ~560 kernels per replay, of which the RMSNorm
chain was the single largest group.

This kernel loads each row into SRAM once, does the whole normalization there,
and writes once — the theoretical minimum HBM traffic, and one launch instead of
six.

Precision note: the accumulation runs in float32 regardless of input dtype,
matching the reference HuggingFace LLaMA implementation. The previous torch path
squared and averaged in the input dtype, so bf16 activations lost precision in
the reduction; this kernel is strictly more accurate, not merely faster.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    out_ptr, in_ptr, w_ptr,
    in_row_stride, out_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row: load row → normalize in SRAM → scale → store."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=0.0).to(tl.float32)
    # Masked lanes load 0.0, contributing nothing to the sum of squares.
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(
        out_ptr + row * out_row_stride + col,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension via the fused Triton kernel.

    Args:
        x:      Input. Shape: (..., n_cols). Last dim must fit one SRAM block.
        weight: Learned gain. Shape: (n_cols,)
        eps:    Stability epsilon.

    Returns:
        Normalized tensor, same shape and dtype as ``x``.
    """
    n_cols = x.shape[-1]
    x2d = x.reshape(-1, n_cols) if x.is_contiguous() else x.contiguous().view(-1, n_cols)
    out = torch.empty_like(x2d)

    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
    _rms_norm_kernel[(x2d.shape[0],)](
        out, x2d, weight,
        x2d.stride(0), out.stride(0),
        n_cols, eps,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return out.view(x.shape)


# Rows wider than this would need more SRAM than a single block gives us; the
# caller falls back to the torch expression there.
MAX_FUSED_COLS = 8192
