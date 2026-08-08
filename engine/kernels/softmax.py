"""Fused, numerically-stable softmax in Triton (Phase 5, Kernel 1).

Softmax over a row does four logical passes: max, subtract+exp, sum, divide. PyTorch's
native softmax launches a couple of kernels and moves the row through HBM more than once.
The fused Triton version loads each row into on-chip SRAM **once**, does max/exp/sum/divide
there, and writes the result **once** — so the operation does the theoretical minimum HBM
traffic (read input + write output). Softmax is memory-bandwidth bound (a handful of flops
per element), so cutting HBM round-trips is exactly the right lever (see DECISIONS.md).

Constraint: this one-program-per-row design keeps a whole row in SRAM, so it requires the row
to fit in one block (BLOCK_SIZE = next_pow2(n_cols)). GPT-2 attention rows are ≤ 1024, well
within SRAM. Rows too large for SRAM need the online-softmax tiling of Flash-Attention (K2).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program per row: load row -> stable softmax in SRAM -> store."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=-float("inf"))  # (BLOCK,)
    x = x - tl.max(x, axis=0)                       # subtract row max for numerical stability
    num = tl.exp(x)                                 # exp in SRAM
    denom = tl.sum(num, axis=0)                     # row sum in SRAM
    y = num / denom                                 # normalize
    tl.store(out_ptr + row * out_row_stride + col, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension via the fused Triton kernel.

    Args:
        x: Input. Shape: (..., n_cols). Last dim must fit one SRAM block.

    Returns:
        Softmax over the last dim, same shape and dtype as ``x``.
    """
    *lead, n_cols = x.shape
    x2d = x.reshape(-1, n_cols).contiguous()        # (n_rows, n_cols)
    n_rows = x2d.shape[0]
    out = torch.empty_like(x2d)

    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
    _softmax_kernel[(n_rows,)](
        out, x2d,
        x2d.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return out.reshape(*lead, n_cols)


if __name__ == "__main__":
    # Smoke test: confirms Triton compiles/runs on this GPU and matches torch.softmax.
    torch.manual_seed(0)
    dev = "cuda"
    for shape in [(128, 128), (2048, 512), (12 * 1024, 1024)]:
        x = torch.randn(shape, device=dev, dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        got = triton_softmax(x)
        max_err = (ref - got).abs().max().item()
        print(f"shape {shape}: max|Δ| = {max_err:.2e}  {'OK' if max_err < 1e-5 else 'FAIL'}")
