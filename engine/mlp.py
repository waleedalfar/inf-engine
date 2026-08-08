"""Position-wise feed-forward (MLP) block: expand -> GELU -> project back."""

from __future__ import annotations

import torch

from engine.layers import conv1d, gelu


def mlp(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Two-layer MLP with GELU, applied independently at each position.

    Args:
        x:       Layer-normed residual stream. Shape: (batch, seq, d_model)
        weights: Block tensors (mlp.c_fc, mlp.c_proj) from ``GPT2Weights.block(i)``.

    Returns:
        MLP output to add back to the residual. Shape: (batch, seq, d_model)
    """
    h = conv1d(x, weights["mlp.c_fc.weight"], weights["mlp.c_fc.bias"])  # (B, T, d_mlp=4*d_model)
    h = gelu(h)                                                          # (B, T, d_mlp)
    out = conv1d(h, weights["mlp.c_proj.weight"], weights["mlp.c_proj.bias"])  # (B, T, d_model)
    return out                                                          # (B, T, d_model)
