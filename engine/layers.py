"""Primitive layers for the Qwen3 / LLaMA inference engine.

Pure tensor functions (not nn.Module) so forward-pass code reads as explicit math.
"""

from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization (LLaMA-style): no mean subtraction, no bias.

    rms_norm(x) = x / sqrt(mean(x²) + eps) * weight

    Args:
        x:      Input. Shape: (..., d_model)
        weight: Learned gain. Shape: (d_model,)
        eps:    Stability epsilon.

    Returns:
        Normalized tensor. Shape: (..., d_model)
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # (..., 1)
    return (x / rms) * weight                                      # (..., d_model)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation used in LLaMA's SwiGLU MLP: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def linear(x: torch.Tensor, weight) -> torch.Tensor:
    """LLaMA linear projection: y = x @ W^T, with INT4 fused-kernel dispatch.

    If ``weight`` is an ``_Int4Weight`` (from quantize.py), delegates to its
    ``fused_linear`` method, which unpacks nibbles inside a Triton kernel without
    writing an intermediate bf16 tensor to HBM.  Otherwise falls back to the
    standard ``x @ weight.T`` path.

    Args:
        x:      Input. Shape: (..., d_in)
        weight: (d_out, d_in) tensor, or an ``_Int4Weight`` object.

    Returns:
        Projected tensor. Shape: (..., d_out)
    """
    if hasattr(weight, "fused_linear"):
        return weight.fused_linear(x)
    return x @ weight.T


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq: int,
    theta: float = 500_000.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cosine/sine tables for RoPE up to ``max_seq`` positions.

    Follows the HuggingFace LLaMA convention: frequencies are computed for
    head_dim/2 pairs, then concatenated to produce full-head_dim cos/sin
    tables so the apply step is a simple pointwise multiply.

    Returns:
        cos, sin — each shape (max_seq, head_dim)
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )                                                              # (head_dim/2,)
    positions = torch.arange(max_seq, device=device).float()      # (max_seq,)
    angles = torch.outer(positions, inv_freq)                     # (max_seq, head_dim/2)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)         # (max_seq, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)         # (max_seq, head_dim)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by 90°: [-x2, x1] where x = [x1, x2] split on last dim."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to Q and K in-place style.

    Args:
        q:            Query vectors.  Shape: (B, n_head,    T, head_dim)
        k:            Key vectors.    Shape: (B, n_kv_heads, T, head_dim)
        cos:          Precomputed cos table. Shape: (max_seq, head_dim)
        sin:          Precomputed sin table. Shape: (max_seq, head_dim)
        position_ids: Token positions. Shape: (T,) or (B, T).

    Returns:
        (q_rot, k_rot) with the same shapes as input.
    """
    if position_ids.dim() == 1:
        c = cos[position_ids][None, None]   # (1, 1, T, head_dim)
        s = sin[position_ids][None, None]
    else:
        c = cos[position_ids][:, None]      # (B, 1, T, head_dim)
        s = sin[position_ids][:, None]
    q_rot = q * c + _rotate_half(q) * s
    k_rot = k * c + _rotate_half(k) * s
    return q_rot, k_rot


