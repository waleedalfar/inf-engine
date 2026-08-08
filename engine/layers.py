"""Primitive layers implemented from scratch: LayerNorm, GELU, Conv1D-style linear.

These are deliberately not ``torch.nn`` modules — they are pure functions over
tensors so the forward pass reads as explicit math with named-dimension shape
comments.
"""

from __future__ import annotations

import math

import torch

# GPT-2 uses the "tanh" GELU approximation (a.k.a. ``gelu_new``). The exact-erf
# GELU produces slightly different activations and would break the token-for-token
# correctness gate, so the constant below is part of the contract, not a tweakable.
_GELU_COEF = 0.044715
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Layer normalization over the last dimension (implemented manually).

    Normalizes each length-``d_model`` vector to zero mean / unit variance, then
    applies a learned affine transform. Variance is the **biased** estimator
    (divide by d_model, not d_model-1), matching ``torch.nn.LayerNorm``.

    Args:
        x:      Input.  Shape: (..., d_model)
        weight: Affine gain.  Shape: (d_model,)
        bias:   Affine shift. Shape: (d_model,)
        eps:    Added to variance before sqrt for numerical stability.

    Returns:
        Normalized tensor. Shape: (..., d_model)
    """
    mean = x.mean(dim=-1, keepdim=True)                      # (..., 1)
    var = x.var(dim=-1, unbiased=False, keepdim=True)        # (..., 1) biased variance
    x_hat = (x - mean) / torch.sqrt(var + eps)               # (..., d_model)
    return x_hat * weight + bias                             # (..., d_model) broadcast affine


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation, tanh approximation (``gelu_new``, as used by GPT-2).

    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

    Args:
        x: Input of any shape.

    Returns:
        Activated tensor, same shape as ``x``.
    """
    inner = _SQRT_2_OVER_PI * (x + _GELU_COEF * x.pow(3))    # same shape as x
    return 0.5 * x * (1.0 + torch.tanh(inner))               # same shape as x


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


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Standard nn.Linear projection: y = x @ W^T. Weight stored as (d_out, d_in).

    LLaMA uses nn.Linear (opposite transpose convention from GPT-2's Conv1D).

    Args:
        x:      Input. Shape: (..., d_in)
        weight: Stored as (d_out, d_in).

    Returns:
        Projected tensor. Shape: (..., d_out)
    """
    return x @ weight.T                                            # (..., d_in) @ (d_in, d_out)


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


def conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GPT-2 Conv1D projection: ``y = x @ W + b`` (weight stored as (d_in, d_out)).

    This is HuggingFace's ``Conv1D`` semantics, the transpose of ``nn.Linear``.
    See ``engine/weights.py`` for why the weight is used without transposing.

    Args:
        x:      Input.  Shape: (..., d_in)
        weight: Stored as (d_in, d_out)  -- NOT transposed.
        bias:   Shape: (d_out,)

    Returns:
        Projected tensor. Shape: (..., d_out)
    """
    return x @ weight + bias                                 # (..., d_in) @ (d_in, d_out) -> (..., d_out)
