"""A single GPT-2 transformer block: pre-norm attention + pre-norm MLP.

GPT-2 is a **pre-norm** transformer: LayerNorm is applied to the input of each
sub-layer, and the sub-layer's output is added back to the (un-normed) residual
stream. This differs from the original "post-norm" Transformer and matters for
matching HuggingFace exactly.

    x = x + attn(ln_1(x))
    x = x + mlp (ln_2(x))
"""

from __future__ import annotations

import torch

from engine.attention import causal_self_attention
from engine.config import GPT2Config
from engine.kv_cache import KVCache
from engine.layers import layer_norm
from engine.mlp import mlp


def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: GPT2Config,
    cache: KVCache | None = None,
    layer_idx: int = 0,
    start_pos: int = 0,
    key_padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one pre-norm transformer block to the residual stream.

    Args:
        x:                Residual stream. Shape: (batch, seq, d_model)
        weights:          Block tensors from ``GPT2Weights.block(i)``.
        config:           Model config.
        cache:            Optional KV cache (threaded into attention).
        layer_idx:        This block's index (used to address the cache).
        start_pos:        Absolute position of the first token in ``x``.
        key_padding_mask: Optional key padding mask (batch, total_len) for static batching.
        attn_mask:        Optional explicit allowed mask (batch, q_len, total_len) for
                          continuous batching; overrides causal/padding when given.

    Returns:
        Updated residual stream. Shape: (batch, seq, d_model)
    """
    eps = config.layer_norm_eps

    # --- attention sub-layer (pre-norm + residual add) ---
    ln1 = layer_norm(x, weights["ln_1.weight"], weights["ln_1.bias"], eps)  # (B, T, d_model)
    x = x + causal_self_attention(
        ln1, weights, config, cache, layer_idx, start_pos, key_padding_mask, attn_mask
    )  # (B, T, d_model)

    # --- MLP sub-layer (pre-norm + residual add) ---
    ln2 = layer_norm(x, weights["ln_2.weight"], weights["ln_2.bias"], eps)  # (B, T, d_model)
    x = x + mlp(ln2, weights)                                               # (B, T, d_model)
    return x                                                                # (B, T, d_model)
