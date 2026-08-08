"""One LLaMA transformer block: pre-norm attention + pre-norm SwiGLU MLP.

    x = x + attn( rms_norm(x) )
    x = x + mlp ( rms_norm(x) )

Identical residual structure to GPT-2's pre-norm blocks, but using
RMSNorm (no mean subtraction, no bias) and the LLaMA attention / MLP.
"""

from __future__ import annotations

import torch

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache
from engine.layers import rms_norm
from engine.llama_attention import llama_attention
from engine.llama_mlp import swiglu_mlp
from engine.llama_moe import moe_mlp


def llama_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: LlamaConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache: LlamaStaticKVCache | None = None,
    layer_idx: int = 0,
    start_pos: int = 0,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one LLaMA pre-norm transformer block.

    Args:
        x:            Residual stream. Shape: (B, T, d_model)
        weights:      Block tensors from ``LlamaWeights.layer(i)``.
        config:       Model config.
        cos:          Full RoPE cosine table. Shape: (n_ctx, head_dim)
        sin:          Full RoPE sine table.   Shape: (n_ctx, head_dim)
        position_ids: Absolute positions for query tokens. Shape: (T,) or (B, T).
        cache:        Optional KV cache.
        layer_idx:    Block index (for cache addressing).
        start_pos:    Absolute position of the first query token.
        attn_mask:    Optional explicit attention mask (continuous batching).

    Returns:
        Updated residual stream. Shape: (B, T, d_model)
    """
    eps = config.norm_eps

    # pre-norm attention sub-layer
    h = rms_norm(x, weights["input_layernorm.weight"], eps)
    x = x + llama_attention(h, weights, config, cos, sin, position_ids, cache, layer_idx, start_pos, attn_mask)

    # pre-norm MLP sub-layer
    h = rms_norm(x, weights["post_attention_layernorm.weight"], eps)
    if config.is_moe:
        x = x + moe_mlp(h, weights, config)
    else:
        x = x + swiglu_mlp(h, weights)

    return x
