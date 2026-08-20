"""Multi-head causal self-attention for LLaMA: GQA + RoPE.

Three differences from the GPT-2 attention (engine/attention.py):

1. **Separate projections** — q_proj, k_proj, v_proj, o_proj instead of a
   fused c_attn.  KV projections produce ``n_kv_heads`` heads (< n_head for GQA).

2. **RoPE** — rotary positional embeddings are applied to Q and K after
   projection; the precomputed cos/sin tables come from the model.

3. **GQA** — ``repeat_kv`` expands the ``n_kv_heads`` KV tensors to ``n_head``
   query heads before the scaled dot-product, so the rest of the attention math
   is unchanged.  The KV cache stores only ``n_kv_heads`` to realize the memory
   saving; expansion happens at attention time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache
from engine.layers import apply_rope, linear, rms_norm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for grouped-query attention.

    Args:
        x:     KV tensor. Shape: (B, n_kv_heads, T, head_dim)
        n_rep: n_head // n_kv_heads (GQA grouping factor).

    Returns:
        Expanded tensor. Shape: (B, n_kv_heads * n_rep, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv, T, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, d)
        .reshape(B, n_kv * n_rep, T, d)
    )


def llama_attention(
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
    """LLaMA multi-head causal self-attention with GQA and RoPE.

    Args:
        x:            RMSNorm'd residual stream. Shape: (B, T_q, d_model)
        weights:      Block tensors from ``LlamaWeights.layer(i)``.
        config:       Model config.
        cos:          Full RoPE cosine table. Shape: (n_ctx, head_dim)
        sin:          Full RoPE sine table.   Shape: (n_ctx, head_dim)
        position_ids: Absolute token positions for the current query tokens.
                      Shape: (T_q,) or (B, T_q).
        cache:        Optional KV cache (LlamaStaticKVCache).
        layer_idx:    Block index used to address the cache.
        start_pos:    Absolute position of the first query token (0 on prefill).
        attn_mask:    Optional caller-supplied allowed mask (B, T_q, T_total);
                      overrides causal logic when given (continuous batching).

    Returns:
        Attention output. Shape: (B, T_q, d_model)
    """
    B, T_q, _ = x.shape
    n_head, n_kv_heads, head_dim = config.n_head, config.n_kv_heads, config.head_dim
    n_kv_groups = config.n_kv_groups

    # --- separate Q / K / V projections (nn.Linear layout: weight is d_out × d_in) ---
    q = linear(x, weights["self_attn.q_proj.weight"])              # (B, T_q, d_model)
    k = linear(x, weights["self_attn.k_proj.weight"])              # (B, T_q, n_kv_heads*head_dim)
    v = linear(x, weights["self_attn.v_proj.weight"])              # (B, T_q, n_kv_heads*head_dim)

    # --- reshape to (B, heads, T, head_dim) ---
    q = q.view(B, T_q, n_head,    head_dim).transpose(1, 2)       # (B, n_head,    T_q, head_dim)
    k = k.view(B, T_q, n_kv_heads, head_dim).transpose(1, 2)      # (B, n_kv_heads, T_q, head_dim)
    v = v.view(B, T_q, n_kv_heads, head_dim).transpose(1, 2)      # (B, n_kv_heads, T_q, head_dim)

    # --- QK-norm (Qwen3): per-head RMSNorm on Q and K before RoPE ---
    if config.qk_norm:
        q = rms_norm(q, weights["self_attn.q_norm.weight"], config.norm_eps)
        k = rms_norm(k, weights["self_attn.k_norm.weight"], config.norm_eps)

    # --- RoPE: rotate Q and K by their absolute positions ---
    q, k = apply_rope(q, k, cos, sin, position_ids)

    # --- KV cache: append new K/V (stored at n_kv_heads), retrieve full history ---
    if cache is not None:
        k, v = cache.extend(layer_idx, k, v, start_pos)           # (B, n_kv_heads, T_total, d)
    T_total = k.shape[2]

    # --- GQA: expand n_kv_heads → n_head before the dot-product ---
    k_exp = repeat_kv(k, n_kv_groups)                             # (B, n_head, T_total, head_dim)
    v_exp = repeat_kv(v, n_kv_groups)                             # (B, n_head, T_total, head_dim)

    # --- SDPA for all query lengths ---
    # FlashAttention2 eliminates the O(N²) score matrix for long prefills.
    # Using SDPA for decode (T_q=1) too so speculative decoding verify and standard
    # decode use the same kernel — required for the correctness guarantee to hold.
    # Note: is_causal=True with T_q=1 is equivalent to no mask (all keys allowed,
    # since mask[0,j] = j ≤ 0 + (T_total-1) = T_total-1 → all True).
    if attn_mask is not None:
        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=attn_mask[:, None].bool()
        )
    else:
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

    # --- merge heads and output projection ---
    out = out.transpose(1, 2).contiguous().view(B, T_q, n_head * head_dim)
    out = linear(out, weights["self_attn.o_proj.weight"])         # (B, T_q, d_model)
    return out
