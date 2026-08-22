"""Multi-head causal self-attention for LLaMA: GQA + RoPE.

Three differences from the GPT-2 attention (engine/attention.py):

1. **Separate projections** — q_proj, k_proj, v_proj, o_proj instead of a
   fused c_attn.  KV projections produce ``n_kv_heads`` heads (< n_head for GQA).

2. **RoPE** — rotary positional embeddings are applied to Q and K after
   projection; the precomputed cos/sin tables come from the model.

3. **GQA** — on the *unmasked* SDPA calls (plain causal / decode-all-cached),
   ``n_kv_heads``-width K/V go straight to
   ``F.scaled_dot_product_attention(..., enable_gqa=True)`` instead of being
   pre-expanded to ``n_head`` width first, letting the flash kernel read the
   narrow K/V directly. When an explicit ``attn_mask`` is involved (continuous
   batching / spec-decode verify), PyTorch's flash/mem-efficient backends
   don't support ``enable_gqa`` + ``attn_mask`` together on this build and
   silently fall back to the much slower reference "math" backend (measured:
   masked decode got ~1.7x *slower* end to end) — those calls still
   pre-expand via ``repeat_kv`` to keep hitting the fused efficient-attention
   kernel. See the branch comments below for which case does which.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache
from engine.layers import apply_rope, linear, rms_norm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for grouped-query attention.

    Only used on the masked SDPA branches (see ``llama_attention`` below) —
    ``enable_gqa=True`` handles the unmasked branches without this copy.

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

    # --- Q / K / V projections (nn.Linear layout: weight is d_out × d_in) ---
    # engine/fuse_weights.py concatenates the three into one wide projection at
    # load time; splitting the result here is free (a view), whereas three
    # separate matmuls over the same x leave the GPU badly under-occupied on the
    # narrow K/V shapes. Falls back to separate weights when not fused.
    qkv_w = weights.get("self_attn.qkv_proj.weight")
    if qkv_w is not None:
        qkv = linear(x, qkv_w)                                     # (B, T_q, (n_head+2*n_kv)*head_dim)
        q_width, kv_width = n_head * head_dim, n_kv_heads * head_dim
        q, k, v = qkv.split([q_width, kv_width, kv_width], dim=-1)
    else:
        q = linear(x, weights["self_attn.q_proj.weight"])          # (B, T_q, d_model)
        k = linear(x, weights["self_attn.k_proj.weight"])          # (B, T_q, n_kv_heads*head_dim)
        v = linear(x, weights["self_attn.v_proj.weight"])          # (B, T_q, n_kv_heads*head_dim)

    # --- split into heads, still token-major: (B, T, heads, head_dim) ---
    # reshape, not view: after a fused-QKV split these are slices of a wider
    # row, so at T_q > 1 they are views rather than contiguous blocks.
    q = q.reshape(B, T_q, n_head,     head_dim)
    k = k.reshape(B, T_q, n_kv_heads, head_dim)
    v = v.reshape(B, T_q, n_kv_heads, head_dim)

    # --- QK-norm (Qwen3): per-head RMSNorm on Q and K before RoPE ---
    # Applied here rather than after the transpose below: RMSNorm reduces over
    # head_dim either way, but on this side the rows are contiguous, so the
    # fused kernel reads them directly instead of forcing a copy.
    if config.qk_norm:
        q = rms_norm(q, weights["self_attn.q_norm.weight"], config.norm_eps)
        k = rms_norm(k, weights["self_attn.k_norm.weight"], config.norm_eps)

    # --- RoPE: rotate Q and K by their absolute positions ---
    # Also applied token-major, for the same reason as QK-norm above: each head
    # vector is one contiguous row, which is what the fused kernel wants.
    q, k = apply_rope(q, k, cos, sin, position_ids)

    # --- to (B, heads, T, head_dim) for attention ---
    q = q.transpose(1, 2)                                          # (B, n_head,     T_q, head_dim)
    k = k.transpose(1, 2)                                          # (B, n_kv_heads, T_q, head_dim)
    v = v.transpose(1, 2)                                          # (B, n_kv_heads, T_q, head_dim)

    # --- KV cache: append new K/V (stored at n_kv_heads), retrieve full history ---
    if cache is not None:
        k, v = cache.extend(layer_idx, k, v, start_pos)           # (B, n_kv_heads, T_total, d)
    T_total = k.shape[2]

    gqa = n_kv_groups != 1

    # --- SDPA — three cases based on T_q and cache state ---
    #
    # PyTorch's is_causal=True uses "upper left" masking: mask[i][j] = (j <= i).
    # With a KV cache where T_total > T_q, this is WRONG — query i can only see
    # keys 0..i instead of keys 0..start_pos+i.
    #
    # Case A — explicit mask provided (continuous batching): pre-expand K/V
    #           (repeat_kv) — enable_gqa=True + attn_mask forces the slow
    #           MATH backend on this torch build, so this branch avoids it.
    # Case B — decode step (T_q=1): all cached keys are already from earlier
    #           positions; is_causal=False (attend to all of them). No mask,
    #           so enable_gqa=True safely reaches the flash kernel.
    # Case C — full prefill without cached prefix (start_pos=0, T_q==T_total):
    #           standard lower-triangular, is_causal=True is correct. No
    #           mask, same as B.
    # Case D — multi-token forward with existing cache (spec verify phase):
    #           query i (abs pos start_pos+i) must see keys 0..start_pos+i.
    #           Builds its own offset bias mask, so same MATH-fallback risk
    #           as Case A — pre-expand here too.
    if attn_mask is not None:
        k_exp, v_exp = repeat_kv(k, n_kv_groups), repeat_kv(v, n_kv_groups)
        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=attn_mask[:, None].bool()
        )
    elif T_q == 1:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=gqa)
    elif start_pos == 0:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    else:
        rows = torch.arange(T_q, device=q.device)
        cols = torch.arange(T_total, device=q.device)
        mask = cols[None, :] <= (rows[:, None] + start_pos)        # (T_q, T_total)
        bias = torch.zeros(1, 1, T_q, T_total, dtype=q.dtype, device=q.device)
        bias.masked_fill_(~mask[None, None], float("-inf"))
        k_exp, v_exp = repeat_kv(k, n_kv_groups), repeat_kv(v, n_kv_groups)
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=bias)

    # --- merge heads and output projection ---
    out = out.transpose(1, 2).contiguous().view(B, T_q, n_head * head_dim)
    out = linear(out, weights["self_attn.o_proj.weight"])         # (B, T_q, d_model)
    return out
