"""Multi-head causal self-attention with explicit Q, K, V projections.

Naive (Phase 1) implementation: it materializes the full (q_len, total_len) attention
score matrix per head. That is exactly the O(T^2) memory cost Phase 5's Flash-Attention
kernel will later remove — we keep this version as the correctness reference.

Phase 2 adds optional KV-cache support. When a cache is supplied, only the *new*
tokens' Q/K/V are computed; the cache returns the full K/V history to attend over. The
causal mask is generalized from a fixed lower-triangular matrix to an absolute-position
comparison, which reduces *exactly* to the lower-triangular case when ``start_pos == 0``
and there is no cache — so the Phase 1 correctness gate is unaffected.
"""

from __future__ import annotations

import math

import torch

from engine.config import GPT2Config
from engine.kv_cache import KVCache
from engine.layers import conv1d


def causal_self_attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: GPT2Config,
    cache: KVCache | None = None,
    layer_idx: int = 0,
    start_pos: int = 0,
    key_padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one block's multi-head causal self-attention.

    Args:
        x:                Layer-normed residual stream. Shape: (batch, q_len, d_model)
        weights:          Block tensors from ``GPT2Weights.block(i)`` (c_attn, c_proj).
        config:           Model config (supplies n_head, head_dim, d_model).
        cache:            Optional KV cache. If given, K/V are appended and the full
                          history is attended over.
        layer_idx:        This block's index (used to address the cache).
        start_pos:        Absolute (buffer) position of the first token in ``x`` (0 unless
                          decoding with a cache that already holds earlier tokens).
        key_padding_mask: Optional bool/0-1 mask over keys. Shape: (batch, total_len).
                          True/1 = real token, False/0 = padding (ignored as a key). Used
                          for static batching of left-padded sequences (Phase 3).
        attn_mask:        Optional caller-provided boolean "allowed" mask, shape
                          (batch, q_len, total_len). When given it *fully overrides* the
                          internal causal/padding logic — used by continuous batching
                          (Phase 4) where each slot attends to its own variable-length
                          history. True = attend, False = mask out.

    Returns:
        Attention output to add back to the residual. Shape: (batch, q_len, d_model)
    """
    batch, q_len, d_model = x.shape                          # (B, T_q, d_model)
    n_head, head_dim = config.n_head, config.head_dim        # H, d_head ; H*d_head == d_model

    # --- QKV projection: one fused Conv1D produces Q, K, V stacked on last dim ---
    qkv = conv1d(x, weights["attn.c_attn.weight"], weights["attn.c_attn.bias"])
    # qkv: (B, T_q, 3*d_model)
    q, k, v = qkv.split(d_model, dim=2)                      # each (B, T_q, d_model)

    # --- split d_model into heads and move head axis next to batch ---
    # (B, T_q, d_model) -> (B, T_q, H, d_head) -> (B, H, T_q, d_head)
    def to_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, q_len, n_head, head_dim).transpose(1, 2)

    q = to_heads(q)                                          # (B, H, T_q, d_head)
    k = to_heads(k)                                          # (B, H, T_q, d_head)
    v = to_heads(v)                                          # (B, H, T_q, d_head)

    # --- KV cache: append new K/V, retrieve full history (no-op when cache is None) ---
    if cache is not None:
        k, v = cache.extend(layer_idx, k, v, start_pos)      # k,v: (B, H, T_total, d_head)
    total_len = k.shape[2]                                   # T_total = start_pos + T_q

    # --- scaled dot-product scores ---
    scale = 1.0 / math.sqrt(head_dim)                        # 1/sqrt(d_head)
    scores = (q @ k.transpose(-2, -1)) * scale               # (B, H, T_q, T_total)

    # --- causal mask by absolute position: query i (abs start_pos+i) may attend to
    #     key j (abs j) iff j <= start_pos + i. With start_pos=0 and total_len=q_len
    #     this is exactly torch.tril, so Phase 1 behavior is byte-identical. ---
    q_abs = torch.arange(q_len, device=x.device) + start_pos      # (T_q,) buffer indices
    k_abs = torch.arange(total_len, device=x.device)              # (T_total,) buffer indices
    allowed = k_abs[None, :] <= q_abs[:, None]                    # (T_q, T_total) bool causal

    if attn_mask is not None:
        # Continuous batching: caller fully specifies which keys each query may attend to.
        scores = scores.masked_fill(~attn_mask[:, None].bool(), float("-inf"))  # (B,1,T_q,T_total) over heads
    elif key_padding_mask is None:
        # Phase 1/2 path: pure causal mask (byte-identical to before).
        scores = scores.masked_fill(~allowed, float("-inf"))      # (B, H, T_q, T_total)
    else:
        # Static batching: also forbid attending to padded keys. A left-padded query
        # that is itself padding could otherwise have *every* key masked -> NaN softmax,
        # so we always allow each query to attend to its own position (the diagonal).
        # Padded query outputs are never read (left padding keeps the last position real)
        # and padded keys are masked for every real query, so this stays isolated.
        kp = key_padding_mask[:, None, :].bool()                  # (B, 1, T_total)
        allowed_b = (allowed[None] & kp)                          # (B, T_q, T_total)
        diag = (k_abs[None, :] == q_abs[:, None])[None]           # (1, T_q, T_total) self
        allowed_b = allowed_b | diag                              # never fully masked
        scores = scores.masked_fill(~allowed_b[:, None], float("-inf"))  # broadcast over heads

    # --- softmax over keys, then weighted sum of values ---
    attn = torch.softmax(scores, dim=-1)                     # (B, H, T_q, T_total)
    out = attn @ v                                           # (B, H, T_q, d_head)

    # --- merge heads back to d_model, then output projection ---
    # (B, H, T_q, d_head) -> (B, T_q, H, d_head) -> (B, T_q, d_model)
    out = out.transpose(1, 2).contiguous().view(batch, q_len, d_model)
    out = conv1d(out, weights["attn.c_proj.weight"], weights["attn.c_proj.bias"])
    return out                                               # (B, T_q, d_model)
