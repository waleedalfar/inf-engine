"""LLaMA language model: forward pass and autoregressive generation.

Mirrors the structure of GPT2Model (engine/model.py) but uses:
  - Token embeddings only (no learned position embedding table; RoPE handles position).
  - RMSNorm instead of LayerNorm.
  - LLaMA transformer blocks (llama_block).
  - Separate lm_head (or weight-tied embed_tokens for 1B/3B).

RoPE frequency tables are precomputed once at construction and stored on the
same device as the weights; they are indexed by position_ids at each forward call.
"""

from __future__ import annotations

import torch

from engine.config import LlamaConfig
from engine.kv_cache import LlamaStaticKVCache
from engine.layers import precompute_rope_freqs, rms_norm
from engine.llama_block import llama_block
from engine.llama_weights import LlamaWeights
from engine.sampling import SamplingConfig, sample_next_token


class LlamaModel:
    """LLaMA language model (no nn.Module, no HuggingFace model class).

    Stateless except for the precomputed RoPE tables.  All autoregressive
    state lives in the optional ``LlamaStaticKVCache`` passed to ``forward``.
    """

    def __init__(self, weights: LlamaWeights, config: LlamaConfig) -> None:
        self.w = weights
        self.config = config
        device = str(weights.embed_tokens.device)
        # Precompute RoPE tables once for the full context length.
        # Shape: (n_ctx, head_dim) each.  Stored as float32 for precision;
        # cast to weights dtype inside forward to stay on the hot path.
        cos, sin = precompute_rope_freqs(
            config.head_dim, config.n_ctx, config.rope_theta, device
        )
        self.rope_cos = cos  # (n_ctx, head_dim)
        self.rope_sin = sin  # (n_ctx, head_dim)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: LlamaStaticKVCache | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-token logits for every input position.

        Args:
            input_ids:    Token ids. Shape: (B, T_q), dtype long.
            cache:        Optional KV cache. When supplied, only the new tokens
                          are processed and their K/V are written at ``start_pos``.
            start_pos:    Absolute position of the first token in ``input_ids``
                          (0 for a full forward or prefill).
            position_ids: Absolute positions for RoPE. Shape: (T_q,) or (B, T_q).
                          Defaults to ``arange(start_pos, start_pos + T_q)``.
            attn_mask:    Optional explicit allowed mask (B, T_q, T_total) for
                          continuous batching; overrides causal logic when given.

        Returns:
            Logits over the vocabulary. Shape: (B, T_q, vocab_size)
        """
        return self.forward_stage(
            input_ids, 0, self.config.n_layer, True, True,
            cache, start_pos, position_ids, attn_mask,
        )

    @torch.no_grad()
    def forward_stage(
        self,
        x: torch.Tensor,
        start_layer: int,
        end_layer: int,
        is_first: bool,
        is_last: bool,
        cache: LlamaStaticKVCache | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run layers ``[start_layer, end_layer)`` of the model.

        Splits ``forward`` at its two natural seams — the embedding lookup and
        the final norm/lm_head — so the model can be executed as a chain of
        stages (e.g. across a pipeline-parallel network boundary) with no
        behavior change when run as a single stage covering every layer.
        ``forward`` is exactly ``forward_stage(input_ids, 0, n_layer, True, True, ...)``.

        Args:
            x:            Token ids (B, T_q), dtype long, when ``is_first`` is
                          True. Otherwise the residual stream handed off by a
                          previous stage, shape (B, T_q, d_model).
            start_layer:  First layer index this stage runs (inclusive).
            end_layer:    Last layer index this stage runs (exclusive).
            is_first:     Whether this stage owns the embedding lookup.
            is_last:      Whether this stage owns the final norm + lm_head.
            cache:        Optional KV cache. Only layers in
                          ``[start_layer, end_layer)`` are touched.
            start_pos:    Absolute position of the first token in this forward.
            position_ids: Absolute positions for RoPE. Shape: (T_q,) or (B, T_q).
                          Defaults to ``arange(start_pos, start_pos + T_q)``.
            attn_mask:    Optional explicit allowed mask (B, T_q, T_total) for
                          continuous batching; overrides causal logic when given.

        Returns:
            Logits (B, T_q, vocab_size) when ``is_last``, otherwise the
            residual stream (B, T_q, d_model) to hand off to the next stage.
        """
        if is_first:
            input_ids = x
            B, T_q = input_ids.shape
        else:
            B, T_q = x.shape[0], x.shape[1]

        if start_pos + T_q > self.config.n_ctx:
            raise ValueError(
                f"position {start_pos + T_q} exceeds n_ctx={self.config.n_ctx}"
            )

        if is_first:
            x = self.w.embed_tokens[input_ids]                         # (B, T_q, d_model)

        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + T_q, device=x.device
            )                                                           # (T_q,)

        # Cast RoPE tables to the weight dtype and move to the right device once.
        cos = self.rope_cos.to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin.to(device=x.device, dtype=x.dtype)

        for i in range(start_layer, end_layer):
            x = llama_block(
                x, self.w.layer(i), self.config,
                cos, sin, position_ids,
                cache, i, start_pos, attn_mask,
            )

        if is_last:
            x = rms_norm(x, self.w.norm_weight, self.config.norm_eps)  # (B, T_q, d_model)
            logits = x @ self.w.lm_head.T                              # (B, T_q, vocab_size)
            return logits
        return x

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids`` without a KV cache (O(T²) reference).

        Args:
            input_ids:      Prompt ids. Shape: (B, T_p)
            max_new_tokens: Tokens to append.
            sampling:       Decoding config; defaults to greedy.

        Returns:
            Extended ids. Shape: (B, T_p + max_new_tokens)
        """
        cfg = sampling or SamplingConfig()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)                       # (B, T, V)
            next_tok = sample_next_token(logits[:, -1, :], cfg)   # (B, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cache: LlamaStaticKVCache,
        sampling: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids`` using a KV cache (O(1) per step).

        Prefills the cache with the full prompt, then decodes one token at a
        time, advancing start_pos each step.

        Args:
            input_ids:      Prompt ids. Shape: (B, T_p)
            max_new_tokens: Tokens to append.
            cache:          Pre-allocated LlamaStaticKVCache.
            sampling:       Decoding config; defaults to greedy.

        Returns:
            Extended ids. Shape: (B, T_p + max_new_tokens)
        """
        cfg = sampling or SamplingConfig()
        if max_new_tokens <= 0:
            return input_ids

        B, T_p = input_ids.shape
        pos = torch.arange(T_p, device=input_ids.device)          # (T_p,)
        logits = self.forward(input_ids, cache=cache, start_pos=0, position_ids=pos)
        next_tok = sample_next_token(logits[:, -1, :], cfg)       # (B, 1)
        out = torch.cat([input_ids, next_tok], dim=1)

        for step in range(1, max_new_tokens):
            start_pos = T_p + step - 1
            pos_s = torch.tensor([start_pos], device=input_ids.device)
            logits = self.forward(next_tok, cache=cache, start_pos=start_pos, position_ids=pos_s)
            next_tok = sample_next_token(logits[:, -1, :], cfg)
            out = torch.cat([out, next_tok], dim=1)

        return out
