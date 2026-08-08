"""The full GPT-2 forward pass and autoregressive generation.

Two decode paths live here on purpose:

* ``generate`` (Phase 1) recomputes the entire sequence every step — no cache. It is
  the O(T) per-step / O(T^2) total reference.
* ``generate_cached`` (Phase 2) computes the prompt once (prefill), then feeds one
  token at a time, reusing cached K/V. Decode is O(1) work per step.

Both must produce identical tokens; ``tests/test_phase2_kv_cache.py`` enforces that.
"""

from __future__ import annotations

import torch

from engine.block import transformer_block
from engine.config import GPT2Config
from engine.kv_cache import KVCache
from engine.layers import layer_norm
from engine.sampling import SamplingConfig, sample_next_token
from engine.weights import GPT2Weights


class GPT2Model:
    """GPT-2 language model as a callable over token ids (no nn.Module, no HF).

    Holds the loaded weights and config. The model itself is stateless; any
    autoregressive state lives in the optional ``KVCache`` passed to ``forward``.
    """

    def __init__(self, weights: GPT2Weights, config: GPT2Config) -> None:
        self.w = weights
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-token logits for every input position.

        Args:
            input_ids:        Token ids. Shape: (batch, q_len), dtype long.
            cache:            Optional KV cache. When given, only ``input_ids`` are
                              processed and their K/V are appended at ``start_pos``.
            start_pos:        Absolute (buffer) position of the first token in
                              ``input_ids``. 0 for a full (Phase 1) forward or prefill.
            position_ids:     Optional explicit positional indices. Shape: (batch, q_len).
                              Needed for static batching of left-padded sequences, where a
                              token's sequence position differs from its buffer index.
                              When None, defaults to ``arange(start_pos, start_pos+q_len)``.
            key_padding_mask: Optional key padding mask (batch, total_len) for batching.
            attn_mask:        Optional explicit allowed mask (batch, q_len, total_len) for
                              continuous batching (overrides causal/padding when given).

        Returns:
            Logits over the vocabulary. Shape: (batch, q_len, vocab_size)
        """
        batch, q_len = input_ids.shape                                # (B, T_q)
        end = start_pos + q_len
        if end > self.config.n_ctx:
            raise ValueError(f"position {end} exceeds n_ctx={self.config.n_ctx}")

        # --- embeddings: token lookup + learned positional lookup ---
        tok_emb = self.w.wte[input_ids]                               # (B, T_q, d_model)
        if position_ids is None:
            positions = torch.arange(start_pos, end, device=input_ids.device)  # (T_q,)
            pos_emb = self.w.wpe[positions]                          # (T_q, d_model) -> broadcasts
        else:
            pos_emb = self.w.wpe[position_ids]                       # (B, T_q, d_model)
        x = tok_emb + pos_emb                                        # (B, T_q, d_model)

        # --- L transformer blocks over the residual stream ---
        for i in range(self.config.n_layer):
            x = transformer_block(
                x, self.w.block(i), self.config, cache, i, start_pos, key_padding_mask, attn_mask
            )  # (B, T_q, d_model)

        # --- final layer norm ---
        x = layer_norm(x, self.w.ln_f_weight, self.w.ln_f_bias, self.config.layer_norm_eps)  # (B, T_q, d_model)

        # --- unembed via weight tying: logits = x @ wte^T (no separate lm_head) ---
        logits = x @ self.w.wte.transpose(0, 1)                      # (B, T_q, d_model) @ (d_model, V) -> (B, T_q, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids`` WITHOUT a KV cache (Phase 1 reference).

        Recomputes the full sequence at every step. Correct but O(T^2) total.

        Args:
            input_ids:      Prompt token ids. Shape: (batch, seq)
            max_new_tokens: Number of tokens to append.
            sampling:       Decoding strategy; defaults to greedy.

        Returns:
            ``input_ids`` with new tokens appended. Shape: (batch, seq + max_new_tokens)
        """
        cfg = sampling if sampling is not None else SamplingConfig()  # default: greedy
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)                         # (B, T, V)
            next_token = sample_next_token(logits[:, -1, :], cfg)    # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)    # (B, T+1)
        return input_ids

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cache: KVCache,
        sampling: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids`` USING a KV cache (Phase 2).

        Prefill processes the whole prompt once (filling the cache), then each decode
        step feeds only the single new token. O(1) compute per step apart from the
        attention dot-products against cached keys.

        Args:
            input_ids:      Prompt token ids. Shape: (batch, prompt_len)
            max_new_tokens: Number of tokens to append.
            cache:          KV cache sized for at least prompt_len + max_new_tokens.
            sampling:       Decoding strategy; defaults to greedy.

        Returns:
            ``input_ids`` with new tokens appended. Shape: (batch, prompt_len + max_new_tokens)
        """
        cfg = sampling if sampling is not None else SamplingConfig()  # default: greedy
        if max_new_tokens <= 0:
            return input_ids

        batch, prompt_len = input_ids.shape                          # (B, T_p)
        # --- prefill: run the whole prompt, cache K/V for positions [0, prompt_len) ---
        logits = self.forward(input_ids, cache=cache, start_pos=0)   # (B, T_p, V)
        next_token = sample_next_token(logits[:, -1, :], cfg)        # (B, 1)
        out = torch.cat([input_ids, next_token], dim=1)              # (B, T_p+1)

        # --- decode: feed one token at a time, advancing start_pos ---
        for step in range(1, max_new_tokens):
            start_pos = prompt_len + step - 1                        # absolute pos of `next_token`
            logits = self.forward(next_token, cache=cache, start_pos=start_pos)  # (B, 1, V)
            next_token = sample_next_token(logits[:, -1, :], cfg)    # (B, 1)
            out = torch.cat([out, next_token], dim=1)                # grow by 1
        return out
