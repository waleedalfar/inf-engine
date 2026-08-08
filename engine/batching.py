"""Static batching baseline: pad N prompts to equal length, decode them in lockstep.

This is the Phase 3 reference point every later optimization is measured against. It is
deliberately the *naive* batching strategy:

* Prompts are **left-padded** to the longest prompt in the batch, so all sequences align
  at the right edge and generate in lockstep, one shared column per step.
* A **key padding mask** stops real tokens from attending to pad, and per-row
  **position ids** give each real token its true sequence position (a token's sequence
  position differs from its buffer index once you left-pad).
* Every sequence runs for the full ``max_new_tokens`` regardless of when it would naturally
  stop — there is no early exit and no slot reuse.

Its two inefficiencies are exactly what Phase 4 (continuous batching) removes:
1. **Padding waste** — short prompts carry dead pad tokens through every layer and occupy
   KV-cache slots that compute nothing useful.
2. **Lockstep / head-of-line blocking** — the batch is only as free as its longest member;
   a finished sequence cannot leave and a new one cannot join until the whole batch is done.
"""

from __future__ import annotations

import torch

from engine.kv_cache import StaticKVCache
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, sample_next_token


def left_pad(
    prompts: list[list[int]],
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad a list of token-id sequences to a rectangular batch.

    Args:
        prompts: List of B token-id lists (varying lengths).
        pad_id:  Token id to pad with (its value is irrelevant — it is masked out).
        device:  Target device.

    Returns:
        input_ids:      (B, T_max) left-padded token ids.
        attention_mask: (B, T_max) 1 for real tokens, 0 for padding.
        prompt_lens:    (B,) true length of each prompt.
    """
    lens = [len(p) for p in prompts]
    t_max = max(lens)
    batch = len(prompts)
    input_ids = torch.full((batch, t_max), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, t_max), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        input_ids[i, t_max - len(p):] = torch.tensor(p, dtype=torch.long, device=device)  # right-align
        attention_mask[i, t_max - len(p):] = 1
    prompt_lens = torch.tensor(lens, dtype=torch.long, device=device)
    return input_ids, attention_mask, prompt_lens


@torch.no_grad()
def generate_batched(
    model: GPT2Model,
    prompts: list[list[int]],
    max_new_tokens: int,
    pad_id: int,
    sampling: SamplingConfig | None = None,
) -> torch.Tensor:
    """Statically-batched greedy/sampled generation over left-padded prompts.

    Args:
        model:          The GPT-2 model.
        prompts:        List of B token-id lists.
        max_new_tokens: Number of tokens to generate for every sequence (no early stop).
        pad_id:         Padding token id.
        sampling:       Decoding config (defaults to greedy).

    Returns:
        Generated tokens only. Shape: (B, max_new_tokens). Row ``i`` is the continuation of
        ``prompts[i]`` and is identical (greedy) to generating that prompt on its own.
    """
    cfg = sampling if sampling is not None else SamplingConfig()
    device = model.w.wte.device
    input_ids, attention_mask, prompt_lens = left_pad(prompts, pad_id, device)
    batch, t_max = input_ids.shape

    if t_max + max_new_tokens > model.config.n_ctx:
        raise ValueError(
            f"padded prompt ({t_max}) + new ({max_new_tokens}) exceeds n_ctx={model.config.n_ctx}"
        )

    cache = StaticKVCache(
        model.config, batch=batch, max_seq=t_max + max_new_tokens,
        device=device, dtype=model.w.wte.dtype,
    )

    # --- prefill: real tokens get true positions; pad gets 0 (masked anyway) ---
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)     # (B, T_max)
    key_padding = attention_mask.clone()                              # (B, T_max) grows during decode
    logits = model.forward(
        input_ids, cache=cache, start_pos=0,
        position_ids=position_ids, key_padding_mask=key_padding,
    )                                                                 # (B, T_max, V)
    next_tok = sample_next_token(logits[:, -1, :], cfg)               # (B, 1) first generated token
    generated = [next_tok]
    next_seq_pos = prompt_lens.clone()                               # (B,) seq position of next_tok

    # --- decode: feed one token per step, in lockstep across the batch ---
    for step in range(max_new_tokens - 1):
        buf_pos = t_max + step                                        # buffer index for next_tok
        key_padding = torch.cat(
            [key_padding, torch.ones((batch, 1), dtype=key_padding.dtype, device=device)], dim=1
        )                                                             # newly generated token is real
        logits = model.forward(
            next_tok, cache=cache, start_pos=buf_pos,
            position_ids=next_seq_pos.unsqueeze(1), key_padding_mask=key_padding,
        )                                                             # (B, 1, V)
        next_tok = sample_next_token(logits[:, -1, :], cfg)           # (B, 1)
        generated.append(next_tok)
        next_seq_pos = next_seq_pos + 1

    return torch.cat(generated, dim=1)                                # (B, max_new_tokens)
