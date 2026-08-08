"""Decoding strategies: greedy, top-k, and top-p (nucleus) sampling.

All three operate on the final-position logits ``(batch, vocab_size)`` and return
the chosen next token ``(batch, 1)``. Greedy is deterministic; the sampled modes
take an optional ``torch.Generator`` so benchmarks and tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class SamplingMode(str, Enum):
    """Which decoding rule to apply to the next-token distribution."""

    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"


@dataclass
class SamplingConfig:
    """Decoding configuration.

    Attributes:
        mode:        Which sampling rule to use.
        temperature: Logit scaling before softmax (>0). 1.0 = unchanged.
                     Ignored for GREEDY.
        top_k:       Number of highest-prob tokens to keep (TOP_K only).
        top_p:       Cumulative-probability nucleus threshold in (0, 1] (TOP_P only).
        generator:   Optional RNG for reproducible sampling.
    """

    mode: SamplingMode = SamplingMode.GREEDY
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    generator: torch.Generator | None = None


def sample_next_token(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Pick the next token from final-position logits.

    Args:
        logits: Next-token logits. Shape: (batch, vocab_size)
        cfg:    Decoding configuration.

    Returns:
        Chosen token ids. Shape: (batch, 1), dtype long.
    """
    if cfg.mode is SamplingMode.GREEDY:
        return logits.argmax(dim=-1, keepdim=True)               # (B, 1)

    if cfg.temperature <= 0.0:
        raise ValueError("temperature must be > 0 for sampling modes")
    scaled = logits / cfg.temperature                            # (B, V)

    if cfg.mode is SamplingMode.TOP_K:
        filtered = _top_k_filter(scaled, cfg.top_k)              # (B, V), masked
    elif cfg.mode is SamplingMode.TOP_P:
        filtered = _top_p_filter(scaled, cfg.top_p)              # (B, V), masked
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown sampling mode: {cfg.mode}")

    probs = torch.softmax(filtered, dim=-1)                      # (B, V)
    return torch.multinomial(probs, num_samples=1, generator=cfg.generator)  # (B, 1)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask all but the ``k`` highest logits per row with -inf.

    Args:
        logits: (batch, vocab_size)
        k:      Number of tokens to keep (clamped to vocab_size).

    Returns:
        Logits with non-top-k entries set to -inf. Shape: (batch, vocab_size)
    """
    vocab_size = logits.size(-1)
    k = max(1, min(k, vocab_size))
    # kth_value is the smallest logit we keep, per row.
    kth_value = torch.topk(logits, k, dim=-1).values[:, -1, None]  # (B, 1)
    mask = logits < kth_value                                      # (B, V) True where dropped
    return logits.masked_fill(mask, float("-inf"))                 # (B, V)


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative prob >= ``p`` (nucleus).

    Args:
        logits: (batch, vocab_size)
        p:      Cumulative-probability threshold in (0, 1].

    Returns:
        Logits with out-of-nucleus entries set to -inf. Shape: (batch, vocab_size)
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)  # (B, V), (B, V)
    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)          # (B, V)

    # Drop tokens once cumulative prob has already passed p; always keep the top-1.
    drop_sorted = cum_probs - torch.softmax(sorted_logits, dim=-1) >= p      # (B, V)
    drop_sorted[:, 0] = False                                                # keep most-likely token

    # Scatter the per-rank mask back to original vocabulary positions.
    drop = torch.zeros_like(drop_sorted).scatter(dim=-1, index=sorted_idx, src=drop_sorted)
    return logits.masked_fill(drop, float("-inf"))                           # (B, V)
