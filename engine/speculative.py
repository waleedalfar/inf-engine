"""Speculative decoding: draft-then-verify for 2–4× throughput on chat workloads.

Algorithm (Leviathan et al., 2023)
------------------------------------
At each speculative step we generate up to K+1 tokens with a single target-model
forward instead of K+1 separate forwards:

1. DRAFT — run the small draft model K times autoregressively to produce
   draft tokens x̃_{L+1}, …, x̃_{L+K}.  Store the full probability distribution
   at each step (needed for the correction formula on rejection).

2. VERIFY — run the large target model ONCE on the K+1-token sequence
   [x_L, x̃_{L+1}, …, x̃_{L+K}] starting at position L.  The K+1 output
   logits give p_target for each of the K draft tokens plus a free bonus
   prediction for position L+K+1.

3. ACCEPT / REJECT — for draft token x̃_{L+j+1}, accept with probability
   min(1, p_target(x̃_{L+j+1}) / p_draft(x̃_{L+j+1})).
   On rejection at position j, sample a correction token from the residual
   distribution max(0, p_target − p_draft) and stop.

4. EMIT — emit all accepted draft tokens (or just the correction), plus the
   free bonus token if all K were accepted.

Cache management
----------------
Both caches start at length L before each speculative step.  The draft cache
is extended K times during drafting.  The target cache is extended once (K+1
tokens) during verification.  On rejection at draft position j, both caches
are rolled back to L+j using ``reset_to``, so the next iteration starts with
the caches aligned at L+j.

Correctness property (greedy)
------------------------------
Under greedy decoding (temperature=0, argmax) speculative decoding produces
IDENTICAL token sequences to standard decoding:
  - draft token x̃ is accepted iff target's argmax agrees (p_target(x̃) = 1)
  - on rejection, correction = target's argmax (same as standard decode)

This gives us a bit-exact correctness gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.kv_cache import LlamaStaticKVCache
from engine.llama_model import LlamaModel
from engine.sampling import (
    SamplingConfig,
    SamplingMode,
    _apply_repetition_penalty,
    sample_next_token,
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class SpecStats:
    """Acceptance statistics for one generation run."""

    n_accepted: int = 0    # draft tokens accepted by the target
    n_rejected: int = 0    # draft tokens rejected (one per speculative step that fails)
    n_bonus: int = 0       # bonus tokens emitted when all K drafts accepted
    n_steps: int = 0       # number of speculative steps taken

    @property
    def acceptance_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        return self.n_accepted / total if total > 0 else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Average new tokens emitted per speculative step."""
        total = self.n_accepted + self.n_rejected + self.n_bonus
        return total / self.n_steps if self.n_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_probs(
    logits: torch.Tensor,
    cfg: SamplingConfig,
    context_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert next-token logits to a probability distribution matching ``cfg``.

    Applies repetition penalty (if configured), temperature scaling, and
    top-k/top-p masking in the same way that ``sample_next_token`` does, so
    draft and target distributions are comparable.

    Args:
        logits:      (vocab_size,) raw logits for one position.
        cfg:         Sampling configuration.
        context_ids: All token ids seen so far (1-D). Used for repetition
                     penalty when ``cfg.repetition_penalty != 1.0``.

    Returns:
        (vocab_size,) probability distribution (sums to 1).
    """
    if cfg.repetition_penalty != 1.0 and context_ids is not None:
        # _apply_repetition_penalty expects (B, V); unsqueeze/squeeze around it.
        logits = _apply_repetition_penalty(
            logits.unsqueeze(0), context_ids, cfg.repetition_penalty
        ).squeeze(0)

    if cfg.mode is SamplingMode.GREEDY:
        # One-hot at argmax — avoids float noise in the correction formula.
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs

    scaled = logits / max(cfg.temperature, 1e-8)

    if cfg.mode is SamplingMode.TOP_K:
        k = max(1, min(cfg.top_k, logits.shape[-1]))
        kth = torch.topk(scaled, k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    elif cfg.mode is SamplingMode.TOP_P:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= cfg.top_p
        drop[0] = False
        mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(0, sorted_idx, drop)
        scaled = scaled.masked_fill(mask, float("-inf"))

    return torch.softmax(scaled, dim=-1)


def _sample_from_probs(probs: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Sample one token from a pre-computed probability distribution.

    Args:
        probs: (vocab_size,) probability distribution.
        cfg:   Sampling config — greedy takes argmax, others use multinomial.

    Returns:
        (1, 1) token id tensor.
    """
    if cfg.mode is SamplingMode.GREEDY:
        return probs.argmax(keepdim=True).unsqueeze(0)   # (1, 1)
    return torch.multinomial(probs.unsqueeze(0), num_samples=1, generator=cfg.generator)  # (1, 1)


def _correction_sample(
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    cfg: SamplingConfig,
) -> torch.Tensor:
    """Sample from the residual distribution max(0, p_target − p_draft) / Z.

    This is the corrected distribution used when a draft token is rejected.
    It guarantees that the marginal distribution of accepted tokens equals
    p_target (the Leviathan et al. unbiasedness result).

    Args:
        p_target: (vocab_size,) target probability distribution.
        p_draft:  (vocab_size,) draft probability distribution.
        cfg:      Sampling config (used only to decide argmax vs multinomial).

    Returns:
        (1, 1) sampled token id.
    """
    residual = (p_target - p_draft).clamp(min=0.0)
    total = residual.sum()
    if total < 1e-9:
        # Edge case: p_target == p_draft everywhere (shouldn't happen in practice).
        residual = p_target
    else:
        residual = residual / total

    if cfg.mode is SamplingMode.GREEDY:
        return residual.argmax(keepdim=True).unsqueeze(0)   # (1, 1)
    return torch.multinomial(residual.unsqueeze(0), num_samples=1, generator=cfg.generator)


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Speculative decoder: small draft model + large target model.

    Generates up to ``n_draft + 1`` tokens per target-model forward pass,
    compared to 1 token per forward in standard decoding.

    Args:
        draft:   Small (fast) language model — e.g., LLaMA 3.2 1B.
        target:  Large (quality) language model — e.g., LLaMA 3 8B.
        n_draft: Number of draft tokens to speculate per step.
    """

    def __init__(
        self,
        draft: LlamaModel,
        target: LlamaModel,
        n_draft: int = 4,
    ) -> None:
        self.draft = draft
        self.target = target
        self.n_draft = n_draft

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        draft_cache: LlamaStaticKVCache,
        target_cache: LlamaStaticKVCache,
        sampling: SamplingConfig | None = None,
        eos_token: int | None = None,
    ) -> tuple[torch.Tensor, SpecStats]:
        """Autoregressively generate ``max_new_tokens`` using speculative decoding.

        Both caches must be freshly allocated (length 0) and large enough for
        ``len(input_ids) + max_new_tokens + 1`` tokens.

        Args:
            input_ids:      Prompt ids. Shape: (1, T_p)
            max_new_tokens: Maximum new tokens to generate.
            draft_cache:    Pre-allocated LlamaStaticKVCache for the draft model.
            target_cache:   Pre-allocated LlamaStaticKVCache for the target model.
            sampling:       Decoding config (default: greedy).
            eos_token:      Optional EOS token id; generation stops on first emit.

        Returns:
            (output_ids, stats): full token sequence (1, T_p + n_new) and stats.
        """
        cfg = sampling or SamplingConfig()
        device = input_ids.device
        B, T_p = input_ids.shape
        assert B == 1, "speculative decoding currently supports batch=1 only"

        # ── Prefill both models with the full prompt ──────────────────────
        pos = torch.arange(T_p, device=device)
        target_logits = self.target.forward(input_ids, cache=target_cache,
                                            start_pos=0, position_ids=pos)
        self.draft.forward(input_ids, cache=draft_cache,
                           start_pos=0, position_ids=pos)

        # Sample first token from target (target is always authoritative).
        # No penalty context yet — penalty tracks generated tokens only.
        first_tok = sample_next_token(target_logits[:, -1, :], cfg)   # (1, 1)
        out = torch.cat([input_ids, first_tok], dim=1)
        if eos_token is not None and first_tok.item() == eos_token:
            return out, SpecStats()

        # State: both caches at T_p; last_tok is at position T_p (not yet cached).
        last_tok = first_tok                                           # (1, 1)
        L = T_p                                                        # cache fill level
        generated = 1
        stats = SpecStats()

        while generated < max_new_tokens:
            K = min(self.n_draft, max_new_tokens - generated)
            # Only generated tokens (not the prompt) as the penalty context.
            # Using the full out[0] would include prompt tokens that appear
            # many times in long system prompts, causing over-suppression.
            step_context = out[0, T_p:]                                # (n_generated,)

            # ── DRAFT PHASE ───────────────────────────────────────────────
            # Run draft K times; store full distributions for the correction formula.
            draft_tokens: list[torch.Tensor] = []    # (1, 1) each
            draft_probs: list[torch.Tensor] = []     # (vocab,) each

            cur = last_tok
            for k in range(K):
                pos_k = torch.tensor([L + k], device=device)
                d_logits = self.draft.forward(cur, cache=draft_cache,
                                              start_pos=L + k, position_ids=pos_k)
                d_probs = _get_probs(d_logits[0, -1], cfg, context_ids=step_context)  # (vocab,)
                tok = _sample_from_probs(d_probs, cfg)                # (1, 1)
                draft_tokens.append(tok)
                draft_probs.append(d_probs)
                cur = tok
            # draft cache now at L + K; draft_tokens = [x̃_{L+1}, …, x̃_{L+K}]

            # ── VERIFY PHASE ──────────────────────────────────────────────
            # Target processes [last_tok, x̃_{L+1}, …, x̃_{L+K}] (K+1 tokens).
            # Output logits[:, j, :] = p_target(· | x_0..x_{L+j})  for j=0..K
            verify_input = torch.cat([last_tok] + draft_tokens, dim=1)   # (1, K+1)
            pos_range = torch.arange(L, L + K + 1, device=device)
            t_logits = self.target.forward(verify_input, cache=target_cache,
                                           start_pos=L, position_ids=pos_range)
            # target cache now at L + K + 1

            # ── ACCEPT / REJECT ───────────────────────────────────────────
            stats.n_steps += 1
            accepted_this_step: list[torch.Tensor] = []
            all_accepted = True

            for j in range(K):
                tok_id = draft_tokens[j].item()
                p_target_j = _get_probs(t_logits[0, j], cfg, context_ids=step_context)  # (vocab,)
                p_draft_j = draft_probs[j]                            # (vocab,)

                accept_prob = min(1.0, (p_target_j[tok_id] / (p_draft_j[tok_id] + 1e-10)).item())

                if torch.rand(1, generator=cfg.generator).item() < accept_prob:
                    accepted_this_step.append(draft_tokens[j])
                    stats.n_accepted += 1
                else:
                    # Correction: sample from residual max(0, p_target − p_draft).
                    corr = _correction_sample(p_target_j, p_draft_j, cfg)
                    accepted_this_step.append(corr)
                    stats.n_rejected += 1

                    # Rollback caches to L+j+1: keep positions 0..L+j (the j+1
                    # processed positions including last_tok at L), discard L+j+1 onwards.
                    target_cache.reset_to(L + j + 1)
                    draft_cache.reset_to(L + j + 1)

                    all_accepted = False
                    break

            if all_accepted:
                # Bonus token from target's prediction at position L+K+1.
                bonus_probs = _get_probs(t_logits[0, K], cfg, context_ids=step_context)
                bonus = _sample_from_probs(bonus_probs, cfg)
                accepted_this_step.append(bonus)
                stats.n_bonus += 1

                # Sync draft cache to L+K (process the last draft token x̃_{L+K}).
                pos_sync = torch.tensor([L + K], device=device)
                self.draft.forward(draft_tokens[-1], cache=draft_cache,
                                   start_pos=L + K, position_ids=pos_sync)
                # draft cache now at L + K + 1, matching target cache.

            # ── EMIT accepted tokens ──────────────────────────────────────
            for tok in accepted_this_step:
                out = torch.cat([out, tok], dim=1)
                generated += 1
                if eos_token is not None and tok.item() == eos_token:
                    return out, stats
                if generated >= max_new_tokens:
                    return out, stats

            # Advance state: last_tok = last emitted, L = new cache length.
            last_tok = accepted_this_step[-1]
            n_emitted = len(accepted_this_step)
            L = L + n_emitted

        return out, stats
