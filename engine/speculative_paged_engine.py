"""Speculative decoding over paged KV caches with CUDA graphs.

``SpeculativePagedEngine`` wraps two ``LlamaPagedEngine`` instances (target and
draft) and runs the Leviathan et al. accept/reject loop.  The draft model uses
graphed single-token steps (``_step_one_graphed``); the target runs an eager
K+1-token verify step (``_step_verify_eager``).

Architecture
------------
- CUDA graph overhead is eliminated for the small draft model (≈0.28 GB INT4
  for Qwen3-0.6B) — each of the K draft steps is a graph replay instead of a
  full eager forward.
- The verify step stays eager because it writes K+1 tokens per call (variable
  q_len), which the current graph infrastructure doesn't support.  Phase 3
  of the roadmap will extend ``extend_static`` to handle q_len > 1.

Correctness property (greedy)
------------------------------
Under ``SamplingMode.GREEDY`` with identical draft and target models, this
engine produces token-for-token identical output to standard ``LlamaPagedEngine``
decode — the standard correctness gate for speculative decoding (see
``tests/test_speculative_paged.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.speculative import SpecStats, _correction_sample, _get_probs, _sample_from_probs


@dataclass
class _SeqState:
    """Per-sequence bookkeeping inside one _generate_one call."""
    t_sid: int                          # target engine seq_id
    d_sid: int                          # draft engine seq_id
    t_saved_next_id: int                # target._next_seq_id to restore after
    d_saved_next_id: int                # draft._next_seq_id to restore after
    req: LlamaRequest
    draft_req: LlamaRequest


class SpeculativePagedEngine:
    """Two-model speculative decode engine backed by paged KV caches.

    Args:
        target:   Fully loaded target ``LlamaPagedEngine`` (large model).
                  ``enable_cuda_graphs`` can be False — verify is always eager.
        draft:    Fully loaded draft ``LlamaPagedEngine`` (small model).
                  Should have ``enable_cuda_graphs=True`` on CUDA for best
                  performance.
        n_draft:  Number of draft tokens to generate per speculative step.
        eos_token: Token id that signals end-of-sequence.  When None, only
                  ``max_new_tokens`` is used as a stopping criterion.
    """

    def __init__(
        self,
        target: LlamaPagedEngine,
        draft: LlamaPagedEngine,
        n_draft: int = 4,
        eos_token: int | None = None,
    ) -> None:
        self.target = target
        self.draft = draft
        self.n_draft = n_draft
        self.eos = eos_token

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_offline(
        self, requests: list[LlamaRequest]
    ) -> tuple[dict[int, list[int]], SpecStats]:
        """Generate all requests sequentially, returning results and stats."""
        combined = SpecStats()
        results: dict[int, list[int]] = {}
        for req in requests:
            gen_ids, stats = self._generate_one(req)
            results[req.req_id] = gen_ids
            combined.n_accepted += stats.n_accepted
            combined.n_rejected += stats.n_rejected
            combined.n_bonus += stats.n_bonus
            combined.n_steps += stats.n_steps
        return results, combined

    # ------------------------------------------------------------------
    # Internal: one request
    # ------------------------------------------------------------------

    def _generate_one(self, req: LlamaRequest) -> tuple[list[int], SpecStats]:
        target, draft = self.target, self.draft
        now = time.perf_counter()

        # ── Prefill target ──────────────────────────────────────────────
        t_sid = target._next_seq_id
        target._prefill(req, now)
        # After _prefill: req.generated = [first_tok], _active[t_sid] set,
        #                 ensure_slot already called for next position.
        if t_sid not in target._active:
            # Finished in prefill (max_new_tokens==1 or hit EOS).
            target.completed.pop()
            return req.generated, SpecStats()

        # ── Prefill draft (shadow request — tokens discarded) ───────────
        draft_req = LlamaRequest(
            req_id=-1,
            prompt_ids=req.prompt_ids,
            max_new_tokens=req.max_new_tokens,
        )
        d_sid = draft._next_seq_id
        draft._prefill(draft_req, now)
        if d_sid not in draft._active:
            draft.completed.pop()

        # Sync draft's active input token to match target's first generated token.
        # Draft's own prefill may have predicted a different first token — we
        # override so both caches are in the same state before the spec loop.
        first_tok_tensor = target._active[t_sid][1]
        draft._active[d_sid] = (draft_req, first_tok_tensor.to(draft.device))

        stats = SpecStats()

        # ── Speculative decode loop ─────────────────────────────────────
        while True:
            remaining = req.max_new_tokens - len(req.generated)
            if remaining <= 0:
                break

            K = min(self.n_draft, remaining - 1)
            if K <= 0:
                # Only one token left — run a single target step and stop.
                target.cache.ensure_slot(t_sid)
                context_ids = _make_ctx(req.generated, target.device)
                tok, _ = target._step_one_eager(t_sid, context_ids)
                req.generated.append(tok)
                break

            # current length in both caches (must be aligned)
            L = target.cache.seq_lens[t_sid]

            # context for repetition penalty (accepted generated tokens only)
            context_ids = _make_ctx(req.generated, target.device)
            draft_ctx = _make_ctx(req.generated, draft.device)

            # ── Draft phase ──────────────────────────────────────────────
            draft_tokens: list[int] = []
            draft_logits: list[torch.Tensor] = []
            for k in range(K):
                tok, logits = draft._step_one_graphed(d_sid, draft_ctx)
                draft_tokens.append(tok)
                draft_logits.append(logits)
                if k < K - 1:
                    draft.cache.ensure_slot(d_sid)

            # ── Verify phase ─────────────────────────────────────────────
            last_target_tok = int(target._active[t_sid][1].item())
            verify_ids = [last_target_tok] + draft_tokens       # K+1 tokens
            t_logits = target._step_verify_eager(t_sid, verify_ids)  # (K+1, vocab)

            # ── Accept / reject ──────────────────────────────────────────
            emitted: list[int] = []
            all_accepted = True

            for j in range(K):
                tok_id = draft_tokens[j]
                p_t = _get_probs(t_logits[j], target.cfg, context_ids)
                p_d = _get_probs(draft_logits[j], draft.cfg, draft_ctx)
                accept_prob = min(1.0, (p_t[tok_id] / (p_d[tok_id] + 1e-10)).item())

                u = torch.rand(1, generator=target.cfg.generator).item()
                if u < accept_prob:
                    emitted.append(tok_id)
                    stats.n_accepted += 1
                else:
                    corr = int(_correction_sample(p_t, p_d, target.cfg).item())
                    emitted.append(corr)
                    stats.n_rejected += 1

                    # Rollback both caches to L+j+1 (KVs for positions 0..L+j kept).
                    # position L is last_target_tok (first verify input), written at pos L.
                    # positions L+1..L+j are accepted draft tokens.
                    # position L+j+1 is where the correction goes next.
                    target.cache.reset_to(t_sid, L + j + 1)
                    draft.cache.reset_to(d_sid, L + j + 1)

                    corr_t = torch.tensor([corr], device=target.device)
                    target._active[t_sid] = (req, corr_t)
                    draft._active[d_sid] = (draft_req, corr_t.to(draft.device))

                    target.cache.ensure_slot(t_sid)
                    draft.cache.ensure_slot(d_sid)
                    all_accepted = False
                    break

            if all_accepted:
                # Sample bonus token from target position K.
                p_bonus = _get_probs(t_logits[K], target.cfg, context_ids)
                bonus = int(_sample_from_probs(p_bonus, target.cfg).item())
                emitted.append(bonus)
                stats.n_bonus += 1

                # Sync draft: advance it from L+K to L+K+1 so it's aligned with target.
                draft.cache.ensure_slot(d_sid)
                draft._step_one_graphed(d_sid, draft_ctx)               # output discarded

                # Both caches are now at L+K+1; point _active at bonus token.
                bonus_t = torch.tensor([bonus], device=target.device)
                target._active[t_sid] = (req, bonus_t)
                draft._active[d_sid] = (draft_req, bonus_t.to(draft.device))

                target.cache.ensure_slot(t_sid)
                draft.cache.ensure_slot(d_sid)

            stats.n_steps += 1

            # ── Emit tokens, check stopping conditions ───────────────────
            done = False
            for tok in emitted:
                req.generated.append(tok)
                if (
                    len(req.generated) >= req.max_new_tokens
                    or tok == self.eos
                    or (req.stop_check is not None and req.stop_check(req.generated))
                ):
                    done = True
                    break
            if done:
                break

        # ── Cleanup ─────────────────────────────────────────────────────
        target._active.pop(t_sid, None)
        target.cache.free_sequence(t_sid)
        target._generated.pop(t_sid, None)
        if d_sid in draft._active:
            draft._active.pop(d_sid)
        if d_sid in draft.cache.seq_lens:
            draft.cache.free_sequence(d_sid)
        draft._generated.pop(d_sid, None)

        # Remove spurious completions added by _prefill (if any re-queued).
        target.completed = [r for r in target.completed if r.req_id != req.req_id]
        draft.completed = [r for r in draft.completed if r.req_id == -1]
        draft.completed.clear()

        return req.generated, stats


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_ctx(generated: list[int], device: str) -> torch.Tensor | None:
    """Build a 1-D context tensor for repetition penalty, or None if empty."""
    if not generated:
        return None
    return torch.tensor(generated, device=device)
