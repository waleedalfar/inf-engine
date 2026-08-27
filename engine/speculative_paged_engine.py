"""Speculative decoding over paged KV caches with CUDA graphs.

``SpeculativePagedEngine`` wraps two ``LlamaPagedEngine`` instances (target and
draft) and runs the Leviathan et al. accept/reject loop.  Both draft and target
use CUDA graphs: the draft model uses ``_step_one_graphed`` (q_len=1) and the
target uses ``_step_verify_graphed`` (q_len=K+1).

Architecture
------------
- CUDA graph overhead is eliminated for the small draft model (≈0.28 GB INT4
  for Qwen3-0.6B) — each of the K draft steps is a graph replay instead of a
  full eager forward.
- The verify step uses a separate graph family keyed by (len_bucket, q_len).
  ``extend_static`` loops over q_len positions at capture time so the Python
  loop unrolls into static CUDA ops.

Correctness property (greedy)
------------------------------
Under ``SamplingMode.GREEDY`` with identical draft and target models, this
engine produces token-for-token identical output to standard ``LlamaPagedEngine``
decode — the standard correctness gate for speculative decoding (see
``tests/test_speculative_paged.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.sampling import SamplingConfig, SamplingMode, _apply_repetition_penalty
from engine.sampling import sample_next_token
from engine.speculative import SpecStats, _correction_sample, _sample_from_probs


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
                  Set ``enable_cuda_graphs=True`` to use the graphed verify path.
        draft:    Fully loaded draft ``LlamaPagedEngine`` (small model).
                  Should have ``enable_cuda_graphs=True`` on CUDA for best
                  performance.
        n_draft:  Draft tokens per speculative step. An int for a fixed value,
                  or a callable ``(context_len) -> int`` to vary it with the
                  sequence length.

                  Varying it matters at long context. Every draft step now pays
                  attention cost proportional to the KV history, so the marginal
                  draft token gets more expensive as context grows while its
                  marginal yield (``p**k`` for acceptance ``p``) keeps shrinking.
                  The optimum therefore falls with length — see
                  ``bench/ctx_scaling_bench.py`` for the sweep that pins it.
        eos_token: Token id that signals end-of-sequence.  When None, only
                  ``max_new_tokens`` is used as a stopping criterion.
    """

    # Below this many shared tokens, rolling back and re-prefilling the suffix
    # costs more bookkeeping than it saves.
    MIN_REUSE_TOKENS = 64

    def __init__(
        self,
        target: LlamaPagedEngine,
        draft: LlamaPagedEngine,
        n_draft: int | Callable[[int], int] = 4,
        eos_token: int | None = None,
    ) -> None:
        self.target = target
        self.draft = draft
        self.n_draft = n_draft
        self.max_n_draft = n_draft if isinstance(n_draft, int) else 16
        self.eos = eos_token
        # Cross-turn prefix reuse: the sequences and the exact token history
        # currently held in both KV caches. None when nothing is resident.
        self._resident: dict | None = None

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
            combined.prefill_s += stats.prefill_s
            combined.decode_s += stats.decode_s
        return results, combined

    def generate_resident(self, req: LlamaRequest) -> tuple[list[int], SpecStats]:
        """Generate, reusing whatever prefix of ``req.prompt_ids`` is already cached.

        A chat turn's prompt is the previous turn's prompt plus its answer plus
        the new message — strictly append-only. Re-prefilling all of it every
        turn is the dominant interactive cost: measured 4430 ms to re-read 7152
        tokens before emitting a single new one, against 27 ms/step of decode.

        Keeps both sequences alive between calls, rolls the caches back to the
        longest common prefix, and prefills only what is new. Falls back to a
        clean full prefill whenever the prefix does not match — a different
        conversation, an edited history, or a cache that was dropped.

        The caller owns the lifetime: call ``release()`` when the session ends.
        """
        target, draft = self.target, self.draft
        res = self._resident
        if res is None:
            return self._generate_one(req, keep_resident=True)

        shared = _common_prefix_len(res["tokens"], req.prompt_ids)
        # Both caches hold every resident token except the last, which is still
        # pending in _active. Never reuse the whole new prompt: at least one
        # token must go through the model to produce logits.
        cached = min(
            shared, res["cached"], len(req.prompt_ids) - 1,
            target.cache.seq_lens.get(t_sid_peek := res["t_sid"], 0),
            draft.cache.seq_lens.get(res["d_sid"], 0),
        )
        if cached < self.MIN_REUSE_TOKENS:
            self.release()
            return self._generate_one(req, keep_resident=True)

        t_sid, d_sid = res["t_sid"], res["d_sid"]
        now = time.perf_counter()
        target.cache.reset_to(t_sid, cached)
        draft.cache.reset_to(d_sid, cached)

        suffix = req.prompt_ids[cached:]
        logits = target._prefill_forward(t_sid, suffix, base=cached)
        draft._prefill_forward(d_sid, suffix, base=cached)

        first = sample_next_token(logits[:, -1, :], target.cfg)      # (1, 1)
        req.generated.append(int(first))
        req.start_time = now

        first_t = first.view(1).to(target.device)
        draft_req = LlamaRequest(req_id=-1, prompt_ids=req.prompt_ids,
                                 max_new_tokens=req.max_new_tokens)
        target._active[t_sid] = (req, first_t)
        draft._active[d_sid] = (draft_req, first_t.to(draft.device))
        target._generated[t_sid] = [int(first)]
        target.cache.ensure_slot(t_sid)
        draft.cache.ensure_slot(d_sid)

        if len(req.generated) >= req.max_new_tokens or int(first) == self.eos:
            self._remember(req, t_sid, d_sid)
            return req.generated, SpecStats(prefill_s=time.perf_counter() - now)

        return self._decode(req, draft_req, t_sid, d_sid, now, keep_resident=True)

    def release(self) -> None:
        """Free the resident sequences. Safe to call when nothing is resident."""
        res, self._resident = self._resident, None
        if res is None:
            return
        for eng, sid in ((self.target, res["t_sid"]), (self.draft, res["d_sid"])):
            eng._active.pop(sid, None)
            if sid in eng.cache.seq_lens:
                eng.cache.free_sequence(sid)
            eng._generated.pop(sid, None)

    def _remember(self, req: LlamaRequest, t_sid: int, d_sid: int) -> None:
        """Record what the caches now hold, for the next turn to match against."""
        # The reusable length is what *both* caches hold. They can differ by a
        # token: the draft's bonus-sync step and the target's verify write on
        # different schedules, and rolling back past either one's contents
        # would reuse KV that was never written.
        self._resident = {
            "t_sid": t_sid,
            "d_sid": d_sid,
            "tokens": list(req.prompt_ids) + list(req.generated),
            "cached": min(self.target.cache.seq_lens.get(t_sid, 0),
                          self.draft.cache.seq_lens.get(d_sid, 0)),
        }

    def _n_draft_for(self, context_len: int) -> int:
        """Draft tokens to speculate at this context length."""
        if callable(self.n_draft):
            return max(1, int(self.n_draft(context_len)))
        return self.n_draft

    # ------------------------------------------------------------------
    # Internal: one request
    # ------------------------------------------------------------------

    def _generate_one(self, req: LlamaRequest,
                      keep_resident: bool = False) -> tuple[list[int], SpecStats]:
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

        return self._decode(req, draft_req, t_sid, d_sid, now, keep_resident)

    def _decode(self, req: LlamaRequest, draft_req: LlamaRequest,
                t_sid: int, d_sid: int, now: float,
                keep_resident: bool) -> tuple[list[int], SpecStats]:
        """The speculative loop, from a prefilled pair of caches to completion.

        Split out of ``_generate_one`` so cross-turn prefix reuse can reach it
        after an incremental prefill instead of a full one.
        """
        target, draft = self.target, self.draft

        # Prefill (both models) is done; everything after this is decode.
        # Separating them matters: a long prompt with a short answer makes
        # end-to-end tok/s look far worse than the decode rate actually is.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_start = time.perf_counter()
        stats = SpecStats(prefill_s=decode_start - now)

        # Repetition-penalty context. Only materialised when a penalty is
        # actually configured — otherwise _get_probs ignores it and rebuilding
        # a growing device tensor every step is pure waste.
        ctx = (
            _RollingCtx(req.generated, req.max_new_tokens + self.max_n_draft + 2,
                        target.device, draft.device)
            if target.cfg.repetition_penalty != 1.0 or draft.cfg.repetition_penalty != 1.0
            else None
        )

        # ── Speculative decode loop ─────────────────────────────────────
        while True:
            remaining = req.max_new_tokens - len(req.generated)
            if remaining <= 0:
                break

            t_ctx = ctx.target if ctx is not None else None
            d_ctx = ctx.draft if ctx is not None else None

            K = min(self._n_draft_for(target.cache.seq_lens[t_sid]), remaining - 1)
            if K <= 0:
                # Only one token left — run a single target step and stop.
                target.cache.ensure_slot(t_sid)
                tok, _ = target._step_one_eager(t_sid, t_ctx)
                req.generated.append(tok)
                break

            # current length in both caches (must be aligned)
            L = target.cache.seq_lens[t_sid]

            # ── Draft phase ──────────────────────────────────────────────
            # One batched allocation covers all K positions, and tokens stay on
            # the GPU as (1,) tensors — no sync anywhere in this loop.
            draft.cache.ensure_slots_for(d_sid, K)
            tok_tensors: list[torch.Tensor] = []
            draft_logits: list[torch.Tensor] = []
            for _ in range(K):
                tok_t, logits = draft._step_one_graphed(d_sid, d_ctx)
                tok_tensors.append(tok_t)
                draft_logits.append(logits)

            # ── Verify phase ─────────────────────────────────────────────
            draft_ids = torch.cat(tok_tensors).to(target.device)          # (K,)
            verify_ids = torch.cat([target._active[t_sid][1], draft_ids])  # (K+1,)
            t_logits = target._step_verify_graphed(t_sid, verify_ids)      # (K+1, vocab)

            # ── Accept / reject, batched on the GPU ──────────────────────
            p_t_all = _get_probs_batch(t_logits, target.cfg, t_ctx)        # (K+1, V)
            p_d = _get_probs_batch(torch.stack(draft_logits), draft.cfg, d_ctx)
            p_d = p_d.to(p_t_all.device)                                   # (K, V)

            rows = torch.arange(K, device=p_t_all.device)
            accept_prob = (
                p_t_all[rows, draft_ids] / (p_d[rows, draft_ids] + 1e-10)
            ).clamp(max=1.0)                                               # (K,)
            # torch.rand must be built on the generator's own device; .to()
            # then matches accept_prob's device and dtype for the comparison.
            gen = target.cfg.generator
            u = torch.rand(
                K, generator=gen, device=gen.device if gen is not None else "cpu"
            ).to(accept_prob)
            rejected = u >= accept_prob                                    # (K,)
            # Leading run of accepts: positions before the first rejection are
            # exactly those whose running rejection count is still zero.
            n_acc_t = (rejected.cumsum(0) == 0).sum()

            # Both continuations are computed unconditionally so the branch can
            # be decided after the single sync below. Each is one vocab-sized
            # op — negligible next to the target forward we just ran.
            j = n_acc_t.clamp(max=K - 1)
            corr_t = _correction_sample(p_t_all[j], p_d[j], target.cfg).view(1)
            bonus_t = _sample_from_probs(p_t_all[K], target.cfg).view(1)

            # ── The one GPU→CPU sync per speculative step ────────────────
            packed = torch.cat([n_acc_t.view(1), draft_ids, corr_t, bonus_t]).tolist()
            n_accepted, draft_tokens = packed[0], packed[1 : 1 + K]
            corr, bonus = packed[1 + K], packed[2 + K]

            if n_accepted < K:
                emitted = draft_tokens[:n_accepted] + [corr]
                stats.n_accepted += n_accepted
                stats.n_rejected += 1

                # Rollback both caches to L+n+1 (KVs for positions 0..L+n kept).
                # position L holds the last target token (first verify input);
                # positions L+1..L+n are the accepted draft tokens;
                # position L+n+1 is where the correction goes next.
                target.cache.reset_to(t_sid, L + n_accepted + 1)
                draft.cache.reset_to(d_sid, L + n_accepted + 1)
                nxt_t = corr_t
            else:
                emitted = draft_tokens + [bonus]
                stats.n_accepted += K
                stats.n_bonus += 1

                # Sync draft: advance it from L+K to L+K+1 so it's aligned with target.
                draft.cache.ensure_slot(d_sid)
                draft._step_one_graphed(d_sid, d_ctx)               # output discarded
                nxt_t = bonus_t

            target._active[t_sid] = (req, nxt_t)
            draft._active[d_sid] = (draft_req, nxt_t.to(draft.device))
            target.cache.ensure_slot(t_sid)
            draft.cache.ensure_slot(d_sid)

            stats.n_steps += 1

            # ── Emit tokens, check stopping conditions ───────────────────
            done = False
            appended: list[int] = []
            for tok in emitted:
                req.generated.append(tok)
                appended.append(tok)
                if (
                    len(req.generated) >= req.max_new_tokens
                    or tok == self.eos
                    or (req.stop_check is not None and req.stop_check(req.generated))
                ):
                    done = True
                    break
            if ctx is not None:
                ctx.extend(appended)
            if done:
                break

        # ── Cleanup ─────────────────────────────────────────────────────
        if keep_resident:
            # Leave both caches populated so the next turn can roll back to the
            # shared prefix instead of re-reading it.
            self._remember(req, t_sid, d_sid)
        else:
            target._active.pop(t_sid, None)
            target.cache.free_sequence(t_sid)
            target._generated.pop(t_sid, None)
            if d_sid in draft._active:
                draft._active.pop(d_sid)
            if d_sid in draft.cache.seq_lens:
                draft.cache.free_sequence(d_sid)
            draft._generated.pop(d_sid, None)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats.decode_s = time.perf_counter() - decode_start

        # Remove spurious completions added by _prefill (if any re-queued).
        target.completed = [r for r in target.completed if r.req_id != req.req_id]
        draft.completed = [r for r in draft.completed if r.req_id == -1]
        draft.completed.clear()

        return req.generated, stats


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _get_probs_batch(
    logits: torch.Tensor,
    cfg: SamplingConfig,
    context_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-wise ``_get_probs`` over a stack of positions.

    Each output row is what ``_get_probs`` would return for that row on its
    own; doing all N at once keeps the accept/reject decision on the GPU
    instead of one softmax-plus-``.item()`` per draft token.

    Args:
        logits:      (N, vocab_size) raw logits.
        cfg:         Sampling configuration.
        context_ids: 1-D ids seen so far, shared by every row. Used for
                     repetition penalty when ``cfg.repetition_penalty != 1.0``.

    Returns:
        (N, vocab_size) probability distributions, each row summing to 1.
    """
    if cfg.repetition_penalty != 1.0 and context_ids is not None:
        logits = _apply_repetition_penalty(logits, context_ids, cfg.repetition_penalty)

    if cfg.mode is SamplingMode.GREEDY:
        # One-hot at argmax — avoids float noise in the correction formula.
        probs = torch.zeros_like(logits)
        probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    scaled = logits / max(cfg.temperature, 1e-8)

    if cfg.mode is SamplingMode.TOP_K:
        k = max(1, min(cfg.top_k, logits.shape[-1]))
        kth = torch.topk(scaled, k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    elif cfg.mode is SamplingMode.TOP_P:
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        drop = (sorted_probs.cumsum(dim=-1) - sorted_probs) >= cfg.top_p
        drop[:, 0] = False
        mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(1, sorted_idx, drop)
        scaled = scaled.masked_fill(mask, float("-inf"))

    return torch.softmax(scaled, dim=-1)


class _RollingCtx:
    """Growing repetition-penalty token buffer, mirrored on target and draft.

    Rebuilding ``torch.tensor(req.generated)`` once per speculative step costs
    a host→device copy that grows with the generation length. This preallocates
    one buffer per device and appends only the tokens emitted this step.
    """

    def __init__(
        self, generated: list[int], capacity: int, t_device: str, d_device: str
    ) -> None:
        self._t = torch.zeros(capacity, dtype=torch.long, device=t_device)
        self._d = (
            self._t if str(d_device) == str(t_device)
            else torch.zeros(capacity, dtype=torch.long, device=d_device)
        )
        self._n = 0
        self.extend(generated)

    def extend(self, tokens: list[int]) -> None:
        if not tokens:
            return
        src = torch.tensor(tokens, dtype=torch.long)
        end = self._n + len(tokens)
        self._t[self._n : end].copy_(src)
        if self._d is not self._t:
            self._d[self._n : end].copy_(src)
        self._n = end

    @property
    def target(self) -> torch.Tensor | None:
        return self._t[: self._n] if self._n else None

    @property
    def draft(self) -> torch.Tensor | None:
        return self._d[: self._n] if self._n else None
