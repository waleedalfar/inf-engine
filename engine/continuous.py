"""Continuous (in-flight) batching with iteration-level scheduling.

Static batching (Phase 3) runs a fixed batch to completion: every sequence waits for the
slowest member, finished sequences keep their slot, and queued requests wait for the whole
batch. Continuous batching instead schedules **per iteration**:

  every step:
    1. ADMIT  — fill any free slot from the queue (prefill its prompt, emit first token)
    2. DECODE — advance every active slot by exactly one token, in one batched forward
    3. EVICT  — a slot that hit its token budget (or EOS) is freed *immediately*

So a sequence leaves the instant it finishes and a queued request takes its slot on the very
next step — no head-of-line blocking, and no padding to a shared length because each slot
attends only to its own KV history (`SlotKVCache` + explicit per-slot ``attn_mask``).

Two schedulers are provided: **FCFS** (arrival order) and **SJF**
(shortest job first, by prompt+output length). The benchmark contrasts their
latency/throughput tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from engine.kv_cache import SlotKVCache
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, sample_next_token


class Policy(str, Enum):
    """Scheduling policy for choosing which queued request to admit next."""

    FCFS = "fcfs"   # first come, first served (arrival order)
    SJF = "sjf"     # shortest job first (by prompt_len + max_new_tokens)


@dataclass
class Request:
    """A generation request and the bookkeeping the engine fills in.

    Attributes:
        req_id:         Caller-assigned id.
        prompt_ids:     Prompt token ids.
        max_new_tokens: Total tokens to generate for this request.
        arrival_time:   Wall-clock submit time (for latency accounting under load).
        generated:      Output tokens (filled by the engine).
        start_time:     When the request was admitted (prefill).
        finish_time:    When the request completed.
    """

    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_time: float = 0.0
    generated: list[int] = field(default_factory=list)
    start_time: float = 0.0
    finish_time: float = 0.0

    @property
    def job_size(self) -> int:
        """SJF cost estimate: prompt length + requested output length."""
        return len(self.prompt_ids) + self.max_new_tokens


class ContinuousBatchingEngine:
    """Iteration-level scheduler over a fixed pool of KV-cache slots."""

    def __init__(
        self,
        model: GPT2Model,
        n_slots: int,
        max_seq: int,
        policy: Policy = Policy.FCFS,
        sampling: SamplingConfig | None = None,
        eos_token: int | None = None,
        max_queue_depth: int = 4096,
    ) -> None:
        self.model = model
        self.config = model.config
        self.n_slots = n_slots
        self.max_seq = max_seq
        self.policy = policy
        self.cfg = sampling if sampling is not None else SamplingConfig()
        self.eos = eos_token
        self.max_queue_depth = max_queue_depth

        device = model.w.wte.device
        dtype = model.w.wte.dtype
        self.device = device
        self.cache = SlotKVCache(self.config, n_slots, max_seq, str(device), dtype)
        self.queue: list[Request] = []
        self.slot_req: list[Request | None] = [None] * n_slots
        self.slot_last: list[torch.Tensor | None] = [None] * n_slots  # last token id per slot
        self.free: list[int] = list(range(n_slots))
        self.completed: list[Request] = []

    # -- queue management ---------------------------------------------------
    def submit(self, req: Request) -> bool:
        """Enqueue a request. Returns False (rejected) if the queue is full."""
        if len(self.queue) >= self.max_queue_depth:
            return False
        self.queue.append(req)
        return True

    @property
    def num_active(self) -> int:
        return self.n_slots - len(self.free)

    def has_work(self) -> bool:
        return bool(self.queue) or self.num_active > 0

    def _pick_next(self) -> Request:
        if self.policy is Policy.FCFS:
            return self.queue.pop(0)
        # SJF: smallest job_size (ties broken by arrival order via index).
        idx = min(range(len(self.queue)), key=lambda i: self.queue[i].job_size)
        return self.queue.pop(idx)

    # -- core steps ---------------------------------------------------------
    @torch.no_grad()
    def _prefill(self, slot: int, req: Request, now: float) -> bool:
        """Prefill a prompt into ``slot`` and emit its first token. Returns done?."""
        ids = torch.tensor([req.prompt_ids], device=self.device)        # (1, prompt_len)
        self.cache.reset_slot(slot)
        self.cache.begin_step(torch.tensor([slot], device=self.device))
        logits = self.model.forward(ids, cache=self.cache, start_pos=0)  # causal prefill
        first = sample_next_token(logits[:, -1, :], self.cfg)            # (1, 1)
        req.generated.append(int(first))
        req.start_time = now
        self.slot_req[slot] = req
        self.slot_last[slot] = first.view(1)
        return len(req.generated) >= req.max_new_tokens or int(first) == self.eos

    def _complete(self, slot: int, req: Request, now: float) -> None:
        req.finish_time = now
        self.completed.append(req)
        self.slot_req[slot] = None
        self.slot_last[slot] = None
        self.free.append(slot)
        self.cache.reset_slot(slot)

    @torch.no_grad()
    def _decode_step(self, now: float) -> list[Request]:
        """Advance every active slot by one token in a single batched forward."""
        active = [s for s in range(self.n_slots) if self.slot_req[s] is not None]
        if not active:
            return []
        active_t = torch.tensor(active, device=self.device)
        self.cache.begin_step(active_t)
        lengths = self.cache.lengths[active_t]                           # (A,) = position of new token
        input_ids = torch.cat([self.slot_last[s] for s in active]).view(len(active), 1)  # (A, 1)
        position_ids = lengths.view(-1, 1)                              # (A, 1) true positions
        max_len = int(lengths.max()) + 1
        ar = torch.arange(max_len, device=self.device)                  # (max_len,)
        # slot i may attend to its own [0 : lengths[i]+1]; everything beyond is masked.
        allowed = (ar[None, :] <= lengths[:, None])                     # (A, max_len) bool
        attn_mask = allowed[:, None, :]                                 # (A, 1, max_len)

        logits = self.model.forward(
            input_ids, cache=self.cache, start_pos=0,
            position_ids=position_ids, attn_mask=attn_mask,
        )                                                               # (A, 1, V)
        nxt = sample_next_token(logits[:, -1, :], self.cfg)             # (A, 1)
        tokens = nxt.view(-1).tolist()                                  # single GPU->CPU sync for the step

        completions: list[Request] = []
        for i, s in enumerate(active):
            tok = tokens[i]
            req = self.slot_req[s]
            assert req is not None
            req.generated.append(tok)
            self.slot_last[s] = nxt[i]                                  # (1,) tensor, kept on GPU
            if len(req.generated) >= req.max_new_tokens or tok == self.eos:
                self._complete(s, req, now)
                completions.append(req)
        return completions

    @torch.no_grad()
    def step(self, now: float = 0.0) -> list[Request]:
        """One scheduler iteration: admit (prefill) → decode → evict. Returns completions."""
        completions: list[Request] = []
        # ADMIT: fill free slots from the queue.
        while self.free and self.queue:
            req = self._pick_next()
            slot = self.free.pop()
            if self._prefill(slot, req, now):                           # finished on first token
                self._complete(slot, req, now)
                completions.append(req)
        # DECODE one token for all active slots; EVICT happens inside.
        completions.extend(self._decode_step(now))
        return completions

    @torch.no_grad()
    def run_offline(self, requests: list[Request]) -> dict[int, list[int]]:
        """Submit all requests at once and run to completion (saturated throughput).

        Returns:
            Mapping req_id -> generated token ids.
        """
        for r in requests:
            self.submit(r)
        while self.has_work():
            self.step(now=0.0)
        return {r.req_id: r.generated for r in self.completed}
