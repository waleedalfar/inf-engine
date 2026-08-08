"""Continuous batching engine for LLaMA backed by a paged KV cache.

Mirrors ``ContinuousBatchingEngine`` (continuous.py, GPT-2) but uses:
  - ``PagedLlamaKVCache`` instead of ``SlotKVCache`` — sequences grow one block at
    a time instead of pre-allocating ``max_seq`` slots per sequence.
  - ``LlamaModel`` (RoPE, GQA, SwiGLU) instead of GPT2Model.
  - FCFS admission gated by block availability — a request is only admitted when
    the pool has enough free blocks for its prompt.

Each iteration:
  1. ADMIT  — prefill any queued request that fits in the block pool.
  2. DECODE — advance all active sequences by one token (single batched forward).
  3. EVICT  — free any sequence that hit EOS or ``max_new_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from engine.llama_model import LlamaModel
from engine.paged_cache import BlockManager, PagedLlamaKVCache
from engine.sampling import SamplingConfig, sample_next_token


@dataclass
class LlamaRequest:
    """A generation request."""

    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_time: float = 0.0
    generated: list[int] = field(default_factory=list)
    start_time: float = 0.0
    finish_time: float = 0.0

    @property
    def job_size(self) -> int:
        return len(self.prompt_ids) + self.max_new_tokens


class LlamaPagedEngine:
    """Iteration-level FCFS scheduler using a paged LLaMA KV cache.

    Args:
        model:           Loaded LlamaModel (weights on GPU/CPU).
        n_total_blocks:  Physical block count for the KV pool.
        block_size:      Tokens per physical block (default 16).
        eos_token:       Optional token id that signals end-of-sequence.
        sampling:        Decoding config (default: greedy).
        max_queue_depth: Maximum outstanding requests before submit() rejects.
    """

    def __init__(
        self,
        model: LlamaModel,
        n_total_blocks: int,
        block_size: int = 16,
        eos_token: int | None = None,
        sampling: SamplingConfig | None = None,
        max_queue_depth: int = 4096,
    ) -> None:
        self.model = model
        device = str(model.w.embed_tokens.device)
        dtype = model.w.embed_tokens.dtype

        self.manager = BlockManager(n_total_blocks, block_size)
        self.cache = PagedLlamaKVCache(model.config, self.manager, device, dtype)
        self.cfg = sampling or SamplingConfig()
        self.eos = eos_token
        self.max_queue_depth = max_queue_depth
        self.device = device

        self.queue: list[LlamaRequest] = []
        # seq_id → (request, last-token tensor on device)
        self._active: dict[int, tuple[LlamaRequest, torch.Tensor]] = {}
        self._next_seq_id: int = 0
        self.completed: list[LlamaRequest] = []

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def submit(self, req: LlamaRequest) -> bool:
        """Enqueue a request; returns False if the queue is full."""
        if len(self.queue) >= self.max_queue_depth:
            return False
        self.queue.append(req)
        return True

    @property
    def has_work(self) -> bool:
        return bool(self.queue) or bool(self._active)

    def _can_admit(self, req: LlamaRequest) -> bool:
        """True when the pool has enough free blocks for the prompt + 1 decode token."""
        needed = self.manager.blocks_needed(len(req.prompt_ids) + 1)
        return self.manager.n_free >= needed

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, req: LlamaRequest, now: float) -> bool:
        """Prefill one request; return True if already done after first token."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        T_p = len(req.prompt_ids)
        self.cache.allocate_sequence(seq_id, T_p)
        self.cache.begin_step([seq_id])

        ids = torch.tensor([req.prompt_ids], device=self.device)      # (1, T_p)
        pos = torch.arange(T_p, device=self.device)

        logits = self.model.forward(ids, cache=self.cache, start_pos=0, position_ids=pos)
        first = sample_next_token(logits[:, -1, :], self.cfg)          # (1, 1)
        req.generated.append(int(first))
        req.start_time = now

        done = len(req.generated) >= req.max_new_tokens or int(first) == self.eos
        if done:
            req.finish_time = now
            self.completed.append(req)
            self.cache.free_sequence(seq_id)
        else:
            self.cache.ensure_slot(seq_id)
            self._active[seq_id] = (req, first.view(1).to(self.device))
        return done

    def _evict(self, seq_id: int, req: LlamaRequest, now: float) -> None:
        req.finish_time = now
        self.completed.append(req)
        self._active.pop(seq_id)
        self.cache.free_sequence(seq_id)

    @torch.no_grad()
    def _decode_step(self, now: float) -> list[LlamaRequest]:
        """Advance all active sequences by one token (single batched forward)."""
        if not self._active:
            return []

        seq_ids = list(self._active.keys())
        A = len(seq_ids)

        # Current length of each sequence = absolute position of the new token.
        seq_lens = torch.tensor(
            [self.cache.seq_lens[sid] for sid in seq_ids],
            device=self.device,
        )                                                               # (A,)

        input_ids = torch.cat(
            [self._active[sid][1] for sid in seq_ids]
        ).view(A, 1)                                                    # (A, 1)
        position_ids = seq_lens.view(-1, 1)                            # (A, 1)

        # Attention mask: sequence i can attend to positions [0..seq_lens[i]].
        max_len = int(seq_lens.max()) + 1
        ar = torch.arange(max_len, device=self.device)
        allowed = ar[None, :] <= seq_lens[:, None]                     # (A, max_len)
        attn_mask = allowed[:, None, :]                                # (A, 1, max_len)

        self.cache.begin_step(seq_ids)
        logits = self.model.forward(
            input_ids, cache=self.cache, start_pos=0,
            position_ids=position_ids, attn_mask=attn_mask,
        )                                                               # (A, 1, V)
        nxt = sample_next_token(logits[:, -1, :], self.cfg)            # (A, 1)
        tokens = nxt.view(-1).tolist()                                 # one GPU→CPU sync per step

        completions: list[LlamaRequest] = []
        evict_ids: list[tuple[int, LlamaRequest]] = []
        for i, sid in enumerate(seq_ids):
            req = self._active[sid][0]
            tok = tokens[i]
            req.generated.append(tok)
            done = len(req.generated) >= req.max_new_tokens or tok == self.eos
            if done:
                evict_ids.append((sid, req))
                completions.append(req)
            else:
                self.cache.ensure_slot(sid)
                self._active[sid] = (req, nxt[i : i + 1].view(1))

        for sid, req in evict_ids:
            self._evict(sid, req, now)

        return completions

    @torch.no_grad()
    def step(self, now: float = 0.0) -> list[LlamaRequest]:
        """One iteration: admit queued requests → decode → evict finished.

        Returns the list of requests completed this step.
        """
        completions: list[LlamaRequest] = []

        # ADMIT: fill free capacity from the queue (skip requests that don't fit).
        i = 0
        while i < len(self.queue):
            req = self.queue[i]
            if self._can_admit(req):
                self.queue.pop(i)
                if self._prefill(req, now):
                    completions.append(req)
                # do not increment i — next request shifted into position i
            else:
                i += 1

        # DECODE: one token for every active sequence.
        completions.extend(self._decode_step(now))
        return completions

    @torch.no_grad()
    def run_offline(self, requests: list[LlamaRequest]) -> dict[int, list[int]]:
        """Submit all requests at once and run until completion.

        Returns:
            Mapping ``req_id → generated token ids``.
        """
        for r in requests:
            self.submit(r)
        while self.has_work:
            self.step()
        return {r.req_id: r.generated for r in self.completed}
