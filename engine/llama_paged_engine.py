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
from typing import Callable

import torch

from engine.llama_model import LlamaModel
from engine.paged_cache import BlockManager, PagedLlamaKVCache
from engine.sampling import SamplingConfig, sample_next_token

# Candidate CUDA-graph batch sizes; the smallest bucket >= the active count
# is used each decode step (extra batch lanes do harmless redundant work).
DEFAULT_GRAPH_BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def _default_graph_len_buckets(n_ctx: int, block_size: int) -> list[int]:
    """Power-of-two, block-aligned KV-gather-length buckets up to ``n_ctx``."""
    buckets = []
    b = block_size
    while b < n_ctx:
        buckets.append(b)
        b *= 2
    last = ((n_ctx + block_size - 1) // block_size) * block_size
    buckets.append(last)
    return sorted(set(buckets))


def _splits_for(capture_len: int, batch: int, n_kv: int, target_ctas: int = 512) -> int:
    """KV splits for a captured graph's shape.

    Must be a Python int fixed at capture time. Chosen so ``batch * n_kv *
    splits`` comfortably fills the GPU, while each split still covers enough
    positions to amortize its share of the combine.
    """
    base = max(1, target_ctas // max(batch * n_kv, 1))
    return max(1, min(base, max(1, capture_len // 128), 64))


def _as_id_row(ids: list[int] | torch.Tensor, device: str) -> torch.Tensor:
    """Normalise a token-id sequence to a ``(1, T)`` long tensor on ``device``.

    Accepts an already-on-device 1-D tensor (reshaped for free, no host round
    trip) or a Python list (copied up).
    """
    if torch.is_tensor(ids):
        return ids.view(1, -1).to(device=device, dtype=torch.long)
    return torch.tensor([ids], dtype=torch.long, device=device)


@dataclass
class _CapturedGraph:
    """One CUDA graph captured for a fixed (batch_bucket, len_bucket) decode shape."""

    graph: "torch.cuda.CUDAGraph"
    input_ids: torch.Tensor          # (bucket, 1) long — static
    position_ids: torch.Tensor       # (bucket, 1) long — static
    block_table_buf: torch.Tensor    # (bucket, capture_len // block_size) long — static
    seq_lens_buf: torch.Tensor       # (bucket,) long — static
    attn_mask_buf: torch.Tensor      # (bucket, 1, capture_len) bool — static
    logits: torch.Tensor             # (bucket, 1, vocab) — static output, captured
    capture_len: int
    bucket_size: int


@dataclass
class _CapturedVerifyGraph:
    """CUDA graph for a fixed (len_bucket, q_len) verify shape (batch=1 always)."""

    graph: "torch.cuda.CUDAGraph"
    input_ids: torch.Tensor          # (1, q_len) long — static
    position_ids: torch.Tensor       # (1, q_len) long — static
    block_table_buf: torch.Tensor    # (1, capture_len // block_size) long — static
    seq_lens_buf: torch.Tensor       # (1,) long — static
    attn_mask_buf: torch.Tensor      # (1, q_len, capture_len) bool — static
    logits: torch.Tensor             # (1, q_len, vocab) — static output, captured
    capture_len: int
    q_len: int


class _StaticBufferCache:
    """Routes ``LlamaModel.forward``'s cache calls through fixed static buffers.

    This is the piece that makes a whole decode/verify forward CUDA-graph
    capturable (see paged_cache.py's "CUDA-graph-safe decode path" for why the
    normal ``extend()`` Python-loop implementation cannot be captured safely).

    Two paths out of here:

    * ``fused_attend`` — writes K/V, then attends against the pool in place with
      the paged flash-decoding kernel. No contiguous gather, no GQA expansion,
      and cost tracks the true sequence length rather than ``capture_len``.
      ``llama_attention`` prefers this whenever it exists.
    * ``extend`` — the older write-then-gather path, kept as the fallback for
      head dims the kernel does not handle.

    ``n_splits`` is fixed at construction because it must be constant across a
    CUDA-graph capture.
    """

    def __init__(
        self,
        real_cache: PagedLlamaKVCache,
        block_table_buf: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        capture_len: int,
    ) -> None:
        self._real = real_cache
        self.block_table_buf = block_table_buf
        self.seq_lens_buf = seq_lens_buf
        self.capture_len = capture_len
        self.n_splits = _splits_for(capture_len, seq_lens_buf.shape[0],
                                    real_cache.config.n_kv_heads)

    def extend(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int = 0):
        return self._real.extend_static(
            layer, k_new, v_new, self.block_table_buf, self.seq_lens_buf, self.capture_len,
        )

    def fused_attend(
        self, layer: int, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> torch.Tensor:
        self._real.write_static(layer, k_new, v_new, self.block_table_buf, self.seq_lens_buf)
        kv_lens = self.seq_lens_buf + k_new.shape[2]
        return self._real.paged_attend(
            layer, q, self.block_table_buf, kv_lens, self.n_splits
        )


# Kept as distinct names so tracebacks and the graph caches stay readable.
class _GraphVerifyCache(_StaticBufferCache):
    """Static-buffer cache for verify steps, where q_len = K+1 > 1."""


class _GraphDecodeCache(_StaticBufferCache):
    """Static-buffer cache for q_len=1 decode steps."""


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
    stop_check: Callable[[list[int]], bool] | None = field(default=None, repr=False)
    """Optional predicate over ``generated`` checked after every new token
    (e.g. "did a <tool_call> block just close?"). In addition to the
    standard EOS / max_new_tokens stop conditions."""

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
        enable_cuda_graphs: Capture/replay a CUDA graph per (batch, KV-length)
                         bucket for the decode step instead of an eager
                         forward. No-op (silently ignored) off CUDA. See
                         paged_cache.py's static-buffer methods for why this
                         is safe under continuous batching's varying batch
                         composition.
        graph_batch_buckets: Candidate decode batch sizes to capture graphs
                         for (default: powers of two, 1..64).
        graph_len_buckets: Candidate KV-gather lengths, must be multiples of
                         block_size (default: power-of-two block multiples
                         up to model.config.n_ctx).
        kv_dtype:        KV pool storage dtype (default: the model's dtype).
                         ``torch.float8_e4m3fn`` halves the KV bytes read per
                         token, which is the dominant cost at long context.
                         Safe to enable on a speculative *draft* engine at
                         essentially no quality cost — the draft only proposes
                         and the target's accept/reject still guarantees the
                         target's exact output distribution. On a target engine
                         it changes the model's own distribution, so gate it on
                         a quality check.
    """

    def __init__(
        self,
        model: LlamaModel,
        n_total_blocks: int,
        block_size: int = 16,
        eos_token: int | None = None,
        sampling: SamplingConfig | None = None,
        max_queue_depth: int = 4096,
        enable_cuda_graphs: bool = False,
        graph_batch_buckets: tuple[int, ...] | None = None,
        graph_len_buckets: list[int] | None = None,
        kv_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        device = str(model.w.embed_tokens.device)
        dtype = model.w.embed_tokens.dtype

        self.manager = BlockManager(n_total_blocks, block_size)
        self.cache = PagedLlamaKVCache(model.config, self.manager, device, dtype,
                                       kv_dtype=kv_dtype)
        self.cfg = sampling or SamplingConfig()
        self.eos = eos_token
        self.max_queue_depth = max_queue_depth
        self.device = device

        self.queue: list[LlamaRequest] = []
        # seq_id → (request, last-token tensor on device)
        self._active: dict[int, tuple[LlamaRequest, torch.Tensor]] = {}
        self._next_seq_id: int = 0
        self.completed: list[LlamaRequest] = []
        # seq_id -> generated token ids (prompt excluded), for repetition
        # penalty context — matches AgentLoop._generate_to_eos's rule that
        # penalty only ever tracks generated tokens, never the prompt.
        self._generated: dict[int, list[int]] = {}

        self.enable_cuda_graphs = enable_cuda_graphs and torch.device(device).type == "cuda"
        self.graph_batch_buckets = sorted(graph_batch_buckets or DEFAULT_GRAPH_BATCH_BUCKETS)
        self.graph_len_buckets = sorted(
            graph_len_buckets or _default_graph_len_buckets(model.config.n_ctx, block_size)
        )
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._verify_graphs: dict[tuple[int, int], _CapturedVerifyGraph] = {}

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

        done = (
            len(req.generated) >= req.max_new_tokens
            or int(first) == self.eos
            or (req.stop_check is not None and req.stop_check(req.generated))
        )
        if done:
            req.finish_time = now
            self.completed.append(req)
            self.cache.free_sequence(seq_id)
        else:
            self.cache.ensure_slot(seq_id)
            self._active[seq_id] = (req, first.view(1).to(self.device))
            self._generated[seq_id] = [int(first)]
        return done

    def _evict(self, seq_id: int, req: LlamaRequest, now: float) -> None:
        req.finish_time = now
        self.completed.append(req)
        self._active.pop(seq_id)
        self.cache.free_sequence(seq_id)
        self._generated.pop(seq_id, None)

    def _repetition_context(self, seq_ids: list[int]) -> torch.Tensor | None:
        """Build a padded (A, T) context-id tensor for the repetition penalty,
        one row per active sequence's own generated tokens (prompt excluded).

        Rows are padded on the left by repeating that sequence's own first
        generated token — a real, already-correctly-penalized id, so padding
        never introduces spurious penalties on an unrelated token (unlike
        padding with a sentinel like 0, which would silently penalize
        whatever real token id 0 happens to be for short sequences).
        Returns None when repetition penalty is off (skips the tensor build).
        """
        if self.cfg.repetition_penalty == 1.0:
            return None
        max_len = max(len(self._generated[sid]) for sid in seq_ids)
        rows = []
        for sid in seq_ids:
            gen = self._generated[sid]
            pad = [gen[0]] * (max_len - len(gen))
            rows.append(pad + gen)
        return torch.tensor(rows, device=self.device)

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
        context_ids = self._repetition_context(seq_ids)
        nxt = sample_next_token(logits[:, -1, :], self.cfg, context_ids)  # (A, 1)
        tokens = nxt.view(-1).tolist()                                 # one GPU→CPU sync per step
        for sid, tok in zip(seq_ids, tokens):
            self._generated[sid].append(tok)

        completions: list[LlamaRequest] = []
        evict_ids: list[tuple[int, LlamaRequest]] = []
        for i, sid in enumerate(seq_ids):
            req = self._active[sid][0]
            tok = tokens[i]
            req.generated.append(tok)
            done = (
                len(req.generated) >= req.max_new_tokens
                or tok == self.eos
                or (req.stop_check is not None and req.stop_check(req.generated))
            )
            if done:
                evict_ids.append((sid, req))
                completions.append(req)
            else:
                self.cache.ensure_slot(sid)
                self._active[sid] = (req, nxt[i : i + 1].view(1))

        for sid, req in evict_ids:
            self._evict(sid, req, now)

        return completions

    # ------------------------------------------------------------------
    # CUDA-graph decode path
    # ------------------------------------------------------------------

    def _pick_bucket(self, n_active: int, max_len_needed: int) -> tuple[int, int] | None:
        """Smallest (batch, len) bucket that fits, or None if too big to have configured."""
        batch_bucket = next((b for b in self.graph_batch_buckets if b >= n_active), None)
        len_bucket = next((l for l in self.graph_len_buckets if l >= max_len_needed), None)
        if batch_bucket is None or len_bucket is None:
            return None
        return batch_bucket, len_bucket

    def _build_bucket_inputs(
        self, seq_ids: list[int], bucket_size: int, capture_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fresh (non-static) tensors for this bucket — used both to seed a new
        graph's static buffers and to refill them before each replay."""
        A = len(seq_ids)
        # Decode graphs write one position per sequence; callers ensure_slot()
        # before every step, including the one that triggers capture.
        block_table_buf, seq_lens_buf = self.cache.build_static_buffers(
            seq_ids, bucket_size, capture_len, self.device, write_len=1
        )
        real_ids = torch.cat([self._active[sid][1] for sid in seq_ids]).view(A, 1)
        if A < bucket_size:
            pad = real_ids[-1:].expand(bucket_size - A, 1)
            real_ids = torch.cat([real_ids, pad], dim=0)
        input_ids = real_ids.contiguous()
        position_ids = seq_lens_buf.view(-1, 1).clone()
        ar = torch.arange(capture_len, device=self.device)
        attn_mask = (ar[None, :] <= seq_lens_buf[:, None])[:, None, :].clone()
        return input_ids, position_ids, block_table_buf, seq_lens_buf, attn_mask

    def _capture_graph(self, batch_bucket: int, len_bucket: int, seq_ids: list[int]) -> _CapturedGraph:
        input_ids, position_ids, block_table_buf, seq_lens_buf, attn_mask = self._build_bucket_inputs(
            seq_ids, batch_bucket, len_bucket
        )
        wrapper = _GraphDecodeCache(self.cache, block_table_buf, seq_lens_buf, len_bucket)

        # Warmup on a side stream (standard torch.cuda.graph capture practice) —
        # the model's forward re-reads the same (unchanged) buffers each time,
        # so repeated writes into the KV pool during warmup are idempotent.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.forward(
                    input_ids, cache=wrapper, start_pos=0,
                    position_ids=position_ids, attn_mask=attn_mask,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = self.model.forward(
                input_ids, cache=wrapper, start_pos=0,
                position_ids=position_ids, attn_mask=attn_mask,
            )

        return _CapturedGraph(
            graph=graph, input_ids=input_ids, position_ids=position_ids,
            block_table_buf=block_table_buf, seq_lens_buf=seq_lens_buf,
            attn_mask_buf=attn_mask, logits=static_logits,
            capture_len=len_bucket, bucket_size=batch_bucket,
        )

    def _fill_graph_inputs(self, cg: _CapturedGraph, seq_ids: list[int]) -> None:
        input_ids, position_ids, block_table_buf, seq_lens_buf, attn_mask = self._build_bucket_inputs(
            seq_ids, cg.bucket_size, cg.capture_len
        )
        cg.input_ids.copy_(input_ids)
        cg.position_ids.copy_(position_ids)
        cg.block_table_buf.copy_(block_table_buf)
        cg.seq_lens_buf.copy_(seq_lens_buf)
        cg.attn_mask_buf.copy_(attn_mask)

    @torch.no_grad()
    def _decode_step_graphed(self, now: float) -> list[LlamaRequest]:
        """Like ``_decode_step`` but replays a captured CUDA graph for the
        active batch's (batch, KV-length) bucket instead of an eager forward.
        Falls back to ``_decode_step`` when the active batch doesn't fit any
        configured bucket (e.g. a sequence longer than the largest len bucket)."""
        if not self._active:
            return []

        seq_ids = list(self._active.keys())
        A = len(seq_ids)
        max_len_needed = max(self.cache.seq_lens[sid] for sid in seq_ids) + 1

        bucket = self._pick_bucket(A, max_len_needed)
        if bucket is None:
            return self._decode_step(now)
        batch_bucket, len_bucket = bucket

        cg = self._graphs.get(bucket)
        if cg is None:
            cg = self._capture_graph(batch_bucket, len_bucket, seq_ids)
            self._graphs[bucket] = cg

        self._fill_graph_inputs(cg, seq_ids)
        cg.graph.replay()

        # cg.logits is a fresh tensor copied out of the graph's static output
        # buffer on each replay — applying repetition penalty here (ordinary
        # eager tensor ops, not part of the captured graph) is safe.
        context_ids = self._repetition_context(seq_ids)
        nxt = sample_next_token(cg.logits[:A, -1, :], self.cfg, context_ids)  # (A, 1)
        tokens = nxt.view(-1).tolist()                                 # one GPU→CPU sync per step
        for sid, tok in zip(seq_ids, tokens):
            self._generated[sid].append(tok)

        # extend_static doesn't touch self.cache.seq_lens (it works off the
        # static buffer, not the dict) — advance it here, same effect as the
        # eager extend()'s end-of-forward update.
        for sid in seq_ids:
            self.cache.seq_lens[sid] += 1

        completions: list[LlamaRequest] = []
        evict_ids: list[tuple[int, LlamaRequest]] = []
        for i, sid in enumerate(seq_ids):
            req = self._active[sid][0]
            tok = tokens[i]
            req.generated.append(tok)
            done = (
                len(req.generated) >= req.max_new_tokens
                or tok == self.eos
                or (req.stop_check is not None and req.stop_check(req.generated))
            )
            if done:
                evict_ids.append((sid, req))
                completions.append(req)
            else:
                self.cache.ensure_slot(sid)
                self._active[sid] = (req, nxt[i : i + 1].view(1))

        for sid, req in evict_ids:
            self._evict(sid, req, now)

        return completions

    # ------------------------------------------------------------------
    # Speculative decode helpers (called by SpeculativePagedEngine)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_one_graphed(
        self, seq_id: int, context_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-sequence q_len=1 decode; returns ``(token_id_1d, logits_row)``.

        The token is returned as a ``(1,)`` device tensor rather than a Python
        int so that a caller running K draft steps back to back pays **zero**
        GPU→CPU syncs inside the loop and can batch them into one ``.tolist()``.

        Uses a captured CUDA graph when ``enable_cuda_graphs`` is True,
        otherwise falls back to eager. ``context_ids`` (1-D tensor of all
        generated token ids so far) is forwarded to repetition penalty sampling.
        Caller must ensure ``cache`` has a free slot before each invocation.
        """
        if not self.enable_cuda_graphs:
            _, logits_row = self._step_one_eager(seq_id, context_ids)
            return self._active[seq_id][1], logits_row

        seq_len = self.cache.seq_lens[seq_id]
        bucket = self._pick_bucket(1, seq_len + 1)
        if bucket is None:
            _, logits_row = self._step_one_eager(seq_id, context_ids)
            return self._active[seq_id][1], logits_row
        batch_bucket, len_bucket = bucket

        cg = self._graphs.get(bucket)
        if cg is None:
            cg = self._capture_graph(batch_bucket, len_bucket, [seq_id])
            self._graphs[bucket] = cg

        self._fill_graph_inputs(cg, [seq_id])
        cg.graph.replay()

        # cg.logits aliases the graph's *static* output buffer, which the next
        # replay overwrites in place.  Speculative decode holds all K draft
        # rows until after the draft loop finishes, so clone out of the buffer
        # — without this every row would read back as the last replay's logits.
        logits_row = cg.logits[0, -1, :].clone()                        # (vocab,)
        ctx_2d = context_ids.unsqueeze(0) if context_ids is not None else None
        tok_t = sample_next_token(logits_row.unsqueeze(0), self.cfg, ctx_2d).view(1)

        self.cache.seq_lens[seq_id] += 1                                 # extend_static doesn't update this
        req = self._active[seq_id][0]
        self._active[seq_id] = (req, tok_t)
        return tok_t, logits_row

    @torch.no_grad()
    def _step_one_eager(
        self, seq_id: int, context_ids: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Eager single-sequence q_len=1 decode. CPU fallback / test path."""
        L = self.cache.seq_lens[seq_id]
        seq_lens_t = torch.tensor([L], device=self.device)
        ar = torch.arange(L + 1, device=self.device)
        attn_mask = (ar[None, :] <= seq_lens_t[:, None])[:, None, :]    # (1, 1, L+1)
        ids = self._active[seq_id][1].view(1, 1)
        pos = seq_lens_t.view(1, 1)
        self.cache.begin_step([seq_id])
        logits = self.model.forward(
            ids, cache=self.cache, start_pos=0,
            position_ids=pos, attn_mask=attn_mask,
        )
        # cache.extend() inside forward already advanced seq_lens[seq_id]
        logits_row = logits[0, -1, :]
        ctx_2d = context_ids.unsqueeze(0) if context_ids is not None else None
        tok = int(sample_next_token(logits_row.unsqueeze(0), self.cfg, ctx_2d).item())
        req = self._active[seq_id][0]
        self._active[seq_id] = (req, torch.tensor([tok], device=self.device))
        return tok, logits_row

    @torch.no_grad()
    def _step_verify_eager(
        self, seq_id: int, verify_ids: list[int] | torch.Tensor
    ) -> torch.Tensor:
        """Eager multi-token forward for the target verify step.

        Writes K+1 KV entries into the cache and returns raw ``(K+1, vocab)``
        logits. Does NOT sample — the caller (SpeculativePagedEngine) runs
        accept/reject on the returned logits.

        ``verify_ids`` may be a Python list or a 1-D device tensor; the tensor
        form lets speculative decode feed draft tokens straight back in without
        a GPU→CPU→GPU round trip.
        """
        K1 = len(verify_ids)
        L = self.cache.seq_lens[seq_id]
        self.cache.ensure_slots_for(seq_id, K1)
        self.cache.begin_step([seq_id])
        ids = _as_id_row(verify_ids, self.device)                       # (1, K+1)
        pos_ids = torch.arange(L, L + K1, device=self.device).unsqueeze(0)  # (1, K+1)
        # No explicit attn_mask: cache.extend() gathers exactly L+K1 positions,
        # so llama_attention's lower-right causal branch applies — same mask,
        # but it reaches the flash backend with GQA instead of materializing an
        # expanded K/V. start_pos carries the offset the mask is built from.
        logits = self.model.forward(
            ids, cache=self.cache, start_pos=L, position_ids=pos_ids,
        )                                                                # (1, K+1, vocab)
        # extend() advanced seq_lens[seq_id] by K1
        return logits[0]                                                 # (K+1, vocab)

    def _capture_verify_graph(self, len_bucket: int, q_len: int, seq_id: int) -> _CapturedVerifyGraph:
        # Warmup and capture both write q_len positions into the real KV pool,
        # so the caller must have allocated them already (write_len asserts it).
        block_table_buf, seq_lens_buf = self.cache.build_static_buffers(
            [seq_id], 1, len_bucket, self.device, write_len=q_len
        )
        L = int(seq_lens_buf[0].item())
        input_ids    = torch.zeros(1, q_len, dtype=torch.long, device=self.device)
        position_ids = torch.arange(L, L + q_len, device=self.device).unsqueeze(0)
        ar      = torch.arange(len_bucket, device=self.device)
        q_abs   = position_ids[0]
        attn_mask = (ar[None, :] <= q_abs[:, None])[None, :, :]           # (1, q_len, len_bucket)

        wrapper = _GraphVerifyCache(self.cache, block_table_buf, seq_lens_buf, len_bucket)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.forward(
                    input_ids, cache=wrapper, start_pos=0,
                    position_ids=position_ids, attn_mask=attn_mask,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = self.model.forward(
                input_ids, cache=wrapper, start_pos=0,
                position_ids=position_ids, attn_mask=attn_mask,
            )

        return _CapturedVerifyGraph(
            graph=graph, input_ids=input_ids, position_ids=position_ids,
            block_table_buf=block_table_buf, seq_lens_buf=seq_lens_buf,
            attn_mask_buf=attn_mask, logits=static_logits,
            capture_len=len_bucket, q_len=q_len,
        )

    @torch.no_grad()
    def _step_verify_graphed(
        self, seq_id: int, verify_ids: list[int] | torch.Tensor
    ) -> torch.Tensor:
        """Graphed verify step; falls back to eager when CUDA graphs are off or bucket too large."""
        if not self.enable_cuda_graphs:
            return self._step_verify_eager(seq_id, verify_ids)

        q_len = len(verify_ids)
        L = self.cache.seq_lens[seq_id]
        bucket = self._pick_bucket(1, L + q_len)
        if bucket is None:
            return self._step_verify_eager(seq_id, verify_ids)
        _, len_bucket = bucket

        # Allocate the q_len slots BEFORE capture, not after. Capture runs
        # warmup forwards that *write* KV at positions L..L+q_len-1, and
        # build_static_buffers pads not-yet-allocated block-table columns with
        # the sequence's own block 0. Padding is harmless for reads (attn_mask
        # hides those positions) but a write through it lands on the sequence's
        # real positions 0..block_size-1 and destroys its prompt KV — silently,
        # and only on the step that first needs a new (len_bucket, q_len).
        self.cache.ensure_slots_for(seq_id, q_len)

        vg = self._verify_graphs.get((len_bucket, q_len))
        if vg is None:
            vg = self._capture_verify_graph(len_bucket, q_len, seq_id)
            self._verify_graphs[(len_bucket, q_len)] = vg

        block_table_buf, seq_lens_buf = self.cache.build_static_buffers(
            [seq_id], 1, len_bucket, self.device, write_len=q_len
        )
        vg.block_table_buf.copy_(block_table_buf)
        vg.seq_lens_buf.copy_(seq_lens_buf)
        vg.input_ids.copy_(_as_id_row(verify_ids, self.device))
        vg.position_ids.copy_(torch.arange(L, L + q_len, device=self.device).unsqueeze(0))
        ar    = torch.arange(len_bucket, device=self.device)
        q_abs = vg.position_ids[0]
        vg.attn_mask_buf.copy_((ar[None, :] <= q_abs[:, None])[None, :, :])

        vg.graph.replay()

        # extend_static doesn't advance seq_lens — do it here, same as _step_one_graphed.
        self.cache.seq_lens[seq_id] += q_len

        return vg.logits[0]   # (q_len, vocab)

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
        if self.enable_cuda_graphs:
            completions.extend(self._decode_step_graphed(now))
        else:
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
