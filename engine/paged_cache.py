"""Paged KV cache for LLaMA: block-based allocation, non-contiguous physical layout.

Motivation
----------
LlamaStaticKVCache pre-allocates ``max_seq`` token slots per sequence up front.
For many concurrent short sequences that reserve a large ``max_seq``, most of those
slots sit empty — wasted VRAM.

A paged KV cache (inspired by vLLM's PagedAttention) splits the physical cache into
fixed-size **blocks** (e.g., 16 tokens each). Each sequence is assigned only as many
blocks as it actually needs, growing one block at a time. When a sequence finishes, its
blocks are returned to the free pool immediately and can be given to new arrivals.

Physical layout (per layer, K and V)
-------------------------------------
    k_pool: (n_layer, n_total_blocks, n_kv_heads, block_size, head_dim)  float

Block table
-----------
    block_table[seq_id] = [phys_b0, phys_b1, ...]
    seq_lens[seq_id]   = current token count (0 before any write)

Token at absolute position ``t`` lives in:
    block index  b = t // block_size
    block offset o = t % block_size
    phys block   = block_table[seq_id][b]
    location     = k_pool[layer, phys, :, o, :]
"""

from __future__ import annotations

import math

import torch

from engine.config import LlamaConfig
from engine.kernels.paged_attention import paged_flash_attention, write_kv_quant

# 8-bit KV storage formats -> largest representable magnitude.
#
# INT8 is the default choice, not FP8. Both are one byte, but with per-token
# amax scaling the exponent range FP8 spends bits on is already supplied by the
# scale, so those bits are wasted: e4m3 keeps 3 mantissa bits against INT8's
# effective 7. Measured on realistic K/V, per-token INT8 lands at 0.64% relative
# error against e4m3's 2.60% — and at 2.6% per layer the 28-layer draft model's
# logits degenerate completely (0.2% argmax agreement with bf16).
_QUANT_MAX = {torch.int8: 127.0}
for _n in ("float8_e4m3fn", "float8_e4m3fnuz"):
    if hasattr(torch, _n):
        _QUANT_MAX[getattr(torch, _n)] = 448.0


class BlockManager:
    """Pool of physical KV-cache blocks shared across all sequences."""

    def __init__(self, n_total: int, block_size: int) -> None:
        if n_total <= 0 or block_size <= 0:
            raise ValueError(f"n_total and block_size must be > 0, got {n_total}, {block_size}")
        self.n_total = n_total
        self.block_size = block_size
        self._free: list[int] = list(range(n_total))

    def allocate(self, n: int) -> list[int]:
        """Reserve ``n`` blocks. Raises RuntimeError on pool exhaustion (KV-cache OOM)."""
        if len(self._free) < n:
            raise RuntimeError(
                f"KV-cache OOM: need {n} blocks, only {len(self._free)} free "
                f"(total={self.n_total}, block_size={self.block_size})"
            )
        return [self._free.pop() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        """Return ``block_ids`` to the free pool."""
        self._free.extend(block_ids)

    @property
    def n_free(self) -> int:
        return len(self._free)

    def blocks_needed(self, token_count: int) -> int:
        """Minimum blocks to hold ``token_count`` tokens."""
        if token_count <= 0:
            return 0
        return math.ceil(token_count / self.block_size)


class PagedLlamaKVCache:
    """Paged KV cache for LLaMA with GQA support.

    Sequences are identified by a caller-assigned integer ``seq_id``.  The cache
    manages their physical block allocation and provides an ``extend`` interface
    compatible with ``LlamaStaticKVCache`` so ``llama_attention`` needs no changes.

    Typical lifecycle per sequence
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        cache.allocate_sequence(seq_id, prompt_len)   # grab initial blocks
        cache.begin_step([seq_id, ...])               # before each forward call
        # llama_attention calls cache.extend() per layer inside model.forward()
        cache.ensure_slot(seq_id)                     # before EACH decode step
        cache.free_sequence(seq_id)                   # when done

    Memory footprint
    ~~~~~~~~~~~~~~~~
    Only allocated (not free) blocks consume VRAM.  The physical pool is the
    ceiling; actual usage tracks the live token population.
    """

    def __init__(
        self,
        config: LlamaConfig,
        manager: BlockManager,
        device: str,
        dtype: torch.dtype,
        owned_layers: range | None = None,
        kv_dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            kv_dtype: Storage dtype for the K/V pool. Defaults to ``dtype``.
                Pass ``torch.int8`` (preferred) or ``torch.float8_e4m3fn``
                to halve the pool — and, more to
                the point, halve the KV traffic that dominates long-context
                decode. FP8 storage keeps a float32 scale per (block, head,
                position): 4 bytes against head_dim bytes of payload, ~3%
                overhead for a 1.94x cut in bytes read per token.

                INT8 is 4x more accurate than FP8 at the same size here — see
                ``_QUANT_MAX``. Only the fused paged-attention path reads 8-bit
                codes directly; the gather path dequantizes, so it stays correct
                but gives the bandwidth back.
            owned_layers: Contiguous layer range this cache stores, e.g.
                ``range(4, 8)`` for a pipeline stage that only owns layers
                4-7. Defaults to every layer (``range(config.n_layer)``,
                single-device behavior). The pool is sized to
                ``len(owned_layers)`` layers, not ``config.n_layer`` — a
                stage only pays VRAM for the layers it actually runs.
                ``extend``/``extend_static`` take the model's *absolute*
                layer index and remap it to a local pool row internally.
        """
        self.config = config
        self.manager = manager
        self.owned_layers = owned_layers if owned_layers is not None else range(config.n_layer)
        self.layer_offset = self.owned_layers.start
        n_total = manager.n_total
        bs = manager.block_size
        shape = (len(self.owned_layers), n_total, config.n_kv_heads, bs, config.head_dim)
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype
        self.quantized = self.kv_dtype in _QUANT_MAX
        self.qmax = _QUANT_MAX.get(self.kv_dtype, 0.0)
        self.is_int = self.kv_dtype is torch.int8
        self.k_pool = torch.zeros(shape, device=device, dtype=self.kv_dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=self.kv_dtype)
        if self.quantized:
            # One scale per (block, head, position) — the granularity the kernel
            # can fold into the score/value math for free.
            self.k_scale = torch.ones(shape[:-1], device=device, dtype=torch.float32)
            self.v_scale = torch.ones(shape[:-1], device=device, dtype=torch.float32)
        else:
            self.k_scale = self.v_scale = None
        self.block_table: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self._active: list[int] = []

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def allocate_sequence(self, seq_id: int, prompt_len: int) -> None:
        """Reserve blocks for a new sequence with a prompt of ``prompt_len`` tokens.

        Allocates ``ceil(prompt_len / block_size)`` physical blocks (at least 1 so
        a short prefill always has room to write).
        """
        if seq_id in self.block_table:
            raise ValueError(f"seq_id={seq_id} is already allocated; call free_sequence first")
        n = self.manager.blocks_needed(max(prompt_len, 1))
        self.block_table[seq_id] = self.manager.allocate(n)
        self.seq_lens[seq_id] = 0

    def ensure_slot(self, seq_id: int) -> None:
        """Guarantee room for one more token in the sequence.

        Call this **before** each decode step (after prefill, not before it —
        the initial allocation already covers the prompt).  If the current token
        count is an exact multiple of ``block_size``, the last block is full and
        a new physical block is appended.
        """
        length = self.seq_lens[seq_id]
        if length > 0 and length % self.manager.block_size == 0:
            self.block_table[seq_id].extend(self.manager.allocate(1))

    def ensure_slots_for(self, seq_id: int, n_tokens: int) -> None:
        """Guarantee room for at least ``n_tokens`` more tokens in the sequence.

        Unlike ``ensure_slot`` (which only allocates at exact block boundaries),
        this allocates however many blocks are needed so that
        ``seq_lens[seq_id] + n_tokens`` tokens can be stored. Used by the
        speculative verify step to pre-allocate K+1 slots in one call.
        """
        length = self.seq_lens[seq_id]
        needed = self.manager.blocks_needed(length + n_tokens)
        current = len(self.block_table[seq_id])
        if needed > current:
            self.block_table[seq_id].extend(self.manager.allocate(needed - current))

    def free_sequence(self, seq_id: int) -> None:
        """Return all physical blocks of ``seq_id`` to the pool."""
        self.manager.free(self.block_table.pop(seq_id, []))
        self.seq_lens.pop(seq_id, None)

    def reset_to(self, seq_id: int, pos: int) -> None:
        """Roll back a sequence to ``pos`` filled tokens.

        Blocks allocated strictly beyond ``pos`` are returned to the free pool.
        Block data is NOT zeroed — the retained blocks still contain valid KV
        values at positions < pos, which is exactly what we need for speculative
        decoding rollback.

        Args:
            seq_id: Sequence to roll back.
            pos:    New token count (must be ≤ current seq_len).
        """
        if seq_id not in self.seq_lens:
            raise ValueError(f"reset_to: seq_id={seq_id} not allocated")
        current = self.seq_lens[seq_id]
        if pos < 0 or pos > current:
            raise ValueError(f"reset_to: pos={pos} out of range [0, {current}]")
        bs = self.manager.block_size
        # Blocks needed to hold pos tokens (at least 1 so the sequence is never
        # left with zero blocks, matching the invariant set by allocate_sequence).
        keep = max(self.manager.blocks_needed(pos), 1)
        excess = self.block_table[seq_id][keep:]
        if excess:
            self.manager.free(excess)
            self.block_table[seq_id] = self.block_table[seq_id][:keep]
        self.seq_lens[seq_id] = pos

    # ------------------------------------------------------------------
    # Per-step interface (call begin_step then model.forward)
    # ------------------------------------------------------------------

    def begin_step(self, seq_ids: list[int]) -> None:
        """Declare which sequences participate in the upcoming model.forward."""
        self._active = seq_ids

    def extend(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V into physical blocks; return gathered K/V histories.

        This is called once per transformer layer by ``llama_attention``.

        Args:
            layer:     Absolute model layer index (not offset by
                       ``owned_layers`` — this cache remaps internally).
            k_new:     New keys.   Shape: (A, n_kv_heads, q_len, head_dim)
            v_new:     New values. Shape: (A, n_kv_heads, q_len, head_dim)
            start_pos: Ignored — each sequence tracks its own write position
                       via ``seq_lens``.  Present only to match the
                       ``LlamaStaticKVCache`` interface.

        Returns:
            ``(k_all, v_all)``, each ``(A, n_kv_heads, max_len, head_dim)``
            where ``max_len = max(new lengths of active sequences)``.
            Positions beyond a sequence's own length are zero.
        """
        if layer not in self.owned_layers:
            raise ValueError(f"layer {layer} not owned by this cache ({self.owned_layers})")
        local_layer = layer - self.layer_offset
        A, _, q_len, _ = k_new.shape
        if A != len(self._active):
            raise RuntimeError(f"batch {A} != begin_step had {len(self._active)} seqs")
        bs = self.manager.block_size

        # Snapshot write bases before updating (seq_lens before this layer's writes).
        write_bases = [self.seq_lens[sid] for sid in self._active]
        new_lens = [wb + q_len for wb in write_bases]

        # ── Write new tokens into physical blocks (block-aligned slices) ──
        if self.quantized:
            # Same fused quantize+scatter the graph path uses. Doing it with a
            # Python loop over blocks made prefill dramatically slower — enough
            # to swamp every read-side saving FP8 buys.
            max_blocks = max(len(self.block_table[sid]) for sid in self._active)
            bt = torch.zeros(A, max_blocks, dtype=torch.long, device=k_new.device)
            for i, sid in enumerate(self._active):
                row = self.block_table[sid]
                bt[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=k_new.device)
            write_kv_quant(
                k_new, v_new,
                self.k_pool[local_layer], self.v_pool[local_layer],
                self.k_scale[local_layer], self.v_scale[local_layer],
                bt, torch.tensor(write_bases, dtype=torch.long, device=k_new.device),
                page_size=bs, qmax=self.qmax, is_int=self.is_int,
            )
        else:
          for i, sid in enumerate(self._active):
              base = write_bases[i]
              phys = self.block_table[sid]
              first_b = base // bs
              last_b = (base + q_len - 1) // bs
              for b_idx in range(first_b, last_b + 1):
                  blk_start = b_idx * bs
                  wrt_start = max(blk_start, base)
                  wrt_end = min(blk_start + bs, base + q_len)
                  off = wrt_start - blk_start     # offset inside block
                  n_t = wrt_end - wrt_start
                  src = wrt_start - base           # index into k_new[i]
                  self.k_pool[local_layer, phys[b_idx], :, off : off + n_t, :] = \
                      k_new[i, :, src : src + n_t, :]
                  self.v_pool[local_layer, phys[b_idx], :, off : off + n_t, :] = \
                      v_new[i, :, src : src + n_t, :]

        # Advance seq_lens once per full forward — after the final layer this
        # cache owns (not necessarily config.n_layer - 1: a non-last pipeline
        # stage's cache never sees layers beyond its own owned_layers).
        if layer == self.owned_layers[-1]:
            for i, sid in enumerate(self._active):
                self.seq_lens[sid] = new_lens[i]

        # ── Gather K/V histories for all active sequences ─────────────
        max_len = max(new_lens)
        n_kv_h = self.config.n_kv_heads
        head_dim = self.config.head_dim
        k_out = k_new.new_zeros(A, n_kv_h, max_len, head_dim)
        v_out = v_new.new_zeros(A, n_kv_h, max_len, head_dim)

        for i, sid in enumerate(self._active):
            slen = new_lens[i]
            phys = self.block_table[sid]
            n_full = slen // bs
            remainder = slen % bs
            n_used = n_full + (1 if remainder else 0)
            idx = torch.tensor(phys[:n_used], dtype=torch.long, device=k_out.device)
            # (n_used, n_kv, bs, D) -> (n_kv, n_used*bs, D), then trim to slen.
            kb = self.k_pool[local_layer, idx]
            vb = self.v_pool[local_layer, idx]
            if self.quantized:
                # One dequantize over the whole gathered history rather than one
                # per physical block — the per-block form dominated prefill.
                kb = self._dequantize(kb, self.k_scale[local_layer, idx])
                vb = self._dequantize(vb, self.v_scale[local_layer, idx])
            kb = kb.permute(1, 0, 2, 3).reshape(n_kv_h, n_used * bs, head_dim)
            vb = vb.permute(1, 0, 2, 3).reshape(n_kv_h, n_used * bs, head_dim)
            k_out[i, :, :slen, :] = kb[:, :slen, :]
            v_out[i, :, :slen, :] = vb[:, :slen, :]

        return k_out, v_out

    # ------------------------------------------------------------------
    # CUDA-graph-safe decode path (fixed-shape tensor gather/scatter)
    # ------------------------------------------------------------------
    #
    # ``extend()`` above loops over Python ints (block ids read out of the
    # ``block_table`` dict) — each iteration becomes a *fixed* CUDA copy
    # kernel at graph-capture time, baked in against whatever physical
    # block addresses happened to be active during capture. Replaying that
    # graph against a different request's block table would read/write the
    # wrong memory. The methods below instead address the pool via tensor
    # *values* (gather/advanced-indexing), so a graph captured once is
    # correct on replay against new buffer *contents* copied in before each
    # replay — the standard approach production paged-attention engines use
    # to make continuous batching CUDA-graph-compatible. Used only for the
    # single-token decode step (q_len == 1); prefill keeps using extend().

    def build_static_buffers(
        self, seq_ids: list[int], bucket_size: int, capture_len: int, device: str,
        write_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded ``(block_table, seq_lens)`` tensors for one decode bucket.

        Real rows mirror the live Python-side ``block_table``/``seq_lens``.
        Padding *rows* (``bucket_size - len(seq_ids)`` of them, when fewer
        sequences are active than the bucket's batch size) repeat the last
        real row — those extra batch lanes do harmless redundant work and
        their outputs are discarded by the caller. Padding *columns* (beyond
        a sequence's currently-allocated block count, up to
        ``capture_len // block_size``) repeat that sequence's own block 0 —
        always a valid, allocated block; ``attn_mask`` hides the unwritten
        positions from SDPA so their content never affects the result.

        **Column padding is safe for reads only.** A forward that *writes* KV
        through a padded column aliases the sequence's own block 0 and silently
        corrupts its first ``block_size`` positions. Callers that write must
        allocate first (``ensure_slot`` / ``ensure_slots_for``) and pass
        ``write_len`` — the number of positions this forward will write starting
        at ``seq_lens[sid]`` — so the invariant is checked rather than assumed.

        Args:
            seq_ids:     Sequences to include, in batch-lane order.
            bucket_size: Batch lanes in the captured graph (rows are padded to this).
            capture_len: KV-gather length of the bucket; must be a multiple of
                         ``block_size``.
            device:      Device for the returned tensors.
            write_len:   Positions this forward will write per sequence (0 for
                         read-only / gather-only uses).
        """
        bs = self.manager.block_size
        if capture_len % bs != 0:
            raise ValueError(f"capture_len={capture_len} must be a multiple of block_size={bs}")
        if not seq_ids:
            raise ValueError("seq_ids must be non-empty")
        n_blocks_cap = capture_len // bs

        rows_blocks: list[list[int]] = []
        rows_lens: list[int] = []
        for sid in seq_ids:
            phys = self.block_table[sid]
            length = self.seq_lens[sid]
            if write_len and self.manager.blocks_needed(length + write_len) > len(phys):
                raise ValueError(
                    f"seq {sid}: writing {write_len} positions from {length} needs "
                    f"{self.manager.blocks_needed(length + write_len)} blocks but only "
                    f"{len(phys)} are allocated — call ensure_slots_for() first. "
                    f"Writing through padded columns would corrupt positions "
                    f"0..{bs - 1} of this sequence."
                )
            row = list(phys[:n_blocks_cap])
            if len(row) < n_blocks_cap:
                row += [phys[0]] * (n_blocks_cap - len(row))
            rows_blocks.append(row)
            rows_lens.append(length)

        while len(rows_blocks) < bucket_size:
            rows_blocks.append(rows_blocks[-1])
            rows_lens.append(rows_lens[-1])

        block_table_buf = torch.tensor(rows_blocks, dtype=torch.long, device=device)
        seq_lens_buf = torch.tensor(rows_lens, dtype=torch.long, device=device)
        return block_table_buf, seq_lens_buf

    def write_static(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_table_buf: torch.Tensor,
        seq_lens_buf: torch.Tensor,
    ) -> None:
        """Write new K/V into physical blocks. Fixed-shape, graph-capturable.

        The write half of ``extend_static``, split out so the fused paged
        attention kernel can consume the pool directly instead of paying for a
        contiguous gather it does not need.

        Args:
            k_new, v_new:    (A, n_kv_heads, q_len, head_dim). q_len is fixed at
                             capture time so this loop unrolls into static ops.
            block_table_buf: (A, capture_len // block_size) physical block ids.
            seq_lens_buf:    (A,) token count *before* this write.
        """
        if layer not in self.owned_layers:
            raise ValueError(f"layer {layer} not owned by this cache ({self.owned_layers})")
        local_layer = layer - self.layer_offset
        bs = self.manager.block_size
        if self.quantized:
            # One fused launch: quantize and scatter together. Doing it with
            # torch ops was measured slower than bf16 outright — the write
            # overhead more than cancelled the read-bandwidth saving.
            write_kv_quant(
                k_new, v_new,
                self.k_pool[local_layer], self.v_pool[local_layer],
                self.k_scale[local_layer], self.v_scale[local_layer],
                block_table_buf, seq_lens_buf,
                page_size=bs, qmax=self.qmax, is_int=self.is_int,
            )
            return
        for q in range(k_new.shape[2]):
            block_idx_q = (seq_lens_buf + q) // bs                           # (A,)
            offset_q    = (seq_lens_buf + q) % bs                            # (A,)
            phys_q = block_table_buf.gather(1, block_idx_q.unsqueeze(1)).squeeze(1)  # (A,)
            self.k_pool[local_layer, phys_q, :, offset_q, :] = k_new[:, :, q, :]
            self.v_pool[local_layer, phys_q, :, offset_q, :] = v_new[:, :, q, :]

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-(sequence, head) FP8 quantization of one token's K or V vector.

        Scaling by ``amax / qmax`` puts the largest magnitude exactly at the
        format's limit, so nothing saturates and the full range is used.
        All tensor ops, no data-dependent branching — safe to graph-capture.

        Args:
            x: (A, n_kv_heads, head_dim)

        Returns:
            ``(quantized, scale)`` with shapes (A, n_kv_heads, head_dim) and
            (A, n_kv_heads).
        """
        scale = (x.abs().amax(dim=-1).float() / self.qmax).clamp(min=1e-12)
        q = x.float() / scale.unsqueeze(-1)
        if self.is_int:
            q = q.round().clamp(-self.qmax, self.qmax)
        return q.to(self.kv_dtype), scale

    def _dequantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Undo ``_quantize`` for the gather path, which reads bf16."""
        return (x.to(torch.float32) * scale.unsqueeze(-1)).to(self.dtype)


    def paged_attend(
        self,
        layer: int,
        q: torch.Tensor,
        block_table_buf: torch.Tensor,
        kv_lens: torch.Tensor,
        n_splits: int | None = None,
    ) -> torch.Tensor:
        """Attend against this layer's pool through the block table.

        Assumes the current step's K/V are already written (``write_static``).
        Reads the pool in place — no contiguous gather, no GQA expansion — so
        cost tracks each sequence's true length rather than the captured bucket.

        Args:
            layer:           Absolute model layer index.
            q:               (A, n_head, q_len, head_dim).
            block_table_buf: (A, n_blocks) physical block ids.
            kv_lens:         (A,) valid KV positions *including* this write.
            n_splits:        KV splits; must be constant across a CUDA-graph
                             capture. None lets the kernel choose.

        Returns:
            (A, n_head, q_len, head_dim), same dtype as ``q``.
        """
        if layer not in self.owned_layers:
            raise ValueError(f"layer {layer} not owned by this cache ({self.owned_layers})")
        local_layer = layer - self.layer_offset
        return paged_flash_attention(
            q, self.k_pool[local_layer], self.v_pool[local_layer],
            block_table_buf, kv_lens,
            page_size=self.manager.block_size,
            k_scale=self.k_scale[local_layer] if self.quantized else None,
            v_scale=self.v_scale[local_layer] if self.quantized else None,
            n_splits=n_splits,
        )

    def extend_static(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_table_buf: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        capture_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-shape write + gather, safe to capture in a CUDA graph.

        Args:
            k_new, v_new:    New keys/values. Shape: (A, n_kv_heads, q_len, head_dim).
                             q_len=1 for decode steps; q_len=K+1 for verify steps.
                             q_len is fixed at graph-capture time so the Python loop
                             below unrolls into static CUDA ops.
            block_table_buf: (A, capture_len // block_size) physical block ids.
            seq_lens_buf:    (A,) each row's token count *before* this write.
            capture_len:     Fixed gather length for this bucket (a multiple
                             of block_size).

        Returns:
            ``(k_out, v_out)``, each ``(A, n_kv_heads, capture_len, head_dim)``.
        """
        self.write_static(layer, k_new, v_new, block_table_buf, seq_lens_buf)

        local_layer = layer - self.layer_offset
        bs = self.manager.block_size
        n_blocks_cap = capture_len // bs
        phys_all = block_table_buf[:, :n_blocks_cap]                      # (A, n_blocks_cap)
        k_gathered = self.k_pool[local_layer, phys_all]      # (A, n_blocks_cap, n_kv_heads, bs, head_dim)
        v_gathered = self.v_pool[local_layer, phys_all]
        if self.quantized:
            # This path hands SDPA a dense bf16 tensor, so the FP8 saving is
            # spent here. It exists for correctness of the fallback only —
            # fused_attend reads the FP8 pool directly and keeps the bandwidth.
            k_gathered = self._dequantize(k_gathered, self.k_scale[local_layer, phys_all])
            v_gathered = self._dequantize(v_gathered, self.v_scale[local_layer, phys_all])
        n_kv_h, head_dim = self.config.n_kv_heads, self.config.head_dim
        k_out = k_gathered.permute(0, 2, 1, 3, 4).reshape(k_new.shape[0], n_kv_h, capture_len, head_dim)
        v_out = v_gathered.permute(0, 2, 1, 3, 4).reshape(k_new.shape[0], n_kv_h, capture_len, head_dim)
        return k_out, v_out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_free_blocks(self) -> int:
        return self.manager.n_free

    def memory_bytes(self) -> int:
        """Total bytes occupied by the physical K/V pool (allocated + free blocks)."""
        return (
            self.k_pool.numel() * self.k_pool.element_size()
            + self.v_pool.numel() * self.v_pool.element_size()
        )
