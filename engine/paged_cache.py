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
    ) -> None:
        self.config = config
        self.manager = manager
        n_total = manager.n_total
        bs = manager.block_size
        shape = (config.n_layer, n_total, config.n_kv_heads, bs, config.head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=dtype)
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

    def free_sequence(self, seq_id: int) -> None:
        """Return all physical blocks of ``seq_id`` to the pool."""
        self.manager.free(self.block_table.pop(seq_id, []))
        self.seq_lens.pop(seq_id, None)

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
            layer:     Layer index.
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
        A, _, q_len, _ = k_new.shape
        if A != len(self._active):
            raise RuntimeError(f"batch {A} != begin_step had {len(self._active)} seqs")
        bs = self.manager.block_size

        # Snapshot write bases before updating (seq_lens before this layer's writes).
        write_bases = [self.seq_lens[sid] for sid in self._active]
        new_lens = [wb + q_len for wb in write_bases]

        # ── Write new tokens into physical blocks (block-aligned slices) ──
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
                self.k_pool[layer, phys[b_idx], :, off : off + n_t, :] = k_new[i, :, src : src + n_t, :]
                self.v_pool[layer, phys[b_idx], :, off : off + n_t, :] = v_new[i, :, src : src + n_t, :]

        # Advance seq_lens once per full forward (after the final layer).
        if layer == self.config.n_layer - 1:
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
            for b_idx in range(n_full):
                dst = b_idx * bs
                k_out[i, :, dst : dst + bs, :] = self.k_pool[layer, phys[b_idx]]
                v_out[i, :, dst : dst + bs, :] = self.v_pool[layer, phys[b_idx]]
            if remainder:
                dst = n_full * bs
                k_out[i, :, dst : dst + remainder, :] = self.k_pool[layer, phys[n_full], :, :remainder, :]
                v_out[i, :, dst : dst + remainder, :] = self.v_pool[layer, phys[n_full], :, :remainder, :]

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
