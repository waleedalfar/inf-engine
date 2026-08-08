"""KV cache: store per-layer keys and values so decode is O(1) per step, not O(T).

## What the cache eliminates

Autoregressive decoding generates one token at a time. The attention at step ``t``
needs the keys and values of *every* previous position, but those K/V vectors are a
pure function of tokens already seen — they never change once computed. The Phase 1
engine recomputes them from scratch at every step (the whole sequence, every layer),
which is the O(T^2) total cost. The cache stores each layer's K and V and only
computes them for the *new* token, then appends. Decode becomes O(1) work per step
(plus the unavoidable O(t) attention dot-products against the cached keys).

## Memory cost — derived from first principles

Each layer stores one K and one V tensor. Per token, per layer, each of K and V holds
``n_head * head_dim`` floats (= d_model for GPT-2). Over the whole stack:

    bytes = 2 * n_layer * n_head * head_dim * seq_len * batch * bytes_per_element
            ^                                                    ^
            K and V                                              4 for fp32, 2 for fp16

For gpt2-small (L=12, H=12, d_head=64) at seq_len=1024, batch=1, fp32:
    2 * 12 * 12 * 64 * 1024 * 1 * 4  = 150,994,944 bytes ≈ 144 MiB per sequence.

This is the exact size of the allocated cache tensors, so empirical
``numel * element_size`` matches the formula to the byte (verified in tests). The CUDA
caching allocator may *reserve* more than this (it rounds allocations up to block
sizes), which is why ``torch.cuda.memory_allocated`` can read higher — that gap is
allocator bookkeeping, not cache data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from engine.config import GPT2Config, LlamaConfig


def kv_cache_bytes(
    config: GPT2Config,
    seq_len: int,
    dtype: torch.dtype,
    batch: int = 1,
) -> int:
    """Theoretical KV-cache size in bytes from the first-principles formula.

    bytes = 2 * n_layer * n_head * head_dim * seq_len * batch * bytes_per_element

    Args:
        config:  Model config (supplies n_layer, n_head, head_dim).
        seq_len: Number of cached token positions.
        dtype:   Element dtype (its ``itemsize`` is bytes_per_element).
        batch:   Number of sequences cached in parallel.

    Returns:
        Exact byte count the K and V tensors occupy.
    """
    bytes_per_element = torch.empty(0, dtype=dtype).element_size()
    return (
        2  # one K tensor + one V tensor
        * config.n_layer
        * config.n_head
        * config.head_dim
        * seq_len
        * batch
        * bytes_per_element
    )


class KVCache(ABC):
    """Per-layer key/value store shared across one autoregressive generation.

    A single forward step calls ``extend`` once per layer (passing that layer's freshly
    computed K/V for the new token(s) and the absolute ``start_pos`` of those tokens),
    and gets back the full K/V history for that layer to attend over.
    """

    def __init__(
        self,
        config: GPT2Config,
        batch: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.batch = batch
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def extend(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's new K/V at ``start_pos`` and return the full history.

        Args:
            layer:     Layer index in [0, n_layer).
            k_new:     New keys.   Shape: (batch, n_head, q_len, head_dim)
            v_new:     New values. Shape: (batch, n_head, q_len, head_dim)
            start_pos: Absolute position of the first new token.

        Returns:
            (k_all, v_all), each (batch, n_head, start_pos + q_len, head_dim).
        """

    @property
    @abstractmethod
    def length(self) -> int:
        """Number of token positions currently cached."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """Bytes actually occupied by the K/V tensors right now."""


class StaticKVCache(KVCache):
    """Pre-allocated cache: one fixed (batch, n_head, max_seq, head_dim) buffer per layer.

    Allocates the full ``max_seq`` up front and writes new K/V into slices. Memory is
    constant from creation (= the formula at ``max_seq``) regardless of how full it is.
    This mirrors how production engines reserve KV blocks ahead of time to avoid
    per-step allocation and fragmentation.
    """

    def __init__(
        self,
        config: GPT2Config,
        batch: int,
        max_seq: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config, batch, device, dtype)
        self.max_seq = max_seq
        shape = (config.n_layer, batch, config.n_head, max_seq, config.head_dim)
        # (L, B, H, max_seq, d_head) for K and V — allocated once, never grows.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self._length = 0

    def extend(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = k_new.shape[2]                                   # new tokens this step
        end = start_pos + q_len
        if end > self.max_seq:
            raise ValueError(f"static cache overflow: {end} > max_seq={self.max_seq}")
        # write new K/V into the reserved slots [start_pos:end]
        self.k[layer, :, :, start_pos:end, :] = k_new           # (B, H, q_len, d_head)
        self.v[layer, :, :, start_pos:end, :] = v_new
        if layer == self.config.n_layer - 1:
            self._length = end                                  # advance once per full step
        return self.k[layer, :, :, :end, :], self.v[layer, :, :, :end, :]  # (B, H, end, d_head)

    def reset_to(self, pos: int) -> None:
        """Roll back the cache to ``pos`` filled tokens.

        K/V data at positions [pos, max_seq) is stale but will be silently
        overwritten on the next write at those positions.  Used by speculative
        decoding to discard rejected draft tokens and their K/V entries.
        """
        if pos < 0 or pos > self.max_seq:
            raise ValueError(f"reset_to pos={pos} out of range [0, {self.max_seq}]")
        self._length = pos

    @property
    def length(self) -> int:
        return self._length

    def memory_bytes(self) -> int:
        # Full pre-allocation: independent of current fill level.
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


class DynamicKVCache(KVCache):
    """Growing cache: each layer's K/V tensor is concatenated as the sequence extends.

    Allocates nothing up front; memory grows with the actual sequence length (= the
    formula at the current length). Simpler and memory-frugal for short/variable
    sequences, but each growth step reallocates and copies — the classic
    flexibility-vs-fragmentation tradeoff versus the static cache.
    """

    def __init__(
        self,
        config: GPT2Config,
        batch: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config, batch, device, dtype)
        # Per-layer K/V tensors, created on first extend. None until then.
        self.k: list[torch.Tensor | None] = [None] * config.n_layer
        self.v: list[torch.Tensor | None] = [None] * config.n_layer
        self._length = 0

    def extend(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_pos == 0 or self.k[layer] is None:
            # First write (prefill): the new K/V *is* the whole history.
            self.k[layer] = k_new                               # (B, H, q_len, d_head)
            self.v[layer] = v_new
        else:
            # Grow by concatenation along the sequence axis.
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)  # (B, H, prev+q_len, d_head)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        if layer == self.config.n_layer - 1:
            self._length = self.k[layer].shape[2]               # advance once per full step
        return self.k[layer], self.v[layer]

    @property
    def length(self) -> int:
        return self._length

    def memory_bytes(self) -> int:
        total = 0
        for t in (*self.k, *self.v):
            if t is not None:
                total += t.numel() * t.element_size()
        return total


class LlamaStaticKVCache:
    """Static KV cache for LLaMA GQA: stores ``n_kv_heads`` (not ``n_head``) per layer.

    Identical interface to ``StaticKVCache`` — ``extend`` writes new K/V at
    ``start_pos`` and returns the full history — but the backing tensors are
    shaped ``(L, B, n_kv_heads, max_seq, head_dim)`` so only the KV heads are
    stored.  The model's attention function calls ``repeat_kv`` after retrieval
    to expand back to ``n_head`` before the scaled dot-product.

    Memory vs StaticKVCache:
        Saving = n_kv_groups× per layer. For LLaMA 3 (32 Q / 8 KV heads):
        4× less KV memory than a naive MHA cache — this is the GQA dividend.
    """

    def __init__(
        self,
        config: LlamaConfig,
        batch: int,
        max_seq: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.batch = batch
        self.max_seq = max_seq
        shape = (config.n_layer, batch, config.n_kv_heads, max_seq, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self._length = 0

    def extend(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = k_new.shape[2]
        end = start_pos + q_len
        if end > self.max_seq:
            raise ValueError(f"LlamaStaticKVCache overflow: {end} > max_seq={self.max_seq}")
        self.k[layer, :, :, start_pos:end, :] = k_new
        self.v[layer, :, :, start_pos:end, :] = v_new
        if layer == self.config.n_layer - 1:
            self._length = end
        return self.k[layer, :, :, :end, :], self.v[layer, :, :, :end, :]

    def reset_to(self, pos: int) -> None:
        """Roll back the cache to ``pos`` filled tokens (speculative decode rollback)."""
        if pos < 0 or pos > self.max_seq:
            raise ValueError(f"reset_to pos={pos} out of range [0, {self.max_seq}]")
        self._length = pos

    @property
    def length(self) -> int:
        return self._length

    def memory_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


class SlotKVCache:
    """KV cache for continuous batching: ``n_slots`` independent, variable-length slots.

    Unlike the static/dynamic caches above (one growing sequence, uniform ``start_pos``),
    each slot here holds its own sequence at its own length and can be reset and reused as
    requests complete. A decode step writes one token into *each active slot at that slot's
    own current length*, so positions differ across the batch — which is why continuous
    batching needs the explicit ``attn_mask`` / ``position_ids`` paths rather than the
    scalar-``start_pos`` machinery.

    Usage per forward step:
        cache.begin_step(active_slot_indices)   # records which slots & their write offsets
        model.forward(..., cache=cache, attn_mask=..., position_ids=...)
        # extend() is called once per layer by the model; lengths advance after the last.
    """

    def __init__(
        self,
        config: GPT2Config,
        n_slots: int,
        max_seq: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.n_slots = n_slots
        self.max_seq = max_seq
        shape = (config.n_layer, n_slots, config.n_head, max_seq, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)   # (L, S, H, max_seq, d_head)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)  # filled len per slot
        self._active: torch.Tensor | None = None      # slot indices for the current step
        self._write_pos: torch.Tensor | None = None   # per-active write offset (len before step)
        self._step_max_len: int | None = None         # memoized max len for the step (avoid per-layer sync)

    def reset_slot(self, slot: int) -> None:
        """Free a slot for reuse: logically clear it (data is overwritten on next prefill)."""
        self.lengths[slot] = 0

    def begin_step(self, active_slots: torch.Tensor) -> None:
        """Declare which slots participate in the upcoming forward and where they write.

        Args:
            active_slots: 1-D LongTensor of slot indices active this step.
        """
        self._active = active_slots
        self._write_pos = self.lengths[active_slots].clone()      # (A,)
        self._step_max_len = None                                 # recomputed once per step in extend

    def extend(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write each active slot's new K/V at its own offset; return active histories.

        Args:
            layer:     Layer index.
            k_new:     New keys.   Shape: (A, n_head, q_len, head_dim)
            v_new:     New values. Shape: (A, n_head, q_len, head_dim)
            start_pos: Ignored (per-slot offsets come from ``begin_step``); kept for the
                       common cache interface.

        Returns:
            (k_all, v_all), each (A, n_head, max_len, head_dim) where
            max_len = max over active slots of (write_offset + q_len).
        """
        assert self._active is not None and self._write_pos is not None, "call begin_step first"
        active, write_pos = self._active, self._write_pos
        q_len = k_new.shape[2]

        if q_len == 1:
            # Decode: vectorized scatter — each slot writes its single token at its offset.
            self.k[layer, active, :, write_pos, :] = k_new[:, :, 0, :]  # (A, H, d_head)
            self.v[layer, active, :, write_pos, :] = v_new[:, :, 0, :]
        else:
            # Prefill: a single slot writes its whole prompt at [0:q_len].
            for i in range(active.shape[0]):
                p = int(write_pos[i])
                self.k[layer, active[i], :, p : p + q_len, :] = k_new[i]
                self.v[layer, active[i], :, p : p + q_len, :] = v_new[i]

        # max_len is identical across layers in a step; compute once to avoid a GPU sync per layer.
        if self._step_max_len is None:
            self._step_max_len = int((write_pos + q_len).max())
        max_len = self._step_max_len
        k_all = self.k[layer, active, :, :max_len, :]             # (A, H, max_len, d_head)
        v_all = self.v[layer, active, :, :max_len, :]
        if layer == self.config.n_layer - 1:
            self.lengths[active] = write_pos + q_len              # advance once per full step
        return k_all, v_all

    def memory_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
