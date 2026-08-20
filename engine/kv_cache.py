"""KV cache for LLaMA GQA inference.

Stores ``n_kv_heads`` (not ``n_head``) per layer — that is the GQA dividend:
4× less KV memory than a naive MHA cache for LLaMA 3 (32 Q / 8 KV heads).

Per forward step, ``extend`` writes new K/V at ``start_pos`` and returns the
full key/value history for that layer to attend over.
"""

from __future__ import annotations

import torch

from engine.config import LlamaConfig


class LlamaStaticKVCache:
    """Static KV cache for LLaMA GQA: pre-allocated (n_kv_heads, max_seq) per layer.

    Memory is fixed at creation (= formula at ``max_seq``) regardless of fill level,
    which avoids per-step reallocation and fragmentation.
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
        """Roll back cache to ``pos`` filled tokens (speculative decode rollback)."""
        if pos < 0 or pos > self.max_seq:
            raise ValueError(f"reset_to pos={pos} out of range [0, {self.max_seq}]")
        self._length = pos

    @property
    def length(self) -> int:
        return self._length

    def memory_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
