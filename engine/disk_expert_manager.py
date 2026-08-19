"""Disk-streaming expert loader for Qwen3-MoE.

Replaces ExpertOffloadManager when system RAM is insufficient to hold all
routed expert weights (~57 GB for Qwen3-30B-A3B in bf16).

At startup: scan safetensors headers to build a lightweight index
(shard path + tensor key per expert tensor).  No tensor data is read.

At each MoE layer forward: read only the active top-K experts from disk,
cast to the inference dtype, move to VRAM.  Peak RAM ≈ 38 MB (one layer's
top-8 experts in flight), compared to 57 GB for full RAM offload.

Async overlap
-------------
``prefetch_async(layer_idx, active_ids)`` launches the disk read in a
background thread and returns a Future.  Calling code should then run the
shared-expert forward (always-active, weights already on VRAM) and only
call ``.result()`` afterward.  On NVMe (~4 GB/s) the 38 MB read takes
~10 ms, which is typically hidden behind the ~10–20 ms shared-expert matmul.

API parity with ExpertOffloadManager
-------------------------------------
Both classes expose ``fetch(layer_idx, expert_ids)`` and
``prefetch_async(layer_idx, expert_ids)``, so either can be passed to
``LlamaMoEOffloadModel`` without changes to the model forward.

Usage::

    from engine.disk_expert_manager import DiskExpertManager

    mgr = DiskExpertManager.from_shards(shard_paths, config,
                                        device="cuda", dtype=torch.bfloat16)

    # Inside a MoE layer forward:
    future     = mgr.prefetch_async(layer_idx, active_ids)
    shared_out = shared_expert_forward(x)   # overlaps with disk I/O
    expert_w   = future.result()            # wait (usually already done)
"""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import LlamaConfig


def _is_expert_key(key: str) -> bool:
    return ".mlp.experts." in key


def _parse_expert_key(key: str) -> tuple[int, int, str]:
    """'model.layers.{i}.mlp.experts.{j}.{proj}.weight' → (layer, eid, proj_key)."""
    parts = key.split(".")
    return int(parts[2]), int(parts[5]), f"{parts[6]}.weight"


class DiskExpertManager:
    """Streams routed expert weights from safetensors shards on demand.

    Args:
        index:  ``{layer_idx: {expert_id: {proj_key: (shard_path, tensor_key)}}}``.
        device: VRAM device string (e.g. ``"cuda"``).
        dtype:  Inference dtype; tensors are cast on load.
    """

    def __init__(
        self,
        index: dict[int, dict[int, dict[str, tuple[Path, str]]]],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self._index = index
        self._device = device
        self._dtype = dtype
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expert-io"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_shards(
        cls,
        shard_paths: list[Path],
        config: LlamaConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DiskExpertManager":
        """Build expert index by scanning shard headers — no tensor data loaded.

        Args:
            shard_paths: Ordered list of ``.safetensors`` file paths.
            config:      Model config (used for validation output only).
            device:      VRAM device for inference-time ``fetch`` calls.
            dtype:       Tensors are cast to this dtype when read from disk.

        Returns:
            Configured ``DiskExpertManager``.
        """
        index: dict[int, dict[int, dict[str, tuple[Path, str]]]] = {}

        for shard_path in shard_paths:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if not _is_expert_key(key):
                        continue
                    layer_idx, eid, proj_key = _parse_expert_key(key)
                    index.setdefault(layer_idx, {}).setdefault(eid, {})[proj_key] = (
                        shard_path,
                        key,
                    )

        n_layers = len(index)
        n_experts = len(next(iter(index.values()))) if index else 0
        print(
            f"  DiskExpertManager: {n_layers} layers × {n_experts} experts indexed "
            f"(0 MB RAM — weights stay on disk)"
        )
        return cls(index, device, dtype)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_experts(
        self,
        layer_idx: int,
        expert_ids: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Read the requested experts from disk → VRAM.

        Reads are grouped by shard file so each file is opened at most once
        per call, minimising filesystem overhead.
        """
        by_shard: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)
        for eid in expert_ids:
            for proj_key, (shard_path, tensor_key) in self._index[layer_idx][eid].items():
                by_shard[shard_path].append((eid, proj_key, tensor_key))

        result: dict[int, dict[str, torch.Tensor]] = {eid: {} for eid in expert_ids}
        for shard_path, reads in by_shard.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for eid, proj_key, tensor_key in reads:
                    t = f.get_tensor(tensor_key).to(dtype=self._dtype)
                    result[eid][proj_key] = t.to(self._device, non_blocking=True)
        return result

    # ------------------------------------------------------------------
    # Fetch API
    # ------------------------------------------------------------------

    def fetch(
        self,
        layer_idx: int,
        expert_ids: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Synchronously read experts from disk and return tensors on VRAM."""
        return self._load_experts(layer_idx, expert_ids)

    def prefetch_async(
        self,
        layer_idx: int,
        expert_ids: list[int],
    ) -> concurrent.futures.Future:
        """Start disk read in a background thread; return a Future.

        Call ``.result()`` to wait for completion.  Running the shared-expert
        forward between this call and ``.result()`` hides most of the I/O
        latency behind VRAM compute.
        """
        return self._executor.submit(self._load_experts, layer_idx, expert_ids)

    def close(self) -> None:
        """Shut down the background I/O thread pool."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def n_layers(self) -> int:
        return len(self._index)

    @property
    def n_experts(self) -> int:
        if not self._index:
            return 0
        return len(next(iter(self._index.values())))
