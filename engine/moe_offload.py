"""CPU expert offload for Qwen3-MoE: store routed experts in RAM, fetch to VRAM on demand.

Problem: Qwen3-30B-A3B has 128 experts × 48 layers × 3 matrices × 768 × 2048 × 2 bytes
≈ 57 GB of expert weights in bf16.  A single 16 GB GPU cannot hold them all.

Solution: keep all routed expert weights as pinned CPU tensors; at each MoE block forward,
fetch only the K activated experts (top-K routing) to VRAM, compute, then let them go.

Memory breakdown (bf16):
  • 128 experts, each ~4.7 MB → 602 MB per layer → 29 GB for 48 layers (on CPU)
  • At decode, one layer's top-8 experts = 8 × 4.7 MB = 38 MB on VRAM at a time
  • Non-expert weights (attn, norm, shared expert, router): ~0.6 GB on VRAM

Usage:
    from engine.moe_offload import ExpertOffloadManager

    # One-time setup (at model load):
    mgr = ExpertOffloadManager.from_weights(weights, config, device="cuda")

    # Per-layer at inference (inside the forward loop):
    block_w = weights.layer(i)               # non-expert weights (on VRAM)
    block_w = mgr.get_layer_weights(i, block_w)   # adds active-expert tensors from CPU
    output  = moe_mlp(hidden, block_w, config)

    # Alternatively, two-step (for explicit control):
    expert_w = mgr.fetch(layer_idx=i, expert_ids=[3, 17, 42, ...])
    for eid, w in expert_w.items():
        block_w[f"mlp.experts.{eid}.gate_proj.weight"] = w["gate_proj.weight"]
        ...
"""

from __future__ import annotations

import torch

from engine.config import LlamaConfig


class ExpertOffloadManager:
    """CPU-pinned expert weight cache with synchronous VRAM fetch.

    All routed expert tensors live in pinned host memory (page-locked RAM).
    ``fetch()`` copies the requested experts to the target device and returns
    them; the caller is responsible for releasing references when done.

    Args:
        layer_experts: Nested dict: layer_idx → expert_id → {proj_key → CPU tensor}.
        device:        Target VRAM device string, e.g. ``"cuda"`` or ``"cuda:0"``.
    """

    def __init__(
        self,
        layer_experts: dict[int, dict[int, dict[str, torch.Tensor]]],
        device: str,
    ) -> None:
        self._experts = layer_experts
        self._device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        weights,            # LlamaWeights | QuantizedLlamaWeights
        config: LlamaConfig,
        device: str,
    ) -> "ExpertOffloadManager":
        """Extract every routed expert's weights into pinned CPU tensors.

        ``weights.layer(i)`` is called once per layer; each expert's three
        matrices are detached, moved to CPU, and (if CUDA is available) pinned.

        Note: the source tensors in ``weights`` are NOT modified or freed — the
        caller is expected to use this manager for inference rather than loading
        the full model to VRAM if they are memory-constrained.

        Args:
            weights: Loaded model weights (plain or quantized).
            config:  Model config for ``n_layer`` and ``n_experts``.
            device:  VRAM device for subsequent ``fetch()`` calls.

        Returns:
            Configured ``ExpertOffloadManager``.
        """
        use_pin = torch.cuda.is_available()
        layer_experts: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

        for layer_idx in range(config.n_layer):
            layer_dict = weights.layer(layer_idx)
            experts: dict[int, dict[str, torch.Tensor]] = {}
            for eid in range(config.n_experts):
                pfx = f"mlp.experts.{eid}."
                w = {
                    "gate_proj.weight": layer_dict[pfx + "gate_proj.weight"].detach().cpu(),
                    "up_proj.weight":   layer_dict[pfx + "up_proj.weight"].detach().cpu(),
                    "down_proj.weight": layer_dict[pfx + "down_proj.weight"].detach().cpu(),
                }
                if use_pin:
                    w = {k: v.pin_memory() for k, v in w.items()}
                experts[eid] = w
            layer_experts[layer_idx] = experts

        return cls(layer_experts, device)

    # ------------------------------------------------------------------
    # Fetch API
    # ------------------------------------------------------------------

    def fetch(
        self,
        layer_idx: int,
        expert_ids: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Synchronously copy selected experts from CPU to VRAM.

        Args:
            layer_idx:  Transformer block index (0-based).
            expert_ids: IDs of experts to bring to device.

        Returns:
            ``{expert_id: {"gate_proj.weight": ..., "up_proj.weight": ...,
            "down_proj.weight": ...}}`` with tensors on ``self.device``.
        """
        result: dict[int, dict[str, torch.Tensor]] = {}
        for eid in expert_ids:
            cpu_w = self._experts[layer_idx][eid]
            result[eid] = {
                k: v.to(self._device, non_blocking=False) for k, v in cpu_w.items()
            }
        return result

    def get_layer_weights(
        self,
        layer_idx: int,
        base_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Augment a block weight dict with all experts for ``layer_idx`` on VRAM.

        Use this when you want to stream one full layer's experts to VRAM at a
        time (e.g. during prefill or when doing layer-by-layer decoding).

        Args:
            layer_idx:    Transformer block index.
            base_weights: Non-expert block tensors already on VRAM (router,
                          shared expert, attention, norms).  Usually the result
                          of ``LlamaWeights.layer(layer_idx)`` minus expert tensors.

        Returns:
            Shallow copy of ``base_weights`` merged with all routed expert
            tensors fetched to VRAM.
        """
        all_ids = list(self._experts[layer_idx].keys())
        expert_w = self.fetch(layer_idx, all_ids)

        result = dict(base_weights)
        for eid, w in expert_w.items():
            for proj_key, tensor in w.items():
                result[f"mlp.experts.{eid}.{proj_key}"] = tensor
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Target VRAM device string."""
        return self._device

    @property
    def n_layers(self) -> int:
        """Number of MoE layers managed."""
        return len(self._experts)

    @property
    def n_experts(self) -> int:
        """Number of routed experts per layer."""
        if not self._experts:
            return 0
        return len(next(iter(self._experts.values())))
