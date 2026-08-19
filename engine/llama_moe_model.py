"""Qwen3-MoE inference with CPU-offloaded routed experts.

Memory layout at inference time (Qwen3-30B-A3B on 16 GB GPU):
  VRAM  ~1.4 GB  non-expert weights: attention, norms, shared expert, router
  VRAM  ~0.2 GB  KV cache (4 096 tokens, bf16)
  RAM  ~57 GB    all routed expert weights in pinned CPU memory

Per decode step, per layer:
  1. Router runs on VRAM (mlp.gate.weight already there).
  2. Only the top-K selected experts (~38 MB) are copied CPU → VRAM.
  3. Expert forward runs, then the tensors go out of scope — VRAM freed.

Usage::

    from engine.llama_moe_model import load_moe_weights
    from engine.config import QWEN3_30B_A3B

    model = load_moe_weights("weights/Qwen--Qwen3-30B-A3B", QWEN3_30B_A3B)
    # model.forward() has the same signature as LlamaModel.forward()
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import LlamaConfig
from engine.disk_expert_manager import DiskExpertManager
from engine.kv_cache import LlamaStaticKVCache
from engine.layers import precompute_rope_freqs, rms_norm
from engine.llama_attention import llama_attention
from engine.llama_weights import LlamaWeights
from engine.moe_offload import ExpertOffloadManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_expert(key: str) -> bool:
    return ".mlp.experts." in key


def _parse_expert_key(key: str) -> tuple[int, int, str]:
    """Parse 'model.layers.{i}.mlp.experts.{j}.{proj}.weight' → (layer, eid, proj_key)."""
    parts = key.split(".")
    # model . layers . {i} . mlp . experts . {j} . {proj} . weight
    #   0       1       2     3       4        5      6        7
    return int(parts[2]), int(parts[5]), f"{parts[6]}.weight"


def _swiglu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    return (torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T


# ---------------------------------------------------------------------------
# Offload-aware MoE block
# ---------------------------------------------------------------------------

def _moe_block_offload(
    x: torch.Tensor,
    base_w: dict[str, torch.Tensor],
    config: LlamaConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    offload_mgr: ExpertOffloadManager,
    layer_idx: int,
    cache: LlamaStaticKVCache | None,
    start_pos: int,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """One MoE transformer block: attention (VRAM only) + offload-aware MoE MLP.

    Runs the router on the gate weight that lives on VRAM, then fetches only
    the top-K selected experts from CPU — not all 128.
    """
    eps = config.norm_eps
    B, T, d = x.shape

    # Attention — all weights already on VRAM
    h = rms_norm(x, base_w["input_layernorm.weight"], eps)
    x = x + llama_attention(
        h, base_w, config, cos, sin, position_ids,
        cache, layer_idx, start_pos, attn_mask,
    )

    # MoE MLP
    h = rms_norm(x, base_w["post_attention_layernorm.weight"], eps)
    h_flat = h.view(B * T, d)                                          # (S, d)

    # Step 1 — router (gate.weight is on VRAM, no expert fetch yet)
    router_probs = torch.softmax(
        h_flat @ base_w["mlp.gate.weight"].T, dim=-1                   # (S, n_experts)
    )
    topk_w, topk_ids = torch.topk(router_probs, config.n_experts_per_tok, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)                # renormalise

    # Step 2 — kick off async expert fetch (disk: triggers I/O; RAM: near-instant)
    active_ids: list[int] = topk_ids.unique().tolist()
    expert_future = offload_mgr.prefetch_async(layer_idx, active_ids)

    # Step 3 — shared expert (always fires, weights on VRAM)
    # Runs concurrently with disk I/O when using DiskExpertManager.
    shared_out = _swiglu(
        h_flat,
        base_w["mlp.shared_expert.gate_proj.weight"],
        base_w["mlp.shared_expert.up_proj.weight"],
        base_w["mlp.shared_expert.down_proj.weight"],
    )

    expert_tensors = expert_future.result()  # wait — usually already done

    # Step 4 — routed experts (only active ones)
    routed_out = torch.zeros_like(h_flat)
    for eid in active_ids:
        w = expert_tensors[eid]
        tok_w = ((topk_ids == eid) * topk_w).sum(dim=-1)              # (S,)
        mask = tok_w > 0
        if not mask.any():
            continue
        expert_out = _swiglu(
            h_flat[mask],
            w["gate_proj.weight"],
            w["up_proj.weight"],
            w["down_proj.weight"],
        )
        routed_out[mask] += expert_out * tok_w[mask, None]

    return x + (shared_out + routed_out).view(B, T, d)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class LlamaMoEOffloadModel:
    """Qwen3-MoE language model: non-expert weights on VRAM, experts in pinned RAM.

    Drop-in replacement for :class:`engine.llama_model.LlamaModel` — same
    ``forward(input_ids, cache, start_pos, position_ids, attn_mask)`` signature,
    compatible with :class:`engine.agent.AgentLoop` without any changes.
    """

    def __init__(
        self,
        vram_weights: LlamaWeights,
        offload_mgr: ExpertOffloadManager,
        config: LlamaConfig,
    ) -> None:
        self.w = vram_weights
        self.offload_mgr = offload_mgr
        self.config = config

        device = str(vram_weights.embed_tokens.device)
        cos, sin = precompute_rope_freqs(
            config.head_dim, config.n_ctx, config.rope_theta, device
        )
        self.rope_cos = cos
        self.rope_sin = sin

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: LlamaStaticKVCache | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q = input_ids.shape
        if start_pos + T_q > self.config.n_ctx:
            raise ValueError(
                f"position {start_pos + T_q} exceeds n_ctx={self.config.n_ctx}"
            )

        x = self.w.embed_tokens[input_ids]

        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + T_q, device=input_ids.device
            )

        cos = self.rope_cos.to(device=input_ids.device, dtype=x.dtype)
        sin = self.rope_sin.to(device=input_ids.device, dtype=x.dtype)

        for i in range(self.config.n_layer):
            base_w = self.w.layer(i)          # non-expert tensors from VRAM
            x = _moe_block_offload(
                x, base_w, self.config, cos, sin, position_ids,
                self.offload_mgr, i, cache, start_pos, attn_mask,
            )

        x = rms_norm(x, self.w.norm_weight, self.config.norm_eps)
        return x @ self.w.lm_head.T


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

def load_moe_weights(
    model_dir: str | Path,
    config: LlamaConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaMoEOffloadModel:
    """Load a Qwen3-MoE model, routing tensors to VRAM or pinned CPU as appropriate.

    Non-expert tensors go directly to ``device``.
    Routed expert tensors go to pinned CPU memory (never touch VRAM during load).

    Args:
        model_dir: Directory with safetensors shards + model.safetensors.index.json.
        config:    Must be a MoE config (``config.is_moe == True``).
        device:    VRAM device string (default ``"cuda"``).
        dtype:     Weight dtype (default ``bfloat16``).

    Returns:
        :class:`LlamaMoEOffloadModel` ready for inference.

    RAM requirements (Qwen3-30B-A3B):
        ~57 GB pinned CPU memory for routed experts.
        ~1.4 GB VRAM for non-expert weights.
    """
    if not config.is_moe:
        raise ValueError(
            f"load_moe_weights requires a MoE config (n_experts > 0); "
            f"got config '{config.name}' with n_experts={config.n_experts}."
        )

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    single = model_dir / "model.safetensors"
    index  = model_dir / "model.safetensors.index.json"

    if single.is_file():
        shard_paths = [single]
    elif index.is_file():
        with open(index) as f:
            shard_map: dict[str, str] = json.load(f)["weight_map"]
        shard_paths = [model_dir / s for s in sorted(set(shard_map.values()))]
    else:
        raise FileNotFoundError(
            f"No safetensors found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )

    pin = torch.cuda.is_available()
    vram_tensors: dict[str, torch.Tensor] = {}
    cpu_experts: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

    n_shards = len(shard_paths)
    for idx, shard_path in enumerate(shard_paths, 1):
        print(f"  [{idx}/{n_shards}] {shard_path.name}", flush=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                t = f.get_tensor(key).to(dtype=dtype)
                if _is_expert(key):
                    layer_idx, eid, proj = _parse_expert_key(key)
                    cpu_t = t.pin_memory() if pin else t
                    cpu_experts.setdefault(layer_idx, {}).setdefault(eid, {})[proj] = cpu_t
                else:
                    vram_tensors[key] = t.to(device)

    vram_mb   = sum(t.numel() * t.element_size() for t in vram_tensors.values()) / 1e6
    expert_gb = sum(
        t.numel() * t.element_size()
        for layer in cpu_experts.values()
        for exp in layer.values()
        for t in exp.values()
    ) / 1e9
    print(f"  VRAM (non-expert): {vram_mb:.0f} MB  |  CPU pinned (experts): {expert_gb:.1f} GB")

    weights    = LlamaWeights(vram_tensors, config)
    offload_mgr = ExpertOffloadManager(cpu_experts, device)
    return LlamaMoEOffloadModel(weights, offload_mgr, config)


def load_moe_weights_disk(
    model_dir: str | Path,
    config: LlamaConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaMoEOffloadModel:
    """Load a Qwen3-MoE model with expert weights streamed from disk.

    Non-expert weights (attention, norms, shared expert, router) go to VRAM.
    Routed expert weights are NOT loaded — only a disk index is built.
    At inference, only the active top-K experts are read per layer.

    Memory requirements (Qwen3-30B-A3B):
        VRAM:  ~1.4 GB  (non-expert weights + KV cache)
        RAM:   ~38 MB   peak per token (one layer's top-8 experts in flight)
        Disk:  ~57 GB   expert weights remain on disk throughout

    Args:
        model_dir: Directory with safetensors shards + index.json.
        config:    Must be a MoE config (``config.is_moe == True``).
        device:    VRAM device string (default ``"cuda"``).
        dtype:     Weight dtype (default ``bfloat16``).

    Returns:
        :class:`LlamaMoEOffloadModel` using a :class:`DiskExpertManager`.
    """
    if not config.is_moe:
        raise ValueError(
            f"load_moe_weights_disk requires a MoE config; "
            f"got '{config.name}' with n_experts={config.n_experts}."
        )

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    single = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"

    if single.is_file():
        shard_paths = [single]
    elif index_file.is_file():
        with open(index_file) as f:
            shard_map: dict[str, str] = json.load(f)["weight_map"]
        shard_paths = [model_dir / s for s in sorted(set(shard_map.values()))]
    else:
        raise FileNotFoundError(
            f"No safetensors found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )

    vram_tensors: dict[str, torch.Tensor] = {}
    n_shards = len(shard_paths)
    print(f"Loading non-expert weights to VRAM ({n_shards} shards) ...")
    for idx, shard_path in enumerate(shard_paths, 1):
        print(f"  [{idx}/{n_shards}] {shard_path.name}", flush=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if not _is_expert(key):
                    vram_tensors[key] = f.get_tensor(key).to(dtype=dtype).to(device)

    vram_mb = sum(t.numel() * t.element_size() for t in vram_tensors.values()) / 1e6
    print(f"  VRAM (non-expert): {vram_mb:.0f} MB  |  experts: on disk")

    print("Building disk expert index ...")
    disk_mgr = DiskExpertManager.from_shards(shard_paths, config, device, dtype)

    weights = LlamaWeights(vram_tensors, config)
    return LlamaMoEOffloadModel(weights, disk_mgr, config)
