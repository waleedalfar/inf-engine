"""Load LLaMA weights from HuggingFace safetensors (single-file or sharded).

Key differences from GPT-2 (engine/weights.py):
  - Separate q_proj / k_proj / v_proj / o_proj instead of fused c_attn.
  - Weights use nn.Linear layout (d_out, d_in) — multiply with W.T, not W.
  - No bias tensors on attention or MLP projections.
  - RMSNorm has weight only (no bias).
  - Sharded models (8B+) store tensors across multiple safetensors files;
    we load all shards and merge into one dict before wrapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import LlamaConfig


class LlamaWeights:
    """Named LLaMA tensors loaded from safetensors.

    All attention/MLP weights are stored as (d_out, d_in) matching nn.Linear.
    Call ``engine.layers.linear(x, w)`` (i.e. ``x @ w.T``) to use them.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], config: LlamaConfig) -> None:
        self._t = tensors
        self.config = config

    @property
    def embed_tokens(self) -> torch.Tensor:
        """Token embedding table. Shape: (vocab_size, d_model)."""
        return self._t["model.embed_tokens.weight"]

    @property
    def lm_head(self) -> torch.Tensor:
        """Unembedding matrix. Shape: (vocab_size, d_model).

        Returns embed_tokens when tie_word_embeddings=True (LLaMA 3.2 1B/3B).
        """
        if self.config.tie_word_embeddings or "lm_head.weight" not in self._t:
            return self._t["model.embed_tokens.weight"]
        return self._t["lm_head.weight"]

    @property
    def norm_weight(self) -> torch.Tensor:
        """Final RMSNorm gain. Shape: (d_model,)."""
        return self._t["model.norm.weight"]

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        """Return all tensors for transformer block ``i``, keys relative to the block.

        Dense keys (shapes — d=d_model, h=n_kv_heads*head_dim, f=intermediate_size):
            input_layernorm.weight              (d,)
            post_attention_layernorm.weight     (d,)
            self_attn.q_proj.weight             (d, d)
            self_attn.k_proj.weight             (h, d)
            self_attn.v_proj.weight             (h, d)
            self_attn.o_proj.weight             (d, d)
            mlp.gate_proj.weight                (f, d)
            mlp.up_proj.weight                  (f, d)
            mlp.down_proj.weight                (d, f)

        Additional MoE keys (when config.is_moe):
            mlp.gate.weight                         (n_experts, d)
            mlp.shared_expert.{gate,up}_proj.weight (shared_inter, d)
            mlp.shared_expert.down_proj.weight      (d, shared_inter)
            mlp.experts.{j}.{gate,up}_proj.weight   (moe_inter, d)   j=0..n_experts-1
            mlp.experts.{j}.down_proj.weight        (d, moe_inter)
        """
        prefix = f"model.layers.{i}."
        return {k[len(prefix):]: v for k, v in self._t.items() if k.startswith(prefix)}

    def expert_weights(self, layer_idx: int, expert_id: int) -> dict[str, torch.Tensor]:
        """Return the three weight tensors for routed expert ``expert_id`` in layer ``layer_idx``.

        Keys: ``'gate_proj.weight'``, ``'up_proj.weight'``, ``'down_proj.weight'``
        """
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
        return {k[len(prefix):]: v for k, v in self._t.items() if k.startswith(prefix)}


def _load_tensors(
    safetensors_path: Path,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            tensors[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    return tensors


def load_llama_weights(
    model_dir: str | Path,
    config: LlamaConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaWeights:
    """Load all LLaMA tensors from ``model_dir`` into ``LlamaWeights``.

    Handles both single-file (``model.safetensors``) and sharded
    (``model-00001-of-00002.safetensors`` + ``model.safetensors.index.json``)
    layouts automatically.

    Args:
        model_dir: Directory containing the safetensors file(s).
        config:    Model config the weights must match.
        device:    Target device for tensors.
        dtype:     Target dtype. bfloat16 is the standard inference dtype for
                   LLaMA 3; use float32 only for the correctness gate.

    Returns:
        ``LlamaWeights`` holding every tensor on ``device`` with ``dtype``.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"

    tensors: dict[str, torch.Tensor] = {}

    if single.is_file():
        tensors = _load_tensors(single, device, dtype)
    elif index.is_file():
        with open(index) as f:
            shard_map: dict[str, str] = json.load(f)["weight_map"]
        shards = sorted(set(shard_map.values()))
        for shard_name in shards:
            shard_path = model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"shard not found: {shard_path}")
            tensors.update(_load_tensors(shard_path, device, dtype))
    else:
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json found in {model_dir}"
        )

    required = ["model.embed_tokens.weight", "model.norm.weight"]
    for key in required:
        if key not in tensors:
            raise KeyError(f"missing tensor '{key}' in {model_dir}")

    if config.qk_norm:
        for i in range(config.n_layer):
            for proj in ("q_norm", "k_norm"):
                key = f"model.layers.{i}.self_attn.{proj}.weight"
                if key not in tensors:
                    raise KeyError(
                        f"config has qk_norm=True but '{key}' not found in {model_dir}. "
                        "Ensure you are loading a Qwen3 checkpoint."
                    )

    return LlamaWeights(tensors, config)
