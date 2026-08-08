"""Post-training INT4 quantization for LlamaModel (W4A16).

Wraps a loaded LlamaModel and replaces every linear projection weight with a
packed INT4 version.  At inference, weights are dequantized just-in-time
(one layer at a time) before the matmul — activations stay in bfloat16.

Memory at decode:
    7B  bf16 = 14 GB    →   7B  INT4 = 3.5 GB  (4× reduction)
    14B bf16 = 28 GB    →  14B  INT4 = 7.0 GB  (4× reduction)

Usage:
    model = LlamaModel(weights, config)           # load full-precision model
    q_model = quantize_llama(model, group_size=128)
    # q_model is a drop-in replacement: forward / generate_cached unchanged.

Design: ``QuantizedLlamaWeights.layer(i)`` dequantizes that layer's INT4
projections back to float on each call, returning an ordinary weight dict.
The attention / MLP / block functions see plain tensors and need no changes.
Peak extra memory = one layer's worth of weights in float (not all layers).

Layers quantized (7 per block):
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
NOT quantized: embed_tokens, lm_head, RMSNorm gains (small, precision-sensitive).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.config import LlamaConfig
from engine.kernels.quant import dequantize_weight_int4, quantize_weight_int4
from engine.llama_model import LlamaModel
from engine.llama_weights import LlamaWeights

_LINEAR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

# MoE expert projection names (relative to each expert prefix).
_EXPERT_PROJ_SUFFIXES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


@dataclass
class _Int4Weight:
    """Packed INT4 weight tensor with its dequantization metadata."""
    packed: torch.Tensor   # (d_out // 2, d_in) int8  [note: transposed vs Linear convention]
    scale: torch.Tensor    # (n_groups, d_in)  float
    group_size: int
    shape: tuple[int, int]  # original (d_out, d_in) — Linear layout

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the reconstructed weight in (d_out, d_in) Linear layout."""
        # packed was quantized as (d_in, d_out) column-major; transpose back.
        w_col = dequantize_weight_int4(self.packed, self.scale.to(dtype), self.group_size)
        return w_col.T.contiguous()  # → (d_out, d_in)


class QuantizedLlamaWeights:
    """LlamaWeights where linear projections are backed by packed INT4 storage.

    ``layer(i)`` dequantizes that block's projections on demand and returns a
    plain tensor dict — identical to what the unquantized ``LlamaWeights.layer``
    returns.  The attention / MLP / block functions require no changes.
    """

    def __init__(
        self,
        orig: LlamaWeights,
        int4: dict[str, _Int4Weight],
        dtype: torch.dtype,
    ) -> None:
        self._orig = orig
        self._int4 = int4           # key: "model.layers.{i}.{suffix}"
        self._dtype = dtype

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self._orig.embed_tokens

    @property
    def lm_head(self) -> torch.Tensor:
        return self._orig.lm_head

    @property
    def norm_weight(self) -> torch.Tensor:
        return self._orig.norm_weight

    @property
    def config(self) -> LlamaConfig:
        return self._orig.config

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        """Return a weight dict for block ``i``, dequantizing INT4 projections."""
        prefix = f"model.layers.{i}."
        result = self._orig.layer(i)          # start with all original tensors

        for suffix in _LINEAR_SUFFIXES:
            key = prefix + suffix
            if key in self._int4:
                result[suffix] = self._int4[key].dequantize(self._dtype)

        if self._orig.config.is_moe:
            n_exp = self._orig.config.n_experts
            # Dequantize shared expert projections.
            for proj in _EXPERT_PROJ_SUFFIXES:
                suffix = f"mlp.shared_expert.{proj}"
                key = prefix + suffix
                if key in self._int4:
                    result[suffix] = self._int4[key].dequantize(self._dtype)
            # Dequantize routed expert projections.
            for j in range(n_exp):
                for proj in _EXPERT_PROJ_SUFFIXES:
                    suffix = f"mlp.experts.{j}.{proj}"
                    key = prefix + suffix
                    if key in self._int4:
                        result[suffix] = self._int4[key].dequantize(self._dtype)

        return result

    def expert_weights(self, layer_idx: int, expert_id: int) -> dict[str, torch.Tensor]:
        """Return dequantized weight dict for routed expert ``expert_id`` in layer ``layer_idx``."""
        prefix = f"model.layers.{layer_idx}."
        result = self._orig.expert_weights(layer_idx, expert_id)
        for proj in _EXPERT_PROJ_SUFFIXES:
            key = prefix + f"mlp.experts.{expert_id}.{proj}"
            if key in self._int4:
                result[proj] = self._int4[key].dequantize(self._dtype)
        return result


def quantize_llama(
    model: LlamaModel,
    group_size: int = 128,
) -> LlamaModel:
    """Quantize all linear projections in ``model`` to INT4 W4A16.

    Args:
        model:      Loaded LlamaModel (any float dtype).
        group_size: Rows per quantization group (128 is standard).

    Returns:
        New ``LlamaModel`` backed by ``QuantizedLlamaWeights``.  The original
        model is unchanged.  generate_cached / forward work identically.
    """
    config = model.config
    orig: LlamaWeights = model.w
    dtype = orig.embed_tokens.dtype

    int4: dict[str, _Int4Weight] = {}
    original_bytes = 0
    quantized_bytes = 0

    for i in range(config.n_layer):
        prefix = f"model.layers.{i}."
        layer_dict = orig.layer(i)

        # Dense attention projections + dense MLP (for non-MoE) projections.
        for suffix in _LINEAR_SUFFIXES:
            if suffix not in layer_dict:
                continue
            w: torch.Tensor = layer_dict[suffix]   # (d_out, d_in) Linear layout
            # quantize_weight_int4 expects (d_in, d_out) column-major; transpose.
            w_col = w.T.contiguous()               # (d_in, d_out)
            packed, scale = quantize_weight_int4(w_col.float(), group_size)
            int4[prefix + suffix] = _Int4Weight(packed, scale, group_size, tuple(w.shape))
            original_bytes += w.numel() * w.element_size()
            quantized_bytes += packed.numel() * packed.element_size() + scale.numel() * scale.element_size()

        if config.is_moe:
            # Shared expert projections.
            for proj in _EXPERT_PROJ_SUFFIXES:
                suffix = f"mlp.shared_expert.{proj}"
                if suffix not in layer_dict:
                    continue
                w = layer_dict[suffix]
                w_col = w.T.contiguous()
                packed, scale = quantize_weight_int4(w_col.float(), group_size)
                int4[prefix + suffix] = _Int4Weight(packed, scale, group_size, tuple(w.shape))
                original_bytes += w.numel() * w.element_size()
                quantized_bytes += packed.numel() * packed.element_size() + scale.numel() * scale.element_size()

            # Routed expert projections.
            for j in range(config.n_experts):
                for proj in _EXPERT_PROJ_SUFFIXES:
                    suffix = f"mlp.experts.{j}.{proj}"
                    if suffix not in layer_dict:
                        continue
                    w = layer_dict[suffix]
                    w_col = w.T.contiguous()
                    packed, scale = quantize_weight_int4(w_col.float(), group_size)
                    int4[prefix + suffix] = _Int4Weight(packed, scale, group_size, tuple(w.shape))
                    original_bytes += w.numel() * w.element_size()
                    quantized_bytes += packed.numel() * packed.element_size() + scale.numel() * scale.element_size()

    reduction = original_bytes / max(quantized_bytes, 1)
    print(
        f"Quantized {len(int4)} tensors across {config.n_layer} layers  "
        f"({original_bytes / 1e9:.2f} GB → {quantized_bytes / 1e9:.2f} GB  "
        f"{reduction:.1f}× reduction)"
    )

    q_weights = QuantizedLlamaWeights(orig, int4, dtype)

    # Re-use LlamaModel with the quantized weight wrapper; RoPE tables transfer.
    q_model = LlamaModel.__new__(LlamaModel)
    q_model.w = q_weights
    q_model.config = config
    q_model.rope_cos = model.rope_cos
    q_model.rope_sin = model.rope_sin
    return q_model
