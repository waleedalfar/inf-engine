"""SwiGLU feed-forward block used by LLaMA.

GPT-2 MLP:  x → c_fc (GELU) → c_proj    (2 weight matrices)
LLaMA MLP:  x → gate_proj → SiLU ⊙ up_proj → down_proj  (3 weight matrices)

The gate controls how much of the up-projected signal passes through; the
element-wise product of SiLU(gate) and up is the gated linear unit (GLU)
variant that gives SwiGLU its name.  intermediate_size is the hidden width
(set per-model in LlamaConfig; not necessarily 4×d_model).
"""

from __future__ import annotations

import torch

from engine.layers import linear, silu


def swiglu_mlp(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """SwiGLU MLP: down_proj( silu(gate_proj(x)) * up_proj(x) ).

    Args:
        x:       RMSNorm'd residual. Shape: (B, T, d_model)
        weights: Block tensors from ``LlamaWeights.layer(i)``
                 (mlp.gate_proj, mlp.up_proj, mlp.down_proj).

    Returns:
        MLP output to add back to the residual. Shape: (B, T, d_model)
    """
    gate = linear(x, weights["mlp.gate_proj.weight"])              # (B, T, intermediate_size)
    up   = linear(x, weights["mlp.up_proj.weight"])                # (B, T, intermediate_size)
    h    = silu(gate) * up                                         # (B, T, intermediate_size)
    return linear(h, weights["mlp.down_proj.weight"])              # (B, T, d_model)
