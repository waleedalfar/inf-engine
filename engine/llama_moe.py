"""Qwen3-MoE MLP: router + top-K routed experts + always-active shared expert.

Architecture (per transformer block, replacing the dense SwiGLU):
  1. Router: linear d_model → n_experts, softmax → top-K selection per token
  2. Shared expert: one SwiGLU always computed for every token
  3. Routed experts: top-K SwiGLU experts, outputs weighted by softmax scores
  4. Return: shared_out + weighted_sum(routed_expert_outs)

Weight keys expected in the block weight dict (relative to the block prefix):
  mlp.gate.weight                        (n_experts, d_model)
  mlp.shared_expert.gate_proj.weight     (shared_inter, d_model)
  mlp.shared_expert.up_proj.weight       (shared_inter, d_model)
  mlp.shared_expert.down_proj.weight     (d_model, shared_inter)
  mlp.experts.{j}.gate_proj.weight       (moe_inter, d_model)   j ∈ 0..n_experts-1
  mlp.experts.{j}.up_proj.weight         (moe_inter, d_model)
  mlp.experts.{j}.down_proj.weight       (d_model, moe_inter)
"""

from __future__ import annotations

import torch

from engine.config import LlamaConfig
from engine.layers import linear, silu


def _expert_forward(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU forward for one expert: down(silu(gate(x)) * up(x)).

    Args:
        x:      Input tokens. Shape: (n_tokens, d_model)
        gate_w: Shape: (moe_inter, d_model)
        up_w:   Shape: (moe_inter, d_model)
        down_w: Shape: (d_model, moe_inter)

    Returns:
        Shape: (n_tokens, d_model)
    """
    return linear(silu(linear(x, gate_w)) * linear(x, up_w), down_w)


def moe_mlp(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: LlamaConfig,
) -> torch.Tensor:
    """Qwen3-MoE MLP forward: shared expert + top-K routed experts.

    Args:
        x:       RMSNorm'd residual. Shape: (B, T, d_model)
        weights: Block tensors from ``LlamaWeights.layer(i)``. Must contain the
                 router, shared expert, and all n_experts routed expert tensors.
        config:  Model config. ``config.is_moe`` must be True.

    Returns:
        MLP output (to add back to residual). Shape: (B, T, d_model)
    """
    B, T, d = x.shape
    x_flat = x.view(B * T, d)       # (S, d)  S = B*T

    # --- Shared expert (fires for every token) ---
    shared_out = _expert_forward(
        x_flat,
        weights["mlp.shared_expert.gate_proj.weight"],
        weights["mlp.shared_expert.up_proj.weight"],
        weights["mlp.shared_expert.down_proj.weight"],
    )                                # (S, d)

    # --- Router: compute top-K expert assignments ---
    router_logits = linear(x_flat, weights["mlp.gate.weight"])   # (S, n_experts)
    router_weights = torch.softmax(router_logits, dim=-1)        # (S, n_experts)

    topk_weights, topk_ids = torch.topk(
        router_weights, config.n_experts_per_tok, dim=-1
    )                                # topk_weights: (S, K),  topk_ids: (S, K) int64
    # Renormalise selected weights so they sum to 1 per token.
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # (S, K)

    # --- Dispatch routed experts ---
    routed_out = torch.zeros_like(x_flat)  # (S, d)

    for eid_t in topk_ids.unique():
        eid = int(eid_t.item())
        # Aggregate routing weight for this expert across all K slots.
        # Each token selects each expert at most once, so only one slot fires.
        token_weights = ((topk_ids == eid) * topk_weights).sum(dim=-1)  # (S,)
        active = token_weights > 0                                       # (S,) bool
        if not active.any():
            continue

        expert_out = _expert_forward(
            x_flat[active],
            weights[f"mlp.experts.{eid}.gate_proj.weight"],
            weights[f"mlp.experts.{eid}.up_proj.weight"],
            weights[f"mlp.experts.{eid}.down_proj.weight"],
        )                            # (n_active, d)
        routed_out[active] += expert_out * token_weights[active, None]

    return (shared_out + routed_out).view(B, T, d)
