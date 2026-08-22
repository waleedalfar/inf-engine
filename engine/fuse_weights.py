"""Fuse per-layer projections that share an input into single wide matmuls.

Q, K and V all read the same RMSNorm'd residual, as do gate and up. Keeping them
as separate weight tensors means separate matmuls over identical activations —
three launches instead of one, and, worse, three *narrow* ones.

Narrow matters more than the launch count. Every INT4 matmul is bound by
streaming its weights from HBM, and the kernel's block grid is ``N / BLOCK_N``
wide. Qwen3-8B's K and V projections are only N=1024, so they fill 32 blocks
against 70 SMs and measured **8% of peak memory bandwidth** — the GPU is mostly
idle waiting on them. Concatenated with Q into one N=6144 projection, the same
bytes move under a grid that actually fills the machine.

The fusion is a pure concatenation along the output dimension: ``qkv(x)`` is
``cat([q(x), k(x), v(x)], dim=-1)``, so callers split the result back apart by
width. Bit-exact, not an approximation.

Call ``fuse_projections(model)`` after loading (and after quantizing, if
quantizing). Attention and the MLP pick up the fused weights automatically when
present and fall back to the separate ones when not.
"""

from __future__ import annotations

import torch

# Fused key -> (source keys, in concatenation order). Order is load-bearing:
# the consumers split the output by width in exactly this sequence.
FUSED_GROUPS: dict[str, tuple[str, ...]] = {
    "self_attn.qkv_proj.weight": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ),
    "mlp.gate_up_proj.weight": (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ),
}


def _cat_int4(parts: list) -> object:
    """Concatenate ``_Int4Weight``s along their output dimension.

    ``packed`` is (d_in//2, d_out) and ``scale`` is (d_in//group, d_out) — the
    output dimension is axis 1 in both, and every part shares d_in and the same
    grouping along it, so this is a plain cat with no requantization.
    """
    from engine.quantize import _Int4Weight

    first = parts[0]
    assert all(p.group_size == first.group_size for p in parts), \
        "cannot fuse INT4 weights quantized with different group sizes"
    assert all(p.shape[1] == first.shape[1] for p in parts), \
        "cannot fuse INT4 weights with different input widths"
    return _Int4Weight(
        packed=torch.cat([p.packed for p in parts], dim=1).contiguous(),
        scale=torch.cat([p.scale for p in parts], dim=1).contiguous(),
        group_size=first.group_size,
        shape=(sum(p.shape[0] for p in parts), first.shape[1]),
    )


def _cat_dense(parts: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate dense (d_out, d_in) Linear-layout weights along d_out."""
    return torch.cat(parts, dim=0).contiguous()


def _fuse_into(store: dict, prefix: str, fused_key: str, sources: tuple[str, ...]) -> bool:
    """Replace ``sources`` in ``store`` with one fused ``fused_key`` entry."""
    keys = [prefix + s for s in sources]
    if not all(k in store for k in keys):
        return False
    parts = [store[k] for k in keys]
    is_int4 = hasattr(parts[0], "fused_linear")
    store[prefix + fused_key] = _cat_int4(parts) if is_int4 else _cat_dense(parts)
    for k in keys:
        del store[k]
    return True


def fuse_projections(model):
    """Fuse Q/K/V and gate/up in every layer of ``model``, in place.

    Works on both dense ``LlamaWeights`` and INT4 ``QuantizedLlamaWeights``. For
    a quantized model the INT4 tensors are fused and the dense originals (kept
    only as a fallback) are dropped alongside them, so the two stay consistent.

    Skipped for MoE models: their experts each hold their own gate/up pair, and
    the routed-expert path loads weights per expert rather than per layer.

    Returns ``model`` for chaining.
    """
    config = model.config
    if config.is_moe:
        return model

    w = model.w
    int4 = getattr(w, "_int4", None)
    dense = w._orig._t if int4 is not None else w._t

    for i in range(config.n_layer):
        prefix = f"model.layers.{i}."
        for fused_key, sources in FUSED_GROUPS.items():
            if int4 is not None and _fuse_into(int4, prefix, fused_key, sources):
                # Drop the dense originals too — layer() merges both dicts, and
                # a stale q_proj there would shadow nothing but waste memory.
                for s in sources:
                    dense.pop(prefix + s, None)
            else:
                _fuse_into(dense, prefix, fused_key, sources)

    return model
