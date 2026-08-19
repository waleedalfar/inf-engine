"""Inspect a HuggingFace model directory and print the dimensions needed
to configure the engine.

Usage:
    python scripts/inspect_model.py weights/Qwen--Qwen3-8B
    python scripts/inspect_model.py weights/Qwen--Qwen3-30B-A3B

Output:
    - Key fields from config.json
    - Actual tensor shapes sampled from the safetensors shards
    - All unique structural key patterns (useful for discovering new architectures)
    - A ready-to-paste LlamaConfig snippet
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open


def _load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    return json.load(open(cfg_path))


def _shard_paths(model_dir: Path) -> list[Path]:
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        wmap = json.load(open(index))["weight_map"]
        return [model_dir / s for s in sorted(set(wmap.values()))]
    return []


def _unique_patterns(model_dir: Path) -> list[str]:
    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index.is_file():
        keys = json.load(open(index))["weight_map"].keys()
    elif single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    else:
        return []

    pats: set[str] = set()
    for k in keys:
        p = re.sub(r"\.layers\.\d+", ".layers.N", k)
        p = re.sub(r"\.experts\.\d+", ".experts.E", p)
        pats.add(p)
    return sorted(pats)


def _sample_shapes(model_dir: Path, targets: set[str]) -> dict[str, tuple]:
    shards = _shard_paths(model_dir)
    found: dict[str, tuple] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in targets:
                    found[k] = tuple(f.get_slice(k).get_shape())
        if len(found) >= len(targets):
            break
    return found


def inspect(model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: directory not found: {model_dir}")
        sys.exit(1)

    cfg = _load_config(model_dir)
    patterns = _unique_patterns(model_dir)
    is_moe = any(".experts.E." in p for p in patterns)
    has_shared = any("shared_expert" in p for p in patterns)
    has_qk_norm = any("q_norm" in p for p in patterns)

    # ------------------------------------------------------------------ config.json
    print("=" * 60)
    print(f"Model: {model_dir.name}")
    print("=" * 60)

    hf_keys = [
        ("hidden_size",                    "d_model"),
        ("num_hidden_layers",              "n_layer"),
        ("num_attention_heads",            "n_head"),
        ("num_key_value_heads",            "n_kv_heads"),
        ("head_dim",                       "head_dim"),
        ("intermediate_size",              "intermediate_size"),
        ("vocab_size",                     "vocab_size"),
        ("max_position_embeddings",        "n_ctx"),
        ("rope_theta",                     "rope_theta"),
        ("rms_norm_eps",                   "norm_eps"),
        ("tie_word_embeddings",            "tie_word_embeddings"),
        ("num_experts",                    "n_experts"),
        ("num_experts_per_tok",            "n_experts_per_tok"),
        ("moe_intermediate_size",          "moe_intermediate_size"),
        ("shared_expert_intermediate_size","shared_expert_intermediate_size"),
    ]
    print("\nconfig.json fields:")
    for hf_key, engine_key in hf_keys:
        if hf_key in cfg:
            print(f"  {hf_key:<40s} {cfg[hf_key]}  →  {engine_key}")

    # ------------------------------------------------------------------ shapes
    layer0_targets = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    if is_moe:
        layer0_targets.add("model.layers.0.mlp.experts.0.gate_proj.weight")
        if has_shared:
            layer0_targets.add("model.layers.0.mlp.shared_expert.gate_proj.weight")
    else:
        layer0_targets.update({
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        })

    shapes = _sample_shapes(model_dir, layer0_targets)
    print("\nKey tensor shapes (layer 0):")
    for k, v in sorted(shapes.items()):
        label = k.replace("model.layers.0.", "")
        print(f"  {label:<55s} {v}")

    # ------------------------------------------------------------------ derived dims
    print("\nDerived dimensions:")
    q_shape = shapes.get("model.layers.0.self_attn.q_proj.weight")
    k_shape = shapes.get("model.layers.0.self_attn.k_proj.weight")
    d_model  = cfg.get("hidden_size")
    if q_shape and d_model:
        q_out = q_shape[0]
        head_dim_cfg = cfg.get("head_dim")
        if head_dim_cfg:
            n_head = q_out // head_dim_cfg
            print(f"  n_head  = q_proj_out({q_out}) / head_dim({head_dim_cfg}) = {n_head}")
        else:
            print(f"  q_proj output dim = {q_out}  (need head_dim to derive n_head)")
    if k_shape and d_model:
        k_out = k_shape[0]
        head_dim_cfg = cfg.get("head_dim")
        if head_dim_cfg:
            n_kv = k_out // head_dim_cfg
            print(f"  n_kv_heads = k_proj_out({k_out}) / head_dim({head_dim_cfg}) = {n_kv}")

    # ------------------------------------------------------------------ architecture flags
    print("\nArchitecture flags:")
    print(f"  is_moe          = {is_moe}")
    print(f"  has_shared_expert = {has_shared}")
    print(f"  has_qk_norm     = {has_qk_norm}")

    # ------------------------------------------------------------------ key patterns
    print("\nUnique structural key patterns:")
    for p in patterns:
        print(f"  {p}")

    # ------------------------------------------------------------------ suggested config
    d = cfg.get("hidden_size", "???")
    nl = cfg.get("num_hidden_layers", "???")
    nh = "???"
    nkv = "???"
    hd = cfg.get("head_dim", 0)
    if q_shape and hd:
        nh = q_shape[0] // hd
    if k_shape and hd:
        nkv = k_shape[0] // hd
    vs = cfg.get("vocab_size", "???")
    nc = cfg.get("max_position_embeddings", 32768)
    rt = cfg.get("rope_theta", 500_000.0)
    ne = cfg.get("rms_norm_eps", cfg.get("norm_eps", 1e-5))
    inter = cfg.get("intermediate_size", 0)
    n_exp = cfg.get("num_experts", 0)
    n_per = cfg.get("num_experts_per_tok", 0)
    moe_i = cfg.get("moe_intermediate_size", 0)
    shared_i = cfg.get("shared_expert_intermediate_size", 0)

    print("\n--- Suggested LlamaConfig ---")
    print(f"LlamaConfig(")
    print(f'    name="{cfg.get("_name_or_path", model_dir.name)}",')
    print(f"    vocab_size={vs},")
    print(f"    n_ctx={nc},")
    print(f"    d_model={d},")
    print(f"    n_layer={nl},")
    print(f"    n_head={nh},")
    print(f"    n_kv_heads={nkv},")
    print(f"    intermediate_size={inter},")
    print(f"    rope_theta={rt},")
    print(f"    norm_eps={ne},")
    if has_qk_norm:
        print(f"    qk_norm=True,")
    if hd and hd != d // (nh if isinstance(nh, int) and nh > 0 else 1):
        print(f"    head_dim_override={hd},")
    if is_moe:
        print(f"    n_experts={n_exp},")
        print(f"    n_experts_per_tok={n_per},")
        print(f"    moe_intermediate_size={moe_i},")
        if has_shared:
            print(f"    shared_expert_intermediate_size={shared_i},")
        else:
            print(f"    shared_expert_intermediate_size=0,  # no shared expert")
    print(f")")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_model.py <model_dir>")
        sys.exit(1)
    inspect(sys.argv[1])
