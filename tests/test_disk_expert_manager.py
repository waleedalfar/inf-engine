"""Tests for DiskExpertManager — disk-streaming expert loader for Qwen3-MoE.

All tests use synthetic mini weights written to a temporary safetensors file;
no real checkpoint is required.

Run:
    pytest tests/test_disk_expert_manager.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

DEVICE = "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_config(n_experts: int = 4, n_layers: int = 2):
    from engine.config import LlamaConfig
    return LlamaConfig(
        name="test-disk-moe",
        vocab_size=64,
        n_ctx=64,
        d_model=16,
        n_layer=n_layers,
        n_head=2,
        n_kv_heads=1,
        intermediate_size=0,
        n_experts=n_experts,
        n_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )


def _write_expert_shard(
    path: Path,
    config,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Write a fake safetensors file with all expert tensors; return the dict."""
    torch.manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    inter = config.moe_intermediate_size
    d = config.d_model
    for i in range(config.n_layer):
        for j in range(config.n_experts):
            p = f"model.layers.{i}.mlp.experts.{j}."
            tensors[p + "gate_proj.weight"] = torch.randn(inter, d)
            tensors[p + "up_proj.weight"]   = torch.randn(inter, d)
            tensors[p + "down_proj.weight"] = torch.randn(d, inter)
    save_file(tensors, str(path))
    return tensors


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def test_from_shards_builds_correct_index():
    """from_shards scans headers and indexes every expert tensor."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=4, n_layers=2)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)

        mgr = DiskExpertManager.from_shards([shard], config, device=DEVICE, dtype=DTYPE)

    assert mgr.n_layers == config.n_layer
    assert mgr.n_experts == config.n_experts


def test_from_shards_indexes_all_proj_keys():
    """Each expert entry has gate_proj, up_proj, and down_proj keys."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=3, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)
        mgr = DiskExpertManager.from_shards([shard], config, device=DEVICE, dtype=DTYPE)

    # Access internal index to verify proj keys
    for eid in range(config.n_experts):
        proj_keys = set(mgr._index[0][eid].keys())
        assert proj_keys == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}


def test_non_expert_keys_are_not_indexed():
    """DiskExpertManager only indexes expert tensors; non-expert keys are ignored."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=2, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        tensors = _write_expert_shard(shard, config)
        # Add a non-expert key to the shard
        tensors["model.norm.weight"] = torch.ones(config.d_model)
        save_file(tensors, str(shard))

        mgr = DiskExpertManager.from_shards([shard], config, device=DEVICE, dtype=DTYPE)

    # Non-expert key should not appear in the index
    for layer_experts in mgr._index.values():
        for expert_projs in layer_experts.values():
            for _, tensor_key in expert_projs.values():
                assert "model.norm.weight" not in tensor_key


# ---------------------------------------------------------------------------
# Fetch correctness
# ---------------------------------------------------------------------------

def test_fetch_returns_correct_tensors():
    """fetch() returns tensors matching the originals written to disk."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=4, n_layers=2)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        originals = _write_expert_shard(shard, config, seed=7)
        mgr = DiskExpertManager.from_shards([shard], config, device=DEVICE, dtype=DTYPE)

        result = mgr.fetch(layer_idx=0, expert_ids=[0, 2])

    assert set(result.keys()) == {0, 2}
    for eid in (0, 2):
        for proj in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            key = f"model.layers.0.mlp.experts.{eid}.{proj}"
            torch.testing.assert_close(result[eid][proj], originals[key].to(DTYPE))


def test_fetch_tensors_on_correct_device():
    """Fetched tensors land on the device specified at construction."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=2, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)
        mgr = DiskExpertManager.from_shards([shard], config, device="cpu", dtype=DTYPE)
        result = mgr.fetch(layer_idx=0, expert_ids=[1])

    for proj, t in result[1].items():
        assert t.device.type == "cpu", f"{proj} not on cpu"


def test_fetch_respects_dtype():
    """Tensors are cast to the manager's dtype on load."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=2, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)  # saved as float32

        mgr = DiskExpertManager.from_shards(
            [shard], config, device="cpu", dtype=torch.float16
        )
        result = mgr.fetch(layer_idx=0, expert_ids=[0])

    for t in result[0].values():
        assert t.dtype == torch.float16


def test_fetch_subset_of_experts():
    """fetch() with a subset of expert IDs only returns those experts."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=8, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)
        mgr = DiskExpertManager.from_shards([shard], config, device="cpu", dtype=DTYPE)
        result = mgr.fetch(layer_idx=0, expert_ids=[3, 5, 7])

    assert set(result.keys()) == {3, 5, 7}


# ---------------------------------------------------------------------------
# Async prefetch
# ---------------------------------------------------------------------------

def test_prefetch_async_result_matches_sync_fetch():
    """prefetch_async().result() returns the same tensors as a sync fetch."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=4, n_layers=2)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config, seed=42)
        mgr = DiskExpertManager.from_shards([shard], config, device="cpu", dtype=DTYPE)

        sync_result  = mgr.fetch(layer_idx=1, expert_ids=[0, 3])
        async_result = mgr.prefetch_async(layer_idx=1, expert_ids=[0, 3]).result()

    for eid in (0, 3):
        for proj in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            torch.testing.assert_close(sync_result[eid][proj], async_result[eid][proj])


def test_prefetch_async_is_non_blocking():
    """prefetch_async returns a Future immediately without blocking the caller."""
    import time
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=2, n_layers=1)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)
        mgr = DiskExpertManager.from_shards([shard], config, device="cpu", dtype=DTYPE)

        t0 = time.perf_counter()
        future = mgr.prefetch_async(layer_idx=0, expert_ids=[0])
        elapsed = time.perf_counter() - t0

        # The call itself should return in well under 1 ms
        assert elapsed < 0.1, f"prefetch_async blocked for {elapsed:.3f}s"
        _ = future.result()  # clean up


# ---------------------------------------------------------------------------
# Multi-shard index
# ---------------------------------------------------------------------------

def test_from_shards_spans_multiple_files():
    """Expert tensors split across two shard files are both indexed correctly."""
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=4, n_layers=2)
    inter, d = config.moe_intermediate_size, config.d_model

    with tempfile.TemporaryDirectory() as tmp:
        shard0 = Path(tmp) / "shard-00001.safetensors"
        shard1 = Path(tmp) / "shard-00002.safetensors"

        # Put layer 0 experts in shard0, layer 1 experts in shard1
        t0, t1 = {}, {}
        for j in range(config.n_experts):
            p0 = f"model.layers.0.mlp.experts.{j}."
            t0[p0 + "gate_proj.weight"] = torch.randn(inter, d)
            t0[p0 + "up_proj.weight"]   = torch.randn(inter, d)
            t0[p0 + "down_proj.weight"] = torch.randn(d, inter)

            p1 = f"model.layers.1.mlp.experts.{j}."
            t1[p1 + "gate_proj.weight"] = torch.randn(inter, d)
            t1[p1 + "up_proj.weight"]   = torch.randn(inter, d)
            t1[p1 + "down_proj.weight"] = torch.randn(d, inter)

        save_file(t0, str(shard0))
        save_file(t1, str(shard1))

        mgr = DiskExpertManager.from_shards(
            [shard0, shard1], config, device="cpu", dtype=DTYPE
        )

        r0 = mgr.fetch(layer_idx=0, expert_ids=[1])
        r1 = mgr.fetch(layer_idx=1, expert_ids=[2])

    assert "gate_proj.weight" in r0[1]
    assert "gate_proj.weight" in r1[2]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_properties():
    from engine.disk_expert_manager import DiskExpertManager

    config = _mini_config(n_experts=6, n_layers=3)
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "model.safetensors"
        _write_expert_shard(shard, config)
        mgr = DiskExpertManager.from_shards([shard], config, device="cpu", dtype=DTYPE)

    assert mgr.device == "cpu"
    assert mgr.n_layers == 3
    assert mgr.n_experts == 6
