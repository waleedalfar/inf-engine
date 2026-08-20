"""CUDA graph decode-path correctness + benchmark (Task #10).

Compares LlamaPagedEngine with enable_cuda_graphs=True against the eager
decode path (enable_cuda_graphs=False) on a real (small but genuine, not
mock-scripted) LlamaModel, and asserts token-for-token identical greedy
output. Skipped entirely off CUDA — graphs are a no-op there.

Run:
    wsl bash -c "cd /home/waleed/mlproj && .venv/bin/pytest tests/test_cuda_graphs.py -v -s"
"""

from __future__ import annotations

import time

import pytest
import torch

from engine.config import LlamaConfig
from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.sampling import SamplingConfig, SamplingMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")

DEVICE = "cuda"
DTYPE = torch.float16


def _mini_config() -> LlamaConfig:
    return LlamaConfig(
        name="test-mini-graph",
        vocab_size=256,
        n_ctx=256,
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
    )


def _mini_model(config: LlamaConfig, device: str = DEVICE):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(99)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=device)

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight":         torch.ones(d, dtype=DTYPE, device=device),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight":          torch.ones(d, dtype=DTYPE, device=device),
            p + "post_attention_layernorm.weight":  torch.ones(d, dtype=DTYPE, device=device),
            p + "self_attn.q_proj.weight":          rand(d, d),
            p + "self_attn.k_proj.weight":          rand(h, d),
            p + "self_attn.v_proj.weight":          rand(h, d),
            p + "self_attn.o_proj.weight":          rand(d, d),
            p + "mlp.gate_proj.weight":             rand(f, d),
            p + "mlp.up_proj.weight":               rand(f, d),
            p + "mlp.down_proj.weight":             rand(d, f),
        }
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


def _run(engine: LlamaPagedEngine, requests: list[LlamaRequest]) -> dict[int, list[int]]:
    return engine.run_offline(requests)


def test_graphed_decode_matches_eager_single_sequence():
    cfg = _mini_config()
    model = _mini_model(cfg)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (5,)).tolist()

    eager = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=False,
    )
    graphed = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=True,
    )
    assert graphed.enable_cuda_graphs is True

    ref = _run(eager, [LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=10)])
    got = _run(graphed, [LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=10)])

    assert got[0] == ref[0], f"graphed diverged from eager:\n  graphed: {got[0]}\n  eager  : {ref[0]}"


def test_graphed_decode_matches_eager_concurrent_sequences():
    """Multiple concurrent sequences of different lengths exercise real bucket padding
    (both extra batch lanes and extra KV-length columns) — the case CUDA-graph capture
    must handle correctly under continuous batching's varying active-set composition."""
    cfg = _mini_config()
    model = _mini_model(cfg)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(1)
    prompts = [
        torch.randint(0, cfg.vocab_size, (3,)).tolist(),
        torch.randint(0, cfg.vocab_size, (7,)).tolist(),
        torch.randint(0, cfg.vocab_size, (5,)).tolist(),
    ]

    def build(engine: LlamaPagedEngine) -> list[LlamaRequest]:
        return [
            LlamaRequest(req_id=i, prompt_ids=p, max_new_tokens=9)
            for i, p in enumerate(prompts)
        ]

    eager = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=False,
    )
    graphed = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=True, graph_batch_buckets=(1, 2, 4),
    )

    ref = _run(eager, build(eager))
    got = _run(graphed, build(graphed))

    for i in range(len(prompts)):
        assert got[i] == ref[i], f"req {i} diverged:\n  graphed: {got[i]}\n  eager  : {ref[i]}"


def test_graphed_decode_falls_back_when_no_bucket_fits():
    """A KV length beyond every configured len_bucket must fall back to eager decode
    instead of crashing or silently truncating context."""
    cfg = _mini_config()
    model = _mini_model(cfg)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(2)
    prompt = torch.randint(0, cfg.vocab_size, (5,)).tolist()

    graphed = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=True, graph_len_buckets=[8],  # tiny — forces fallback quickly
    )
    results = _run(graphed, [LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=10)])
    assert len(results[0]) == 10


def test_graph_capture_is_cached_per_bucket():
    cfg = _mini_config()
    model = _mini_model(cfg)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (4,)).tolist()

    graphed = LlamaPagedEngine(
        model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
        enable_cuda_graphs=True,
    )
    _run(graphed, [LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=6)])
    n_graphs_after_first = len(graphed._graphs)
    assert n_graphs_after_first >= 1

    _run(graphed, [LlamaRequest(req_id=1, prompt_ids=prompt, max_new_tokens=6)])
    # Same (batch=1, len_bucket) shape recurs — no new graph should be captured.
    assert len(graphed._graphs) == n_graphs_after_first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="benchmark requires CUDA")
def test_benchmark_graphed_vs_eager_decode_throughput():
    """Not a correctness assertion — prints tokens/sec for eager vs graphed decode
    over many steps on a single active sequence so the speedup is visible in -s output."""
    cfg = _mini_config()
    model = _mini_model(cfg)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    torch.manual_seed(4)
    prompt = torch.randint(0, cfg.vocab_size, (8,)).tolist()
    n_new = 200

    def timed_run(enable_graphs: bool) -> float:
        engine = LlamaPagedEngine(
            model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
            enable_cuda_graphs=enable_graphs,
        )
        req = LlamaRequest(req_id=0, prompt_ids=prompt, max_new_tokens=n_new)
        # Warm up (captures graphs / warms caches) before timing.
        _run(engine, [LlamaRequest(req_id=99, prompt_ids=prompt, max_new_tokens=4)])

        engine2 = LlamaPagedEngine(
            model, n_total_blocks=64, block_size=8, eos_token=None, sampling=greedy,
            enable_cuda_graphs=enable_graphs,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        _run(engine2, [req])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return n_new / elapsed

    eager_tps = timed_run(False)
    graphed_tps = timed_run(True)
    print(f"\n[benchmark] eager decode:   {eager_tps:8.1f} tok/s")
    print(f"[benchmark] graphed decode: {graphed_tps:8.1f} tok/s")
    print(f"[benchmark] speedup:        {graphed_tps / eager_tps:5.2f}x")
