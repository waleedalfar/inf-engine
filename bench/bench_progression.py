"""Phase 6: before/after progression across all optimization phases.

Measures effective throughput (tok/s) and peak memory on a consistent basis for:
  naive (no cache) -> KV cache -> static batching -> continuous batching.

Note the column means "tokens/sec the system delivers": phases 1-2 are single-stream
(one request), phases 3-4 are aggregate over concurrency. That is the honest progression —
KV cache fixes per-stream decode cost, batching multiplies throughput across streams. The
Triton-kernel phase (5) is a component-level result (Flash-Attention memory/throughput) and
is reported separately, since the kernels are benchmarked standalone (see ARCHITECTURE.md).

Usage:
    python bench/bench_progression.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.batching import generate_batched  # noqa: E402
from engine.config import get_config  # noqa: E402
from engine.continuous import ContinuousBatchingEngine, Policy, Request  # noqa: E402
from engine.kv_cache import StaticKVCache  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import load_weights  # noqa: E402

PROMPT_LEN = 64
NEW_TOKENS = 128
STATIC_BATCH = 16
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)
MB = 1024 * 1024


def _timed(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / runs / 1e3


def _peak_mib(fn) -> float:
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / MB


def run(model_name: str) -> dict:
    device = "cuda"
    config = get_config(model_name)
    wpath = REPO_ROOT / "weights" / model_name / "model.safetensors"
    model = GPT2Model(load_weights(wpath, config, device=device, dtype=torch.float32), config)
    prompt = torch.zeros((1, PROMPT_LEN), dtype=torch.long, device=device)

    rows = []

    # Phase 1: naive, no cache, single stream (recompute every step).
    t = _timed(lambda: model.generate(prompt, NEW_TOKENS, GREEDY), 1, 3)
    mem = _peak_mib(lambda: model.generate(prompt, NEW_TOKENS, GREEDY))
    rows.append({"phase": "1 naive (no cache)", "tok_s": NEW_TOKENS / t, "peak_mib": mem,
                 "mode": "single-stream"})

    # Phase 2: KV cache, single stream.
    def kv():
        c = StaticKVCache(config, 1, PROMPT_LEN + NEW_TOKENS, device, torch.float32)
        model.generate_cached(prompt, NEW_TOKENS, c, GREEDY)
    t = _timed(kv, 3, 10)
    mem = _peak_mib(kv)
    rows.append({"phase": "2 + KV cache", "tok_s": NEW_TOKENS / t, "peak_mib": mem,
                 "mode": "single-stream"})

    # Phase 3: static batching, aggregate over STATIC_BATCH streams.
    prompts = [[0] * PROMPT_LEN] * STATIC_BATCH
    def static():
        generate_batched(model, prompts, NEW_TOKENS, 50256, GREEDY)
    t = _timed(static, 3, 10)
    mem = _peak_mib(static)
    rows.append({"phase": f"3 + static batch x{STATIC_BATCH}", "tok_s": STATIC_BATCH * NEW_TOKENS / t,
                 "peak_mib": mem, "mode": "aggregate"})

    # Phase 4: continuous batching on the SAME uniform workload as static, for an
    # apples-to-apples row (16 identical requests). On uniform load continuous ~ static;
    # its real win is on variable-length load (+1.24x, see bench_continuous.py / Phase 4).
    uniform = [([0] * PROMPT_LEN, NEW_TOKENS) for _ in range(STATIC_BATCH)]
    max_seq = PROMPT_LEN + NEW_TOKENS

    def continuous():
        eng = ContinuousBatchingEngine(model, STATIC_BATCH, max_seq, policy=Policy.FCFS, sampling=GREEDY)
        reqs = [Request(req_id=i, prompt_ids=p, max_new_tokens=n) for i, (p, n) in enumerate(uniform)]
        eng.run_offline(reqs)
    total_tokens = sum(n for _, n in uniform)
    torch.cuda.synchronize(); t0 = time.perf_counter(); continuous(); torch.cuda.synchronize()
    cont_t = time.perf_counter() - t0
    mem = _peak_mib(continuous)
    rows.append({"phase": "4 + continuous batch", "tok_s": total_tokens / cont_t, "peak_mib": mem,
                 "mode": "aggregate (uniform; +1.24x on variable load)"})

    base_tok_s = rows[0]["tok_s"]
    for r in rows:
        r["speedup_vs_naive"] = r["tok_s"] / base_tok_s
        print(f"{r['phase']:28s} | {r['tok_s']:8.0f} tok/s | peak {r['peak_mib']:7.1f} MiB | "
              f"{r['speedup_vs_naive']:6.1f}x vs naive | {r['mode']}", flush=True)

    return {"model": model_name, "prompt_len": PROMPT_LEN, "new_tokens": NEW_TOKENS, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()
    torch.manual_seed(0)
    results = run(args.model)
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"progression_{args.model}.json").write_text(json.dumps(results, indent=2))
