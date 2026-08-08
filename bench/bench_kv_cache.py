"""Phase 2 benchmarks: KV cache throughput (with vs without) and memory (static vs dynamic).

NOTE: this is the *direct* (brute-force) harness — it actually generates to each length
with and without a cache. The no-cache path is O(T^2) and takes ~15-20 min at seq=2048, and
single-run timings are skewed by per-shape cuBLAS autotuning. Prefer the fast, warmed
``bench/bench_decode_latency.py`` (per-token latency sweep + reconstructed totals) and
``bench/bench_kv_memory.py`` (memory) for the numbers in BENCHMARKS.md. This file is kept
for direct validation of short sequences and as the home of the ``_bench_model`` helper.


Produces:
  * plots/kv_cache_time.png   -- total generation time vs sequence length: the
                                 no-cache curve is quadratic, the cached curve linear.
  * plots/kv_cache_memory.png -- KV-cache memory vs sequence length: theoretical
                                 formula, static peak, dynamic peak.
  * bench/results/kv_cache.json -- raw numbers.
  * a Markdown table on stdout for BENCHMARKS.md.

Protocol: CUDA-event timing with a sync; warmup runs discarded; mean ± std reported.
The cached path is cheap so it uses the full 3-warmup / 10-run protocol; the no-cache
path is O(T^2) and uses fewer timed runs (documented) to keep total runtime sane.

n_ctx note: GPT-2 only has 1024 trained position embeddings. For the 2048 *timing*
point we extend the position table by tiling so the compute/memory path runs — outputs
are meaningless past 1024, but FLOPs and bytes are representative. This is a timing
harness, not a generation-quality test (that is the correctness gate).

Usage:
    python bench/bench_kv_cache.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.config import GPT2Config, get_config  # noqa: E402
from engine.kv_cache import DynamicKVCache, StaticKVCache, kv_cache_bytes  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import GPT2Weights, load_weights  # noqa: E402

SEQ_LENS = [128, 512, 1024, 2048]
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)
MB = 1024 * 1024


def _bench_model(base: GPT2Model, config: GPT2Config, max_len: int, device: str, dtype: torch.dtype):
    """Return a model whose position table covers ``max_len`` positions.

    For max_len <= n_ctx the original model is returned unchanged. Beyond n_ctx we
    build a synthetic, larger position table by tiling (timing-only; see module note).
    """
    if max_len <= config.n_ctx:
        return base, config
    wpe = base.w.wpe                                            # (n_ctx, d_model)
    reps = (max_len + config.n_ctx - 1) // config.n_ctx
    wpe_ext = wpe.repeat(reps, 1)[:max_len].contiguous()        # (max_len, d_model) synthetic
    tensors = dict(base.w._t)
    tensors["wpe.weight"] = wpe_ext
    big_cfg = replace(config, n_ctx=max_len)
    return GPT2Model(GPT2Weights(tensors, big_cfg), big_cfg), big_cfg


def _time_generation(fn, warmup: int, runs: int) -> tuple[float, float]:
    """Time a no-arg generation closure with CUDA events. Returns (mean_s, std_s)."""
    use_cuda = torch.cuda.is_available()
    times: list[float] = []
    for i in range(warmup + runs):
        if use_cuda:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0          # ms -> s
        else:
            import time
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
        if i >= warmup:
            times.append(elapsed)
    t = torch.tensor(times)
    return t.mean().item(), t.std(unbiased=False).item()


def run(model_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    config = get_config(model_name)
    weights_path = REPO_ROOT / "weights" / model_name / "model.safetensors"
    model = GPT2Model(load_weights(weights_path, config, device=device, dtype=dtype), config)

    prompt = torch.tensor([[model.w.wte.new_zeros(1).long().item()]], device=device)  # 1-token prompt
    results: dict = {"model": model_name, "device": device, "dtype": str(dtype), "seqs": []}

    for seq_len in SEQ_LENS:
        new_tokens = seq_len - 1                                # generate up to total length seq_len
        bmodel, bcfg = _bench_model(model, config, seq_len, device, dtype)

        # --- throughput: cached (full protocol) ---
        def cached_gen():
            cache = StaticKVCache(bcfg, batch=1, max_seq=seq_len, device=device, dtype=dtype)
            return bmodel.generate_cached(prompt, new_tokens, cache, GREEDY)

        cache_mean, cache_std = _time_generation(cached_gen, warmup=3, runs=10)

        # --- throughput: no cache (expensive O(T^2); fewer runs, documented) ---
        # The no-cache cost grows quadratically: a single 2048-token no-cache run is
        # minutes long. At seq>=1024 the cache-vs-no-cache effect is order-of-magnitude,
        # so a single timed run is sufficient (run-to-run variance is immaterial next to
        # a >10x gap). Short sequences keep the full multi-run protocol.
        nc_warmup, nc_runs = (0, 1) if seq_len >= 1024 else (2, 5)

        def no_cache_gen():
            return bmodel.generate(prompt, new_tokens, GREEDY)

        nc_mean, nc_std = _time_generation(no_cache_gen, warmup=nc_warmup, runs=nc_runs)

        # --- memory: theoretical + peak CUDA allocated during gen for each cache type ---
        theo_bytes = kv_cache_bytes(bcfg, seq_len=seq_len, dtype=dtype, batch=1)

        def peak_during(make_cache) -> int:
            if device != "cuda":
                return 0
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            cache = make_cache()
            bmodel.generate_cached(prompt, new_tokens, cache, GREEDY)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            del cache
            return peak

        static_peak = peak_during(lambda: StaticKVCache(bcfg, 1, seq_len, device, dtype))
        dynamic_peak = peak_during(lambda: DynamicKVCache(bcfg, 1, device, dtype))

        row = {
            "seq_len": seq_len,
            "new_tokens": new_tokens,
            "synthetic_positions": seq_len > config.n_ctx,
            "cache_time_s": cache_mean,
            "cache_time_std_s": cache_std,
            "cache_tok_s": new_tokens / cache_mean,
            "no_cache_time_s": nc_mean,
            "no_cache_time_std_s": nc_std,
            "no_cache_tok_s": new_tokens / nc_mean,
            "speedup": nc_mean / cache_mean,
            "theoretical_kv_bytes": theo_bytes,
            "static_peak_bytes": static_peak,
            "dynamic_peak_bytes": dynamic_peak,
        }
        results["seqs"].append(row)
        print(
            f"seq={seq_len:>4} | cache {cache_mean*1e3:7.1f}ms ({row['cache_tok_s']:6.0f} tok/s) "
            f"| no-cache {nc_mean*1e3:8.1f}ms ({row['no_cache_tok_s']:6.0f} tok/s) "
            f"| speedup {row['speedup']:5.1f}x | KV {theo_bytes/MB:6.1f}MB"
            + ("  [synthetic pos]" if row["synthetic_positions"] else ""),
            flush=True,
        )

    return results


def make_plots(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [r["seq_len"] for r in results["seqs"]]
    cache_t = [r["cache_time_s"] for r in results["seqs"]]
    nocache_t = [r["no_cache_time_s"] for r in results["seqs"]]
    theo_mb = [r["theoretical_kv_bytes"] / MB for r in results["seqs"]]
    static_mb = [r["static_peak_bytes"] / MB for r in results["seqs"]]
    dyn_mb = [r["dynamic_peak_bytes"] / MB for r in results["seqs"]]

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- time vs seq length: quadratic (no cache) vs linear (cache) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(seqs, nocache_t, "o-", label="no cache (O(T²))", color="crimson")
    ax.plot(seqs, cache_t, "o-", label="KV cache (O(T))", color="seagreen")
    ax.set_xlabel("sequence length (tokens generated to)")
    ax.set_ylabel("total generation time (s)")
    ax.set_title(f"{results['model']}: KV cache removes the quadratic decode cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "kv_cache_time.png", dpi=130)
    plt.close(fig)

    # --- memory vs seq length: formula vs static/dynamic peak ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(seqs, theo_mb, "s--", label="theoretical KV (formula)", color="black")
    ax.plot(seqs, static_mb, "o-", label="static cache peak", color="steelblue")
    ax.plot(seqs, dyn_mb, "o-", label="dynamic cache peak", color="darkorange")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("memory (MiB)")
    ax.set_title(f"{results['model']}: KV cache memory grows linearly in sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "kv_cache_memory.png", dpi=130)
    plt.close(fig)
    print(f"wrote {plots_dir/'kv_cache_time.png'} and {plots_dir/'kv_cache_memory.png'}")


def print_markdown(results: dict) -> None:
    print("\n### Throughput: with vs without KV cache\n")
    print("| seq | tok/s (cache) | tok/s (no cache) | speedup | cache ms | no-cache ms |")
    print("|----:|-------------:|-----------------:|--------:|---------:|------------:|")
    for r in results["seqs"]:
        note = " *" if r["synthetic_positions"] else ""
        print(
            f"| {r['seq_len']}{note} | {r['cache_tok_s']:.0f} | {r['no_cache_tok_s']:.0f} "
            f"| {r['speedup']:.1f}× | {r['cache_time_s']*1e3:.1f} ± {r['cache_time_std_s']*1e3:.1f} "
            f"| {r['no_cache_time_s']*1e3:.0f} ± {r['no_cache_time_std_s']*1e3:.0f} |"
        )
    print("\n### Memory: static vs dynamic cache\n")
    print("| seq | theoretical KV (MiB) | static peak (MiB) | dynamic peak (MiB) |")
    print("|----:|---------------------:|------------------:|-------------------:|")
    for r in results["seqs"]:
        print(
            f"| {r['seq_len']} | {r['theoretical_kv_bytes']/MB:.2f} "
            f"| {r['static_peak_bytes']/MB:.2f} | {r['dynamic_peak_bytes']/MB:.2f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()

    torch.manual_seed(0)
    results = run(args.model)

    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"kv_cache_{args.model}.json").write_text(json.dumps(results, indent=2))

    make_plots(results)
    print_markdown(results)
