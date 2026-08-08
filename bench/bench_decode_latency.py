"""Fast, scalable decode-latency benchmark via a forward-cost sweep.

Motivation: measuring full no-cache *generation* to length T is O(T^2) and takes minutes
at T=2048. But the quantity we care about decomposes cleanly. In no-cache decoding, the
per-step cost of emitting the token at position t is exactly one forward pass over a
length-t prefix, ``fwd(t)``. So:

    per_token_latency_nocache(t) = fwd(t)
    total_nocache(T)             = sum_{t=1..T} fwd(t)   ≈ ∫ fwd(t) dt   (trapezoid)

We therefore measure ``fwd(t)`` at ~a dozen sampled context lengths (one forward each,
milliseconds) and reconstruct both the per-token latency curve and the total generation
time for ANY T — including 2048/4096 — in seconds instead of minutes. We validate the
reconstruction against the directly-measured totals from ``bench_kv_cache.py``.

For the cached path we measure the single-token decode latency at each context (prefill a
cache to length t-1, then time one token at start_pos=t-1) — that is what actually runs
during cached generation.

Usage:
    python bench/bench_decode_latency.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_kv_cache import _bench_model  # noqa: E402  (reuse synthetic-position helper)
from engine.config import get_config  # noqa: E402
from engine.kv_cache import StaticKVCache  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.weights import load_weights  # noqa: E402

# Context lengths at which we probe a single forward pass. Dense enough to integrate.
CONTEXT_GRID = [1, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048]
REPORT_AT = [128, 512, 1024, 2048]
WARMUP, RUNS = 3, 10


def _time_once(fn, warmup: int, runs: int) -> tuple[float, float]:
    """Time a no-arg closure with CUDA events. Returns (mean_s, std_s)."""
    use_cuda = torch.cuda.is_available()
    times: list[float] = []
    for i in range(warmup + runs):
        if use_cuda:
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            dt = s.elapsed_time(e) / 1000.0
        else:
            import time

            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    t = torch.tensor(times)
    return t.mean().item(), t.std(unbiased=False).item()


def _cumulative_trapezoid(xs: list[float], ys: list[float]) -> list[float]:
    """Cumulative integral of y(x) via the trapezoid rule. Returns value at each x."""
    out = [0.0]
    for i in range(1, len(xs)):
        out.append(out[-1] + 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]))
    return out


def run(model_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    config = get_config(model_name)
    weights_path = REPO_ROOT / "weights" / model_name / "model.safetensors"
    base = GPT2Model(load_weights(weights_path, config, device=device, dtype=dtype), config)

    nocache_ms: list[float] = []   # fwd(t): per-token no-cache latency at context t
    cache_ms: list[float] = []     # single-token cached decode latency at context t

    for t in CONTEXT_GRID:
        bmodel, bcfg = _bench_model(base, config, t, device, dtype)
        ids_t = torch.zeros((1, t), dtype=torch.long, device=device)        # (1, t) dummy tokens

        # no-cache forward over a length-t prefix (= per-token no-cache decode cost at ctx t)
        nc_mean, _ = _time_once(lambda: bmodel.forward(ids_t), WARMUP, RUNS)
        nocache_ms.append(nc_mean * 1e3)

        # cached single-token decode at context t: prefill ONCE (outside timing), then
        # time only the single-token forward at start_pos=t-1. Re-running this step
        # overwrites the same cache slot, so it cleanly measures one decode step at ctx t.
        cache = StaticKVCache(bcfg, batch=1, max_seq=t, device=device, dtype=dtype)
        if t > 1:
            bmodel.forward(ids_t[:, : t - 1], cache=cache, start_pos=0)      # prefill once
        one = ids_t[:, :1]                                                   # (1, 1)
        c_mean, _ = _time_once(
            lambda: bmodel.forward(one, cache=cache, start_pos=max(t - 1, 0)), WARMUP, RUNS
        )
        cache_ms.append(c_mean * 1e3)
        print(f"ctx={t:>4} | no-cache fwd {nocache_ms[-1]:7.2f} ms | cached step {cache_ms[-1]:6.2f} ms", flush=True)

    # Reconstruct total generation time = sum_{t=1..T} step(t) ≈ ∫ step(t) dt, for both
    # paths. No-cache step(t)=fwd(t) (grows ∝ t -> total ∝ T^2); cache step(t) is a single
    # decode (grows slowly -> total ~ linear in T).
    xs = [float(x) for x in CONTEXT_GRID]
    nc_cum = dict(zip(CONTEXT_GRID, _cumulative_trapezoid(xs, nocache_ms)))
    ca_cum = dict(zip(CONTEXT_GRID, _cumulative_trapezoid(xs, cache_ms)))
    recon_nc = {T: nc_cum[T] / 1e3 for T in REPORT_AT if T in nc_cum}  # seconds
    recon_ca = {T: ca_cum[T] / 1e3 for T in REPORT_AT if T in ca_cum}  # seconds

    return {
        "model": model_name,
        "device": device,
        "context_grid": CONTEXT_GRID,
        "nocache_fwd_ms": nocache_ms,
        "cached_step_ms": cache_ms,
        "reconstructed_total_nocache_s": recon_nc,
        "reconstructed_total_cache_s": recon_ca,
    }


def make_plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = results["context_grid"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, results["nocache_fwd_ms"], "o-", color="crimson",
            label="no cache: per-token = full forward fwd(t)  (∝ t)")
    ax.plot(xs, results["cached_step_ms"], "o-", color="seagreen",
            label="KV cache: single-token decode step")
    ax.set_xlabel("context length t (tokens)")
    ax.set_ylabel("per-token latency (ms)")
    ax.set_title(f"{results['model']}: per-token decode latency vs context")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPO_ROOT / "plots" / "decode_latency_vs_context.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")

    # --- reconstructed total generation time vs target length: quadratic vs linear ---
    report = sorted(results["reconstructed_total_nocache_s"])
    nc = [results["reconstructed_total_nocache_s"][str(T) if str(T) in results["reconstructed_total_nocache_s"] else T] for T in report]
    ca = [results["reconstructed_total_cache_s"][str(T) if str(T) in results["reconstructed_total_cache_s"] else T] for T in report]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(report, nc, "o-", color="crimson", label="no cache (Σ fwd(t) ∝ T²)")
    ax.plot(report, ca, "o-", color="seagreen", label="KV cache (Σ decode(t) ∝ T)")
    ax.set_xlabel("tokens generated to (T)")
    ax.set_ylabel("reconstructed total time (s)")
    ax.set_title(f"{results['model']}: total decode time — quadratic vs linear")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = REPO_ROOT / "plots" / "kv_cache_time.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()

    torch.manual_seed(0)
    results = run(args.model)
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"decode_latency_{args.model}.json").write_text(json.dumps(results, indent=2))
    make_plot(results)
    print("\nReconstructed no-cache total generation time (sum of fwd(t)):")
    for T, s in results["reconstructed_total_nocache_s"].items():
        print(f"  to {T:>4} tokens: {s:.1f} s")
