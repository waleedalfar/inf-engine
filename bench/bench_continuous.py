"""Phase 4 benchmark: continuous batching vs the static baseline, under load.

Produces, all on one mixed workload (varied prompt + output lengths):
  * saturated throughput (req/s, tok/s) and GPU utilization: continuous vs static,
  * latency distribution p50/p95/p99 for FCFS vs SJF,
  * queue depth over time under Poisson arrivals at 50/80/100% of measured capacity
    (plots/continuous_queue_depth.png),
  * a throughput/utilization bar plot (plots/continuous_vs_static.png).

Static baseline here = Phase 3 static batching applied to the workload: requests are run in
fixed groups, each group padded to its longest prompt and decoded in lockstep until the
group's *longest* output finishes (shorter requests waste compute). That waste — plus the
inability to admit a queued request until the whole group finishes — is what continuous
batching removes.

Usage:
    python bench/bench_continuous.py --model gpt2
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

from bench.bench_static_batching import GpuUtilSampler  # noqa: E402
from engine.batching import generate_batched  # noqa: E402
from engine.config import get_config  # noqa: E402
from engine.continuous import ContinuousBatchingEngine, Policy, Request  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import load_weights  # noqa: E402

N_REQUESTS = 48
POISSON_N = 120         # more requests for the load test so the queue reaches steady state
N_SLOTS = 16            # concurrency for both continuous slots and static batch size
PROMPT_MIN, PROMPT_MAX = 8, 24
# Output lengths are deliberately high-variance: most requests are short, a minority are
# long. This is the realistic regime (chat: many short replies, a few long ones) and the
# one where static batching's lockstep hurts — a group is held hostage by its longest
# member while short requests sit finished-but-stuck. Continuous batching frees them.
OUT_SHORT = (8, 24)
OUT_LONG = (150, 220)
LONG_FRACTION = 0.2
MAX_SEQ = PROMPT_MAX + OUT_LONG[1] + 8
PAD_ID = 50256
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)


def make_workload(seed: int = 0, n: int = N_REQUESTS) -> list[tuple[list[int], int]]:
    """Deterministic high-variance workload of (prompt_ids, max_new_tokens)."""
    rng = random.Random(seed)
    work = []
    for _ in range(n):
        plen = rng.randint(PROMPT_MIN, PROMPT_MAX)
        if rng.random() < LONG_FRACTION:
            out = rng.randint(*OUT_LONG)
        else:
            out = rng.randint(*OUT_SHORT)
        prompt = [rng.randint(0, 50256) for _ in range(plen)]   # ids don't affect timing
        work.append((prompt, out))
    return work


def _new_requests(work) -> list[Request]:
    return [Request(req_id=i, prompt_ids=p, max_new_tokens=n) for i, (p, n) in enumerate(work)]


def run_static(model: GPT2Model, work, batch_size: int) -> tuple[float, int]:
    """Static batching over the workload: fixed groups, lockstep to the group's longest.

    Returns (wall_time_s, useful_tokens).
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    useful = 0
    for i in range(0, len(work), batch_size):
        group = work[i : i + batch_size]
        max_new = max(n for _, n in group)                      # lockstep to longest output
        prompts = [p for p, _ in group]
        generate_batched(model, prompts, max_new, PAD_ID, GREEDY)
        useful += sum(n for _, n in group)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, useful


def run_continuous_offline(model: GPT2Model, work, policy: Policy) -> tuple[float, int, list[float]]:
    """Saturated continuous batching: submit all at t=0, run to completion.

    Returns (wall_time_s, useful_tokens, per_request_latencies_s).
    """
    engine = ContinuousBatchingEngine(model, N_SLOTS, MAX_SEQ, policy=policy, sampling=GREEDY)
    reqs = _new_requests(work)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for r in reqs:
        r.arrival_time = 0.0
        engine.submit(r)
    while engine.has_work():
        engine.step(now=time.perf_counter() - t0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    useful = sum(n for _, n in work)
    latencies = sorted(r.finish_time - r.arrival_time for r in engine.completed)
    return wall, useful, latencies


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    t = torch.tensor(sorted(xs))
    q = torch.quantile(t, torch.tensor([0.50, 0.95, 0.99]))
    return {"p50": q[0].item(), "p95": q[1].item(), "p99": q[2].item(), "mean": t.mean().item()}


def run_poisson(model: GPT2Model, work, rate_req_s: float, policy: Policy):
    """Drive the engine in real time with Poisson(rate) arrivals; log queue depth.

    Returns (samples[(t, qdepth, active)], latencies_s).
    """
    engine = ContinuousBatchingEngine(model, N_SLOTS, MAX_SEQ, policy=policy, sampling=GREEDY)
    rng = random.Random(1)
    # exponential inter-arrival times -> Poisson process
    arrivals, t = [], 0.0
    for _ in range(len(work)):
        t += rng.expovariate(rate_req_s)
        arrivals.append(t)

    reqs = _new_requests(work)
    for r, a in zip(reqs, arrivals):
        r.arrival_time = a                                      # scheduled arrival (for latency)

    samples: list[tuple[float, int, int]] = []
    t0 = time.perf_counter()
    i = 0
    while i < len(reqs) or engine.has_work():
        now = time.perf_counter() - t0
        while i < len(reqs) and arrivals[i] <= now:
            engine.submit(reqs[i])
            i += 1
        if engine.has_work():
            engine.step(now=time.perf_counter() - t0)
        else:
            # idle until the next arrival
            if i < len(reqs):
                time.sleep(max(0.0, arrivals[i] - (time.perf_counter() - t0)))
        samples.append((time.perf_counter() - t0, len(engine.queue), engine.num_active))
    latencies = sorted(r.finish_time - r.arrival_time for r in engine.completed)
    return samples, latencies


def run(model_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(model_name)
    weights_path = REPO_ROOT / "weights" / model_name / "model.safetensors"
    model = GPT2Model(load_weights(weights_path, config, device=device, dtype=torch.float32), config)
    work = make_workload()

    # warmup
    run_continuous_offline(model, work[:8], Policy.FCFS)

    # --- saturated throughput + utilization: static vs continuous ---
    with GpuUtilSampler() as s:
        static_wall, static_useful = run_static(model, work, N_SLOTS)
    static_util = s.stats()[0]

    with GpuUtilSampler() as s:
        cont_wall, cont_useful, cont_lat = run_continuous_offline(model, work, Policy.FCFS)
    cont_util = s.stats()[0]

    capacity = N_REQUESTS / cont_wall                           # req/s, saturated continuous

    # --- FCFS vs SJF latency distribution (saturated) ---
    _, _, fcfs_lat = run_continuous_offline(model, work, Policy.FCFS)
    _, _, sjf_lat = run_continuous_offline(model, work, Policy.SJF)

    # --- Poisson arrivals at 50/80/100% of capacity (larger workload for steady state) ---
    poisson_work = make_workload(seed=2, n=POISSON_N)
    poisson = {}
    for frac in (0.5, 0.8, 1.0):
        samples, lat = run_poisson(model, poisson_work, rate_req_s=frac * capacity, policy=Policy.FCFS)
        poisson[frac] = {"samples": samples, "lat": percentiles(lat)}
        print(f"poisson {int(frac*100)}% cap: max qdepth={max(q for _,q,_ in samples)} "
              f"p99 latency={poisson[frac]['lat']['p99']*1e3:.0f} ms", flush=True)

    results = {
        "model": model_name, "n_requests": N_REQUESTS, "n_slots": N_SLOTS,
        "static": {"wall_s": static_wall, "req_s": N_REQUESTS / static_wall,
                   "tok_s": static_useful / static_wall, "util": static_util},
        "continuous": {"wall_s": cont_wall, "req_s": capacity,
                       "tok_s": cont_useful / cont_wall, "util": cont_util},
        "throughput_speedup": static_wall / cont_wall,
        "capacity_req_s": capacity,
        "latency_fcfs": percentiles(fcfs_lat),
        "latency_sjf": percentiles(sjf_lat),
        "poisson": {str(k): {"lat": v["lat"],
                             "max_qdepth": max(q for _, q, _ in v["samples"])}
                    for k, v in poisson.items()},
    }
    print(f"\nstatic:     {results['static']['req_s']:.1f} req/s, "
          f"{results['static']['tok_s']:.0f} tok/s, util {static_util:.0f}%")
    print(f"continuous: {results['continuous']['req_s']:.1f} req/s, "
          f"{results['continuous']['tok_s']:.0f} tok/s, util {cont_util:.0f}%  "
          f"({results['throughput_speedup']:.2f}x throughput)")
    print(f"latency FCFS p50/p95/p99 = {fcfs_lat and percentiles(fcfs_lat)}")
    print(f"latency SJF  p50/p95/p99 = {sjf_lat and percentiles(sjf_lat)}")

    _make_plots(results, poisson, fcfs_lat, sjf_lat)
    return results


def _make_plots(results, poisson, fcfs_lat, sjf_lat) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = REPO_ROOT / "plots"
    plots.mkdir(exist_ok=True)

    # queue depth over time at 50/80/100% capacity
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.5: "seagreen", 0.8: "darkorange", 1.0: "crimson"}
    for frac, c in colors.items():
        samples = poisson[frac]["samples"]
        ts = [s[0] for s in samples]
        qs = [s[1] for s in samples]
        ax.plot(ts, qs, color=c, label=f"{int(frac*100)}% capacity", linewidth=1.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("queue depth (waiting requests)")
    ax.set_title(f"{results['model']}: queue depth under Poisson load (FCFS)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(plots / "continuous_queue_depth.png", dpi=130); plt.close(fig)

    # throughput + utilization bars: static vs continuous
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["static", "continuous"]
    ax1.bar(labels, [results["static"]["tok_s"], results["continuous"]["tok_s"]],
            color=["gray", "steelblue"])
    ax1.set_ylabel("useful throughput (tok/s)"); ax1.set_title("throughput")
    ax2.bar(labels, [results["static"]["util"], results["continuous"]["util"]],
            color=["gray", "steelblue"])
    ax2.set_ylabel("GPU utilization (%)"); ax2.set_ylim(0, 100); ax2.set_title("utilization")
    fig.tight_layout(); fig.savefig(plots / "continuous_vs_static.png", dpi=130); plt.close(fig)
    print(f"wrote {plots/'continuous_queue_depth.png'} and {plots/'continuous_vs_static.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()
    torch.manual_seed(0)
    results = run(args.model)
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"continuous_{args.model}.json").write_text(json.dumps(results, indent=2))
