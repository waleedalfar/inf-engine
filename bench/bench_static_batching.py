"""Phase 3 baseline benchmark: static batching throughput / latency / memory / utilization.

This establishes the reference numbers every later phase reports deltas against. For each
batch size we replicate a fixed-length prompt B times and generate a fixed number of tokens
in lockstep (the naive static-batching strategy), measuring:

  * throughput  = (B * new_tokens) / total_time           [tokens/sec, aggregate]
  * latency     = decode_time / new_tokens                [ms per decode step]
  * peak memory = torch.cuda.max_memory_allocated          [accurate]
  * GPU util    = sampled nvidia-smi utilization.gpu       [WDDM caveat — see BENCHMARKS.md]

We also derive the theoretical maximum batch size from the KV-cache memory formula and
compare with what fits empirically.

Usage:
    python bench/bench_static_batching.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.batching import generate_batched  # noqa: E402
from engine.config import get_config  # noqa: E402
from engine.kv_cache import kv_cache_bytes  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import load_weights  # noqa: E402

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
PROMPT_LEN = 128
NEW_TOKENS = 128
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)
MB = 1024 * 1024
GIB = 1024 ** 3


class GpuUtilSampler:
    """Background thread that polls `nvidia-smi` GPU utilization while active."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0,
                )
                self._samples.append(float(out.stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self) -> "GpuUtilSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def stats(self) -> tuple[float, float]:
        if not self._samples:
            return (float("nan"), float("nan"))
        t = torch.tensor(self._samples)
        return (t.mean().item(), t.max().item())


def _weights_bytes(model: GPT2Model) -> int:
    return sum(v.numel() * v.element_size() for v in model.w._t.values())


def _time_batched(model: GPT2Model, prompts, pad_id, warmup: int, runs: int) -> tuple[float, float]:
    """Time full batched generation (prefill+decode). Returns (mean_s, std_s)."""
    times: list[float] = []
    for i in range(warmup + runs):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        generate_batched(model, prompts, NEW_TOKENS, pad_id, GREEDY)
        e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(s.elapsed_time(e) / 1000.0)
    t = torch.tensor(times)
    return t.mean().item(), t.std(unbiased=False).item()


def run(model_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    config = get_config(model_name)
    weights_path = REPO_ROOT / "weights" / model_name / "model.safetensors"
    model = GPT2Model(load_weights(weights_path, config, device=device, dtype=dtype), config)

    pad_id = 50256
    base_prompt = list(range(PROMPT_LEN))  # arbitrary ids; values don't affect timing
    seq_total = PROMPT_LEN + NEW_TOKENS
    weights_b = _weights_bytes(model)

    # --- derive theoretical max batch from the KV-cache memory formula ---
    total_vram = torch.cuda.get_device_properties(0).total_memory if device == "cuda" else 16 * GIB
    kv_per_seq = kv_cache_bytes(config, seq_len=seq_total, dtype=dtype, batch=1)
    usable = total_vram - weights_b - 1 * GIB  # reserve ~1 GiB for activations/CUDA context
    max_batch_seq256 = int(usable // kv_per_seq)
    kv_per_seq_1024 = kv_cache_bytes(config, seq_len=1024, dtype=dtype, batch=1)
    max_batch_seq1024 = int((total_vram - weights_b - 1 * GIB) // kv_per_seq_1024)

    results: dict = {
        "model": model_name, "device": device, "prompt_len": PROMPT_LEN,
        "new_tokens": NEW_TOKENS, "seq_total": seq_total,
        "weights_mib": weights_b / MB, "total_vram_gib": total_vram / GIB,
        "kv_bytes_per_seq_at_seq256": kv_per_seq, "kv_bytes_per_seq_at_seq1024": kv_per_seq_1024,
        "derived_max_batch_seq256": max_batch_seq256, "derived_max_batch_seq1024": max_batch_seq1024,
        "rows": [],
    }
    print(f"weights={weights_b/MB:.0f} MiB | VRAM={total_vram/GIB:.1f} GiB | "
          f"KV/seq@256={kv_per_seq/MB:.2f} MiB | derived max batch @seq256≈{max_batch_seq256}, "
          f"@seq1024≈{max_batch_seq1024}", flush=True)

    for bs in BATCH_SIZES:
        prompts = [base_prompt] * bs

        mean_s, std_s = _time_batched(model, prompts, pad_id, warmup=3, runs=10)
        throughput = (bs * NEW_TOKENS) / mean_s              # aggregate tokens/sec
        ms_per_step = mean_s / NEW_TOKENS * 1e3              # per decode-step latency (prefill incl.)

        # peak memory for one run
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        generate_batched(model, prompts, NEW_TOKENS, pad_id, GREEDY)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() - base_mem

        # GPU utilization sampled over ~1.5 s of sustained generation
        with GpuUtilSampler() as sampler:
            t_end = time.time() + 1.5
            while time.time() < t_end:
                generate_batched(model, prompts, NEW_TOKENS, pad_id, GREEDY)
                torch.cuda.synchronize()
        util_mean, util_max = sampler.stats()

        row = {
            "batch": bs, "time_s": mean_s, "time_std_s": std_s,
            "throughput_tok_s": throughput, "latency_ms_per_step": ms_per_step,
            "peak_mem_mib": peak_mem / MB, "util_mean": util_mean, "util_max": util_max,
        }
        results["rows"].append(row)
        print(f"B={bs:>2} | {throughput:7.0f} tok/s | {ms_per_step:6.2f} ms/step | "
              f"peak {peak_mem/MB:7.1f} MiB | util {util_mean:4.0f}% (max {util_max:.0f}%)", flush=True)

    return results


def make_plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["rows"]
    bs = [r["batch"] for r in rows]
    thr = [r["throughput_tok_s"] for r in rows]
    util = [r["util_mean"] for r in rows]
    mem = [r["peak_mem_mib"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(bs, thr, "o-", color="steelblue", label="throughput")
    ax1.set_xlabel("batch size"); ax1.set_ylabel("throughput (tok/s)")
    ax1b = ax1.twinx()
    ax1b.plot(bs, util, "s--", color="darkorange", label="GPU util %")
    ax1b.set_ylabel("GPU utilization (%)"); ax1b.set_ylim(0, 100)
    ax1.set_title(f"{results['model']}: throughput & utilization vs batch")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")

    ax2.plot(bs, mem, "o-", color="seagreen", label="peak memory")
    ax2.set_xlabel("batch size"); ax2.set_ylabel("peak memory (MiB)")
    ax2.set_title("peak GPU memory vs batch (linear in batch)")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    out = REPO_ROOT / "plots" / "static_batching.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()
    torch.manual_seed(0)
    results = run(args.model)
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"static_batching_{args.model}.json").write_text(json.dumps(results, indent=2))
    make_plot(results)
