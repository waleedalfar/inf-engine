"""Phase 5 K2 benchmark: Flash-Attention vs naive attention — memory and throughput.

Naive attention materializes the (B,H,N,N) score matrix, so its peak memory grows as O(N²).
Flash-Attention streams K/V and keeps only an O(N) accumulator, so its peak is flat in N
(just Q,K,V,O). We verify that empirically and time both. FLOPs are identical (≈ 4·B·H·N²·d),
so any speedup comes from avoiding the N² HBM round-trip — attention is memory-bound.

Usage:
    python bench/bench_flash_attention.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.kernels.flash_attention import flash_attention, naive_attention  # noqa: E402

B, H, D = 2, 8, 64
SEQS = [512, 1024, 2048, 4096]
WARMUP, RUNS = 5, 20
MB = 1024 * 1024


def _time(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(RUNS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / RUNS / 1e3


def _peak_mib(fn, baseline: int) -> float:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / MB


def run() -> dict:
    rows = []
    for n in SEQS:
        q = torch.randn(B, H, n, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, n, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, n, D, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        flops = 4.0 * B * H * n * n * D                      # QKᵀ (2BHN²d) + PV (2BHN²d)

        flash_peak = _peak_mib(lambda: flash_attention(q, k, v, causal=True), baseline)
        # naive only up to where the N² scores fit comfortably
        try:
            naive_peak = _peak_mib(lambda: naive_attention(q, k, v, causal=True), baseline)
            naive_ms = _time(lambda: naive_attention(q, k, v, causal=True)) * 1e3
            naive_tflops = flops / (naive_ms / 1e3) / 1e12
        except torch.cuda.OutOfMemoryError:
            naive_peak = float("nan"); naive_ms = float("nan"); naive_tflops = float("nan")
            torch.cuda.empty_cache()

        flash_ms = _time(lambda: flash_attention(q, k, v, causal=True)) * 1e3
        flash_tflops = flops / (flash_ms / 1e3) / 1e12

        rows.append({
            "seq": n,
            "flash_peak_mib": flash_peak, "naive_peak_mib": naive_peak,
            "flash_ms": flash_ms, "naive_ms": naive_ms,
            "flash_tflops": flash_tflops, "naive_tflops": naive_tflops,
            "mem_ratio": (naive_peak / flash_peak) if flash_peak else float("nan"),
            "speedup": (naive_ms / flash_ms) if flash_ms else float("nan"),
        })
        print(f"seq={n:>4} | flash {flash_ms:7.2f}ms peak {flash_peak:8.1f}MiB | "
              f"naive {naive_ms:7.2f}ms peak {naive_peak:9.1f}MiB | "
              f"mem {rows[-1]['mem_ratio']:5.1f}x  speed {rows[-1]['speedup']:.2f}x", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    return {"B": B, "H": H, "D": D, "rows": rows}


def make_plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["rows"]
    seqs = [r["seq"] for r in rows]
    fp = [r["flash_peak_mib"] for r in rows]
    npk = [r["naive_peak_mib"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(seqs, npk, "o-", color="crimson", label="naive (O(N²) scores)")
    ax1.plot(seqs, fp, "o-", color="seagreen", label="flash (O(N))")
    ax1.set_xlabel("sequence length"); ax1.set_ylabel("attention peak memory (MiB)")
    ax1.set_title("Flash-Attention peak memory is O(N), naive is O(N²)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(seqs, [r["naive_ms"] for r in rows], "o-", color="crimson", label="naive")
    ax2.plot(seqs, [r["flash_ms"] for r in rows], "o-", color="seagreen", label="flash")
    ax2.set_xlabel("sequence length"); ax2.set_ylabel("latency (ms)")
    ax2.set_title("attention latency vs sequence length")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPO_ROOT / "plots" / "flash_attention.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    results = run()
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flash_attention.json").write_text(json.dumps(results, indent=2))
    make_plot(results)
