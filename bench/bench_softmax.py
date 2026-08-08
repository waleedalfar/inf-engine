"""Phase 5 K1 benchmark: fused Triton softmax vs PyTorch native softmax.

Softmax is memory-bandwidth bound: ~a handful of flops per element, but it must read the
input and write the output through HBM. So we report achieved HBM bandwidth (GB/s) and its
fraction of the GPU's theoretical peak — that is the metric that matters, not FLOP/s.

bytes moved = read input + write output = 2 * n_rows * n_cols * 4 (fp32).
achieved bandwidth = bytes / time.

Usage:
    python bench/bench_softmax.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.kernels.softmax import triton_softmax  # noqa: E402

SHAPES = [(8192, 512), (8192, 1024), (8192, 2048), (8192, 4096)]
WARMUP, RUNS = 10, 50


def peak_bandwidth_gb_s() -> float:
    """Theoretical HBM peak from device properties: 2 (DDR) * clock * bus_width / 8."""
    p = torch.cuda.get_device_properties(0)
    clock_hz = p.memory_clock_rate * 1e3            # torch reports kHz
    return 2.0 * clock_hz * (p.memory_bus_width / 8.0) / 1e9


def l2_bytes() -> int:
    return getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)


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
    return s.elapsed_time(e) / RUNS / 1e3  # seconds per call


def run() -> dict:
    peak = peak_bandwidth_gb_s()
    l2 = l2_bytes()
    print(f"peak HBM ~ {peak:.0f} GB/s | L2 = {l2/1024/1024:.0f} MiB", flush=True)
    rows = []
    for n_rows, n_cols in SHAPES:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        bytes_moved = 2 * n_rows * n_cols * 4                # read + write, fp32
        l2_resident = bytes_moved < l2                       # working set fits in L2 cache

        t_triton = _time(lambda: triton_softmax(x))
        t_torch = _time(lambda: torch.softmax(x, dim=-1))

        bw_triton = bytes_moved / t_triton / 1e9
        bw_torch = bytes_moved / t_torch / 1e9
        rows.append({
            "n_rows": n_rows, "n_cols": n_cols, "l2_resident": l2_resident,
            "triton_us": t_triton * 1e6, "torch_us": t_torch * 1e6,
            "triton_gbps": bw_triton, "torch_gbps": bw_torch,
            "triton_pct_peak": 100 * bw_triton / peak,
            "torch_pct_peak": 100 * bw_torch / peak,
            "speedup": t_torch / t_triton,
        })
        tag = " [L2-resident: BW is cache, not HBM]" if l2_resident else ""
        print(f"cols={n_cols:>4} | triton {t_triton*1e6:7.1f}us {bw_triton:6.0f} GB/s "
              f"({100*bw_triton/peak:4.0f}% peak) | torch {t_torch*1e6:7.1f}us "
              f"{bw_torch:6.0f} GB/s ({100*bw_torch/peak:4.0f}%) | {t_torch/t_triton:.2f}x{tag}",
              flush=True)
    return {"peak_bw_gb_s": peak, "l2_bytes": l2, "rows": rows}


if __name__ == "__main__":
    torch.manual_seed(0)
    results = run()
    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "softmax.json").write_text(json.dumps(results, indent=2))
