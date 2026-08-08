"""Phase 5 roofline analysis: place each kernel on the GPU's roofline model.

The roofline model bounds achievable performance by two ceilings:
  * compute:  perf <= peak_flops
  * memory:   perf <= arithmetic_intensity * peak_bandwidth
The crossover (ridge point) is peak_flops / peak_bandwidth. A kernel with arithmetic
intensity (flop/byte) left of the ridge is memory-bound; right of it, compute-bound.

We compute the arithmetic intensity and achieved FLOP/s for each Phase 5 kernel and plot
them against the roofline, which tells us the *direction* of further optimization.

Usage:
    python bench/roofline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.kernels.flash_attention import flash_attention  # noqa: E402
from engine.kernels.quant import int8_matmul, quantize_weight_int8  # noqa: E402
from engine.kernels.softmax import triton_softmax  # noqa: E402

# RTX 5070 Ti (Blackwell, 70 SM): ~44 TFLOP/s FP32 peak (70 SM * 128 FP32 lanes * 2 * ~2.45 GHz).
PEAK_FLOPS = 44e12


def _peak_bw() -> float:
    p = torch.cuda.get_device_properties(0)
    return 2.0 * (p.memory_clock_rate * 1e3) * (p.memory_bus_width / 8.0)  # bytes/s


def _time(fn, warmup=10, runs=50) -> float:
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


def measure() -> list[dict]:
    pts = []

    # --- fused softmax: ~5 flop/elem, 8 byte/elem (fp32 read+write) -> AI ~ 0.6 ---
    n_rows, n_cols = 8192, 2048
    x = torch.randn(n_rows, n_cols, device="cuda")
    t = _time(lambda: triton_softmax(x))
    flops = 5.0 * n_rows * n_cols
    bytes_ = 2.0 * n_rows * n_cols * 4
    pts.append({"name": "fused softmax", "ai": flops / bytes_, "perf": flops / t})

    # --- flash attention: flops 4*B*H*N^2*d; HBM bytes ~ Q,K,V,O = 4*B*H*N*d*4 ---
    B, H, N, D = 2, 8, 2048, 64
    q = torch.randn(B, H, N, D, device="cuda")
    k = torch.randn(B, H, N, D, device="cuda")
    v = torch.randn(B, H, N, D, device="cuda")
    t = _time(lambda: flash_attention(q, k, v, causal=True))
    flops = 4.0 * B * H * N * N * D
    bytes_ = 4.0 * B * H * N * D * 4
    pts.append({"name": "flash attention", "ai": flops / bytes_, "perf": flops / t})

    # --- int8 matmul (decode GEMM): flops 2*M*K*N; bytes int8 weights + fp16 act/out ---
    M, K, Nn = 16, 768, 2304
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = (torch.randn(K, Nn, device="cuda") * 0.1)
    wq, scale = quantize_weight_int8(w)
    t = _time(lambda: int8_matmul(a, wq, scale))
    flops = 2.0 * M * K * Nn
    bytes_ = K * Nn * 1 + M * K * 2 + M * Nn * 2
    pts.append({"name": "int8 matmul", "ai": flops / bytes_, "perf": flops / t})
    return pts


def main() -> None:
    peak_bw = _peak_bw()
    ridge = PEAK_FLOPS / peak_bw
    pts = measure()
    for p in pts:
        bound = "memory-bound" if p["ai"] < ridge else "compute-bound"
        print(f"{p['name']:18s} AI={p['ai']:8.2f} flop/byte  perf={p['perf']/1e12:6.2f} TFLOP/s  "
              f"-> {bound}", flush=True)
    print(f"ridge point = {ridge:.1f} flop/byte (peak {PEAK_FLOPS/1e12:.0f} TFLOP/s / "
          f"{peak_bw/1e9:.0f} GB/s)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    ai = np.logspace(-1, 4, 200)
    roof = np.minimum(PEAK_FLOPS, ai * peak_bw) / 1e12
    ax.plot(ai, roof, "k-", linewidth=2, label="roofline")
    ax.axvline(ridge, color="gray", linestyle=":", label=f"ridge {ridge:.0f} flop/byte")
    for p in pts:
        ax.plot(p["ai"], p["perf"] / 1e12, "o", markersize=10)
        ax.annotate(p["name"], (p["ai"], p["perf"] / 1e12),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (flop/byte)")
    ax.set_ylabel("performance (TFLOP/s)")
    ax.set_title("RTX 5070 Ti roofline — Phase 5 kernels")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    out = REPO_ROOT / "plots" / "roofline.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")

    (REPO_ROOT / "bench" / "results" / "roofline.json").write_text(
        json.dumps({"peak_flops": PEAK_FLOPS, "peak_bw": peak_bw, "ridge": ridge, "points": pts}, indent=2)
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
