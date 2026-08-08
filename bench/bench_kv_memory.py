"""KV-cache memory benchmark: theoretical formula vs static/dynamic peak (fast).

Only cached generations are run (each ~4.5 ms/token), so the whole sweep is seconds,
not minutes. For each sequence length we record:
  * theoretical KV bytes from the first-principles formula,
  * peak CUDA memory attributable to a STATIC cache during generation,
  * peak CUDA memory attributable to a DYNAMIC cache during generation.

The static cache pre-allocates once (peak ≈ formula). The dynamic cache grows by
torch.cat, so its transient peak is higher (old + new buffer coexist during each copy)
and it fragments — the static-vs-dynamic gap is the cost of that growth strategy.

Usage:
    python bench/bench_kv_memory.py --model gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_kv_cache import _bench_model  # noqa: E402
from engine.config import get_config  # noqa: E402
from engine.kv_cache import DynamicKVCache, StaticKVCache, kv_cache_bytes  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import load_weights  # noqa: E402

SEQ_LENS = [128, 512, 1024, 2048]
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)
MB = 1024 * 1024


def run(model_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    config = get_config(model_name)
    weights_path = REPO_ROOT / "weights" / model_name / "model.safetensors"
    base = GPT2Model(load_weights(weights_path, config, device=device, dtype=dtype), config)
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    rows = []
    for seq_len in SEQ_LENS:
        new_tokens = seq_len - 1
        bmodel, bcfg = _bench_model(base, config, seq_len, device, dtype)
        theo = kv_cache_bytes(bcfg, seq_len=seq_len, dtype=dtype, batch=1)

        def peak_during(make_cache) -> int:
            if device != "cuda":
                return 0
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            cache = make_cache()
            bmodel.generate_cached(prompt, new_tokens, cache, GREEDY)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base_mem
            del cache
            return peak

        static_peak = peak_during(lambda: StaticKVCache(bcfg, 1, seq_len, device, dtype))
        dynamic_peak = peak_during(lambda: DynamicKVCache(bcfg, 1, device, dtype))
        rows.append({
            "seq_len": seq_len,
            "theoretical_kv_bytes": theo,
            "static_peak_bytes": static_peak,
            "dynamic_peak_bytes": dynamic_peak,
        })
        print(
            f"seq={seq_len:>4} | theoretical {theo/MB:7.2f} MiB | "
            f"static peak {static_peak/MB:7.2f} MiB | dynamic peak {dynamic_peak/MB:7.2f} MiB",
            flush=True,
        )
    return {"model": model_name, "device": device, "rows": rows}


def make_plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["rows"]
    seqs = [r["seq_len"] for r in rows]
    theo = [r["theoretical_kv_bytes"] / MB for r in rows]
    stat = [r["static_peak_bytes"] / MB for r in rows]
    dyn = [r["dynamic_peak_bytes"] / MB for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(seqs, theo, "s--", color="black", label="theoretical KV (formula)")
    ax.plot(seqs, stat, "o-", color="steelblue", label="static cache peak")
    ax.plot(seqs, dyn, "o-", color="darkorange", label="dynamic cache peak")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("memory (MiB)")
    ax.set_title(f"{results['model']}: KV-cache memory grows linearly in sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPO_ROOT / "plots" / "kv_cache_memory.png"
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
    (out_dir / f"kv_memory_{args.model}.json").write_text(json.dumps(results, indent=2))
    make_plot(results)
