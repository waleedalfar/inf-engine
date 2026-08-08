"""Phase 6: torch.profiler trace of the engine hot path + top bottlenecks.

Profiles cached single-stream decode (the real serving hot path: one token per step over a
KV cache) and prints the ops sorted by total CUDA time. The top entries are the bottlenecks
Phase 6 annotates in DECISIONS.md. A Chrome trace is exported for visual inspection.

Usage:
    python bench/profile_engine.py --model gpt2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.config import get_config  # noqa: E402
from engine.kv_cache import StaticKVCache  # noqa: E402
from engine.model import GPT2Model  # noqa: E402
from engine.sampling import SamplingConfig, SamplingMode  # noqa: E402
from engine.weights import load_weights  # noqa: E402

PROMPT_LEN = 64
NEW_TOKENS = 128
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)


def main(model_name: str) -> None:
    device = "cuda"
    config = get_config(model_name)
    wpath = REPO_ROOT / "weights" / model_name / "model.safetensors"
    model = GPT2Model(load_weights(wpath, config, device=device, dtype=torch.float32), config)
    prompt = torch.zeros((1, PROMPT_LEN), dtype=torch.long, device=device)

    def gen():
        cache = StaticKVCache(config, 1, PROMPT_LEN + NEW_TOKENS, device, torch.float32)
        model.generate_cached(prompt, NEW_TOKENS, cache, GREEDY)

    for _ in range(3):  # warmup (compile/autotune/clocks)
        gen()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        gen()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    out = REPO_ROOT / "bench" / "results" / "engine_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"\nChrome trace: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()
    torch.manual_seed(0)
    main(args.model)
