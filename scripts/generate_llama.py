"""Generate text with LlamaModel.

Usage:
    python scripts/generate_llama.py --prompt "The capital of France is" --n 30
    python scripts/generate_llama.py --model meta-llama/Llama-3.2-3B --mode top_p --top-p 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engine.config import LLAMA_3_2_1B, get_llama_config
from engine.kv_cache import LlamaStaticKVCache
from engine.llama_model import LlamaModel
from engine.llama_weights import load_llama_weights
from engine.sampling import SamplingConfig, SamplingMode

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n", type=int, default=50, help="new tokens to generate")
    p.add_argument("--mode", choices=["greedy", "top_k", "top_p"], default="greedy")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    try:
        config = get_llama_config(args.model)
    except KeyError:
        print(f"Unknown model '{args.model}', using Llama-3.2-1B config as base.")
        config = LLAMA_3_2_1B

    model_dir = WEIGHTS_DIR / args.model.replace("/", "--")
    if not model_dir.is_dir():
        raise SystemExit(
            f"Weights not found at {model_dir}.\n"
            f"Run:  python scripts/download_llama.py {args.model}"
        )

    print(f"Loading {args.model} ...", flush=True)
    weights = load_llama_weights(model_dir, config, device=device, dtype=dtype)
    model = LlamaModel(weights, config)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    prompt_len = input_ids.shape[1]
    max_seq = prompt_len + args.n + 1

    cache = LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=dtype)

    mode_map = {
        "greedy": SamplingMode.GREEDY,
        "top_k": SamplingMode.TOP_K,
        "top_p": SamplingMode.TOP_P,
    }
    gen = torch.manual_seed(args.seed) if args.mode != "greedy" else None
    sampling = SamplingConfig(
        mode=mode_map[args.mode], top_p=args.top_p, top_k=args.top_k, generator=gen
    )

    print(f"Prompt: {args.prompt!r}")
    out_ids = model.generate_cached(input_ids, args.n, cache, sampling)
    new_ids = out_ids[0, prompt_len:].tolist()
    print("Output:", tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
