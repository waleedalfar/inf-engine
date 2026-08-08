"""Interactive text generation with the from-scratch GPT-2 engine.

Usage:
    python scripts/generate.py --prompt "Once upon a time" --mode greedy --n 40
    python scripts/generate.py --prompt "Once upon a time" --mode top_p --top-p 0.9 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/generate.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import get_config
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, SamplingMode
from engine.tokenizer import GPT2Tokenizer
from engine.weights import load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with from-scratch GPT-2.")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--n", type=int, default=40, help="number of new tokens")
    parser.add_argument("--mode", default="greedy", choices=[m.value for m in SamplingMode])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = get_config(args.model)
    weights_path = REPO_ROOT / "weights" / args.model / "model.safetensors"
    weights = load_weights(weights_path, config, device=args.device, dtype=torch.float32)
    model = GPT2Model(weights, config)
    tok = GPT2Tokenizer()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    sampling = SamplingConfig(
        mode=SamplingMode(args.mode),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        generator=generator,
    )

    ids = torch.tensor([tok.encode(args.prompt)], device=args.device)  # (1, T)
    out = model.generate(ids, max_new_tokens=args.n, sampling=sampling)  # (1, T+n)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
