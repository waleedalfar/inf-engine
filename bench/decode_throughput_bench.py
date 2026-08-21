"""Decode throughput benchmark for main.py's actual interactive path:
LlamaStaticKVCache + plain eager model.forward, mirroring
engine.agent.AgentLoop._generate_to_eos exactly (no CUDA graphs, no
attn_mask — this is the T_q==1/is_causal=False SDPA branch). This is the
code path that produced the original 42.06 tok/s baseline, and the one the
GQA repeat_kv fix (enable_gqa=True on the unmasked branches) targets.

Usage:
    python -m bench.decode_throughput_bench --model-dir weights/Qwen--Qwen3-8B
"""

from __future__ import annotations

import argparse
import random
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-len", type=int, default=777)
    p.add_argument("--decode-tokens", type=int, default=200)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--quantize-lm-head", action="store_true")
    args = p.parse_args()

    import main as cli
    from engine.kv_cache import LlamaStaticKVCache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Requires CUDA.")

    dtype = torch.bfloat16
    config = cli.detect_config(args.model_dir)
    model = cli.load_model(args.model_dir, config, device, dtype, quantize=not args.no_quantize,
                            quantize_lm_head=args.quantize_lm_head)

    rng = random.Random(0)
    prompt_ids = [rng.randrange(0, config.vocab_size) for _ in range(args.prompt_len)]
    max_seq = args.prompt_len + args.decode_tokens + 8
    cache = LlamaStaticKVCache(config, batch=1, max_seq=max_seq, device=device, dtype=dtype)

    ids_t = torch.tensor([prompt_ids], device=device)
    pos_t = torch.arange(args.prompt_len, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model.forward(ids_t, cache=cache, start_pos=0, position_ids=pos_t)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    next_tok = int(logits[:, -1, :].argmax(dim=-1))
    pos = args.prompt_len

    # Warmup step (JIT/autotune settle) — excluded from steady-state timing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tok_t = torch.tensor([[next_tok]], device=device)
    pos_s = torch.tensor([pos], device=device)
    logits = model.forward(tok_t, cache=cache, start_pos=pos, position_ids=pos_s)
    next_tok = int(logits[:, -1, :].argmax(dim=-1))
    pos += 1
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    n = args.decode_tokens - 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        tok_t = torch.tensor([[next_tok]], device=device)
        pos_s = torch.tensor([pos], device=device)
        logits = model.forward(tok_t, cache=cache, start_pos=pos, position_ids=pos_s)
        next_tok = int(logits[:, -1, :].argmax(dim=-1))
        pos += 1
    torch.cuda.synchronize()
    steady_s = time.perf_counter() - t0

    tok_per_s = n / steady_s
    print(f"\n=== {config.name}  prompt_len={args.prompt_len}  decode_tokens={args.decode_tokens} "
          f"(LlamaStaticKVCache, eager, no mask) ===")
    print(f"prefill={prefill_s*1000:.1f}ms  warmup_tok={warmup_s*1000:.1f}ms  "
          f"steady={n} tok in {steady_s*1000:.1f}ms  => {tok_per_s:.2f} tok/s  "
          f"({steady_s/n*1000:.2f} ms/tok)")


if __name__ == "__main__":
    main()
