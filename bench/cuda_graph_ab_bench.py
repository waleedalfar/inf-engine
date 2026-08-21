"""A/B benchmark: eager decode vs CUDA-graph-replayed decode, single active
sequence, real Qwen3-8B INT4 weights.

Motivation: `LlamaPagedEngine`'s CUDA graph decode path (`enable_cuda_graphs`)
is fully implemented (`_capture_graph`/`_decode_step_graphed` in
`engine/llama_paged_engine.py`) but is never actually turned on in either
`main.py` (doesn't use `LlamaPagedEngine` at all) or `server.py` (constructs
`LlamaPagedEngine` without passing `enable_cuda_graphs=True`). The measured
42.06 tok/s baseline came from `main.py`'s fully eager path, which dispatches
roughly 1500-2000 individual CUDA kernel launches per decode token (2 RMSNorm
+ QKVO + gate/up/down projections + attention + RoPE, all unfused, x36
layers). This script isolates that one variable — eager vs graph-replayed
decode — with everything else held fixed (same model, same weights, same
prompt length, same sampling config) to test whether kernel-launch overhead
is really the dominant cost, as hypothesized in the decode-bottleneck
analysis (see memory/project_overview.md "Decode throughput investigation").

Usage:
    python -m bench.cuda_graph_ab_bench --model-dir weights/Qwen--Qwen3-8B
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
from engine.sampling import SamplingConfig, SamplingMode


def _run(
    model,
    n_total_blocks: int,
    block_size: int,
    prompt_len: int,
    decode_tokens: int,
    enable_cuda_graphs: bool,
    vocab_size: int,
    seed: int = 0,
) -> dict:
    engine = LlamaPagedEngine(
        model,
        n_total_blocks=n_total_blocks,
        block_size=block_size,
        eos_token=None,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
        enable_cuda_graphs=enable_cuda_graphs,
    )

    rng = random.Random(seed)
    prompt_ids = [rng.randrange(0, vocab_size) for _ in range(prompt_len)]
    req = LlamaRequest(req_id=0, prompt_ids=prompt_ids, max_new_tokens=decode_tokens)
    engine.submit(req)

    # Step 1: prefill + first decode token (and, if graphed, lazily captures
    # the graph on this same call) — excluded from the steady-state timing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    # Steady-state decode: remaining tokens, timed as a block.
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_done = 1
    while engine.has_work and n_done < decode_tokens:
        engine.step()
        n_done += 1
    torch.cuda.synchronize()
    steady_s = time.perf_counter() - t1

    steady_tokens = n_done - 1
    tok_per_s = steady_tokens / steady_s if steady_s > 0 else float("nan")
    return {
        "warmup_s": warmup_s,
        "steady_s": steady_s,
        "steady_tokens": steady_tokens,
        "tok_per_s": tok_per_s,
        "ms_per_tok": (steady_s / steady_tokens * 1000) if steady_tokens else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-len", type=int, default=777, help="Matches the earlier measured baseline's prompt length.")
    p.add_argument("--decode-tokens", type=int, default=200)
    p.add_argument("--n-total-blocks", type=int, default=256)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--quantize-lm-head", action="store_true")
    args = p.parse_args()

    import main as cli  # reuse detect_config / load_model exactly as main.py does

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("This benchmark requires CUDA — it measures kernel-launch overhead, meaningless on CPU.")

    dtype = torch.bfloat16
    config = cli.detect_config(args.model_dir)
    quantize = not args.no_quantize
    model = cli.load_model(args.model_dir, config, device, dtype, quantize=quantize,
                            quantize_lm_head=args.quantize_lm_head)

    needed_blocks = (args.prompt_len + args.decode_tokens) // args.block_size + 4
    n_total_blocks = max(args.n_total_blocks, needed_blocks)

    print(f"\n=== {config.name}  prompt_len={args.prompt_len}  decode_tokens={args.decode_tokens} ===\n")

    print("--- eager (enable_cuda_graphs=False) ---")
    eager = _run(model, n_total_blocks, args.block_size, args.prompt_len, args.decode_tokens,
                 enable_cuda_graphs=False, vocab_size=config.vocab_size)
    print(f"warmup(prefill+1st tok)={eager['warmup_s']*1000:.1f}ms  "
          f"steady={eager['steady_tokens']} tok in {eager['steady_s']*1000:.1f}ms  "
          f"=> {eager['tok_per_s']:.2f} tok/s  ({eager['ms_per_tok']:.2f} ms/tok)")

    print("\n--- CUDA graph (enable_cuda_graphs=True) ---")
    graphed = _run(model, n_total_blocks, args.block_size, args.prompt_len, args.decode_tokens,
                    enable_cuda_graphs=True, vocab_size=config.vocab_size)
    print(f"warmup(prefill+1st tok, incl. graph capture)={graphed['warmup_s']*1000:.1f}ms  "
          f"steady={graphed['steady_tokens']} tok in {graphed['steady_s']*1000:.1f}ms  "
          f"=> {graphed['tok_per_s']:.2f} tok/s  ({graphed['ms_per_tok']:.2f} ms/tok)")

    speedup = graphed["tok_per_s"] / eager["tok_per_s"] if eager["tok_per_s"] else float("nan")
    print(f"\n=== speedup: {speedup:.2f}x ===")


if __name__ == "__main__":
    main()
