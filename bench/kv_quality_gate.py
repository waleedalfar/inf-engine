"""Quality gate for quantizing the TARGET model's KV cache (plan item A1).

Draft-side KV quantization is free: the draft only proposes, and the target's
accept/reject step still yields the target's exact output distribution. Measured
acceptance was unchanged (66.7% vs 65.6%).

Target-side is a different decision — it changes the model's own distribution —
so it needs evidence. It is also close to mandatory at long context: the target's
bf16 KV is 4.6 GB at 30k tokens, which is what pushed the 32K benchmark row to
85% of VRAM and made its timing meaningless.

Three measurements, increasing in strictness:

  1. **Perplexity** on held-out text — the standard scalar for "did the model get
     worse". Teacher-forced, so per-step error is not compounded by divergence.
  2. **Top-1 agreement** over the same teacher-forced positions — how often the
     argmax matches when both are fed identical context.
  3. **Greedy divergence point** — where a free-running generation first picks a
     different token. Divergence at token 3 and at token 180 mean very different
     things, and divergence inside the first few tokens usually means a bug
     rather than quantization error (0.66% injected KV noise leaves argmax
     agreement at 100%; even 10% leaves it at 78%).

Usage:
    python -m bench.kv_quality_gate --model-dir weights/Qwen--Qwen3-8B --ctx 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Perplexity within this ratio of bf16 counts as no meaningful regression.
PPL_TOLERANCE = 1.02
# Top-1 agreement below this is a real behavioural change, not rounding.
TOP1_FLOOR = 0.95


def _corpus(tokenizer, need: int) -> list[int]:
    text = ""
    for pattern in ("README.md", "docs/*.md", "engine/**/*.py"):
        for f in sorted(Path(".").glob(pattern)):
            text += f.read_text(errors="ignore") + "\n\n"
    ids: list[int] = []
    while len(ids) < need:
        ids += tokenizer.encode(text, add_special_tokens=True)
    return ids


def _greedy(model, cfg, ids, kv_dtype, n_new):
    """Free-running greedy continuation, plus peak VRAM for the run."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.sampling import SamplingConfig, SamplingMode

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n_blocks = (len(ids) + n_new + 64) // 16 + 64
    eng = LlamaPagedEngine(
        model, n_total_blocks=n_blocks, block_size=16, eos_token=None,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
        enable_cuda_graphs=True, kv_dtype=kv_dtype,
    )
    req = LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=n_new)
    eng.run_offline([req])
    peak = torch.cuda.max_memory_allocated() / 1e9
    del eng
    torch.cuda.empty_cache()
    return req.generated, peak


@torch.no_grad()
def _teacher_forced(model, cfg, ids, kv_dtype, chunk=512):
    """Perplexity and per-position argmax over ``ids``, teacher-forced.

    Returns:
        ``(perplexity, argmax_ids)`` — argmax at every predicted position, so two
        runs can be compared without either being allowed to diverge.
    """
    from engine.paged_cache import BlockManager, PagedLlamaKVCache

    n_blocks = len(ids) // 16 + 16
    cache = PagedLlamaKVCache(cfg, BlockManager(n_blocks, 16), "cuda",
                              torch.bfloat16, kv_dtype=kv_dtype)
    cache.allocate_sequence(0, len(ids))

    total_nll, count = 0.0, 0
    argmaxes: list[int] = []
    for start in range(0, len(ids) - 1, chunk):
        piece = ids[start:start + chunk]
        cache.ensure_slots_for(0, len(piece))
        cache.begin_step([0])
        x = torch.tensor([piece], device="cuda")
        pos = torch.arange(start, start + len(piece), device="cuda")
        logits = model.forward(x, cache=cache, start_pos=start, position_ids=pos)
        tgt = ids[start + 1:start + len(piece) + 1]
        n = len(tgt)
        row = logits[0, :n].float()
        argmaxes += row.argmax(dim=-1).tolist()
        lp = torch.log_softmax(row, dim=-1)
        total_nll += -lp[torch.arange(n, device="cuda"),
                         torch.tensor(tgt, device="cuda")].sum().item()
        count += n
        del logits, row, lp
    torch.cuda.empty_cache()
    ppl = float(torch.exp(torch.tensor(total_nll / max(count, 1))))
    return ppl, argmaxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ctx", type=int, default=8192,
                    help="Prompt length for the free-running greedy comparison.")
    ap.add_argument("--n-new", type=int, default=200)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    args = ap.parse_args()

    import main as cli
    from engine.qwen_tokenizer import QwenTokenizer

    tokenizer = QwenTokenizer(args.model_dir)
    cfg = cli.detect_config(args.model_dir)
    model = cli.load_model(args.model_dir, cfg, "cuda", torch.bfloat16,
                           quantize=True, quantize_lm_head=True)

    ids = _corpus(tokenizer, max(args.ctx, args.ppl_tokens) + 16)

    print(f"\ntarget-KV quality gate — ctx={args.ctx}, {args.n_new} generated, "
          f"ppl over {args.ppl_tokens} tokens\n")

    out = {}
    for label, kv in (("bf16", None), ("int8", torch.int8)):
        gen, peak = _greedy(model, cfg, ids[:args.ctx], kv, args.n_new)
        ppl, am = _teacher_forced(model, cfg, ids[:args.ppl_tokens], kv)
        out[label] = (gen, peak, ppl, am)
        print(f"  {label:<5} ppl={ppl:8.4f}   peak={peak:5.2f} GB")

    g_bf, peak_bf, ppl_bf, am_bf = out["bf16"]
    g_i8, peak_i8, ppl_i8, am_i8 = out["int8"]

    first = next((i for i, (a, b) in enumerate(zip(g_bf, g_i8)) if a != b), None)
    top1 = sum(a == b for a, b in zip(am_bf, am_i8)) / max(len(am_bf), 1)
    ppl_ratio = ppl_i8 / ppl_bf

    print(f"\n  perplexity ratio        : {ppl_ratio:.4f}   (tolerance {PPL_TOLERANCE})")
    print(f"  top-1 agreement         : {top1:.2%}   (floor {TOP1_FLOOR:.0%})")
    print(f"  greedy diverges at token: {first if first is not None else 'never'}"
          f" of {len(g_bf)}")
    print(f"  VRAM                    : {peak_bf:.2f} -> {peak_i8:.2f} GB "
          f"({peak_bf - peak_i8:+.2f})")

    ppl_ok = ppl_ratio <= PPL_TOLERANCE
    top1_ok = top1 >= TOP1_FLOOR
    verdict = "PASS" if (ppl_ok and top1_ok) else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    if not ppl_ok:
        print(f"    perplexity {ppl_ratio:.3f}x exceeds {PPL_TOLERANCE}")
    if not top1_ok:
        print(f"    top-1 agreement {top1:.1%} below {TOP1_FLOOR:.0%}")
    if first is not None and first < 5:
        print("    NOTE: diverges within the first few tokens. That pattern is a bug"
              " signature, not quantization error — investigate before trusting ppl.")


if __name__ == "__main__":
    main()
