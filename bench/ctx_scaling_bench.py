"""Speculative decode throughput as a function of context length.

The project goal is long-context throughput (see CLAUDE.md), so this — not
``paged_spec_bench`` — is the benchmark that matters. It reports prefill and
decode separately, because they answer different questions:

* **prefill** is a one-time cost paid before the first token (TTFT).
* **decode tok/s** is the steady-state rate the target ladder is stated in.
* **end-to-end** folds prefill in, and therefore depends on how many tokens you
  asked for. It is reported for honesty, not for comparison against targets.

Prompts are real prose sliced to length. Random token ids would depress the
draft's acceptance rate and confound throughput with a sampling artifact.

Usage:
    python -m bench.ctx_scaling_bench \\
        --model-dir weights/Qwen--Qwen3-8B \\
        --draft-model-dir weights/Qwen--Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

# Committed targets from CLAUDE.md, for at-a-glance comparison.
TARGETS = {4096: (90, 120), 16384: (80, 90), 32768: (55, 65), 65536: (40, 42)}


def _nearest_target(ctx: int):
    best = min(TARGETS, key=lambda t: abs(t - ctx))
    return TARGETS[best] if abs(best - ctx) <= best * 0.4 else None


def _time_prefill(engine, ids: list[int]) -> float:
    """Milliseconds for one prefill of ``ids``, reusing ``engine``'s KV pool.

    Warms up once, then times a second pass, releasing the sequence each time so
    the pool is left exactly as it was found.
    """
    from engine.llama_paged_engine import LlamaRequest

    def once() -> float:
        sid = engine._next_seq_id
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine._prefill(LlamaRequest(req_id=-99, prompt_ids=ids, max_new_tokens=1), 0.0)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        engine._active.pop(sid, None)
        if sid in engine.cache.seq_lens:
            engine.cache.free_sequence(sid)
        engine._generated.pop(sid, None)
        engine.completed = [r for r in engine.completed if r.req_id != -99]
        return elapsed

    once()
    return once()


def _corpus_ids(tokenizer, need: int) -> list[int]:
    """Real prose, long enough to slice any requested prompt length from."""
    text = ""
    for p in sorted(Path(".").glob("*.md")) + sorted(Path("docs").rglob("*.md")):
        text += p.read_text(errors="ignore") + "\n\n"
    if not text:
        raise SystemExit("no markdown found to build prompts from")
    ids: list[int] = []
    while len(ids) < need:
        ids += tokenizer.encode(text, add_special_tokens=True)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--draft-model-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--n-draft", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[256, 1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--draft-kv-int8", action="store_true",
                    help="Store the draft's KV cache in INT8 — halves the traffic "
                         "term that dominates long context. Safe: the draft only "
                         "proposes; accept/reject still yields the target's exact "
                         "distribution.")
    ap.add_argument("--target-kv-int8", action="store_true",
                    help="Also store the TARGET's KV in INT8. This changes the "
                         "model's own output distribution — quality-gate it.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    import main as cli
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.qwen_tokenizer import QwenTokenizer
    from engine.sampling import SamplingConfig, SamplingMode
    from engine.speculative_paged_engine import SpeculativePagedEngine

    d_kv = torch.int8 if args.draft_kv_int8 else None
    t_kv = torch.int8 if args.target_kv_int8 else None

    tokenizer = QwenTokenizer(args.model_dir)
    tcfg = cli.detect_config(args.model_dir)
    dcfg = cli.detect_config(args.draft_model_dir)
    print(f"target n_ctx={tcfg.n_ctx}  draft n_ctx={dcfg.n_ctx}")

    target = cli.load_model(args.model_dir, tcfg, "cuda", torch.bfloat16,
                            quantize=True, quantize_lm_head=True)
    draft = cli.load_model(args.draft_model_dir, dcfg, "cuda", torch.bfloat16,
                           quantize=False)

    N = args.max_new_tokens
    all_ids = _corpus_ids(tokenizer, max(args.lengths) + 16)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    print(f"\nn_draft={args.n_draft}, generating {N} tokens per point, "
          f"draft KV={'int8' if d_kv else 'bf16'}, target KV={'int8' if t_kv else 'bf16'}")
    print(f"{'prompt':>8}{'end ctx':>9}{'prefill ms':>12}{'decode ms/step':>16}"
          f"{'decode tok/s':>14}{'e2e tok/s':>11}{'accept':>8}{'tok/step':>10}{'target':>12}")
    print("-" * 100)

    for plen in args.lengths:
        if plen + N + 64 > tcfg.n_ctx:
            print(f"{plen:>8}   skipped — exceeds n_ctx={tcfg.n_ctx} "
                  f"(needs RoPE scaling; see CLAUDE.md)")
            continue
        ids = all_ids[:plen]
        n_blocks = (plen + N + args.n_draft + 64) // 16 + 64

        def build():
            t = LlamaPagedEngine(target, n_total_blocks=n_blocks, block_size=16,
                                 eos_token=None, sampling=greedy,
                                 enable_cuda_graphs=True, kv_dtype=t_kv)
            d = LlamaPagedEngine(draft, n_total_blocks=n_blocks, block_size=16,
                                 eos_token=None, sampling=greedy,
                                 enable_cuda_graphs=True, kv_dtype=d_kv)
            return SpeculativePagedEngine(t, d, n_draft=args.n_draft, eos_token=None)

        eng = build()
        # Warm up so graph capture is not billed to the measured run.
        eng.run_offline([LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=N)])

        # Prefill alone, timed on the engine we already built. Standing up a
        # second engine for this allocates another full KV pool — 2.6 GB a copy
        # at 16K — and three copies pushed a 16 GB card into allocator
        # thrashing that read as a 65x decode slowdown, i.e. as an engine
        # regression rather than a harness bug.
        prefill_ms = _time_prefill(eng.target, ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res, st = eng.run_offline([LlamaRequest(req_id=1, prompt_ids=ids, max_new_tokens=N)])
        torch.cuda.synchronize()
        e2e = time.perf_counter() - t0

        n = len(res[1])
        decode_s = max(e2e - prefill_ms / 1000, 1e-6)
        tgt = _nearest_target(plen + n)
        tgt_s = f"{tgt[0]}-{tgt[1]}" if tgt else "-"
        print(f"{plen:>8}{plen + n:>9}{prefill_ms:>12.0f}"
              f"{decode_s / st.n_steps * 1000:>16.2f}{n / decode_s:>14.1f}"
              f"{n / e2e:>11.1f}{st.acceptance_rate:>8.1%}{st.tokens_per_step:>10.2f}{tgt_s:>12}")

        del eng
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
