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
import hashlib
import json
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


# Prompt source: the repo's own prose and source. Prose alone was 11,829 tokens,
# so a 24k prompt repeated it twice and the draft was predicting text it had
# already seen — acceptance hit 98.8%, which flatters tok/s badly at exactly the
# lengths this benchmark exists to measure. Prose + source is ~168k tokens, long
# enough for 64k with nothing repeated.
_CORPUS_GLOBS = ("README.md", "docs/*.md", "engine/**/*.py", "tests/*.py")

# Encoded corpus is snapshotted here on first use. Building it from live files
# every run made the benchmark change whenever the repo did: acceptance moved
# 63% -> 74% at 4K across two runs with no engine change. Delete this file to
# re-snapshot after deliberately changing the corpus.
_CORPUS_CACHE = Path("bench/.corpus_cache.json")


def _corpus_ids(tokenizer, need: int) -> tuple[list[int], str]:
    """Prompt tokens to slice from, plus a fingerprint identifying them.

    Returns:
        ``(ids, digest)``. Acceptance rate and tok/s are comparable across runs
        only when the digest matches; ms/step is unaffected either way.
    """
    ids: list[int] = []
    if _CORPUS_CACHE.exists():
        ids = json.loads(_CORPUS_CACHE.read_text())["ids"]

    if len(ids) < need:
        text = ""
        for pattern in _CORPUS_GLOBS:
            for f in sorted(Path(".").glob(pattern)):
                text += f.read_text(errors="ignore") + "\n\n"
        if not text:
            raise SystemExit(f"no corpus files matched {_CORPUS_GLOBS}")
        ids = []
        while len(ids) < need:
            ids += tokenizer.encode(text, add_special_tokens=True)
        _CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CORPUS_CACHE.write_text(json.dumps({"ids": ids}))

    digest = hashlib.sha256(",".join(map(str, ids[:need])).encode()).hexdigest()[:12]
    return ids, digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--draft-model-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--n-draft", type=int, nargs="+", default=[4],
                    help="One value, or several to sweep at each context length.")
    ap.add_argument("--slices", type=int, default=1,
                    help="Distinct prompt slices to measure per point, taken at "
                         "evenly spaced offsets through the corpus. Acceptance "
                         "varies a lot with content — the repo's Python source "
                         "yields 96%% where prose yields 80%% — so one slice is "
                         "one sample, not the throughput at that context length. "
                         "Every slice is reported; nothing is averaged away.")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[256, 1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--draft-kv-int8", action="store_true",
                    help="Store the draft's KV cache in INT8 — halves the traffic "
                         "term that dominates long context. Safe: the draft only "
                         "proposes; accept/reject still yields the target's exact "
                         "distribution.")
    ap.add_argument("--draft-window", type=int, default=0,
                    help="Sliding-window attention for the DRAFT only (0 = full "
                         "context). The draft proposes and the target verifies, so "
                         "accept/reject still yields the target's exact "
                         "distribution — this costs acceptance rate, never "
                         "correctness. At 32k the draft is 65%% of per-step KV "
                         "traffic despite being a 0.6B model.")
    ap.add_argument("--target-kv-int8", action="store_true",
                    help="Also store the TARGET's KV in INT8. This changes the "
                         "model's own output distribution — quality-gate it.")
    ap.add_argument("--n-ctx", type=int, default=0,
                    help="Override both models' n_ctx (0 = use config default). "
                         "Raise past 32768 to reach 64K; needs YaRN below.")
    ap.add_argument("--rope-scaling-factor", type=float, default=1.0,
                    help="YaRN extension ratio (4.0 to run a 32768-trained model "
                         "at 131072). 1.0 disables scaling.")
    ap.add_argument("--rope-original-n-ctx", type=int, default=0,
                    help="Context the model was trained for; required when "
                         "--rope-scaling-factor > 1 (32768 for Qwen3).")
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

    import dataclasses

    def _apply_ctx(cfg):
        changes = {}
        if args.n_ctx:
            changes["n_ctx"] = args.n_ctx
        if args.rope_scaling_factor != 1.0:
            changes["rope_scaling_factor"] = args.rope_scaling_factor
            changes["rope_original_n_ctx"] = args.rope_original_n_ctx or 32768
        return dataclasses.replace(cfg, **changes) if changes else cfg

    tokenizer = QwenTokenizer(args.model_dir)
    tcfg = _apply_ctx(cli.detect_config(args.model_dir))
    dcfg = _apply_ctx(cli.detect_config(args.draft_model_dir))
    print(f"target n_ctx={tcfg.n_ctx}  draft n_ctx={dcfg.n_ctx}  "
          f"rope_scale={tcfg.rope_scaling_factor}")

    target = cli.load_model(args.model_dir, tcfg, "cuda", torch.bfloat16,
                            quantize=True, quantize_lm_head=True)
    draft = cli.load_model(args.draft_model_dir, dcfg, "cuda", torch.bfloat16,
                           quantize=False)

    N = args.max_new_tokens
    all_ids, corpus_digest = _corpus_ids(tokenizer, max(args.lengths) + 16)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    print(f"\ncorpus {corpus_digest} (acceptance is comparable across runs only "
          f"when this matches)")
    print(f"generating {N} tokens per point, "
          f"draft KV={'int8' if d_kv else 'bf16'}, target KV={'int8' if t_kv else 'bf16'}, "
          f"draft window={args.draft_window or 'full'}")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{'prompt':>8}{'end ctx':>9}{'K':>4}{'sl':>4}{'prefill ms':>12}{'decode ms/step':>16}"
          f"{'decode tok/s':>14}{'e2e tok/s':>11}{'accept':>8}{'tok/step':>10}"
          f"{'peak GB':>9}{'target':>12}")
    print("-" * 112)

    points = [(p, k, i) for p in args.lengths for k in args.n_draft
              for i in range(args.slices)]
    for plen, n_draft, slice_i in points:
        if plen + N + 64 > tcfg.n_ctx:
            print(f"{plen:>8}   skipped — exceeds n_ctx={tcfg.n_ctx} "
                  f"(needs RoPE scaling; see CLAUDE.md)")
            continue
        # Evenly spaced offsets so slices sample different material.
        span = max(len(all_ids) - plen - 1, 1)
        offset = (span // max(args.slices, 1)) * slice_i
        ids = all_ids[offset:offset + plen]
        n_blocks = (plen + N + n_draft + 64) // 16 + 64

        def build():
            t = LlamaPagedEngine(target, n_total_blocks=n_blocks, block_size=16,
                                 eos_token=None, sampling=greedy,
                                 enable_cuda_graphs=True, kv_dtype=t_kv)
            d = LlamaPagedEngine(draft, n_total_blocks=n_blocks, block_size=16,
                                 eos_token=None, sampling=greedy,
                                 enable_cuda_graphs=True, kv_dtype=d_kv,
                                 attn_window=args.draft_window)
            return SpeculativePagedEngine(t, d, n_draft=n_draft, eos_token=None)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        eng = build()
        # Warm up so graph capture is not billed to the measured run.
        eng.run_offline([LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=N)])

        # Split prefill from decode WITHIN the measured run, by stamping the
        # wall clock at the first decode op (the first draft step of the spec
        # loop, which runs only after both prefills finish). A separately-timed
        # prefill pass — the previous approach — reads ~2.6 s low at 32K because
        # its KV blocks are already allocated and its allocator is warm, while
        # the real prefill inside run_offline pays those costs fresh. At 32K that
        # 2.6 s error, divided across ~76 decode steps, inflated the reported
        # decode ms/step from a true 29 to 63 — i.e. it manufactured the entire
        # "32K throughput gap". Measuring the run's own prefill removes it.
        decode_start = {"t": None}
        orig_draft_step = eng.draft._step_one_graphed

        def _stamped_draft_step(*a, **k):
            if decode_start["t"] is None:
                torch.cuda.synchronize()
                decode_start["t"] = time.perf_counter()
            return orig_draft_step(*a, **k)

        eng.draft._step_one_graphed = _stamped_draft_step

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res, st = eng.run_offline([LlamaRequest(req_id=1, prompt_ids=ids, max_new_tokens=N)])
        torch.cuda.synchronize()
        end = time.perf_counter()
        eng.draft._step_one_graphed = orig_draft_step

        e2e = end - t0
        # First draft step is the true decode start; everything before it is prefill.
        prefill_ms = (decode_start["t"] - t0) * 1000 if decode_start["t"] else 0.0
        n = len(res[1])
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        decode_s = max(end - decode_start["t"], 1e-6) if decode_start["t"] else max(e2e, 1e-6)
        tgt = _nearest_target(plen + n)
        tgt_s = f"{tgt[0]}-{tgt[1]}" if tgt else "-"
        print(f"{plen:>8}{plen + n:>9}{n_draft:>4}{slice_i:>4}{prefill_ms:>12.0f}"
              f"{decode_s / st.n_steps * 1000:>16.2f}{n / decode_s:>14.1f}"
              f"{n / e2e:>11.1f}{st.acceptance_rate:>8.1%}{st.tokens_per_step:>10.2f}"
              f"{peak_gb:>9.1f}{tgt_s:>12}")
        if peak_gb > 0.80 * total_gb:
            print(f"{'':>8}  ^ peak is {peak_gb / total_gb:.0%} of the {total_gb:.1f} GB card — "
                  "timings past ~80% are allocator thrash, not engine cost")
        if st.acceptance_rate > 0.95:
            print(f"{'':>8}  ^ acceptance {st.acceptance_rate:.0%} is implausibly high — the "
                  "prompt is probably repeating corpus the draft has already seen. "
                  "tok/s is inflated; compare ms/step instead")

        del eng
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
