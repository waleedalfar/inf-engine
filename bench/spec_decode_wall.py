"""Decode-only wall ms/step on the REAL engine, no prefill subtraction.

Wraps the target's verify step to stamp the wall clock on the first decode
step and count steps. Decode wall = (final sync) - (first verify), divided by
steps. This touches neither a reconstruction of the loop nor a separate prefill
timing, so it is immune to the two failure modes that made the slope probe and
ctx_scaling_bench disagree.
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--draft-model-dir", required=True)
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--n-draft", type=int, default=4)
    ap.add_argument("--draft-window", type=int, default=4096)
    ap.add_argument("--draft-ring", action="store_true",
                    help="Bound the draft's KV pool to its window. Here so the "
                         "ring can be cross-checked against ctx_scaling_bench "
                         "with a decode-only wall, rather than trusted from one "
                         "harness.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    import main as cli
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.qwen_tokenizer import QwenTokenizer
    from engine.sampling import SamplingConfig, SamplingMode
    from engine.speculative_paged_engine import SpeculativePagedEngine
    from bench.ctx_scaling_bench import _corpus_ids

    tokenizer = QwenTokenizer(args.model_dir)
    tcfg = cli.detect_config(args.model_dir)
    dcfg = cli.detect_config(args.draft_model_dir)
    target_m = cli.load_model(args.model_dir, tcfg, "cuda", torch.bfloat16,
                              quantize=True, quantize_lm_head=True)
    draft_m = cli.load_model(args.draft_model_dir, dcfg, "cuda", torch.bfloat16,
                             quantize=False)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    plen = args.ctx
    all_ids, digest = _corpus_ids(tokenizer, plen + 16)
    ids = all_ids[:plen]
    n_blocks = (plen + args.max_new_tokens * (args.n_draft + 1) + 64) // 16 + 128

    t = LlamaPagedEngine(target_m, n_total_blocks=n_blocks, block_size=16,
                         eos_token=None, sampling=greedy,
                         enable_cuda_graphs=True, kv_dtype=torch.int8)
    d = LlamaPagedEngine(draft_m, n_total_blocks=n_blocks, block_size=16,
                         eos_token=None, sampling=greedy,
                         enable_cuda_graphs=True, kv_dtype=torch.int8,
                         attn_window=args.draft_window,
                         window_ring=args.draft_ring)
    eng = SpeculativePagedEngine(t, d, n_draft=args.n_draft, eos_token=None)

    # Warm up graph capture.
    eng.run_offline([LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=40)])

    # Reproduce the bench's separate prefill estimate for comparison.
    from bench.ctx_scaling_bench import _time_prefill
    bench_prefill_ms = _time_prefill(t, ids)

    # Instrument: stamp wall at the first verify (decode start) and count verifies.
    state = {"t_first": None, "n": 0}
    orig_verify = t._step_verify_graphed

    def wrapped(seq_id, verify_ids):
        if state["t_first"] is None:
            torch.cuda.synchronize()
            state["t_first"] = time.perf_counter()
        state["n"] += 1
        return orig_verify(seq_id, verify_ids)

    t._step_verify_graphed = wrapped

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    _, st = eng.run_offline([LlamaRequest(req_id=1, prompt_ids=ids,
                                          max_new_tokens=args.max_new_tokens)])
    torch.cuda.synchronize()
    end = time.perf_counter()
    decode_wall = end - state["t_first"]
    true_prefill_ms = (state["t_first"] - t_start) * 1000
    e2e_ms = (end - t_start) * 1000
    bench_decode_ms_step = (e2e_ms - bench_prefill_ms) / st.n_steps

    # n verify calls == number of speculative steps that reached verify.
    steps = state["n"]
    ms_step = decode_wall / steps * 1000
    tok_per_step = st.tokens_per_step
    print(f"corpus {digest}  ctx={plen}  window={args.draft_window}  n_draft={args.n_draft}")
    print(f"  verify-counted steps: {steps}  (stats n_steps={st.n_steps})")
    print(f"  accept={st.acceptance_rate:.1%}  tok/step={tok_per_step:.2f}")
    print(f"  decode wall (first verify -> end): {decode_wall*1000:.1f} ms")
    print(f"  decode ms/step (TRUE): {ms_step:.2f}   tok/s: {tok_per_step/(ms_step/1000):.1f}")
    print()
    print(f"  --- prefill-accounting bug check ---")
    print(f"  true in-run prefill wall:   {true_prefill_ms:8.0f} ms")
    print(f"  bench _time_prefill est.:   {bench_prefill_ms:8.0f} ms  "
          f"(undershoots by {true_prefill_ms - bench_prefill_ms:.0f} ms)")
    print(f"  bench decode ms/step (buggy): {bench_decode_ms_step:.2f}  "
          f"vs TRUE {ms_step:.2f}")


if __name__ == "__main__":
    main()
