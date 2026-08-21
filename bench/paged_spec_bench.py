"""Paged-engine speculative decoding benchmark.

Compares three paths on the same prompt:
  1. LlamaPagedEngine + CUDA graphs (single-model baseline, ~53 tok/s)
  2. SpeculativePagedEngine: graphed draft + graphed verify (the new path)
  3. Old eager spec decode (SpeculativeDecoder, no graphs) for reference

Usage:
    python -m bench.paged_spec_bench \\
        --model-dir weights/Qwen--Qwen3-8B \\
        --draft-model-dir weights/Qwen--Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time

import torch


_PROMPT = (
    "Write a Python function that takes a list of integers and returns the "
    "second largest unique value. Include a docstring and a few test cases "
    "using assert statements. Explain your approach briefly before the code."
)


def _paged_baseline(model, config, tokenizer, ids, max_new_tokens, device):
    """LlamaPagedEngine + CUDA graphs, target model only."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.sampling import SamplingConfig, SamplingMode

    n_blocks = (len(ids) + max_new_tokens + 16) // 16 + 64
    engine = LlamaPagedEngine(
        model,
        n_total_blocks=n_blocks,
        block_size=16,
        eos_token=tokenizer.eos_token_id,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
        enable_cuda_graphs=True,
    )

    # Warmup: full-length run to capture all graph buckets up to max_new_tokens.
    wup_req = LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=max_new_tokens)
    engine.run_offline([wup_req])
    engine.completed.clear()

    torch.cuda.synchronize()
    req = LlamaRequest(req_id=1, prompt_ids=ids, max_new_tokens=max_new_tokens)
    t0 = time.perf_counter()
    results = engine.run_offline([req])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen = results[1]
    return len(gen), elapsed


def _paged_spec(target_model, draft_model, target_config, draft_config,
                tokenizer, ids, max_new_tokens, n_draft, device):
    """SpeculativePagedEngine: graphed draft + graphed verify."""
    from engine.llama_paged_engine import LlamaPagedEngine, LlamaRequest
    from engine.sampling import SamplingConfig, SamplingMode
    from engine.speculative_paged_engine import SpeculativePagedEngine

    sampling = SamplingConfig(mode=SamplingMode.GREEDY)
    n_blocks = (len(ids) + max_new_tokens + n_draft + 32) // 16 + 64

    def make_engines():
        t_eng = LlamaPagedEngine(
            target_model, n_total_blocks=n_blocks, block_size=16,
            eos_token=tokenizer.eos_token_id, sampling=sampling,
            enable_cuda_graphs=True,
        )
        d_eng = LlamaPagedEngine(
            draft_model, n_total_blocks=n_blocks, block_size=16,
            eos_token=tokenizer.eos_token_id, sampling=sampling,
            enable_cuda_graphs=True,
        )
        return SpeculativePagedEngine(t_eng, d_eng, n_draft=n_draft,
                                      eos_token=tokenizer.eos_token_id)

    # Warmup: full-length run to capture all graph buckets.
    spec_engine = make_engines()
    wup_req = LlamaRequest(req_id=0, prompt_ids=ids, max_new_tokens=max_new_tokens)
    spec_engine.run_offline([wup_req])

    torch.cuda.synchronize()
    req = LlamaRequest(req_id=1, prompt_ids=ids, max_new_tokens=max_new_tokens)
    t0 = time.perf_counter()
    results, stats = spec_engine.run_offline([req])  # all graphs pre-captured
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen = results[1]
    return len(gen), elapsed, stats


def _eager_spec(target_model, draft_model, target_config, draft_config,
                tokenizer, ids, max_new_tokens, n_draft, device):
    """Old SpeculativeDecoder (no CUDA graphs) for reference."""
    from engine.kv_cache import LlamaStaticKVCache
    from engine.sampling import SamplingConfig, SamplingMode
    from engine.speculative import SpeculativeDecoder

    sampling = SamplingConfig(mode=SamplingMode.GREEDY)
    ids_t = torch.tensor([ids], device=device)
    max_seq = len(ids) + max_new_tokens + n_draft + 8

    decoder = SpeculativeDecoder(draft=draft_model, target=target_model, n_draft=n_draft)

    # Warmup
    dc = LlamaStaticKVCache(draft_config, batch=1, max_seq=max_seq, device=device, dtype=torch.bfloat16)
    tc = LlamaStaticKVCache(target_config, batch=1, max_seq=max_seq, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        decoder.generate(ids_t, 10, dc, tc, sampling=sampling, eos_token=tokenizer.eos_token_id)

    dc = LlamaStaticKVCache(draft_config, batch=1, max_seq=max_seq, device=device, dtype=torch.bfloat16)
    tc = LlamaStaticKVCache(target_config, batch=1, max_seq=max_seq, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out, stats = decoder.generate(ids_t, max_new_tokens, dc, tc,
                                      sampling=sampling, eos_token=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_gen = out.shape[1] - len(ids)
    return n_gen, elapsed, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--draft-model-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--n-draft", type=int, default=4)
    p.add_argument("--skip-eager-spec", action="store_true",
                   help="Skip the old eager SpeculativeDecoder reference run")
    args = p.parse_args()

    import main as cli

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Requires CUDA.")

    dtype = torch.bfloat16
    from engine.qwen_tokenizer import QwenTokenizer
    tokenizer = QwenTokenizer(args.model_dir)

    target_config = cli.detect_config(args.model_dir)
    draft_config = cli.detect_config(args.draft_model_dir)

    print(f"Loading target: {target_config.name} (INT4) ...")
    target = cli.load_model(args.model_dir, target_config, device, dtype,
                            quantize=True, quantize_lm_head=True)
    print(f"Loading draft: {draft_config.name} (bf16) ...")
    draft = cli.load_model(args.draft_model_dir, draft_config, device, dtype, quantize=False)

    ids = tokenizer.encode(_PROMPT, add_special_tokens=True)
    prompt_len = len(ids)
    max_new = args.max_new_tokens
    n_draft = args.n_draft

    print(f"\nPrompt: {prompt_len} tokens, max_new={max_new}, n_draft={n_draft}")
    print("=" * 65)

    # 1. Paged baseline (CUDA graphs, target only)
    print("\n[1/3] LlamaPagedEngine + CUDA graphs (target only) ...")
    n1, t1 = _paged_baseline(target, target_config, tokenizer, ids, max_new, device)
    tps1 = n1 / t1
    print(f"      {n1} tokens in {t1*1000:.1f}ms  =>  {tps1:.2f} tok/s")

    # 2. Paged spec (graphed draft + graphed verify)
    print("\n[2/3] SpeculativePagedEngine (graphed draft + graphed verify) ...")
    n2, t2, stats2 = _paged_spec(target, draft, target_config, draft_config,
                                  tokenizer, ids, max_new, n_draft, device)
    tps2 = n2 / t2
    print(f"      {n2} tokens in {t2*1000:.1f}ms  =>  {tps2:.2f} tok/s")
    print(f"      accept={stats2.acceptance_rate:.1%}  tok/step={stats2.tokens_per_step:.2f}  "
          f"steps={stats2.n_steps}  bonus={stats2.n_bonus}")

    # 3. Old eager spec (reference)
    if not args.skip_eager_spec:
        print("\n[3/3] Old SpeculativeDecoder (eager, no graphs) ...")
        n3, t3, stats3 = _eager_spec(target, draft, target_config, draft_config,
                                      tokenizer, ids, max_new, n_draft, device)
        tps3 = n3 / t3
        print(f"      {n3} tokens in {t3*1000:.1f}ms  =>  {tps3:.2f} tok/s")
        print(f"      accept={stats3.acceptance_rate:.1%}  tok/step={stats3.tokens_per_step:.2f}  "
              f"steps={stats3.n_steps}")

    print("\n" + "=" * 65)
    print(f"BASELINE (paged+graphs, no draft):  {tps1:.2f} tok/s")
    print(f"PAGED SPEC (graphed draft+verify):  {tps2:.2f} tok/s  ({tps2/tps1:.2f}x baseline)")
    if not args.skip_eager_spec:
        print(f"EAGER SPEC (old, no graphs):        {tps3:.2f} tok/s  ({tps3/tps1:.2f}x baseline)")
    print()


if __name__ == "__main__":
    main()
