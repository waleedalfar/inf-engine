"""Speculative decoding throughput benchmark: Qwen3-8B target (INT4) +
Qwen3-0.6B draft, real prompt (acceptance rate is meaningless on random
token ids — the draft model only agrees with the target on natural
continuations). Compares against plain (non-speculative) target-only decode
at the same prompt, same GQA/attention fixes already in the codebase (both
paths route through the same engine.llama_model.LlamaModel.forward).

Usage:
    python -m bench.spec_decode_bench --model-dir weights/Qwen--Qwen3-8B \\
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--draft-model-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--n-draft", type=int, default=4)
    args = p.parse_args()

    import main as cli
    from engine.kv_cache import LlamaStaticKVCache
    from engine.qwen_tokenizer import QwenTokenizer
    from engine.sampling import SamplingConfig, SamplingMode
    from engine.speculative import SpeculativeDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Requires CUDA.")

    dtype = torch.bfloat16
    tokenizer = QwenTokenizer(args.model_dir)
    target_config = cli.detect_config(args.model_dir)
    draft_config = cli.detect_config(args.draft_model_dir)

    target = cli.load_model(args.model_dir, target_config, device, dtype, quantize=True)
    draft = cli.load_model(args.draft_model_dir, draft_config, device, dtype, quantize=False)

    ids = tokenizer.encode(_PROMPT, add_special_tokens=True)
    prompt_len = len(ids)
    ids_t = torch.tensor([ids], device=device)
    max_seq = prompt_len + args.max_new_tokens + args.n_draft + 8

    sampling = SamplingConfig(mode=SamplingMode.GREEDY)

    # --- plain target-only decode (baseline, matches bench/decode_throughput_bench.py) ---
    cache = LlamaStaticKVCache(target_config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    pos = torch.arange(prompt_len, device=device)
    with torch.no_grad():
        logits = target.forward(ids_t, cache=cache, start_pos=0, position_ids=pos)
    next_tok = int(logits[:, -1, :].argmax(dim=-1))
    p_ = prompt_len
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = args.max_new_tokens - 1
    with torch.no_grad():
        for _ in range(n):
            tok_t = torch.tensor([[next_tok]], device=device)
            pos_s = torch.tensor([p_], device=device)
            logits = target.forward(tok_t, cache=cache, start_pos=p_, position_ids=pos_s)
            next_tok = int(logits[:, -1, :].argmax(dim=-1))
            p_ += 1
    torch.cuda.synchronize()
    plain_s = time.perf_counter() - t0
    plain_tok_s = n / plain_s

    # --- speculative decoding ---
    decoder = SpeculativeDecoder(draft=draft, target=target, n_draft=args.n_draft)
    draft_cache = LlamaStaticKVCache(draft_config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    target_cache2 = LlamaStaticKVCache(target_config, batch=1, max_seq=max_seq, device=device, dtype=dtype)

    # Warmup (first call pays any JIT/compile cost).
    with torch.no_grad():
        decoder.generate(ids_t, 8, draft_cache, target_cache2, sampling=sampling,
                          eos_token=tokenizer.eos_token_id)

    draft_cache = LlamaStaticKVCache(draft_config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    target_cache2 = LlamaStaticKVCache(target_config, batch=1, max_seq=max_seq, device=device, dtype=dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out, stats = decoder.generate(
            ids_t, args.max_new_tokens, draft_cache, target_cache2,
            sampling=sampling, eos_token=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    spec_s = time.perf_counter() - t0
    n_generated = out.shape[1] - prompt_len
    spec_tok_s = n_generated / spec_s

    print(f"\n=== {target_config.name} + {draft_config.name} draft, "
          f"prompt_len={prompt_len}, max_new_tokens={args.max_new_tokens}, n_draft={args.n_draft} ===")
    print(f"plain target-only decode:  {plain_tok_s:.2f} tok/s  ({n} tok in {plain_s*1000:.1f}ms)")
    print(f"speculative decode:        {spec_tok_s:.2f} tok/s  ({n_generated} tok in {spec_s*1000:.1f}ms)")
    print(f"acceptance rate: {stats.acceptance_rate*100:.1f}%  "
          f"tokens/step: {stats.tokens_per_step:.2f}  steps: {stats.n_steps}  "
          f"accepted: {stats.n_accepted}  rejected: {stats.n_rejected}  bonus: {stats.n_bonus}")
    print(f"speedup vs plain: {spec_tok_s/plain_tok_s:.2f}x")


if __name__ == "__main__":
    main()
