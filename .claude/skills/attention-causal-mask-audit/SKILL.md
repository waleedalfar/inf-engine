---
name: attention-causal-mask-audit
description: Use when a KV-cached transformer (this engine's LLaMA/Qwen3 attention, or any similar hand-rolled attention) produces coherent output on the first token then degenerates into garbage/repetition, when speculative-decoding accept rate is near 0%, or when adding/changing any `is_causal=True` call site near a KV cache. Also use BEFORE adding or strengthening any custom sampling-side penalty (repetition penalty, frequency penalty, or similar) to fight a degenerate-generation symptom — read the "diagnose before patching" section first. Diagnoses and prevents the PyTorch SDPA "upper-left causal mask" pitfall, and documents why a custom exponential repetition penalty was tried and reverted in favor of the standard flat/windowed approach.
---

# Attention causal-mask audit

## The bug class

`torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`
always applies an **"upper-left" causal mask**: `mask[i][j] = (j <= i)`, indexed
by the *local* query/key row and column — NOT by absolute sequence position.

This is only correct when `T_q == T_total` (query 0 really is absolute
position 0). It silently breaks whenever a KV cache holds more history than
the current query block:

- **Decode step** (`T_q=1`, cache already holds `T_total` prior tokens):
  `mask[0][j] = (j <= 0)` → the single query can attend to **key position 0
  only**. The entire KV cache (everything except the very first cached
  token) is masked out.
- **Speculative-decode verify phase** (`T_q=K+1` queries starting at
  `start_pos > 0`): query row `i` (absolute position `start_pos+i`) is
  restricted to keys `0..i`, instead of `0..start_pos+i`. Almost all of the
  prompt/context is invisible to every verified token.

Both cases fail *silently* — no shape error, no NaN, no exception. The
forward pass runs and returns plausible-looking logits, just computed from
nearly no context.

## Symptoms that point here

- First generated token (from a full, uncached prefill) is coherent; every
  token after it is garbage — mixed languages, LaTeX fragments, code tokens,
  or a single repeated token (e.g. endless `)`).
- Speculative decoding acceptance rate stuck near 0% (draft and target
  models never agree, because the target's verify-phase logits are computed
  from almost no context).
- A cache-free reference path (`model.generate()`, O(T²), no KV cache)
  produces correct output while the cached path (`generate_cached()`,
  `forward(..., cache=...)`) diverges after a token or two.
- Any test comparing cached vs. uncached generation "diverges at token 2" or
  similar — that's the first decode step hitting the bug.

Repetition-penalty / sampling-mode bugs can *look* similar (loops, garbage)
but are distinguishable: they degrade gracefully with temperature/penalty
tuning. This bug does not — turning up penalty just produces *different*
garbage, because the underlying logits are wrong, not just over-peaked.

## Diagnosis: isolate the kernel, don't debug the whole pipeline

Don't try to find this by staring at the model forward pass. Write a
standalone SDPA probe using an identity-like value tensor so the output
directly reveals which key positions were attended:

```python
import torch, torch.nn.functional as F
q = torch.randn(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)   # 1 query
k = torch.randn(1, 1, 5, 64, device="cuda", dtype=torch.bfloat16)   # 5 cached keys
v = torch.eye(5, 64, device="cuda", dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out[0, 0, 0, :5])   # [1.0, 0.0, 0.0, 0.0, 0.0] -> BUG: only key 0 visible
```

If the output is one-hot on key 0 instead of a weighted blend of all 5 keys,
`is_causal=True` is masking almost everything.

## The fix

Branch on `T_q` and `start_pos`, not just "is this decode or prefill":

```python
if attn_mask is not None:
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask[:, None].bool())
elif T_q == 1:
    # Decode: cache already enforces causal order — every cached key is valid.
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
elif start_pos == 0:
    # Full prefill, no cached prefix: standard lower-triangular is correct.
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
else:
    # Multi-token forward with an existing cache prefix (e.g. speculative
    # verify): build an explicit offset mask, mask[i][j] = j <= i + start_pos.
    rows = torch.arange(T_q, device=q.device)
    cols = torch.arange(T_total, device=q.device)
    mask = cols[None, :] <= (rows[:, None] + start_pos)
    bias = torch.zeros(1, 1, T_q, T_total, dtype=q.dtype, device=q.device)
    bias.masked_fill_(~mask[None, None], float("-inf"))
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=bias)
```

Reference implementation: `engine/llama_attention.py`.

## Audit checklist when touching attention/KV-cache code

- [ ] Every `is_causal=True` call site: confirm `T_q == T_total` (no cached
      prefix) at that call. If not, it needs the offset-mask branch above.
- [ ] Every decode-step call (`T_q == 1`) with a cache: confirm it does NOT
      pass `is_causal=True`.
- [ ] Any new attention variant (sliding window, chunked prefill, continuous
      batching `attn_mask`) — same rule applies to whatever masking utility
      it uses, not just this SDPA call.
- [ ] After changing masking logic, run the cached-vs-uncached divergence
      test (`generate()` vs `generate_cached()` on the same prompt, compare
      token-by-token) rather than only checking that output "looks fine" —
      this bug produces fluent-looking prose for the first 1-2 tokens.
- [ ] For speculative decoding specifically, watch the acceptance-rate stat
      after any attention change — a healthy run should be 40-90%+ on
      typical text; near-0% is a strong signal this bug (or the repetition
      penalty being unwired) has regressed.

---

## Related lesson: don't patch degenerate output with a custom penalty — diagnose first

This attention bug was originally masked by a chain of sampling-side fixes
that made the symptom *look* different without curing it, and one of those
fixes (an exponential frequency-weighted repetition penalty) became a
*second*, independent bug once the real cause was found. Full incident
narrative (private, not for the public repo): `docs/lessons/2026-08-20-garbage-output-and-repetition-penalty.md`.

**The mistake, concretely:** a repetition loop (endless `)` tokens, 0% spec
accept) was diagnosed as "the penalty isn't strong enough" and answered by
inventing a custom formula, `effective_penalty = penalty ** count`
(unbounded exponential). This didn't fix the loop — the attention bug did,
once found — but the exponential penalty stuck around as the default and
later broke long code generations on its own: tokens that are
*legitimately* frequent in code (newline, indentation, `:`, `(`, `)`,
common identifiers) accumulate count over a long generation, and
`1.1**count` blows up (`1.1^50 ≈ 117×` suppression on an totally ordinary
token), distorting the whole sampling distribution and producing malformed
syntax, literal `\r` in place of `\n`, or spam loops of fabricated content.
A first attempt to cap the exponent (`penalty ** min(count, 4)`) was still
measured to reproduce the failure — a capped exponential was still the
wrong tool, not just wrongly tuned.

**Current, correct implementation** (`engine/sampling.py`,
`_apply_repetition_penalty`): flat, one-time penalty per unique token
present in the last `_REPETITION_PENALTY_WINDOW` (256) generated tokens —
i.e. exactly llama.cpp's `repeat_penalty` + `repeat_last_n` semantics, and
equivalent to HuggingFace `transformers`' `repetition_penalty`. Not scaled
by occurrence count. If you are ever tempted to make this penalty stronger
or count-weighted again, that is a signal to re-read this section first.

### Standing principle: prefer the industry-standard mechanism; treat "needs a custom formula" as a diagnostic red flag

Before adding or strengthening *any* custom sampling-side workaround
(repetition penalty, frequency penalty, presence penalty, custom logit
bias, etc.) to suppress degenerate generation:

1. **Confirm the standard mechanism is even correctly wired first.**
   Bypassed code paths (penalty not passed into the speculative-decode
   path) and wrong default configs (sampling mode left at `GREEDY` so
   `temperature`/`top_p` are silently inert) are common, cheap to check,
   and were both real bugs in this incident — found and fixed *before* the
   exponential-penalty detour, but after it had already been tried.
2. **If the standard, correctly-wired mechanism still shows severe,
   near-total degeneration** (thousands of identical tokens, ~0% spec
   accept, not just "generation is a bit repetitive") — suspect a
   correctness bug in the forward pass (attention masking, KV cache,
   position ids) before reaching for a non-standard, more aggressive
   sampling formula. Genuine model repetition tendencies are soft/gradual;
   near-total lock-in on a single token is a strong signal that the model
   is reasoning from wrong/missing context, not merely under-penalized.
   Isolate with a minimal kernel-level probe (see the SDPA identity-value
   test above) rather than debugging through the full agent/CLI pipeline.
3. **Default to the well-tested industry formula** (flat penalty + recency
   window, standard top-k/top-p, standard causal masking) over inventing
   one. Production inference engines (llama.cpp, vLLM, HF `transformers`)
   do not need exponential frequency-weighted penalties for ordinary
   chat/code workloads — if a custom formula seems necessary to fix
   something, treat that as a prompt to re-check step 2, not as license to
   ship the custom formula.
4. **Validate fixes against a harder case, not just the original repro.**
   The original `)`-loop repro (a short prompt) no longer showed the
   problem after the attention fix, and would have looked "fixed" — the
   exponential-penalty regression only appeared on longer, more complex
   generations (a 700+ token multi-file code request, an O(1) LRU cache
   implementation). Always retest with a harder prompt before declaring a
   generation-quality fix done.
