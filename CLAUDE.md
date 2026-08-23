# Project goal: long-context speculative decode throughput

## THE COMMITTED MILESTONE

This is the plan. Do not descope it, do not substitute an easier target, and do
not stop at the first number in the table.

| context | committed tok/s |
|---|---|
| ~4K  | **90–120+** |
| ~16K | **80–90** |
| ~32K | **55–65** |
| ~64K | **40–42** |

Model pair: Qwen3-8B INT4 (target) + Qwen3-0.6B (draft), n_draft tuned per
context. Hardware: RTX 5070 Ti, 16 GB, 896 GB/s.

**All four rows are committed, not just 4K.** 4K is the only one reachable with
PyTorch SDPA; the rest require the custom kernel described below. It is accepted
that the kernel's final efficiency is not knowable in advance. That uncertainty
is not a reason to renegotiate the targets — it is a reason to build the kernel
and measure. Give it the full attempt.

---

## Progress

| date | change | 4296 ctx | 8392 ctx |
|---|---|---|---|
| baseline | SDPA gather + expand | 16.7 | 10.6 |
| 2026-08-22 | paged flash-decoding kernel | 71.7 | 53.3 |
| 2026-08-22 | + INT8 draft KV | **82.2** | **73.9** |

(decode tok/s, 200 generated; acceptance 63–67% on these prompts, so at the
3.16 tok/step a well-matched prompt gives, 4296 is ≈100.)

Done: the custom kernel (item 3), INT8 KV (item 4). The kernel subsumed items 1
and 2 — it reads the block table directly, so there is no gather to pad and no
expansion to avoid; splits past the true length exit immediately.

Remaining: context-dependent `n_draft` (item 5), 64K enablement (item 6), and
the INT4 matmul, which is now ~69% of the target verify step and is the largest
context-independent cost.

## Why the old numbers don't transfer

Everything before 2026-08-22 optimized a **239-token** context (39-token prompt,
200 generated). At that length the step is dominated by weight streaming, which
is constant in context length. That work is done and it got short-context decode
to ~104 tok/s.

It does not generalize. Measured throughput vs. context (same model pair, 200
generated tokens):

| end ctx | tok/s | ms/step |
|---|---|---|
| 239 | 84.6 | 29.6 |
| 400 | 101.9 | 38.5 |
| 700 | 54.7 | 45.1 |
| 1200 | 37.0 | 62.1 |
| 2200 | 27.8 | 97.1 |
| 4200 | 16.7 | 161.6 |

**Treat any tok/s figure in this repo without a stated context length as
short-context and therefore not evidence about this goal.**

---

## The two facts that drive the design

### 1. The draft model is 77% of per-step KV traffic

Qwen3-0.6B has the **same** 8 KV heads × 128 head_dim as the 8B target, and 28
vs 36 layers — 112 KiB per context-token vs the target's 144 KiB. But it runs
**4.35× per step** (n_draft draft steps + the bonus-sync forward).

At long context the "small" draft model costs over 3× what the 8B target costs.
Any long-context optimization that ignores the draft is optimizing 23% of the
problem.

### 2. SDPA is ~25× off the bandwidth floor at these shapes

At 4200 ctx (8192 bucket), attention + K/V expansion costs **33 ms**. The pure
bandwidth floor for that attention is **~1.3 ms**:

| factor | cause | fix |
|---|---|---|
| ~2× | bucket padding — 8192 gathered for 4200 real | finer `len_bucket` granularity |
| ~4× | `repeat_kv` expands 8 KV heads to 32 | GQA-native attention |
| ~3× | SDPA mem-efficient backend at `q_len=5` | **custom kernel — no SDPA path fixes this** |

PyTorch's flash backend rejects arbitrary `attn_mask`; the mem-efficient backend
accepts it but is poor at 1–5 row queries; and neither reads FP8 K/V, so KV
quantization cannot pay off through SDPA (dequantizing to bf16 before the call
gives back exactly the bandwidth it saved).

---

## The work, in order

### 1-2. Bucket padding and the GQA fold — SUBSUMED by the kernel
Both existed to work around the gather. The kernel reads the block table
directly, so there is no contiguous gather to pad and no expansion to avoid, and
splits past a sequence's true length exit immediately. Neither was implemented;
neither is needed.

### 3. Custom flash-decoding attention kernel — DONE (`engine/kernels/paged_attention.py`)
Triton, in `engine/kernels/`. Requirements:
- **GQA-native**: reads `n_kv` heads directly, never materializes an expansion.
- **Split over the KV axis** (flash-decoding): a 1–5 row query cannot fill 70 SMs
  by itself. Partition KV into chunks, run per-chunk online softmax, combine
  with the standard log-sum-exp merge. This is what makes short queries saturate
  the GPU and is the single most important property of the kernel.
- **FP8 KV capable**: read FP8 K/V from HBM, dequantize in registers. Halves the
  dominant traffic term. Must be a kernel-level feature — see above for why it
  cannot be bolted on outside.
- **Offset-causal masking**: query row `i` attends keys `0..start_pos+i`, plus a
  true-length bound so bucket padding is excluded. Masking is the #1 correctness
  risk here — see the audit skill before writing a line of it.
- **CUDA-graph capturable**: static shapes per bucket, no host syncs.

`engine/kernels/flash_attention.py` is the existing reference for the online
softmax, but it is square-shaped, fp32, non-GQA — it is a teaching implementation,
not a starting point to extend.

### 4. 8-bit KV cache — DONE (`kv_dtype=torch.int8`)
Per-(block, head, position) scales, read natively by the kernel. Landed on the
draft, where it is free: the draft only proposes and accept/reject still yields
the target's exact distribution. Acceptance was unchanged (66.7% vs 65.6%).

**Use INT8, not FP8.** Both are one byte, but per-token amax scaling already
supplies the exponent range FP8 spends bits on, so e4m3's 3 mantissa bits are
strictly worse than INT8's effective 7.

Target-side KV quantization changes the model's own distribution — still a
separate, quality-gated decision (`--target-kv-int8` exists but is ungated).

### 5. Context-dependent `n_draft`
Optimal `n_draft` falls as context grows, because every draft step now pays
attention cost. At 32K, `n_draft=2` models ~10% ahead of `n_draft=4` despite
lower tokens-per-step. Make it a function of context length, not a constant.

### 6. 64K enablement
`n_ctx` is **32768** for both models. 64K needs YaRN/RoPE scaling configured.
Budget: ~9.7 GB KV (FP8 halves it) + ~5.2 GB weights against 16 GB.

---

## Non-negotiable practices

**Every performance claim states its context length.** A number without one is
not a result.

**A/B in a single session.** Absolute throughput drifts ~15% between sessions;
only back-to-back comparisons are evidence.

**Benchmarks do not prove correctness.** `bench/paged_spec_bench` never inspects
generated text. A KV-corruption bug once survived an entire optimization run at a
healthy-looking 77% accept rate. After any change to attention, the KV cache, or
graph capture: diff greedy speculative output against greedy baseline on a **real**
model over >50 tokens. Greedy spec must be **token-identical** to greedy decode.
The mini-model tests generate too few tokens to cross a `len_bucket` boundary.

**Cold-cache microbenchmarks only.** Re-timing one weight tensor in a loop reads
L2 (64 MB on GB203), not HBM. A hot sweep once predicted 1.16× and delivered ~0
in-graph. Rotate through ≥3× L2 of distinct tensors, and cross-check against
in-graph `torch.profiler` time before believing any win.

**Verify optimized paths actually run.** Assert the activation state and a
side effect only the new path produces — not just that output matches the
fallback, which passes just as well when the feature is silently off.

**Never pass `.clone()`d tensors to a kernel test.** Cloning makes them
contiguous and hides every stride bug. The quantized-KV write kernel addressed V
through K's strides — which genuinely differ in the model, since K goes through
QK-norm and RoPE and V does not — and four rounds of isolation tests all passed
because each cloned its inputs. Test with tensors carrying the layout the caller
actually produces, transposes and slices included.

**Near-total output collapse is a bug, not quantization error.** Injected KV
noise of 0.66% leaves argmax agreement at 100%, and even 10% leaves it at 78%.
So 0.4% agreement is never "the format is too coarse" — measure the equivalent
noise before accepting a precision explanation.

**Suspect the measurement apparatus first.** Every wrong conclusion on
2026-08-22/23 came from the harness, not the code under test: an L2-cached
kernel sweep that mis-ranked tile configs, `.clone()`d test tensors that hid a
stride bug, and a benchmark that stood up three KV pools and read its own
allocator thrashing as a 65x engine regression. Before believing a dramatic
result, price out what the *measurement* costs in memory and bandwidth.

**Budget VRAM before benchmarking at length.** A KV pool is
`n_layer × blocks × n_kv × block_size × head_dim × 2 × bytes`; at 16K that is
2.6 GB per target engine. Anything that instantiates a second engine — timing
prefill separately, A/B-ing two configs in one process — doubles or triples it.
Reuse one engine and release sequences instead.

**Both engine bugs were found by checking a second thing**, not by staring at
the suspect: the int32 overflow surfaced because the *baseline* was diffed
alongside speculative decode, and the quantization question was settled by
injecting equivalent noise into the unquantized path. When something looks
broken, find the comparison that discriminates between your hypotheses.

---

## Commands

```bash
# short-context throughput (the old regime — not the goal)
.venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4 --skip-eager-spec

# the goal: throughput vs context length
.venv/bin/python -m bench.ctx_scaling_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B

.venv/bin/pytest -q
```
