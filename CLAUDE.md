# Project goal: long-context speculative decode throughput

## THE COMMITTED MILESTONE

This is the plan. Do not descope it, do not substitute an easier target, and do
not stop at the first row.

| context | committed tok/s | measured | status |
|---|---|---|---|
| ~4K  | **90–120+** | 118.8 | **met** |
| ~16K | **80–90**   | 65.3  | **missing by 18%** |
| ~32K | **55–65**   | not yet measurable | VRAM-bound |
| ~64K | **40–42**   | not yet attempted | needs INT8 target KV |

Model pair: Qwen3-8B INT4 target + Qwen3-0.6B draft, INT8 draft KV, n_draft=4.
Hardware: RTX 5070 Ti, 16 GB, 896 GB/s.

**All four rows are committed, not just the ones that are close.**

---

## REPORT MEASURED NUMBERS ONLY

Do not divide a measured `ms/step` by an assumed tokens-per-step to produce a
"normalized" throughput figure. That number was never measured by anything, and
in this project it landed closer to target than the real one every single time
— which is what motivated reasoning looks like from the inside.

If a comparison needs a number, **measure it**. `--slices N` exists precisely so
a flattering single sample cannot be reported as the result.

The same rule covers forward-looking claims. "This should land near 45 ms/step"
is the same error in future tense.

---

## Measured results

`bench/ctx_scaling_bench.py`, corpus `d122f3bde4eb`, 200 generated tokens,
INT8 draft KV, n_draft=4:

| end ctx | ms/step | decode tok/s | accept | tok/step | peak GB | guard fired |
|---|---|---|---|---|---|---|
| 4,296  | 30.60 | 118.8 | 84.2% | 3.62 | 8.3  | — |
| 16,584 | 51.03 | 65.3  | 80.2% | 3.30 | 11.0 | — |
| 24,776 | 77.82 | 59.8  | 96.3% | 4.63 | 12.7 | acceptance |
| 32,200 | 76.89 | 33.3  | 65.9% | 2.54 | 14.5 | VRAM 85% |

Rows with a guard fired are **not results**. 24K's acceptance is unrepresentative
of general text (that offset is deep in the repo's Python source, which the draft
predicts unusually well); 32K ran at 85% of VRAM, where timing measures allocator
pressure.

### Where the 16K step goes

Profiled per-kernel on isolated graph replays:

| item | ms/step | context-dependent? |
|---|---|---|
| `_int4_matmul_kernel` (target verify) | 9.15 | no |
| draft gemv (1.59 × 4.96 calls) | 7.89 | no |
| draft attention (1.93 × 4.96 calls) | 9.57 | yes |
| target attention (2.90 + 1.00) | 3.90 | yes |

~17 ms/step is context-independent. `_int4_matmul_kernel` alone is 59.3% of the
target verify replay.

---

## What is built

| piece | where | effect |
|---|---|---|
| Paged flash-decoding attention | `engine/kernels/paged_attention.py` | 4.5–11.6× over SDPA at 4K–32K |
| INT8/FP8 KV cache | `engine/paged_cache.py` | halves KV traffic; acceptance unchanged |
| Chunked prefill | `LlamaPagedEngine._prefill_forward` | 30K prefill 447 s → 19.2 s |
| Last-position-only prefill logits | `LlamaModel.forward(n_logits=)` | −9.1 GB at 30K |
| Fused RMSNorm / RoPE | `engine/kernels/{rms_norm,rope}.py` | one launch instead of ~six |
| Q/K/V and gate/up fusion | `engine/fuse_weights.py` | one wide matmul instead of narrow ones |
| YaRN RoPE scaling | `engine/layers.py` | makes >32768 context possible at all |
| Context-dependent `n_draft` | `SpeculativePagedEngine` | mechanism only; no schedule measured yet |

**Use INT8, not FP8, for KV.** Both are one byte, but per-token amax scaling
already supplies the exponent range FP8 spends bits on, so e4m3's 3 mantissa bits
are strictly worse than INT8's effective 7.

---

## Remaining work, in measured-value order

1. **`_int4_matmul_kernel`** — 9.15 ms/step at 16K and context-independent, so it
   taxes every row including 4K. Split-K is the candidate: `BLOCK_N=128` for full
   cache lines plus K-splits for a full grid. Needs fp32 atomics or a partials
   buffer, so budget an extra kernel launch against the gain.
2. **INT8 target KV** — halves the target's 4.6 GB at 30K. This is what makes 32K
   measurable and 64K possible, so it is now a prerequisite rather than an
   optimization. Changes the model's own distribution, so it needs a quality gate
   (perplexity + greedy divergence vs bf16) before use.
3. **`n_draft` schedule** — the mechanism accepts a callable; the curve has not
   been measured. Sweep with `--n-draft 2 3 4 6` per context.
4. **64K** — YaRN is in. Set `rope_scaling_factor=4`, `rope_original_n_ctx=32768`,
   raise `n_ctx`. VRAM is the open question.

---

## Non-negotiable practices

**Every performance claim states its context length *and* its acceptance rate.**
tok/s is proportional to acceptance, which varies with prompt content — 80.2% on
prose, 96.3% on this repo's Python source. A number without both is not a result.

**A/B in a single session.** Absolute throughput drifts ~15% between sessions.

**Benchmarks do not prove correctness.** `ctx_scaling_bench` never inspects
generated text. A KV-corruption bug once survived an entire optimization run at a
healthy-looking 77% acceptance. After any change to attention, the KV cache, or
graph capture: diff greedy speculative output against greedy baseline on a
**real** model over >50 tokens. They must be token-identical. Check the
single-model baseline too — an int32 overflow in the INT4 matmul was found only
because the baseline was diffed alongside speculative decode.

**Cold-cache microbenchmarks only.** Re-timing one weight tensor in a loop reads
L2 (64 MB on GB203), not HBM. A hot sweep once predicted 1.16× and delivered ~0
in-graph. Rotate through ≥3× L2 of distinct tensors.

**Never pass `.clone()`d tensors to a kernel test.** Cloning makes them
contiguous and hides stride bugs. The quantized-KV write kernel addressed V
through K's strides — which genuinely differ, since K goes through QK-norm and
RoPE and V does not — and four rounds of isolation tests passed because each
cloned its inputs.

**Near-total output collapse is a bug, not quantization error.** 0.66% injected
KV noise leaves argmax agreement at 100%; even 10% leaves it at 78%. So 0.4%
agreement is never "the format is too coarse".

**Suspect the measurement apparatus first.** Every wrong conclusion in this
project came from the harness, not the code under test: an L2-cached kernel
sweep, `.clone()`d test tensors, a benchmark that stood up three KV pools and
read its own allocator thrashing as a 65× engine regression, a corpus that
changed whenever the repo did, and a corpus short enough that long prompts
repeated themselves into 100% acceptance.

**Budget VRAM before benchmarking at length.** A KV pool is
`n_layer × blocks × n_kv × block_size × head_dim × 2 × bytes` — 2.6 GB per target
engine at 16K. The benchmark reports peak VRAM per row and warns past 80%.

---

## Commands

```bash
# the goal: throughput vs context length
.venv/bin/python -m bench.ctx_scaling_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --draft-kv-int8 --lengths 4096 16384 32000 --slices 3

# short-context reference (the pre-2026-08-22 regime, not the goal)
.venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4 --skip-eager-spec

.venv/bin/pytest -q
```
