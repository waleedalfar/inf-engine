# Project plan: long-context throughput now, distributed later

## THE COMMITTED MILESTONE

Reach this ladder on a single RTX 5070 Ti. Do not descope it, do not substitute
an easier target, and do not stop at the rows that already pass.

| context | committed tok/s | measured (2 slices) | status |
|---|---|---|---|
| ~4K  | **90–120+** | 134.2 / 125.3 | ✅ **above band** |
| ~16K | **80–90**   | 73.4 / 66.8   | ❌ −10 to −17% |
| ~32K | **55–65**   | 35.4 / 40.1   | ❌ −27 to −36% |
| ~64K | **40–42**   | —             | ⛔ not attempted |

Qwen3-8B INT4 target + Qwen3-0.6B draft, INT8 draft KV, n_draft=4.
16 GB VRAM, 896 GB/s.

**Distributed multi-machine work is Phase B and comes after this ladder.** Its
purpose is fitting *larger models* (Qwen3-32B/72B), not making this one faster —
pipeline parallelism cannot accelerate a single sequence, and an M2 is 2–9× slower
per layer than the 5070 Ti. Phase A must not break the seams Phase B needs; see
*Distributed-readiness invariants*.

---

## REPORT MEASURED NUMBERS ONLY

Never divide a measured `ms/step` by an assumed tokens-per-step to produce a
"normalized" throughput figure. No benchmark emits that number, and in this
project it landed closer to target than the real one every single time — which is
what motivated reasoning looks like from the inside. Same rule in future tense:
"this should land near 45 ms/step" is the same error.

If a comparison needs a number, **measure it**. `--slices N` exists so a
flattering single sample cannot be reported as the result.

---

## Phase A — reach the ladder

### A1. INT8 target KV + quality gate — ✅ DONE
Target KV is 4.6 GB bf16 at 30K; INT8 halves it. This is a **prerequisite for
measuring 32K at all** (currently 85% of VRAM, where timings are allocator
pressure) and for 64K fitting. Mechanism is built (`kv_dtype=torch.int8`, kernel
reads it natively); what is missing is evidence, because target-side quantization
changes the model's own output distribution — unlike draft-side, which is free.

**Result (`bench/kv_quality_gate.py`, 8192 ctx, 4096-token ppl window):**
perplexity ratio 0.9981 (tolerance 1.02), top-1 agreement 97.58% (floor 95%),
greedy diverges at token 27 of 200, peak 7.01 → 6.34 GB. **PASS.**

Delivered what it was for: 32K peak went 14.5 → 12.2 GB and became measurable.
It did not close throughput gaps — it was a memory fix, not a compute one.

### A2. Split-K for `_int4_matmul_kernel` — ✅ DONE
Cold: **2.03× on the model's matmuls at M=5**, 1.37× at M=1. `lm_head` stays
tiled (0.94×). Partials reduce in **fixed order, not atomics** — atomic ordering
is unspecified and float addition is not associative, which would make the
token-identical greedy gate flaky.

**End-to-end: −3.5 ms/step at 4K (acceptance matched within 0.5 points), −4.35 ms
at 16K slice 1, ~0 at 32K.** The saving is a fixed cost, so its share shrinks as
attention grows — it moved 4K above its band and did nothing measurable at 32K.

⚠️ **Split-K changes accumulation order, so generated text differs from the tiled
path.** Acceptance moved 72.9% → 81.0% on the *same* 16K prompt slice. tok/s is
therefore not directly comparable across a numerics change; compare at matched
acceptance or use isolated kernel timings.

### A3. `n_draft` schedule — IN PROGRESS
Now the main lever. A1 and A2 both addressed costs that are flat in context;
the draft is **77% of per-step KV traffic** and at 32K each of ~3.5 draft
forwards per step attends over 32k tokens, so this is the only remaining lever
that scales *with* context. Sweeping `--n-draft 2 3 4 6` at 16K and 32K.

### A4. 64K enablement
YaRN is implemented. Set `rope_scaling_factor=4`, `rope_original_n_ctx=32768`,
raise `n_ctx`. Blocked on A1 for VRAM.

### A5. Full ladder, multi-slice
Re-measure 4K/16K/32K/64K with `--slices 3+` once A1–A3 land. One slice is one
sample; the spread is the result.

---

## Phase B — distributed, for models that do not fit

Deferred until the ladder is met. Existing scaffolding: `engine/distributed/`
(TCP stage workers, wire protocol), `LlamaModel.forward_stage` seams,
`PagedLlamaKVCache(owned_layers=...)` ranged cache.

**Known gaps, recorded now so Phase A does not widen them:**

- **B1.** Remote stages use a raw `PagedLlamaKVCache`, which has no
  `fused_attend`, so they fall back to the gather+SDPA path and miss the paged
  flash-decoding kernel entirely. Fix: give the raw cache a `fused_attend` (it
  already has `paged_attend`; it needs the block-table/seq-lens plumbing the
  graph wrapper supplies).
- **B2.** Chunked prefill lives in `LlamaPagedEngine._prefill_forward`. A
  distributed engine needs its own equivalent or the activation-memory problem
  returns per stage.
- **B3.** `DistributedPagedEngine` with continuous batching does not exist; the
  worker is one-connection/one-sequence/synchronous.
- **B4.** The network hop is synchronous, so the local stage idles during it.

### Distributed-readiness invariants — do not break these in Phase A

1. **Everything layer-indexed goes through `owned_layers` / `layer_offset`.** A
   stage owns a contiguous slice, not all layers. `paged_attend` and
   `write_static` already do this; keep it that way.
2. **`forward_stage(x, start_layer, end_layer, is_first, is_last, ...)` is the
   seam.** New parameters get defaults and keyword-only call sites so a remote
   worker built against the old signature still runs. `n_logits` was added this
   way, and only takes effect under `is_last`.
3. **No process-global state keyed by shape or layer.** Workspace buffers belong
   to the object that owns the graphs. A module-global attention scratch cache
   handed one engine's graph a buffer from another engine's private CUDA-graph
   pool and caused an illegal memory access — the fix (ownership on the KV cache)
   is also exactly what multi-stage needs.
4. **Kernels take pools and block tables as arguments**, never reach for a global
   model or cache.

---

## Measured results

`bench/ctx_scaling_bench.py`, corpus `d122f3bde4eb`, 200 generated tokens,
INT8 draft KV, n_draft=4, single slice:

Latest: A2 (split-K) + A1 (both KVs INT8), 2 slices, n_draft=4:

| end ctx | ms/step | decode tok/s | accept | tok/step | peak GB |
|---|---|---|---|---|---|
| 4,296  | 24.83 / 24.56 | 134.2 / 125.3 | 80.3 / 76.6% | 3.32 / 3.06 | 7.9 |
| 16,584 | 46.99 / 38.89 | 73.4 / 66.8   | 81.0 / 66.5% | 3.43 / 2.57 | 9.7 |
| 32,200 | 72.34 / 86.10 | 35.4 / 40.1   | 66.1 / 81.0% | 2.55 / 3.43 | 12.0 |

Progression at 16K: 65.3 (bf16 KV) → 63.4–63.6 (A1) → 66.8–73.4 (A2).
At 32K: unmeasurable → 33.3–39.5 (A1) → 35.4–40.1 (A2).

**tok/s tracks acceptance, which varies by slice and shifts whenever numerics
change.** Two slices at 32K differ by 15 points of acceptance and 5 tok/s. Never
compare a single slice across a code change without checking acceptance moved
less than the effect being claimed.

### Where the 16K step goes (per-kernel, isolated graph replays)

| item | ms/step | context-dependent? |
|---|---|---|
| `_int4_matmul_kernel` (target verify) | 9.15 | no |
| draft gemv (1.59 × 4.96 calls) | 7.89 | no |
| draft attention (1.93 × 4.96 calls) | 9.57 | yes |
| target attention (2.90 + 1.00) | 3.90 | yes |

~17 ms/step is context-independent. That is why A2 matters at every length.

---

## What is built

| piece | where | effect |
|---|---|---|
| Paged flash-decoding attention | `engine/kernels/paged_attention.py` | 4.5–11.6× over SDPA, 4K–32K |
| INT8/FP8 KV cache | `engine/paged_cache.py` | halves KV traffic; acceptance unchanged |
| Chunked prefill | `LlamaPagedEngine._prefill_forward` | 30K prefill 447 s → 19.2 s |
| Last-position prefill logits | `LlamaModel.forward(n_logits=)` | −9.1 GB at 30K |
| Fused RMSNorm / RoPE | `engine/kernels/{rms_norm,rope}.py` | one launch instead of ~six |
| Q/K/V and gate/up fusion | `engine/fuse_weights.py` | one wide matmul, not narrow ones |
| YaRN RoPE scaling | `engine/layers.py` | makes >32768 context possible |
| Context-dependent `n_draft` | `SpeculativePagedEngine` | mechanism only; curve unmeasured |

**Use INT8, not FP8, for KV.** Both are one byte, but per-token amax scaling
already supplies the exponent range FP8 spends bits on, so e4m3's 3 mantissa bits
are strictly worse than INT8's effective 7.

---

## Non-negotiable practices

**Every performance claim states its context length *and* its acceptance rate.**
tok/s is proportional to acceptance, which varies with content — 80.2% on prose,
96.3% on this repo's Python. A number without both is not a result.

**A/B in a single session.** Throughput drifts ~15% between sessions.

**Benchmarks do not prove correctness.** After any change to attention, the KV
cache, or graph capture: diff greedy speculative output against greedy baseline
on a **real** model over >50 tokens — they must be token-identical — and check
the single-model baseline too. An int32 overflow in the INT4 matmul was found
only because the baseline was diffed alongside speculative decode.

**Cold-cache microbenchmarks only.** Re-timing one weight tensor reads L2 (64 MB
on GB203), not HBM. A hot sweep predicted 1.16× and delivered ~0 in-graph.

**Never pass `.clone()`d tensors to a kernel test.** Cloning makes them
contiguous and hides stride bugs. The quantized-KV write kernel addressed V
through K's strides — which genuinely differ, since K goes through QK-norm and
RoPE and V does not — and four rounds of isolation tests passed because each
cloned its inputs.

**Near-total output collapse is a bug, not quantization error.** 0.66% injected
KV noise leaves argmax agreement at 100%; even 10% leaves it at 78%.

**Suspect the measurement apparatus first.** Every wrong conclusion in this
project came from the harness: an L2-cached kernel sweep, `.clone()`d test
tensors, a benchmark that stood up three KV pools and read its own allocator
thrashing as a 65× engine regression, a corpus that changed whenever the repo
did, and a corpus short enough that long prompts repeated into 100% acceptance.

**Budget VRAM before benchmarking at length.** A KV pool is
`n_layer × blocks × n_kv × block_size × head_dim × 2 × bytes` — 2.6 GB per target
engine at 16K. The benchmark reports peak VRAM per row and warns past 80%.

---

## Progress log

Keep this current. One line per landed change, newest last.

- 2026-08-22 — Paged flash-decoding kernel; 4200 ctx 16.7 → 71.7 tok/s
- 2026-08-22 — INT8 draft KV; 8392 ctx 62.1 → 73.9 tok/s, acceptance unchanged
- 2026-08-22 — Fixed KV corruption from capturing a verify graph before allocating
- 2026-08-23 — Fixed int32 overflow in INT4 matmul; prefills ≥14k were all-zero logits
- 2026-08-23 — Last-position prefill logits; −9.1 GB at 30K
- 2026-08-23 — Chunked prefill; 30K prefill 447 s → 19.2 s, step 774 → 104.67 ms
- 2026-08-23 — YaRN RoPE scaling (enables >32768)
- 2026-08-23 — Benchmark: peak-VRAM + acceptance guards, pinned 168k corpus, `--slices`
- 2026-08-23 — Cleared stale docs; CLAUDE.md rebuilt around measured results
- 2026-08-23 — A1: target-KV INT8 passes quality gate; 32K peak 14.5 → 12.2 GB, now measurable
- 2026-08-23 — A2: split-K INT4 matmul (deterministic reduce); 4K 115.7 → 134.2 tok/s

---

## Commands

```bash
# the goal
.venv/bin/python -m bench.ctx_scaling_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --draft-kv-int8 --lengths 4096 16384 32000 --slices 3

.venv/bin/pytest -q
```
