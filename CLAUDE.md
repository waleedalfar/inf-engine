# Project plan: long-context throughput now, distributed later

## START HERE — state as of 2026-08-27

**The ladder is met and Phase A is done.** 4K, 16K and 32K all pass, and 32K is
now VRAM-robust rather than VRAM-tight. **64K was formally dropped from the
milestone on 2026-08-27** by the project owner, on the evidence in *A4* — it is
not a pending task and should not be reinstated without new information.

**What changed this session:**

1. **INT4 draft KV: built, gated, REJECTED on measurement.** Correct but
   unusable — it takes the draft's top-1 to 2.17 % and acceptance to 1.3 %. It
   is a property of the format, not a bug, and no grouping fixes it. The code is
   kept and off by default. **Do not retry this.** A4 has the numbers.
2. **Ring-buffer draft cache: landed and measured.** `--draft-ring` bounds the
   draft's KV *pool* to its attention window. 32K peak VRAM stops accumulating
   across slices (12.0→16.1 GB becomes flat 10.5→11.3 GB) and the third slice
   no longer thrashes (79.6 → 33.3 ms/step, prefill 121 s → 26 s).

**Next action — pick one, nothing is blocking:**

- **A5**: re-measure 4K/16K with `--slices 3 --draft-ring` now that the ring
  exists. Only 32K has been measured with it.
- Decide whether `--fast` should imply `--draft-ring`. It currently does **not**,
  deliberately: `--fast` is documented as the fastest *measured* config, and the
  ring buys robustness, not decode speed (it costs ~11 % on prefill). The case
  for folding it in is that on a 16 GB card the ring is what keeps long context
  from degenerating on the second or third turn.
- **Phase B (distributed)** — the ladder no longer blocks it at all. Its known
  gaps (B1–B4) are already written up below.

**Do not re-derive these; they are settled and written up in A4:** INT4 KV is
dead for this draft; the ring's `_grow_ring` one-block slack is deliberate and
load-bearing; the ring pool must be sized `sinks + window + prefill_chunk`.

**Repo state:** branch `main`, 382 tests pass (10 skipped), no xfail. **Working tree is dirty
— this session's work is uncommitted.** `feature/windowed-draft-cache` is
*behind* main, not ahead — ignore it.

**Read `.claude/skills/context-discipline/SKILL.md` before editing anything.**
Its first rule (Edit/Write, never shell heredocs) is a correctness rule: a
heredoc's `str.replace` silently no-ops on a stale anchor, which shipped a real
bug in this repo.

**To run it yourself (agentic, 32K):**
```bash
.venv/bin/python main.py --model-dir weights/Qwen--Qwen3-8B \
    --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --fast --draft-ring --max-ctx 32768 --max-turns 12 --workspace .
```
`--fast` = INT8 draft+target KV, draft window 4096. `--draft-ring` adds the
bounded draft pool; the banner must say `draft window=4096 (ring pool)` or it did
not engage (it needs `--draft-window`, which `--fast` supplies). Tools available
to the model: `read_file`, `write_file`, `list_dir`, `search_files`, `run_shell`,
`run_python`, all sandboxed to `--workspace`. The readout prints decode, prefill
and e2e separately — **only the decode figure is comparable to the numbers
below.**

---

## THE COMMITTED MILESTONE

Reach this ladder on a single RTX 5070 Ti. Do not descope it, do not substitute
an easier target, and do not stop at the rows that already pass. **The ladder is
now the three rows below; 64K was formally dropped on 2026-08-27 (see under the
table). That decision is made — do not re-add the row, and do not read the rule
above as licence to drop another one.**

| context | committed tok/s | measured decode (clean run) | status |
|---|---|---|---|
| ~4K  | **90–120+** | 124–131  (23.9–24.4 ms/step) | ✅ **above band** |
| ~16K | **80–90**   | 85–128   (27–30 ms/step)     | ✅ **met** |
| ~32K | **55–65**   | 92–111   (27.9–33.3 ms/step) | ✅ **met, VRAM-robust** |

**The ladder is met. Phase A is done.**

**64K — DROPPED 2026-08-27 by the project owner.** Not "deferred", not
"blocked": removed from the milestone. Read *A4* before proposing it back. Two
points in particular: 64K was never a throughput problem (extrapolating the
monotone ms/step curve puts it near 32–35 ms/step, comfortably inside its old
40–42 tok/s band), and the best case after every available VRAM lever was ~83 %
of the card — exactly the tightness 32K used to misbehave at. Dropped on
evidence, not on difficulty.

Best config: INT8 draft + target KV, split-K INT4 matmul, **draft attention
window 4096 with 4 sink tokens**, n_draft=4.

**The "32K gap" was a benchmark artifact, not an engine deficit (2026-08-23).**
`ctx_scaling_bench` computed decode as `e2e − separate_prefill_estimate`, and its
`_time_prefill` helper ran on an already-warm allocator with blocks pre-allocated,
so it read the 32K prefill ~2.6 s *low*. That fixed 2.6 s, divided across ~76
decode steps, added ~35 ms/step of phantom cost — manufacturing the entire gap
(true 28 → reported 63). Proven by decomposing prefill vs decode *inside one run*
(`bench/spec_decode_wall.py`): true in-run prefill 23 285 ms vs the helper's
20 653 ms. The bug is invisible at 4K (small prefill) and grows with context,
which is why only the long rows looked short. Once measured correctly, ms/step is
**monotonic in context** (24 → 27 → 28), as the physics requires. Fixed: the
benchmark now stamps the decode start inside the run instead of subtracting a
separate estimate. tok/s within a row still tracks acceptance (content), so the
range is acceptance spread, not timing noise.

**32K's VRAM headroom is FIXED (2026-08-27) by the ring-buffer draft cache.**
It used to sit at 83–94 % and the third back-to-back slice collapsed. Measured
`--slices 3` at 32K, same session, ring off vs on:

| slice | prefill ms | ms/step | accept | peak GB |
|---|---|---|---|---|
| off 0 / 1 / 2 | 23 788 / 24 104 / **120 980** | 30.6 / 33.9 / **79.6** | 66.8 / 82.9 / 82.7 % | 12.0 / 14.1 / **16.1** |
| on  0 / 1 / 2 | 26 420 / 27 090 / **26 290** | 27.9 / 32.8 / **33.3** | 66.1 / 82.9 / 86.2 % | 10.5 / 10.9 / **11.3** |

**Peak VRAM stops accumulating** — flat 10.5→11.3 GB instead of climbing 12.0→16.1
(98.8 % of the card). Slice 2's prefill goes 121 s → 26 s and its ms/step 79.6 →
33.3. On the *healthy* slices ms/step is unchanged within the 7.1 % noise floor
(30.6→27.9, 33.9→32.8), so this bought robustness, not speed. Acceptance is
neutral (−0.7 / 0 / +3.5 points); the shifts are real, not noise, because ring
mode also makes the draft's *prefill* windowed, which changes its proposals.
The one cost is **prefill ~11 % slower on healthy slices** (bookkeeping plus the
windowed prefill mask) — worth it against a 4.6× win on the slice that thrashed.

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

### A3. `n_draft` schedule — ✅ DONE, small
Measured monotone at 32K (n_draft 2/3/4/6 → 38.9/37.2/35.7/32.0 tok/s at matched
acceptance) but **flat at 16K** (72.3–78.6, inside a 7.1% noise floor). +9% at
32K. Real but nowhere near sufficient; it scales the *number* of draft forwards
and cannot touch what each one costs. No schedule shipped — with A6 in place,
n_draft=4 measured best at both 16K and 32K.

### A6. Sliding-window draft attention — ✅ DONE, **the lever that met 16K**
The draft is **65% of per-step KV traffic** at 32K despite being a 0.6B model,
because Qwen3-0.6B carries the same 8 KV heads × 128 head_dim as the 8B target
(56 KiB/token vs 72) and runs several forwards per step.

The draft does not need full context: it only *proposes*, and the target's
accept/reject still yields the target's exact distribution. A windowed draft is
**correct**, it just accepts slightly less — measured 81.5% vs 81.0% at 16K, i.e.
essentially free.

| ctx | window | ms/step | tok/s | accept |
|---|---|---|---|---|
| 16,584 | full | 45.17 | 76.3 | 81.0% |
| 16,584 | **4096** | **40.70** | **84.7** | 81.5% |
| 32,200 | full | 70.71 / 87.60 | 36.3 / 39.4 | 66.1 / 81.0% |
| 32,200 | **4096** | 61.62 / 62.91 | **42.7 / 50.5** | 66.8 / 76.4% |
| 32,200 | 2048 | 60.61 | 42.9 | 66.7% |

2048 ≈ 4096, so 4096 is past the knee. `WINDOW`/`N_SINK` are constexpr so the
branch compiles away for the target; the saving comes from splits outside the
window skipping their loads, not from masking after the fact.

### A7. Close the remaining 32K gap — ✅ RESOLVED (no gap existed)
The "gap" was `ctx_scaling_bench`'s prefill accounting, not the engine. Measured
four independent ways (decode-wall, slope, fixed single-slice bench, 2-slice
slice-0), 32K decodes at 26.5–28.8 ms/step = 91–99 tok/s, above the 55–65 band.
The "~38 ms unexplained" was the misattributed prefill. Benchmark fixed; see the
milestone note above. What remains at 32K is **VRAM headroom**, not throughput.

### A4. 64K enablement — ✅ CLOSED (row dropped 2026-08-27; ring buffer landed)

*Kept in full because it is the evidence for dropping 64K and for never retrying
INT4 KV. The ring-buffer work it scoped did land, and fixed 32K.*
YaRN is implemented and wired into the bench (`--n-ctx 131072 --rope-scaling-factor
4 --rope-original-n-ctx 32768`). **Measured: 64K does not fit.** A probe pinned the
GPU at 93 % VRAM (15.2 / 16.3 GB), 100 % util, grinding — it thrashes before it's
slow, exactly as the VRAM budget predicts.

**Root cause is the draft KV, not the target.** Qwen3-0.6B carries the same 8 KV
heads × 128 head_dim as the 8B target, so at 64K its INT8 KV is ~3.65 GB (28
layers) — nearly as large as the target's 5.0 GB (36 layers) — even though the
draft only ever attends to its 4096-token window. That ~3.4 GB of never-read KV is
the whole overflow: target 5.0 + draft 3.65 + weights 5.4 = 14 GB before
activations, and prefill pushes it over.

**Fix: bound the draft's KV *allocation* to window + sinks** (~0.27 GB), not just
its reads. The VRAM lever is concrete — the pool is one `torch.zeros((n_layers,
n_total_blocks, ...))` sized at construction (`paged_cache.py:165`), so the draft
engine just needs a small `BlockManager(n_total ≈ ceil((sinks+window)/bs)+margin)`.
Making a long sequence live in that small pool is the ring buffer, and tracing it
end-to-end shows **three coupled pieces**, in the order to build+gate them:

1. **Decode-side recycling** (tractable core). In windowed mode, `ensure_slot`
   recycles out-of-window non-sink physical blocks instead of `allocate`-ing new:
   repoint their logical `block_table` entries at a pinned sink block (a valid pool
   index the windowed kernel never reads, exactly like the existing column
   padding), free the physical, reuse it for the new position. `reset_to` and the
   `block_table` full-length list stay as-is; only physical residency shrinks.
   Unit-test: the ring's gathered/attended KV for in-window+sink positions is
   bit-identical to a full cache's, across many block recycles.

2. **Bounded prefill.** `_prefill_forward` allocates all blocks and gathers full
   history via eager `extend()` — no windowing, and the flash-*decoding*
   `paged_attend` (q_len 1–5) is the wrong tool for 2048-token chunks. The draft
   prefill must either window the eager path (recycle between chunks + mask beyond
   window) or prefill only `[sinks]+[last window]` tokens. Prefill peak, not just
   steady state, must fit the small pool.

3. **Spec-loop length decoupling.** `_generate_one` assumes draft and target
   caches share length `L` and rolls both back to `L+n+1`
   (`speculative_paged_engine.py:175,234`). A shorter draft cache needs its own
   length tracked and its own rollback offset.

**Full correctness gate after each piece** (greedy speculative == single-model
baseline, token-identical > 50 tokens) — this is a KV-cache change, the exact
class the gate exists for. Do it on a branch; keep `main` green.

**INT4 draft KV: built, gated, and REJECTED on measurement (2026-08-27).**

The 2026-08-23 decision was to try INT4 draft KV before the ring buffer, on the
reasoning that "the draft's KV precision is nearly free by construction — the
target verifies every token, so it costs acceptance rate and never correctness."
**The first half is true and the second half is the trap.** It costs acceptance
rate, and it costs essentially *all* of it.

Implemented as packed nibbles (`INT4` sentinel in `paged_cache.py`, the pool
stays int8 with last dim `head_dim/2`; `_write_kv_int4_kernel` packs, the
`PACK4` branch of `_paged_flash_decode_kernel` unpacks; `--draft-kv-int4` on the
bench and `main.py`). Measured on Qwen3-8B + Qwen3-0.6B, 4096 ctx, greedy:

| draft KV | greedy output vs baseline | acceptance | tok/step |
|---|---|---|---|
| int8 | token-identical | **71.8 %** | 2.89 |
| int4 | token-identical | **1.3 %** | 1.01 |

The correctness gate *passes* — speculative decoding is exact no matter how bad
the draft is — and that is exactly why the gate could not catch this. tok/step
1.01 means four draft forwards and a verify per emitted token: strictly worse
than not speculating.

**It is the format, not the kernel.** Measured through the draft's own
teacher-forced logits, with no speculative machinery in the path (4096 tokens,
bf16 weights, KV format the only variable):

| draft KV | ppl | top-1 vs bf16 |
|---|---|---|
| bf16 | 16.19 | — |
| int8 | 16.57 | 90.13 % |
| int4 | 265 335 | **2.17 %** |

2.17 % top-1 predicts the 1.3 % acceptance directly — the draft proposes its
argmax and the target accepts it only when it agrees. The kernel itself is
unit-tested against a torch reference at head_dim 64 *and* 128
(`tests/test_int4_kv.py`): round-trip within one quantization step, and the read
kernel agreeing with the independent torch gather path.

**No amount of finer scaling rescues it.** Relative RMS error on gaussian K/V by
group size (measured), against the byte cost per 128 channels:

| group | int8 err | int4 err | int4 bytes/128ch |
|---|---|---|---|
| 128 (shipped granularity) | 0.65 % | **11.7 %** | 68 |
| 32 | 0.53 % | 9.7 % | 80 |
| 8 | 0.40 % | 7.2 % | 128 |
| 4 | 0.31 % | 5.6 % | 192 |

INT8 at 0.65 % gives 90 % top-1. FP8 at 2.6 % gave 0.2 % top-1 on this same
28-layer draft (see `_QUANT_MAX`). **The cliff sits under ~2 %, and INT4's floor
is 5.6 % even at group=4 — which costs 192 bytes, more than INT8's 132.** Four
bits over an amax-scaled range is ~amax/7 per step, so the RMS error cannot go
below roughly σ/15 regardless of grouping. There is no INT4 configuration that
is both smaller than INT8 and accurate enough for this draft. Dead end; don't
re-derive it.

The code is kept — it is correct, tested, and costs nothing switched off — but
**do not enable `--draft-kv-int4`.**

**Ring-buffer draft cache — ✅ DONE (2026-08-27), and it is what fixed 32K.**
`--draft-ring` (bench, `main.py`, `spec_decode_wall.py`) bounds the draft's KV
*pool*, not just its reads. Unlike INT4 it costs no draft precision by
construction: the draft already only reads its 4096-token window, so bounding
what it *stores* to that window changes nothing it computes. Measured 32K draft
pool 2157 blocks / 1.98 GB → **401 blocks / 0.37 GB**; the slice table in the
milestone section has the throughput and peak-VRAM effect.

Three things about it that are settled and should not be re-litigated:

- **`_grow_ring`'s one-block slack is deliberate.** It recycles against the
  length *before* a grow, because the first query of a multi-token grow (a
  2048-token prefill chunk) still attends back to `seq_lens - window + 1`.
  `tests/test_windowed_ring_cache.py` used to assert the tightest possible bound
  and was marked strict-xfail against it; the *test* was wrong, and asserting
  that bound would have been asserting a bug. It now allows one block of slack
  and separately checks that recycling actually happens.
- **Pool size is `sinks + window + prefill_chunk`, not `sinks + window`**
  (`ring_pool_blocks` in `llama_paged_engine.py`). `_prefill_forward` grows a
  whole chunk at a time and nothing can be reclaimed until the next call, so
  omitting the chunk term survives construction and then OOMs partway through
  the first long prefill. 401 blocks at the shipped config.
- **Cross-turn prefix reuse works with it** (`tests/test_prefix_reuse.py`):
  rollback never resurrects a recycled position or double-frees an alias.

**64K — DROPPED 2026-08-27 by the project owner.** Removed from the milestone,
not parked. This section is the evidence behind that call. What was measured:
with INT4 draft KV
the 64K run sat at **15.3 GB / 94 %** at 99 % util and never emerged from
prefill — the same signature as the earlier INT8 probe at 93 %. The ring is worth
~1.7 GB more than INT4 on the draft pool, which would put 64K near 13.6 GB ≈
83 % — i.e. exactly the tightness 32K used to have and misbehave at. That is a
thin margin to build on.

Two caveats recorded so nobody re-derives them: (1) the ~2.8 GB gap between the
predicted 12.5 GB and the observed 15.3 GB was never traced — presumably prefill
activations plus fragmentation, and some of it may be addressable independently
of the draft KV, so 64K is *not proven impossible*. (2) **64K was never a
throughput problem.** Extrapolating the monotone ms/step curve (24 → 27 → 28.8 at
4/16/32K) puts 64K near 32–35 ms/step, comfortably above its 40–42 tok/s band.
It is a memory problem end to end; do not chase it as a speed one.

### A5. Full ladder, multi-slice
32K is done with `--slices 3`, both with and without the ring (milestone
section). **4K and 16K have not been re-measured with `--draft-ring`.** Use a
**fresh engine per slice** or the allocator fragments and the later slices read
as thrash — this is exactly what ring-off slice 2 shows above. One slice is one
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

`bench/ctx_scaling_bench.py` (**decode-start now stamped inside the run** — see
the prefill-accounting note in the milestone section), corpus `d122f3bde4eb`,
200 generated tokens, INT8 draft + target KV, split-K matmul, draft window=4096,
n_draft=4:

| end ctx | ms/step | decode tok/s | accept | tok/step | peak GB |
|---|---|---|---|---|---|
| 4,296  | 23.9 / 24.4  | 130.9 / 123.5 | 77.6 / 76.4% | 3.11 / 3.02 | 7.9 |
| 16,584 | 28.9 / 29.9  | 119.2 / 84.6  | 81.5 / 65.0% | 3.43 / 2.51 | 9.7–10.8 |
| 32,200 | 26.5 / 28.8  | 99.3 / 90.8   | 66.8 / 66.8% | 2.62        | 14.1 |

Second 32K column is `bench/spec_decode_wall.py` (decode-only wall, no prefill
subtraction) — the arbiter that exposed the accounting bug. The two agree, which
is the point. A back-to-back second 32K slice thrashed to 141 ms/step at 94 %
VRAM; discard it (allocator, not engine).

Progression at 32K decode ms/step: reported-63 (accounting bug) → **true 26–29**
once prefill was split correctly. No engine change moved it — the number was
always this; the harness was lying.

**tok/s tracks acceptance, which varies by slice and shifts whenever numerics
change.** ms/step barely moves across slices (28.9 vs 29.9 at 16K); tok/s swings
35 points on the same rows because acceptance does. Compare ms/step across a code
change, tok/s only at matched acceptance.

### Where the 16K step goes (per-kernel, isolated graph replays, pre-A6 full-context draft)

| item | ms/step | context-dependent? |
|---|---|---|
| `_int4_matmul_kernel` (target verify) | 9.15 | no |
| draft gemv (1.59 × 4.96 calls) | 7.89 | no |
| draft attention (1.93 × 4.96 calls) | 9.57 | yes |
| target attention (2.90 + 1.00) | 3.90 | yes |

~17 ms/step is context-independent. That is why A2 matters at every length.
Draft attention row reflects full-context draft; with window=4096 (A6) this cost is cut proportionally to the window/context ratio (~4× at 16K).

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
| Sliding-window draft attention | `engine/kernels/paged_attention.py`, `SpeculativePagedEngine` | draft KV traffic cut ~10× at 32K; met 16K target |
| INT4 packed-nibble KV | `engine/paged_cache.py` (`INT4`), `paged_attention.py` | halves the pool again — **but unusable, see A4. Off by default.** |
| Ring-buffer draft KV pool | `paged_cache.py:_grow_ring`, `ring_pool_blocks` | 32K draft pool 1.98 → 0.37 GB; peak stops accumulating across slices |

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

**A passing correctness gate can hide a total collapse — on the draft side it is
structurally blind.** Speculative decoding is exact for *any* draft, so the
greedy-identity gate returns "token-identical" whether the draft is excellent or
outputting noise. INT4 draft KV passed it while acceptance sat at 1.3 %. Any
change to the draft — its KV format, its window, its weights — must be gated on
**acceptance rate**, measured against the same prompt with the change off. Pair
the identity gate with an acceptance number or you have checked nothing about
whether the change was worth making.

**Cold-cache microbenchmarks only.** Re-timing one weight tensor reads L2 (64 MB
on GB203), not HBM. A hot sweep predicted 1.16× and delivered ~0 in-graph.

**Never pass `.clone()`d tensors to a kernel test.** Cloning makes them
contiguous and hides stride bugs. The quantized-KV write kernel addressed V
through K's strides — which genuinely differ, since K goes through QK-norm and
RoPE and V does not — and four rounds of isolation tests passed because each
cloned its inputs.

**Near-total output collapse is a bug, not quantization error.** 0.66% injected
KV noise leaves argmax agreement at 100%; even 10% leaves it at 78%.

**A single anomalous row is usually a transient — verify before acting on it.**
A 16K row read 82.40 ms/step against 43.89 for the same config; windowing can
only remove work, so instead of reverting, the draft replay was profiled
directly and found 22% *faster*. The clean re-run gave 40.70 ms — the row that
met the 16K target. Trusting the anomaly would have discarded the winning change.
Run-to-run variance on identical work measured **7.1%**; treat anything smaller
as noise, and anything wildly larger as suspect rather than real.

**Suspect the measurement apparatus first.** Every wrong conclusion in this
project came from the harness: an L2-cached kernel sweep, `.clone()`d test
tensors, a benchmark that stood up three KV pools and read its own allocator
thrashing as a 65× engine regression, a corpus that changed whenever the repo
did, a corpus short enough that long prompts repeated into 100% acceptance, and
— most expensively — a decode figure computed as `e2e − separate_prefill_estimate`
where the estimate ran warm and read ~2.6 s low, inventing a 32K throughput gap
that three planned optimizations (A7) were about to chase. It cost nothing to
build a second measurement (`spec_decode_wall.py`, decode-only wall) and diff it;
the disagreement was the whole finding. **Never subtract one measurement from
another when you can measure the thing directly** — the errors don't cancel, they
land wherever the arithmetic sends them, and here that was every long-context row.

**A number that violates a monotonicity you know must hold is the measurement's
bug, not physics'.** Decode ms/step *must* rise with context (more KV to attend).
The old bench had 4K < 16K but 16K < 32K inverted once you looked; the corrected
one is monotone 24 → 27 → 28. When a curve bends the wrong way, check the ruler.

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
- 2026-08-23 — A3: n_draft sweep — monotone at 32K (+9%), flat at 16K; n_draft=4 kept
- 2026-08-23 — A6: sliding-window draft attention; **16K met at 84.7 tok/s**, 32K 36.3 → 42.7
- 2026-08-23 — A7: found the "32K gap" was a bench prefill-accounting bug, not the engine.
  True decode 26–29 ms/step (91–99 tok/s), above the 55–65 band. Fixed the bench to
  stamp decode-start in-run; added `spec_decode_wall.py`. **Ladder met at 4K/16K/32K.**
  Remaining: 64K (blocked on VRAM — 32K already 83–94 %).
- 2026-08-23 — main.py: `--fast` exposes the measured-best config; interactive
  readout now splits decode / prefill / e2e (it had reported e2e only, which read
  as a 5x regression against CLAUDE.md's decode numbers)
- 2026-08-23 — Cross-turn prefix reuse (`generate_resident`); interactive turns
  no longer re-prefill the whole conversation. `_prefill_forward(base=)` carries
  the absolute position so RoPE stays aligned
- 2026-08-23 — A4 probe: wired YaRN into the bench (`--n-ctx/--rope-scaling-factor/
  --rope-original-n-ctx`). 64K thrashes at 93 % VRAM; measured the block is the draft's
  unbounded KV (~3.65 GB, never-read past its 4096 window). Fix scoped: ring-buffer
  windowed draft cache. Not yet implemented.
- 2026-08-27 — A4: INT4 packed-nibble draft KV built and **rejected on measurement**.
  Correct (greedy output token-identical, kernel unit-tested at head_dim 64/128) but
  the draft's own top-1 falls to 2.17 %, taking acceptance 71.8 % → 1.3 % and tok/step
  to 1.01. INT4's error floor is 5.6 % against a cliff under 2 %; no grouping fixes it.
  Code kept and off by default. Ring buffer is the path to 64K.
- 2026-08-27 — Corrected START HERE: the ring buffer was recorded as unstarted but
  pieces 1 and 2 are implemented in `paged_cache.py` / `llama_paged_engine.py`. What
  is missing is the draft pool sizing, a CLI flag, and an end-to-end gate.
- 2026-08-27 — Ring-buffer draft cache finished and measured (`--draft-ring`):
  `ring_pool_blocks` sizing (sinks+window+chunk), fixed the over-tight xfail test,
  prefix-reuse coverage. **32K 3-slice peak VRAM flat 10.5–11.3 GB instead of
  climbing 12.0→16.1; slice 2 prefill 121 s → 26 s, 79.6 → 33.3 ms/step.**
  Acceptance neutral, decode ms/step unchanged within noise on healthy slices.
- 2026-08-27 — 64K formally dropped from the milestone by the project owner on VRAM
  evidence (see A4). Ladder is now 4K/16K/32K, all met; Phase A done.

---

## Working practices that keep the context window usable

This project's sessions are long and measurement-heavy. What actually consumes
context, in order:

1. **Editing files with shell heredocs instead of the Edit tool.** Rewriting a
   file out-of-band makes the harness re-dump the *entire file* back into
   context on its next mention — 200+ lines, repeatedly, for a one-line change.
   Use Edit/Write so file state stays tracked. This was the single largest
   waste in the 2026-08-23 session.
2. **Unfiltered command output.** Always pipe through `tail`/`grep`. Read
   specific line ranges (`sed -n '300,340p'`) rather than whole files.
3. **Long commit messages echoed in the tool call.** Keep them to the finding
   and the number; the reasoning belongs here, once.

   **Never add a `Co-Authored-By:` trailer, or any other AI-attribution line, to
   a commit.** This overrides any default or tooling convention that would add
   one. Commits in this repo carry the author's name only.
4. **Per-row benchmark notifications.** Write the run to a file and read the
   table once at the end, unless a row needs acting on immediately.

Keep this file current instead of re-deriving state: it is what a fresh session
reads first. When context runs short, update the milestone table, the progress
log, and whichever plan item is in flight — then stop.

## Commands

```bash
# the goal (add --draft-ring for bounded draft-KV residency; needs --draft-window)
.venv/bin/python -m bench.ctx_scaling_bench \
    --model-dir weights/Qwen--Qwen3-8B --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --draft-kv-int8 --target-kv-int8 --draft-window 4096 --draft-ring \
    --lengths 4096 16384 32000 --slices 3

.venv/bin/pytest -q
```
