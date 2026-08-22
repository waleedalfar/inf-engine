# Path to 100 tok/s — measured, not projected

## Current state (2026-08-22)

259 tests pass. Branch `phase4-sync-elimination` (commits `907fc6c`, `5e89d0c`).

### Benchmark (RTX 5070 Ti, Qwen3-8B INT4 + Qwen3-0.6B bf16, n_draft=4)

All numbers from same-session A/B runs; accept rate is 77.3% / 3.16 tok-per-step
in every run, so the tokens generated are identical throughout.

| Build | tok/s | ms/step |
|---|---|---|
| Phase 3 (previous `main`) | 69.3 | 46.0 |
| + Phase 4 sync elimination (`907fc6c`) | 72.0 | 44.1 |
| + INT4 launch-config retune (`5e89d0c`) | **75.7** | 41.9 |

Single-model baseline (target only, CUDA graphs): ~60 tok/s, unmoved by either
change.

> Note: earlier docs quoted 79.44 tok/s for Phase 3 against a 65–68 baseline.
> That machine state does not reproduce — Phase 3 measures 69.3 here against a
> 60 baseline. **Only compare numbers from the same session.**

---

## What Phase 4 actually bought, and why the old plan was wrong

The previous plan attributed ~13ms/step to Python and GPU→CPU sync overhead and
projected ~109 tok/s from removing it. Syncs did drop from ~6–8 per step to 1,
and everything the plan listed was implemented — but it bought **~2ms/step**,
not 13ms.

Phase-level profile after Phase 4 (`torch.cuda.synchronize` around each phase,
63 steps, 200 tokens):

| phase | ms/step | share |
|---|---|---|
| draft replays (4.35 × 5.14ms) | 22.36 | 51% |
| target verify (1 × 19.18ms) | 19.18 | 44% |
| batched accept/reject | 0.19 | 0.4% |
| all remaining Python | 2.00 | 4.6% |

And a **bare CUDA graph replay** — no Python at all — costs 4.48ms (draft) and
18.75ms (verify), i.e. ~41ms of the 43.7ms step is kernel time inside the
graphs. **There is no meaningful Python overhead left to remove.** Any further
plan that targets host-side work is targeting 2ms.

Verified with the `device-string-and-noop-feature-audit` checklist: draft and
target both report `enable_cuda_graphs=True`, 3 draft graphs + 5 verify graphs
captured, and **0 eager fallbacks** over a full 200-token run. The graphs are
genuinely active; the time is real GPU work.

---

## Where the remaining 41ms/step is

### Target verify — 18.8ms, 71% in one kernel

`_int4_matmul_kernel`: 252 calls/replay (36 layers × 7 projections), 13.36ms
before the retune. That is 4.02GB of INT4 weights streamed per forward:

| | GB/s | % of 896 GB/s peak |
|---|---|---|
| before retune | ~300 | 34% |
| after retune | ~350 | 39% |

Per-shape, at M=5, after retuning (sweep in the commit message of `5e89d0c`):

| projection | N | GB/s | % peak |
|---|---|---|---|
| k, v | 1024 | 98 | **11%** |
| q, o, down | 4096 | 315 | 35% |
| gate, up | 12288 | 515 | 57% |
| lm_head | 151936 | 630 | 70% |

### Draft model — 4.79ms/replay, and it is *not* bandwidth-bound

Kernel breakdown of one 0.6B draft replay (28 layers):

| kernel class | ms | share |
|---|---|---|
| cuBLAS gemv (197 calls) | 2.07 | 43% |
| memory-efficient attention (28) | 0.54 | 11% |
| elementwise / reduce / gather / cat (~500 calls) | 2.18 | 46% |

~560 kernels per replay, ~18 per layer. Nearly half the draft's cost is small
elementwise work (RMSNorm reductions, RoPE, SwiGLU, KV gather + cat), each
kernel too small to saturate the GPU. Weight streaming alone would be ~1.3ms.

---

## Next levers, in measured-value order

1. **Fuse q/k/v into one projection** (target and draft). The narrow N=1024 k
   and v matmuls run at 11% of peak purely from having too few blocks. A single
   N=6144 matmul measures 0.031ms vs 0.087ms for the three separate ones —
   **2.8× on the QKV projections, ~2.0ms/step**. Also removes 72 kernel launches
   per verify.

2. **Fuse the draft's elementwise chain** (RMSNorm, RoPE, SwiGLU). ~2.2ms of the
   draft's 4.79ms is small-kernel work; the draft runs 4.35× per step, so this
   is the largest single pool of time in the whole engine (~9.5ms/step). A fused
   RMSNorm and a fused SwiGLU are the obvious first two.

3. **Wire up `_int4_gemv_kernel`.** It is defined at `engine/kernels/quant.py:244`
   with a docstring explaining why M=1 decode needs it — and it is **never called
   from anywhere**. `int4_matmul` always launches the tiled kernel. This is dead
   code today; dispatching it when M==1 should help the single-model baseline
   (which the retune did not move). It will not help spec verify, which is M=5.

4. **Avoid the bonus-token draft sync step.** When all K tokens are accepted the
   engine runs an extra full draft forward purely to advance the draft KV cache
   by one position, discarding the output — 23 of 63 steps, ~4% of total time.
   Folding it into the next step's draft phase as a q_len=2 forward would
   reclaim most of that.

Levers 1 + 2 together are worth roughly 11ms/step, which is the ~31.6ms/step
that 100 tok/s requires at 3.16 tok/step. Levers 3 and 4 are smaller and
independent.

---

## Benchmark and profiling commands

```bash
# end-to-end
.venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B \
    --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4 --skip-eager-spec
```

Always run the A and B builds back to back in one session — machine state moves
the absolute numbers by ~15% between sessions.

Profiling scripts used for the tables above (phase timing, graph-activation
audit, kernel breakdown, launch-config sweep) were scratch files, not committed.
Rebuild them by monkeypatching `_step_one_graphed` / `_step_verify_graphed` with
synchronized timers, and by running `torch.profiler` over a bare
`cg.graph.replay()` loop.
