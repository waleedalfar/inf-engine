# 100 tok/s: reached

## Result (2026-08-22)

**103.8 tok/s** speculative decode on RTX 5070 Ti (Qwen3-8B INT4 target +
Qwen3-0.6B bf16 draft, n_draft=4), up from 69.3 at the start of the day.
271 tests pass.

| Step | tok/s | ms/step | commit |
|---|---|---|---|
| Phase 3 (starting point) | 69.3 | 46.0 | `54954ed` |
| Phase 4: one GPU sync per step | 72.0 | 44.1 | `907fc6c` |
| INT4 matmul launch-config retune | 75.7 | 41.9 | `5e89d0c` |
| Fused RMSNorm kernel | 81.6 | 38.8 | `5e80d73` |
| Fused Q/K/V and gate/up projections | 88.6 | 35.3 | `6441b71` |
| Fused RoPE kernel + cached RoPE tables | **103.8** | **30.4** | `37a1069` |

Single-model baseline (target only, CUDA graphs) went 60 → 80 tok/s over the
same work.

**Always A/B in one session.** Absolute numbers drift ~15% between sessions;
every figure above is from a back-to-back comparison.

---

## The correctness bug found along the way

Greedy speculative decode must be token-identical to greedy decode. It was not:
real-model output tracked the baseline for ~10 tokens, then collapsed into
`the capital of the capital of the capital of …`.

`_step_verify_graphed` captured its CUDA graph **before** calling
`ensure_slots_for`. Capture runs warmup forwards that *write* KV, and
`build_static_buffers` pads block-table columns beyond a sequence's allocated
blocks with **that sequence's own block 0**. Padding is safe for reads —
`attn_mask` hides those positions — but a write through a padded column lands on
the sequence's real positions `0..block_size-1`. The first step needing a new
`(len_bucket, q_len)` therefore overwrote its own prompt KV.

Fixed in `686a73d`: allocate before capture, and `build_static_buffers` now
takes `write_len` and raises when the caller has not allocated what it is about
to write.

**Why every test missed it:** the suite generated at most 15 tokens with
`block_size=16`, so it never crossed a bucket boundary. The regression test now
generates 60 tokens to cross three, and asserts `len(_verify_graphs) > 1` so it
cannot silently stop exercising the boundary.

The bug predates the optimization work — confirmed by reproducing it at
`0d43348`. Worth remembering that the whole speedup run happened on top of it
without anyone noticing, because throughput benchmarks do not check output.

---

## Where the time goes now (30.4 ms/step)

| phase | ms/step | share |
|---|---|---|
| draft replays (4.35 × ~3.5ms) | ~15.2 | 50% |
| target verify (1 × ~13ms) | ~13.0 | 43% |
| batched accept/reject | 0.2 | 0.5% |
| Python | 1.9 | 6% |

Inside the target verify, `_int4_matmul_kernel` is still ~70%, now 145 calls
(36 layers × 4 fused projections + lm_head) instead of 251.

### What did and didn't work

- **Python sync elimination gave +3.9%, not the projected +45%.** The old plan
  assumed ~13ms/step of host overhead; it was ~2ms. A *bare* CUDA graph replay
  (no Python at all) accounted for ~41ms of the then-43.7ms step.
- **Kernel count, not bandwidth, dominated the small draft model.** ~560 kernels
  per replay for a 0.6B model; RMSNorm alone was 113 invocations × 6 kernels.
  Fusing RMSNorm and RoPE was worth more than anything done to the matmuls.
- **Tile tuning was measured wrong the first time.** Re-timing one weight tensor
  in a loop reads it from L2 (64MB on GB203), not HBM. The hot sweep predicted
  1.16x and delivered ~0 in-graph. Any kernel sweep here must rotate through
  enough distinct weights to bust L2.
- **The narrow projections were an occupancy problem no tile size could fix.**
  K/V at N=1024 ran at 8% of peak: `BLOCK_N=32` wastes 3/4 of every 128-byte
  cache line, and `BLOCK_N=128` reads whole lines but leaves only 32 blocks for
  70 SMs. Fusing to N=6144 got both.

---

## Remaining levers, in measured-value order

1. **Split-K for the INT4 matmul.** Still the single biggest kernel. The
   coalescing/occupancy tradeoff above is exactly what split-K breaks:
   `BLOCK_N=128` for full cache lines, ×4 K-splits for a full grid. Needs fp32
   atomics or a partials buffer, so budget an extra kernel per matmul (~4µs ×
   145) against the gain.
2. **Fused SwiGLU** (`silu(gate) * up` in one kernel) — the same
   many-tiny-kernels problem RMSNorm and RoPE had, now among the largest
   remaining elementwise groups.
3. **Drop the discarded bonus-token draft forward.** When all K tokens are
   accepted the engine runs a full draft forward purely to advance the draft KV
   cache one position, throwing the output away — 24 of 64 steps, ~1.5ms/step.
   Folding it into the next step's draft phase as a q_len=2 forward reclaims most.
4. **Wire up `_int4_gemv_kernel`** (`engine/kernels/quant.py:244`). Written and
   documented for the M=1 decode case, never called from anywhere.
   `int4_matmul` always launches the tiled kernel. Helps the single-model
   baseline, not spec verify (M=5).

---

## Commands

```bash
# throughput
.venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B \
    --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4 --skip-eager-spec

# correctness — greedy spec must be token-identical to greedy baseline
.venv/bin/pytest tests/test_speculative_paged.py -q
```

Throughput benchmarks do not validate output. After any change to attention,
the KV cache, or graph capture, check greedy spec against greedy baseline on a
real model and a >50-token generation, not just the test suite.
