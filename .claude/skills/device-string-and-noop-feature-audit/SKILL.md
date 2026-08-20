---
name: device-string-and-noop-feature-audit
description: Use when adding or reviewing any torch device-dispatch gate (`if device == "cuda"`, `if device == "cpu"`, etc.), and more generally whenever adding a feature flag / optimized code path (CUDA graphs, torch.compile, fused kernels, quantization) that should be verified as actually active — not just verified by comparing its output to the fallback path. Also use when a benchmark for a claimed speedup comes back suspiciously close to 1.0x.
---

# Device-string comparisons and silently-inactive feature flags

## The bug class

`str(tensor.device)` on a real CUDA tensor renders as `"cuda:0"` (device index
included), never bare `"cuda"`. Any gate written as:

```python
device = str(model.some_tensor.device)
enabled = flag and device == "cuda"          # BUG: always False on real GPUs
```

silently evaluates `False` forever — no exception, no warning, the code just
takes the fallback path unconditionally regardless of what the caller passed.

**Fix:** compare device *kind*, not the raw string:

```python
enabled = flag and torch.device(device).type == "cuda"   # "cuda:0" -> "cuda"
```

Audit any `== "cuda"` / `== "cpu"` string comparison in the codebase whenever
touching device-dispatch logic — grep for `== "cuda"` and `== "cpu"`.

## The broader lesson: output-equality tests can't catch "this path never ran"

This exact bug shipped past working tests. The tests compared a
feature-enabled engine's *output* against a feature-disabled engine's output
and asserted equality — but both were silently running the same (disabled)
code path, so the comparison was a tautology that always passes. It was only
caught because:

1. One test happened to assert the flag's own internal state directly
   (`assert engine.enable_cuda_graphs is True`) — this failed with the
   feature silently off.
2. Another test asserted a side effect that only the new path produces
   (`assert len(engine._graphs) >= 1` after a run that should have captured
   a graph) — this failed with `0 >= 1`.
3. A throughput benchmark for the claimed optimization came back at ~1.03x
   instead of the expected multi-x speedup — a red flag that should be
   investigated as "is this even running the new code," not written off as
   "the synthetic benchmark is too small to show gains."

**When adding any optimized/accelerated code path (CUDA graphs,
`torch.compile`, fused kernels, INT4/INT8 quant, speculative decoding,
whatever):**

- Assert the path's own activation state directly, not just its output vs.
  the fallback (e.g. a boolean flag reflecting reality, a populated cache, an
  incremented counter, a captured graph count).
- Assert at least one side effect that ONLY the new path produces.
- If benchmarking a claimed speedup, treat a near-1.0x result as a
  correctness bug to investigate first, a disappointing-but-real result
  second.
