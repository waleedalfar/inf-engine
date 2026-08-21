# Continuing to 100+ tok/s

## Current state (2026-08-20)

All Phase 1+2 code is committed to main. 233 tests pass.

### Benchmark results (RTX 5070 Ti, Qwen3-8B INT4 + Qwen3-0.6B bf16 draft)

| Path | tok/s | Notes |
|---|---|---|
| LlamaPagedEngine + CUDA graphs (target only) | **65.35** | With quantize_lm_head=True |
| SpeculativePagedEngine (graphed draft + eager verify) | **61.48** | SLOWER than baseline |
| Old SpeculativeDecoder (eager, no graphs) | 39.72 | Reference |

**Spec is 6% slower than baseline.** Root cause: the `_step_verify_eager` method runs
an eager forward through all 36 layers of the 8B model, generating ~288 Python tensor
dispatches per spec step. The graphed baseline has 1 dispatch (`graph.replay()`). This
Python overhead (~20ms) eats the spec speedup from 77% acceptance rate.

Acceptance rate: 77.3%, tok/step: 3.16 — this is GOOD. The algorithm works, the
implementation is correct. It's purely the eager verify overhead killing throughput.

---

## What needs to happen: Phase 3 — Graph the verify step

### Root cause analysis

Per spec step (51.6ms total):
- 4 graphed draft steps (0.6B bf16): ~10ms (bandwidth-limited)
- 1 eager verify (8B INT4, K+1=5 tokens): ~20ms bandwidth + **~20ms Python overhead**
- Accept/reject Python: ~2ms

Break-even with baseline: 15.3ms (baseline) × 3.16 tok/step = 48.4ms per spec step.
We need to shave ~3ms off. Graphing the verify saves ~20ms → puts us well above 100 tok/s.

### Expected result after Phase 3

- 4 graphed draft steps: ~10ms
- 1 GRAPHED verify (K+1=5 tokens): ~15ms (same bandwidth, Python overhead → ~1ms)
- Accept/reject: ~2ms
- Total: ~27ms per spec step
- 3.16 tok/step / 0.027s = **~117 tok/s** (exceeds 100 tok/s target)

---

## Phase 3 implementation plan

### Step 1: Generalize `extend_static` for q_len > 1

**File:** `engine/paged_cache.py`, method `extend_static` (line ~349)

Currently handles q_len=1 only:
```python
block_idx = seq_lens_buf // bs          # (A,)
offset = seq_lens_buf % bs             # (A,)
phys = block_table_buf.gather(1, block_idx.unsqueeze(1)).squeeze(1)
self.k_pool[local_layer, phys, :, offset, :] = k_new[:, :, 0, :]
self.v_pool[local_layer, phys, :, offset, :] = v_new[:, :, 0, :]
```

Replace with a loop over q positions (safe to CUDA-graph because q_len is fixed at
capture time, so Python loop unrolls into static CUDA ops):
```python
q_len = k_new.shape[2]
for q in range(q_len):
    block_idx_q = (seq_lens_buf + q) // bs          # (A,)
    offset_q = (seq_lens_buf + q) % bs              # (A,)
    phys_q = block_table_buf.gather(1, block_idx_q.unsqueeze(1)).squeeze(1)  # (A,)
    self.k_pool[local_layer, phys_q, :, offset_q, :] = k_new[:, :, q, :]
    self.v_pool[local_layer, phys_q, :, offset_q, :] = v_new[:, :, q, :]
```

The existing q_len=1 path keeps working (loop runs once). No callers change.

### Step 2: Add `_GraphVerifyCache` adapter

**File:** `engine/llama_paged_engine.py`

New class (alongside `_GraphDecodeCache`):
```python
class _GraphVerifyCache:
    """Like _GraphDecodeCache but for q_len=K+1 verify steps."""
    def __init__(self, real_cache, block_table_buf, seq_lens_buf, capture_len):
        self._real = real_cache
        self.block_table_buf = block_table_buf
        self.seq_lens_buf = seq_lens_buf
        self.capture_len = capture_len

    def extend(self, layer, k_new, v_new, start_pos=0):
        return self._real.extend_static(
            layer, k_new, v_new, self.block_table_buf, self.seq_lens_buf, self.capture_len,
        )
```

This is essentially identical to `_GraphDecodeCache`. Could unify them, but keeping separate
is cleaner since the verify graph has different static buffer shapes.

### Step 3: Add `_CapturedVerifyGraph` dataclass

```python
@dataclass
class _CapturedVerifyGraph:
    """CUDA graph for a fixed (len_bucket, q_len) verify shape (batch=1 always)."""
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor          # (1, q_len) long
    position_ids: torch.Tensor       # (1, q_len) long
    block_table_buf: torch.Tensor    # (1, capture_len // block_size) long
    seq_lens_buf: torch.Tensor       # (1,) long
    attn_mask_buf: torch.Tensor      # (1, q_len, capture_len) bool
    logits: torch.Tensor             # (1, q_len, vocab)
    capture_len: int
    q_len: int
```

### Step 4: Add `_verify_graphs` dict and `_capture_verify_graph` method to `LlamaPagedEngine`

In `__init__`: add `self._verify_graphs: dict[tuple[int, int], _CapturedVerifyGraph] = {}`
                                               (len_bucket, q_len) → graph

New method `_capture_verify_graph(len_bucket, q_len, seq_id)`:
```python
def _capture_verify_graph(self, len_bucket, q_len, seq_id):
    # Build static buffers for batch=1
    block_table_buf, seq_lens_buf = self.cache.build_static_buffers(
        [seq_id], 1, len_bucket, self.device
    )
    # seq_lens_buf holds current length L; input spans positions L..L+q_len-1
    L = int(seq_lens_buf[0].item())
    input_ids = torch.zeros(1, q_len, dtype=torch.long, device=self.device)
    position_ids = torch.arange(L, L + q_len, device=self.device).unsqueeze(0)
    ar = torch.arange(len_bucket, device=self.device)
    q_abs = position_ids[0]
    attn_mask = (ar[None, :] <= q_abs[:, None])[None, :, :]  # (1, q_len, len_bucket)

    wrapper = _GraphVerifyCache(self.cache, block_table_buf, seq_lens_buf, len_bucket)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            self.model.forward(input_ids, cache=wrapper, start_pos=0,
                               position_ids=position_ids, attn_mask=attn_mask)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_logits = self.model.forward(
            input_ids, cache=wrapper, start_pos=0,
            position_ids=position_ids, attn_mask=attn_mask,
        )

    return _CapturedVerifyGraph(
        graph=graph, input_ids=input_ids, position_ids=position_ids,
        block_table_buf=block_table_buf, seq_lens_buf=seq_lens_buf,
        attn_mask_buf=attn_mask, logits=static_logits,
        capture_len=len_bucket, q_len=q_len,
    )
```

### Step 5: Add `_step_verify_graphed` method

```python
@torch.no_grad()
def _step_verify_graphed(self, seq_id: int, verify_ids: list[int]) -> torch.Tensor:
    """Graphed verify step. Falls back to eager when CUDA graphs not available."""
    if not self.enable_cuda_graphs:
        return self._step_verify_eager(seq_id, verify_ids)

    q_len = len(verify_ids)
    L = self.cache.seq_lens[seq_id]
    seq_len_after = L + q_len
    bucket = self._pick_bucket(1, seq_len_after)
    if bucket is None:
        return self._step_verify_eager(seq_id, verify_ids)
    _, len_bucket = bucket

    vg = self._verify_graphs.get((len_bucket, q_len))
    if vg is None:
        vg = self._capture_verify_graph(len_bucket, q_len, seq_id)
        self._verify_graphs[(len_bucket, q_len)] = vg

    # Fill static input buffers
    self.cache.ensure_slots_for(seq_id, q_len)
    block_table_buf, seq_lens_buf = self.cache.build_static_buffers(
        [seq_id], 1, len_bucket, self.device
    )
    vg.block_table_buf.copy_(block_table_buf)
    vg.seq_lens_buf.copy_(seq_lens_buf)
    vg.input_ids.copy_(torch.tensor([verify_ids], dtype=torch.long, device=self.device))
    vg.position_ids.copy_(torch.arange(L, L + q_len, device=self.device).unsqueeze(0))
    ar = torch.arange(len_bucket, device=self.device)
    q_abs = vg.position_ids[0]
    vg.attn_mask_buf.copy_((ar[None, :] <= q_abs[:, None])[None, :, :])

    vg.graph.replay()

    # extend_static doesn't advance seq_lens — do it manually
    self.cache.seq_lens[seq_id] += q_len

    return vg.logits[0]  # (q_len, vocab)
```

### Step 6: Route `SpeculativePagedEngine` to use graphed verify

In `engine/speculative_paged_engine.py`, `_generate_one`:
Change:
```python
t_logits = target._step_verify_eager(t_sid, verify_ids)
```
To:
```python
t_logits = target._step_verify_graphed(t_sid, verify_ids)
```

That's it. `_step_verify_graphed` falls back to eager automatically when needed.

### Step 7: Also enable CUDA graphs on target engine in `main.py`

Currently `main.py` sets `target_engine` with `enable_cuda_graphs=False`. Change to `True`
so `_step_verify_graphed` actually uses the graph path.

```python
target_engine = LlamaPagedEngine(
    model,
    ...
    enable_cuda_graphs=True,   # was False — needed for graphed verify
)
```

---

## Additional quick wins (try after Phase 3)

1. **Quantize draft model**: currently bf16 (1.2GB), INT4 would be ~0.28GB.
   `load_model(..., quantize=True)` for draft. Each of 4 draft steps becomes ~3× faster.
   In `bench/paged_spec_bench.py` and `main.py` spec+graphs branch.

2. **n_draft tuning**: try n_draft=6 or 8. At 77% acceptance, higher K increases tok/step.
   With graphed verify (fixed overhead), more tokens per step = more throughput.

---

## Files to change for Phase 3

| File | Change |
|---|---|
| `engine/paged_cache.py` | `extend_static`: loop over q_len positions (5-line change) |
| `engine/llama_paged_engine.py` | Add `_GraphVerifyCache`, `_CapturedVerifyGraph`, `_verify_graphs`, `_capture_verify_graph`, `_step_verify_graphed` |
| `engine/speculative_paged_engine.py` | Call `_step_verify_graphed` instead of `_step_verify_eager` |
| `main.py` | Set `enable_cuda_graphs=True` on target engine in spec+graphs branch |
| `tests/test_speculative_paged.py` | Add test for `_step_verify_graphed` (same as eager but graphed path) |

---

## Benchmark command to verify

```bash
wsl bash -c "cd /home/waleed/mlproj && .venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B \
    --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4"
```

Goal: paged spec > 100 tok/s (baseline currently at 65.35 tok/s).

## Key correctness invariant to verify

After graphing the verify step, the `seq_lens[seq_id]` must be advanced by q_len manually
(same as `_step_one_graphed` advances by 1). The line:
```python
self.cache.seq_lens[seq_id] += q_len
```
in `_step_verify_graphed` handles this. Verify with the same greedy-identity tests in
`test_speculative_paged.py`.
