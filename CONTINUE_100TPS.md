# Phase 4: Eliminate Python overhead to hit 100+ tok/s

## Current state (2026-08-20)

Phase 3 done (commit 8a12bfe). 243 tests pass.

### Benchmark (RTX 5070 Ti, Qwen3-8B INT4 + Qwen3-0.6B bf16, n_draft=4)
| Path | tok/s |
|---|---|
| LlamaPagedEngine + CUDA graphs (baseline) | 65–68 |
| SpeculativePagedEngine (graphed draft + graphed verify) | **79.44** |

n_draft tuning result: n_draft=4 is already optimal. Going higher costs more step time
than it gains in tok/step (n_draft=6: 71.7 tok/s, n_draft=8: 60.7 tok/s).

---

## Root cause: Python overhead per spec step

Measured step times:
- n_draft=4: **~40ms/step** at 3.16 tok/step → 79 tok/s
- Estimated compute: ~12ms (4 draft steps) + ~15ms (verify) = ~27ms
- **~13ms is pure Python/sync overhead per step**

### Overhead sources in `SpeculativePagedEngine._generate_one`

**1. GPU→CPU syncs in accept/reject loop (up to K=4 per step)**
```python
accept_prob = min(1.0, (p_t[tok_id] / p_d[tok_id]).item())  # sync
u = torch.rand(1).item()                                      # CPU random
corr = int(_correction_sample(...).item())                    # sync on reject
```
Each `.item()` stalls until the GPU kernel finishes. Up to K+1 syncs per step.

**2. Draft step `.item()` syncs (K per step)**
`_step_one_graphed` calls `sample_next_token(...).item()` to convert the sampled token
to a Python int, then wraps it back in a `torch.tensor([tok])` for `_active`. This is
a GPU→CPU→GPU round trip for every draft step.

**3. `build_static_buffers` called K+1 times per step**
Called once per draft step inside `_step_one_graphed` + once for the verify step.
Each call is a Python loop over the block table.

**4. `ensure_slot` called K times per step (once before each draft step)**
Could be batched to one `ensure_slots_for(d_sid, K)` call at the start of each spec
loop iteration.

**5. `_make_ctx` rebuilds GPU tensor every step**
`torch.tensor(generated, device=...)` copies a growing Python list to GPU each step.

---

## Implementation plan

### Step 1: Batch slot pre-allocation
**File:** `engine/speculative_paged_engine.py`, `_generate_one`

Replace the K individual `draft.cache.ensure_slot(d_sid)` calls (one before each
draft step inside the loop) with a single call at the top of the spec iteration:
```python
draft.cache.ensure_slots_for(d_sid, K)
target.cache.ensure_slots_for(t_sid, K + 1)  # K draft positions + 1 bonus
```
Remove the per-step `draft.cache.ensure_slot(d_sid)` and `target.cache.ensure_slot`
calls inside the loop body.

### Step 2: Defer draft `.item()` syncs
**File:** `engine/llama_paged_engine.py`, `_step_one_graphed`

Change the return type to yield a GPU tensor instead of a Python int:
```python
# Before:
tok = int(sample_next_token(...).item())
self._active[seq_id] = (req, torch.tensor([tok], device=self.device))
return tok, logits_row

# After:
tok_t = sample_next_token(logits_row.unsqueeze(0), self.cfg, ctx_2d).view(1)  # GPU tensor
self._active[seq_id] = (req, tok_t)
return tok_t, logits_row   # tok_t is (1,) GPU tensor
```

Signature change: `_step_one_graphed` returns `(torch.Tensor, torch.Tensor)` instead
of `(int, torch.Tensor)`.

In `_generate_one`, collect `tok_t` tensors across K steps, then call `.tolist()` ONCE:
```python
draft_tok_tensors: list[torch.Tensor] = []
draft_logits: list[torch.Tensor] = []
for k in range(K):
    tok_t, logits = draft._step_one_graphed(d_sid, draft_ctx)
    draft_tok_tensors.append(tok_t)
    draft_logits.append(logits)
draft_tokens = torch.stack(draft_tok_tensors).view(-1).tolist()  # ONE sync for all K
```

Callers of `_step_one_graphed` outside `_generate_one`:
- `_generate_one` bonus-sync step: currently calls `draft._step_one_graphed(d_sid, draft_ctx)`
  and discards output — still works, just ignores the returned tensor.
- The K<=0 path uses `_step_one_eager`, which stays unchanged (returns int).

### Step 3: GPU-batched accept/reject
**File:** `engine/speculative_paged_engine.py`, `_generate_one`

Replace the per-token accept/reject loop with a single GPU pass:
```python
# Stack draft logits and compute probs — all on GPU
draft_ids_gpu = torch.tensor(draft_tokens, device=target.device)      # (K,)
p_t_batch = _get_probs_batch(t_logits[:K], target.cfg, context_ids)   # (K, vocab)
p_d_batch = _get_probs_batch(
    torch.stack(draft_logits), draft.cfg, draft_ctx)                   # (K, vocab)

p_t_at_draft = p_t_batch[torch.arange(K, device=target.device), draft_ids_gpu]  # (K,)
p_d_at_draft = p_d_batch[torch.arange(K, device=target.device), draft_ids_gpu]  # (K,)
accept_probs = torch.clamp(p_t_at_draft / (p_d_at_draft + 1e-10), max=1.0)      # (K,)
u = torch.rand(K, device=target.device, generator=target.cfg.generator)           # (K,)
accepted = (u < accept_probs)  # (K,) bool — stays on GPU

# ONE sync: find first rejection
rej_indices = accepted.logical_not().nonzero(as_tuple=False)
n_accepted = int(rej_indices[0, 0].item()) if len(rej_indices) > 0 else K  # single sync
```

Add helper `_get_probs_batch` to `speculative_paged_engine.py`:
```python
def _get_probs_batch(logits: torch.Tensor, cfg: SamplingConfig,
                     context_ids: torch.Tensor | None) -> torch.Tensor:
    """Batched version of _get_probs: logits (K, vocab) → probs (K, vocab)."""
    # Apply repetition penalty row-wise if needed, then softmax.
    # For greedy/top-p without rep penalty: just softmax.
    return torch.softmax(logits.float(), dim=-1)
```
Full implementation needs to handle rep penalty per-row (context_ids stays 1-D shared
across all K positions, same as current single-token version).

After finding `n_accepted`:
```python
emitted = draft_tokens[:n_accepted]
if n_accepted < K:
    # Sample correction from adjusted distribution
    corr = int(_correction_sample(
        p_t_batch[n_accepted], p_d_batch[n_accepted], target.cfg).item())  # 1 sync
    emitted.append(corr)
    # Rollback caches
    target.cache.reset_to(t_sid, L + n_accepted + 1)
    draft.cache.reset_to(d_sid, L + n_accepted + 1)
    ...
else:
    # Bonus token from t_logits[K]
    p_bonus = _get_probs(t_logits[K], target.cfg, context_ids)
    bonus = int(_sample_from_probs(p_bonus, target.cfg).item())  # 1 sync
    emitted.append(bonus)
    ...
```

### Step 4: Rolling context tensor
**File:** `engine/speculative_paged_engine.py`, `_generate_one`

Replace `_make_ctx(req.generated, device)` with a maintained GPU tensor:
```python
# Before spec loop:
ctx_gpu = torch.tensor(req.generated, device=target.device) if req.generated else None

# After emitting tokens each step (instead of rebuilding from scratch):
new_toks = torch.tensor(emitted, device=target.device)
ctx_gpu = torch.cat([ctx_gpu, new_toks]) if ctx_gpu is not None else new_toks
```
Pass `ctx_gpu` directly where `context_ids` was passed. Eliminates one `torch.tensor(list)`
copy per step (grows with generation length).

---

## Sync count: before vs after

| Source | Before | After |
|---|---|---|
| Draft token `.item()` | K (one per step) | 0 (deferred to one `.tolist()`) |
| Draft `.tolist()` batch | — | 1 |
| Accept/reject per-token | Up to K+1 | 0 |
| Find first rejection | — | 1 |
| Correction sample | 0 or 1 | 0 or 1 |
| Bonus token | 0 or 1 | 0 or 1 |
| **Total** | **~6–8** | **~2–3** |

---

## Files to change

| File | Change |
|---|---|
| `engine/llama_paged_engine.py` | `_step_one_graphed`: return `(tok_t: Tensor, logits_row)` instead of `(int, logits_row)` |
| `engine/speculative_paged_engine.py` | `_generate_one`: Steps 1–4 above; add `_get_probs_batch` |
| `tests/test_speculative_paged.py` | Update tests that inspect `_step_one_graphed` return type; add timing assertion |

---

## Expected result

Syncs: 6–8 → 2–3 per step. Estimated step time: 40ms → ~28–30ms.
Projected: 3.16 tok/step / 0.029s = **~109 tok/s** (above 100 tok/s target).

## Benchmark command
```bash
wsl bash -c "cd /home/waleed/mlproj && .venv/bin/python -m bench.paged_spec_bench \
    --model-dir weights/Qwen--Qwen3-8B \
    --draft-model-dir weights/Qwen--Qwen3-0.6B \
    --max-new-tokens 200 --n-draft 4 --skip-eager-spec"
```
