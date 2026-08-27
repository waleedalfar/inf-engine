---
name: context-discipline
description: Use at the START of any working session in this repo, and before any batch of file edits, file reads, benchmark runs, or commits. Covers how to edit files (Edit/Write, never shell heredocs — they silently no-op and shipped a real bug), how to read them (line ranges, not whole modules), how to run commands (filtered output), how to commit (short messages), and how to collect benchmark results (batched to a file). These habits are what keep long measurement-heavy sessions from exhausting the context window; the 2026-08-23 session burned most of its budget on avoidable re-dumps.
---

# Context discipline

Sessions in this repo are long and measurement-heavy: kernel work, benchmark
sweeps, profiling, doc updates. The context window is a real budget, and it is
almost always spent on avoidable mechanics rather than on thinking.

These are ordered by how much they actually cost, measured against the
2026-08-23 session.

## 1. Edit files with Edit/Write — never shell heredocs

**This is the big one, and it is a correctness rule before it is a context rule.**

Do not do this:

```bash
python3 - <<'PYEOF'
p = 'engine/llama_paged_engine.py'
s = open(p).read()
s = s.replace("old text", "new text")
open(p, 'w').write(s)
PYEOF
```

Two things go wrong:

**It silently no-ops.** `str.replace` on a non-matching pattern changes nothing
and reports nothing. On 2026-08-23 a `base` parameter was added to
`_prefill_forward`'s signature while the body patch failed to match — the file
had been rewritten underneath by other work. The result accepted the argument
and ignored it. Cross-turn prefix reuse then produced *fluent but wrong* output
(94–152% logit error) with nothing raised. `Edit` errors on a failed match; the
heredoc shipped the bug.

**It blows the context budget.** Writing a file outside the harness's tracking
makes it re-dump the *entire file* on its next mention — 200+ lines of
`CLAUDE.md`, `llama_paged_engine.py`, `paged_attention.py`, `main.py`,
repeatedly, for one-line changes. This was the single largest consumer of
context in that session.

Use `Edit` for changes, `Write` for new files or full rewrites. Both keep file
state tracked, and `Edit` fails loudly when its anchor is gone — which is
exactly the signal you want when a file has moved under you.

## 2. Read line ranges, not whole modules

`Read` on a 700-line module to check one function costs 700 lines of context.

```bash
sed -n '300,340p' engine/llama_paged_engine.py     # the function
grep -n "def _prefill_forward" -A 25 engine/llama_paged_engine.py
grep -n "^def \|^class " engine/paged_cache.py     # structure first, then narrow
```

Read the whole file only when you genuinely need the whole file — reviewing it,
or about to rewrite it.

## 3. Filter every command's output

Never let raw output land unfiltered.

```bash
pytest -q 2>&1 | tail -3                  # not the full run
pytest -q 2>&1 | grep "^FAILED"           # when something broke
nvidia-smi --query-gpu=memory.used --format=csv,noheader
git log --oneline -3
```

A full `pytest -v` on 369 tests is hundreds of lines that say nothing a
three-line tail does not.

## 4. Short commit messages

The message is echoed in full in the tool call, so a 25-line commit body costs
25 lines twice. Keep it to the finding and the number:

```
Fix int32 overflow in INT4 matmul at long prefill

offs_m * stride_cm reaches M * vocab_size; past int32 at M >= ~14134 for a
151936 vocab, so prefills of ~14k tokens returned all-zero logits. Row indices
are now int64.
```

The reasoning, the measurements, and the history belong in `CLAUDE.md` — written
once, where the next session will actually read them.

## 5. Batch benchmark output

A long sweep emitting one notification per row costs a message per row. Write
the run to a file and read the table once:

```bash
nohup .venv/bin/python -u -m bench.ctx_scaling_bench ... > /tmp/run.txt 2>&1 &
# ... later, once ...
tail -12 /tmp/run.txt
```

Per-row streaming is worth it only when a row would change what you do next —
an OOM, a guard firing, a result that should abort the sweep. Otherwise collect
and read once.

---

## When context does run short

Update `CLAUDE.md` — the milestone table, the progress log, and whichever plan
item is in flight — then stop. That file plus the git log is the handoff; a
fresh session reads it first and should not have to re-derive anything.
