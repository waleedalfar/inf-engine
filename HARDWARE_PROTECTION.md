# Hardware Protection Plan

## Goals
1. GPU never enters a sustained 100% utilisation hang that cannot be interrupted.
2. VRAM is never over-committed (OOM deadlock = GPU stuck at 100%, requires hard reboot).
3. CPU/RAM are never fully saturated by the WSL2 VM.
4. Every protection is enforced in code — not just documentation.

---

## Threat → Guard mapping

| Threat | Symptom | Guard |
|---|---|---|
| VRAM over-commit post-quantisation | 20 GB allocated on 16 GB card; GPU deadlock at 100% | `check_vram_headroom()` before quantise |
| Silent decode hang | Model appears frozen; no output for minutes | Streaming token output (`VerboseAgentLoop._generate_to_eos`) |
| Decode watchdog needed | No way to tell if model is generating or stuck | Per-token timeout via `signal.alarm` |
| Prefill of enormous prompt | OOM during prefill (no KV cache relief) | `check_prompt_length()` before `model.forward()` |
| WSL2 VmmemWSL unbounded | Windows swap exhausted; system freeze | `.wslconfig` memory cap (`memory=16GB`) |
| GPU temp spike | Unchecked long generation at high power | VRAM headroom check (proxy; dedicated temp polling is future work) |

---

## Implemented

### 1. Quantisation VRAM check (`engine/guards.py` — to be added)

```python
def check_vram_headroom(required_gb: float, label: str = "") -> None:
    """Raise before an operation that would exceed VRAM."""
    if not torch.cuda.is_available():
        return
    free = torch.cuda.mem_get_info()[0] / 1e9
    if free < required_gb:
        raise RuntimeError(
            f"Insufficient VRAM for {label}: need {required_gb:.1f} GB, "
            f"only {free:.1f} GB free. Lower --quantize group size or use a smaller model."
        )
```

Call sites:
- Before `quantize_llama()` in `main.py`: require ≥ 4 GB free for 8B INT4 dequant scratch.
- Before `load_llama_weights()`: require ≥ model_size_gb + 1.5 GB headroom.

### 2. Streaming token output (implemented in `main.py`)

`VerboseAgentLoop._generate_to_eos` prints each token immediately as it is
sampled.  The user sees output within the first second of generation, so a
slow decode (INT4 JIT dequant = ~1 tok/s) is visible rather than appearing
frozen.

### 3. WSL2 memory cap (implemented in `C:\Users\walee\.wslconfig`)

```ini
[wsl2]
memory=16GB
swap=8GB
```

Apply with `wsl --shutdown` in PowerShell, then restart WSL.
This prevents VmmemWSL from consuming 100% of system RAM and making Windows
unresponsive even when the GPU is fine.

---

## Planned (Phase 7+ guards)

### `engine/guards.py`

```python
import signal, torch

def vram_headroom(required_gb, label=""):
    ...  # see above

def prompt_length_check(n_tokens, max_ctx):
    if n_tokens > max_ctx:
        raise ValueError(
            f"Prompt is {n_tokens} tokens but model n_ctx={max_ctx}. "
            "Use /clear to reset or shorten your message."
        )

class GenerationWatchdog:
    """Raises TimeoutError if no token is produced within `timeout` seconds."""
    def __init__(self, timeout=30):
        self.timeout = timeout
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.timeout)
    def __exit__(self, *_):
        signal.alarm(0)
    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Generation watchdog: no token produced in time. GPU may be hung.")
```

Wrap `_generate_to_eos` inner loop with `GenerationWatchdog(timeout=30)` —
if a single decode step takes > 30 s (GPU deadlock), the watchdog fires,
Python exits cleanly, and CUDA resets itself on the next launch.

### `tests/test_hardware_guards.py`

- `test_vram_check_raises_when_not_enough_free` — mock `mem_get_info`
- `test_vram_check_passes_when_enough_free`
- `test_prompt_length_check_raises`
- `test_generation_watchdog_fires` — mock `model.forward` to sleep forever

### `tests/test_quantize_memory.py`

- `test_bf16_originals_freed_after_quantize` — assert `orig._t` has no INT4-keyed entries after `quantize_llama()`
- `test_vram_drops_after_quantize` — compare `cuda.memory_allocated()` before/after

---

## Quick-reference: what to run if GPU hangs

1. **In WSL2**: `kill -9 $(pgrep python)` — kills the Python process; CUDA driver
   cleans up on its own.
2. **In PowerShell**: `wsl --shutdown` — terminates the entire VM and frees all VRAM
   and pinned RAM instantly.  Restart WSL before next run.
3. **If Windows is unresponsive**: hold power button 5 s → hard reboot is safe for
   SSD; GPU/CPU are not harmed by a power-cycle.

Never leave a suspected-hung GPU running overnight — sustained 100% utilisation
at high temperature shortens GPU lifespan.
