"""Phase 4 correctness gate: continuous batching == per-request single-sequence output.

The invariant: scheduling only changes *when* tokens are produced, never *which* tokens.
With greedy decoding, each request's continuous-batching output must equal generating that
prompt on its own (Phase 2 cached path), token-for-token — even when:
  * slots are fewer than requests (forces slot recycling / cache dealloc+realloc), and
  * prompts and output lengths vary (variable-length slots in the same batch), and
  * the scheduler is FCFS or SJF (output content must be policy-independent).

Run:  pytest tests/test_phase4_continuous.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.continuous import ContinuousBatchingEngine, Policy, Request
from engine.config import GPT2_SMALL
from engine.kv_cache import StaticKVCache
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, SamplingMode
from engine.tokenizer import GPT2Tokenizer
from engine.weights import load_weights

MODEL_NAME = "gpt2"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / MODEL_NAME / "model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)
MAX_SEQ = 256

# Varying prompt lengths and output budgets so slots are genuinely variable-length.
SPECS = [
    ("The capital of France is", 25),
    ("Once upon a time, in a land far away,", 30),
    ("def fibonacci(n):", 20),
    ("I think therefore", 35),
    ("Machine learning is", 28),
]


@pytest.fixture(scope="module")
def model() -> GPT2Model:
    if not WEIGHTS_PATH.is_file():
        pytest.skip(f"weights missing: {WEIGHTS_PATH}")
    weights = load_weights(WEIGHTS_PATH, GPT2_SMALL, device=DEVICE, dtype=DTYPE)
    return GPT2Model(weights, GPT2_SMALL)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def _reference(model: GPT2Model, ids: list[int], n: int) -> list[int]:
    """Per-request ground truth via the Phase 2 single-sequence cached path."""
    x = torch.tensor([ids], device=DEVICE)
    cache = StaticKVCache(GPT2_SMALL, 1, len(ids) + n, DEVICE, DTYPE)
    out = model.generate_cached(x, n, cache, GREEDY)
    return out[0, len(ids):].tolist()


@pytest.mark.parametrize("policy", [Policy.FCFS, Policy.SJF])
def test_continuous_matches_individual(model: GPT2Model, tokenizer: GPT2Tokenizer, policy: Policy) -> None:
    reqs = [
        Request(req_id=i, prompt_ids=tokenizer.encode(text), max_new_tokens=n)
        for i, (text, n) in enumerate(SPECS)
    ]
    # n_slots < number of requests -> forces recycling and cache realloc.
    engine = ContinuousBatchingEngine(
        model, n_slots=2, max_seq=MAX_SEQ, policy=policy, sampling=GREEDY, eos_token=None
    )
    outputs = engine.run_offline(reqs)

    assert len(outputs) == len(SPECS)
    for i, (text, n) in enumerate(SPECS):
        ref = _reference(model, tokenizer.encode(text), n)
        assert outputs[i] == ref, (
            f"[{policy}] req {i} ({text!r}) diverged:\n"
            f" continuous: {tokenizer.decode(outputs[i])!r}\n"
            f" reference : {tokenizer.decode(ref)!r}"
        )


def test_fcfs_and_sjf_same_content(model: GPT2Model, tokenizer: GPT2Tokenizer) -> None:
    """Both policies must yield identical per-request tokens (only ordering differs)."""
    def run(policy: Policy) -> dict[int, list[int]]:
        reqs = [
            Request(req_id=i, prompt_ids=tokenizer.encode(t), max_new_tokens=n)
            for i, (t, n) in enumerate(SPECS)
        ]
        eng = ContinuousBatchingEngine(model, 3, MAX_SEQ, policy=policy, sampling=GREEDY)
        return eng.run_offline(reqs)

    assert run(Policy.FCFS) == run(Policy.SJF)
