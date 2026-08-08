"""Phase 3 correctness gate: static batching must equal per-sequence generation.

The invariant: padding, the key padding mask, and per-row position ids must make a
left-padded batch produce, for each row, exactly the tokens that prompt would produce on
its own. If padding leaked into attention or positions were wrong, mixed-length batches
would diverge — so we test with prompts of deliberately different lengths.

Checks (greedy, fixed token count):
  1. batch=1 == single-sequence cached generation.
  2. mixed-length batch: each row == that prompt generated alone.

Run:  pytest tests/test_phase3_batching.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.batching import generate_batched
from engine.config import GPT2_SMALL
from engine.kv_cache import StaticKVCache
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, SamplingMode
from engine.tokenizer import GPT2Tokenizer
from engine.weights import load_weights

MODEL_NAME = "gpt2"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / MODEL_NAME / "model.safetensors"
# Deliberately different lengths so left-padding actually does something.
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "def fibonacci(n):",
    "I think therefore",
]
NEW_TOKENS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)


@pytest.fixture(scope="module")
def model() -> GPT2Model:
    if not WEIGHTS_PATH.is_file():
        pytest.skip(f"weights missing: {WEIGHTS_PATH}")
    weights = load_weights(WEIGHTS_PATH, GPT2_SMALL, device=DEVICE, dtype=DTYPE)
    return GPT2Model(weights, GPT2_SMALL)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def _single(model: GPT2Model, ids: list[int]) -> torch.Tensor:
    """Generate NEW_TOKENS greedily for one prompt via the Phase 2 cached path."""
    x = torch.tensor([ids], device=DEVICE)
    cache = StaticKVCache(GPT2_SMALL, 1, len(ids) + NEW_TOKENS, DEVICE, DTYPE)
    out = model.generate_cached(x, NEW_TOKENS, cache, GREEDY)
    return out[:, len(ids):]                                # (1, NEW_TOKENS) generated only


def test_batch_of_one_matches_single(model: GPT2Model, tokenizer: GPT2Tokenizer) -> None:
    ids = tokenizer.encode(PROMPTS[0])
    batched = generate_batched(model, [ids], NEW_TOKENS, tokenizer.eot_token, GREEDY)  # (1, N)
    single = _single(model, ids)
    assert torch.equal(batched, single)


def test_mixed_length_batch_matches_individual(model: GPT2Model, tokenizer: GPT2Tokenizer) -> None:
    prompt_ids = [tokenizer.encode(p) for p in PROMPTS]
    batched = generate_batched(model, prompt_ids, NEW_TOKENS, tokenizer.eot_token, GREEDY)  # (B, N)

    for i, ids in enumerate(prompt_ids):
        single = _single(model, ids)                        # (1, N)
        assert torch.equal(batched[i : i + 1], single), (
            f"row {i} ({PROMPTS[i]!r}) diverged:\n"
            f" batched: {tokenizer.decode(batched[i].tolist())!r}\n"
            f" single : {tokenizer.decode(single[0].tolist())!r}"
        )
