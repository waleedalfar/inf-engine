"""Phase 2 correctness gate: KV-cache decoding must be exactly equivalent.

Three things must hold before Phase 3:
  1. Cached greedy decode == Phase 1 no-cache greedy decode, token-for-token.
  2. Static cache == dynamic cache (same tokens).
  3. Cached decode still matches the HuggingFace oracle (transitively, via #1, but we
     check directly too for defense in depth).
  4. Empirical cache byte size matches the first-principles formula
     ``2 * n_layer * n_head * head_dim * seq_len * batch * bytes`` exactly.

Run:  pytest tests/test_phase2_kv_cache.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.config import GPT2_SMALL
from engine.kv_cache import DynamicKVCache, StaticKVCache, kv_cache_bytes
from engine.model import GPT2Model
from engine.sampling import SamplingConfig, SamplingMode
from engine.tokenizer import GPT2Tokenizer
from engine.weights import load_weights

MODEL_NAME = "gpt2"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / MODEL_NAME / "model.safetensors"
PROMPTS = [
    "The capital of France is",
    "In a shocking turn of events, scientists discovered",
    "def fibonacci(n):",
    "Once upon a time",
]
NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
GREEDY = SamplingConfig(mode=SamplingMode.GREEDY)


@pytest.fixture(scope="module")
def model() -> GPT2Model:
    if not WEIGHTS_PATH.is_file():
        pytest.skip(f"weights missing: {WEIGHTS_PATH} (run scripts/download_weights.py)")
    weights = load_weights(WEIGHTS_PATH, GPT2_SMALL, device=DEVICE, dtype=DTYPE)
    return GPT2Model(weights, GPT2_SMALL)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_static_cache_matches_no_cache(model: GPT2Model, tokenizer: GPT2Tokenizer, prompt: str) -> None:
    """Static-cache greedy decode == no-cache greedy decode, token-for-token."""
    ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)  # (1, T_p)
    no_cache = model.generate(ids, max_new_tokens=NEW_TOKENS, sampling=GREEDY)

    max_seq = ids.shape[1] + NEW_TOKENS
    cache = StaticKVCache(GPT2_SMALL, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE)
    cached = model.generate_cached(ids, max_new_tokens=NEW_TOKENS, cache=cache, sampling=GREEDY)

    assert torch.equal(no_cache, cached), (
        f"static cache diverged on {prompt!r}:\n"
        f" no-cache: {tokenizer.decode(no_cache[0].tolist())!r}\n"
        f" cached  : {tokenizer.decode(cached[0].tolist())!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_dynamic_matches_static(model: GPT2Model, tokenizer: GPT2Tokenizer, prompt: str) -> None:
    """Dynamic cache produces the same tokens as the static cache."""
    ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)
    max_seq = ids.shape[1] + NEW_TOKENS

    static_cache = StaticKVCache(GPT2_SMALL, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE)
    dyn_cache = DynamicKVCache(GPT2_SMALL, batch=1, device=DEVICE, dtype=DTYPE)
    out_static = model.generate_cached(ids, NEW_TOKENS, static_cache, GREEDY)
    out_dyn = model.generate_cached(ids, NEW_TOKENS, dyn_cache, GREEDY)

    assert torch.equal(out_static, out_dyn)
    # Sanity: dynamic cache grew to exactly prompt_len + NEW_TOKENS - 1 cached positions
    # (the last sampled token is not fed back, so it is never cached).
    assert dyn_cache.length == ids.shape[1] + NEW_TOKENS - 1


def test_cached_matches_huggingface(model: GPT2Model, tokenizer: GPT2Tokenizer) -> None:
    """Defense in depth: cached decode matches the HF oracle directly."""
    transformers = pytest.importorskip("transformers")
    hf = transformers.GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager", torch_dtype=DTYPE
    ).to(DEVICE).eval()

    prompt = PROMPTS[0]
    ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)
    cache = StaticKVCache(GPT2_SMALL, batch=1, max_seq=ids.shape[1] + NEW_TOKENS, device=DEVICE, dtype=DTYPE)
    cached = model.generate_cached(ids, NEW_TOKENS, cache, GREEDY)

    hf_ids = ids.clone()
    with torch.no_grad():
        for _ in range(NEW_TOKENS):
            nxt = hf(hf_ids).logits[:, -1, :].argmax(-1, keepdim=True)
            hf_ids = torch.cat([hf_ids, nxt], dim=1)

    assert torch.equal(cached, hf_ids)


@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
@pytest.mark.parametrize("batch", [1, 4])
def test_memory_formula_exact(seq_len: int, batch: int) -> None:
    """Allocated static-cache bytes equal the first-principles formula exactly."""
    expected = kv_cache_bytes(GPT2_SMALL, seq_len=seq_len, dtype=DTYPE, batch=batch)
    cache = StaticKVCache(GPT2_SMALL, batch=batch, max_seq=seq_len, device=DEVICE, dtype=DTYPE)
    assert cache.memory_bytes() == expected, (
        f"seq={seq_len} batch={batch}: measured {cache.memory_bytes()} != formula {expected}"
    )
