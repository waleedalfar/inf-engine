"""Phase 1 LLaMA correctness gate: our forward pass vs HuggingFace, token-for-token.

Same methodology as test_phase1_forward.py — HuggingFace is the oracle only,
never imported by engine/.

Checks (fp32, fixed inputs, across prompts):
  1. Teacher-forced logits: per-position argmax identical; max |Δlogit| reported.
  2. Greedy decode: identical token ids for >= 20 steps.

Skip behaviour: tests are skipped (not failed) when weights are absent so CI
still passes on machines that haven't downloaded the gated model.

Run:
    pytest tests/test_llama_forward.py -v -s

Prerequisite:
    python scripts/download_llama.py          # requires HF login + Meta license
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_DIR = WEIGHTS_DIR / MODEL_NAME.replace("/", "--")

PROMPTS = [
    "The capital of France is",
    "In a shocking discovery, researchers found",
    "def fibonacci(n):",
    "Once upon a time",
]
GREEDY_STEPS = 20
LOGIT_ATOL = 1e-2   # fp32; bfloat16 weights introduce ~1e-2 error vs HF fp32 ref
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32   # use fp32 for the correctness gate (matches HF default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _skip_if_missing() -> None:
    if not (MODEL_DIR / "model.safetensors").is_file() and not (
        MODEL_DIR / "model.safetensors.index.json"
    ).is_file():
        pytest.skip(
            f"LLaMA weights not found at {MODEL_DIR}\n"
            f"  Run:  python scripts/download_llama.py {MODEL_NAME}\n"
            f"  (Requires HuggingFace login and Meta license acceptance)"
        )


@pytest.fixture(scope="module")
def tokenizer():
    _skip_if_missing()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


@pytest.fixture(scope="module")
def hf_model():
    _skip_if_missing()
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(DEVICE).eval()


@pytest.fixture(scope="module")
def ours():
    _skip_if_missing()
    from engine.config import LLAMA_3_2_1B
    from engine.llama_model import LlamaModel
    from engine.llama_weights import load_llama_weights
    weights = load_llama_weights(MODEL_DIR, LLAMA_3_2_1B, device=DEVICE, dtype=DTYPE)
    return LlamaModel(weights, LLAMA_3_2_1B)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match(prompt, tokenizer, hf_model, ours):
    """Per-position argmax of logits must match; max |Δlogit| is reported."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        hf_logits = hf_model(ids).logits                          # (1, T, V)
        our_logits = ours.forward(ids)                            # (1, T, V)

    max_err = (hf_logits - our_logits).abs().max().item()
    print(f"\n  [{prompt!r}]  max|Δlogit| = {max_err:.2e}")

    hf_top = hf_logits.argmax(-1)
    our_top = our_logits.argmax(-1)
    assert torch.equal(hf_top, our_top), (
        f"Argmax mismatch at positions: {(hf_top != our_top).nonzero().tolist()}"
    )
    assert max_err < LOGIT_ATOL, f"max|Δlogit| {max_err:.2e} > {LOGIT_ATOL}"


@pytest.mark.parametrize("prompt", PROMPTS[:2])
def test_greedy_decode_matches(prompt, tokenizer, hf_model, ours):
    """Greedy decode must produce identical token ids for GREEDY_STEPS steps."""
    from engine.kv_cache import LlamaStaticKVCache
    from engine.config import LLAMA_3_2_1B
    from engine.sampling import SamplingConfig, SamplingMode

    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    T_p = ids.shape[1]
    max_seq = T_p + GREEDY_STEPS + 1

    # --- ours ---
    cache = LlamaStaticKVCache(
        LLAMA_3_2_1B, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE
    )
    our_out = ours.generate_cached(
        ids, GREEDY_STEPS, cache, SamplingConfig(mode=SamplingMode.GREEDY)
    )
    our_new = our_out[0, T_p:].tolist()

    # --- HF greedy ---
    with torch.no_grad():
        hf_out = hf_model.generate(
            ids,
            max_new_tokens=GREEDY_STEPS,
            do_sample=False,
        )
    hf_new = hf_out[0, T_p:].tolist()

    print(f"\n  [{prompt!r}]")
    print(f"  HF : {tokenizer.decode(hf_new)!r}")
    print(f"  Our: {tokenizer.decode(our_new)!r}")

    assert our_new == hf_new, (
        f"Greedy decode diverged:\n  HF : {hf_new}\n  Our: {our_new}"
    )
