"""LLaMA architecture correctness gate using SmolLM2-1.7B (Apache 2.0, no gating).

Validates the full Phase 1 implementation (RoPE, RMSNorm, GQA, SwiGLU) against
HuggingFace as the oracle, using a freely downloadable model so the test can run
without a Meta license.  Once LLaMA 3.2-1B access is approved, the same engine
code is validated by tests/test_llama_forward.py.

SmolLM2-1.7B uses LlamaForCausalLM internally — identical weight key names and
architecture to LLaMA 3.2, just different hyperparameters and vocabulary.

Run:
    python scripts/download_llama.py               # downloads SmolLM2-1.7B (~3.4 GB)
    pytest tests/test_llama_smollm2.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_DIR = WEIGHTS_DIR / MODEL_NAME.replace("/", "--")

PROMPTS = [
    "The capital of France is",
    "In a shocking discovery, researchers found",
    "def fibonacci(n):",
    "Once upon a time",
]
GREEDY_STEPS = 20
LOGIT_ATOL = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _skip_if_missing() -> None:
    if not (MODEL_DIR / "model.safetensors").is_file() and not (
        MODEL_DIR / "model.safetensors.index.json"
    ).is_file():
        pytest.skip(
            f"SmolLM2 weights not found at {MODEL_DIR}\n"
            f"  Run:  python scripts/download_llama.py"
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
    from engine.config import SMOLLM2_1_7B
    from engine.llama_model import LlamaModel
    from engine.llama_weights import load_llama_weights
    weights = load_llama_weights(MODEL_DIR, SMOLLM2_1_7B, device=DEVICE, dtype=DTYPE)
    return LlamaModel(weights, SMOLLM2_1_7B)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match(prompt, tokenizer, hf_model, ours):
    """Per-position argmax must match HuggingFace; max |Δlogit| is reported."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        hf_logits = hf_model(ids).logits
        our_logits = ours.forward(ids)

    max_err = (hf_logits - our_logits).abs().max().item()
    print(f"\n  [{prompt!r}]  max|Δlogit| = {max_err:.2e}")

    assert torch.equal(hf_logits.argmax(-1), our_logits.argmax(-1)), (
        "Argmax mismatch — forward pass has a bug."
    )
    assert max_err < LOGIT_ATOL, f"max|Δlogit| {max_err:.2e} exceeds tolerance {LOGIT_ATOL}"


@pytest.mark.parametrize("prompt", PROMPTS[:2])
def test_greedy_decode_matches(prompt, tokenizer, hf_model, ours):
    """Greedy decode must be token-for-token identical to HuggingFace."""
    from engine.config import SMOLLM2_1_7B
    from engine.kv_cache import LlamaStaticKVCache
    from engine.sampling import SamplingConfig, SamplingMode

    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    T_p = ids.shape[1]
    max_seq = T_p + GREEDY_STEPS + 1

    cache = LlamaStaticKVCache(SMOLLM2_1_7B, batch=1, max_seq=max_seq, device=DEVICE, dtype=DTYPE)
    our_out = ours.generate_cached(ids, GREEDY_STEPS, cache, SamplingConfig(mode=SamplingMode.GREEDY))
    our_new = our_out[0, T_p:].tolist()

    with torch.no_grad():
        hf_out = hf_model.generate(ids, max_new_tokens=GREEDY_STEPS, do_sample=False)
    hf_new = hf_out[0, T_p:].tolist()

    print(f"\n  [{prompt!r}]")
    print(f"  HF : {tokenizer.decode(hf_new)!r}")
    print(f"  Our: {tokenizer.decode(our_new)!r}")

    assert our_new == hf_new, f"Greedy decode diverged:\n  HF : {hf_new}\n  Our: {our_new}"
