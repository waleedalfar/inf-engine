"""Phase 1 correctness gate: our GPT-2 forward pass vs HuggingFace, token-for-token.

HuggingFace is used here **only as the ground-truth oracle** — it never appears in
``engine/``. We pin ``attn_implementation="eager"`` so HF runs the same algorithm
our naive attention does (not a fused SDPA backend), making the comparison apples
to apples.

The hard gate (must pass before Phase 2):
  * per-position argmax of logits is identical on a teacher-forced sequence, and
  * autoregressive greedy decoding produces identical token ids for >= 50 steps,
    across several prompts.
We also report max abs logit difference as a soft numerical check.

Run:  pytest tests/test_phase1_forward.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.config import GPT2_SMALL
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
GREEDY_STEPS = 50
LOGIT_ATOL = 1e-3  # soft check; the real gate is exact argmax / token-id match
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


@pytest.fixture(scope="module")
def ours() -> GPT2Model:
    if not WEIGHTS_PATH.is_file():
        pytest.skip(f"weights missing: {WEIGHTS_PATH} (run scripts/download_weights.py)")
    weights = load_weights(WEIGHTS_PATH, GPT2_SMALL, device=DEVICE, dtype=DTYPE)
    return GPT2Model(weights, GPT2_SMALL)


@pytest.fixture(scope="module")
def hf_model():
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager", torch_dtype=DTYPE
    )
    model.to(DEVICE).eval()
    return model


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def _hf_logits(hf_model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return hf_model(input_ids).logits  # (B, T, V)


def test_tokenizer_parity(tokenizer: GPT2Tokenizer) -> None:
    """tiktoken gpt2 ids must equal HuggingFace GPT-2 tokenizer ids."""
    transformers = pytest.importorskip("transformers")
    hf_tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    for text in PROMPTS:
        assert tokenizer.encode(text) == hf_tok.encode(text), f"token mismatch on: {text!r}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match(ours: GPT2Model, hf_model, tokenizer: GPT2Tokenizer, prompt: str) -> None:
    """Teacher-forced logits: exact per-position argmax + small numerical diff."""
    ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)  # (1, T)
    ours_logits = ours.forward(ids)                                # (1, T, V)
    hf_logits = _hf_logits(hf_model, ids)                          # (1, T, V)

    max_diff = (ours_logits - hf_logits).abs().max().item()
    print(f"\n[{prompt!r}] max|Δlogit| = {max_diff:.2e}")

    # Hard gate: argmax identical at every position.
    assert torch.equal(ours_logits.argmax(-1), hf_logits.argmax(-1)), "argmax mismatch"
    # Soft numerical check.
    assert max_diff < LOGIT_ATOL, f"max logit diff {max_diff:.2e} >= {LOGIT_ATOL}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_decode_matches(ours: GPT2Model, hf_model, tokenizer: GPT2Tokenizer, prompt: str) -> None:
    """Autoregressive greedy decoding must match HF token-for-token for 50 steps."""
    start = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)  # (1, T)

    ours_ids = ours.generate(
        start, max_new_tokens=GREEDY_STEPS, sampling=SamplingConfig(mode=SamplingMode.GREEDY)
    )

    # Manual greedy on HF using the same loop, isolating the forward pass.
    hf_ids = start.clone()
    for _ in range(GREEDY_STEPS):
        nxt = _hf_logits(hf_model, hf_ids)[:, -1, :].argmax(-1, keepdim=True)  # (1, 1)
        hf_ids = torch.cat([hf_ids, nxt], dim=1)

    assert torch.equal(ours_ids, hf_ids), (
        f"greedy divergence on {prompt!r}:\n"
        f" ours: {tokenizer.decode(ours_ids[0].tolist())!r}\n"
        f" hf  : {tokenizer.decode(hf_ids[0].tolist())!r}"
    )
