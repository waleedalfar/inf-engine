"""Phase 5 (scale-up) Qwen3 dense architecture tests.

Validates:
  - qk_norm flag wiring: applied when True, skipped when False, correct output shape.
  - QK-norm changes forward-pass output (the code path is actually taken).
  - Missing q_norm / k_norm weights raise a clear error.
  - Config constants have the right field values.
  - QwenTokenizer loads, encodes, and decodes correctly (two paths tested).

All model tests use synthetic mini-LLaMA weights — no real checkpoint required.
Tokenizer tests use a temporary directory with a minimal tiktoken vocab.

Run:
    pytest tests/test_qwen3_dense.py -v -s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from engine.config import LlamaConfig, QWEN3_0_6B, QWEN3_8B, QWEN3_14B

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Mini-model helpers
# ---------------------------------------------------------------------------

def _mini_config(qk_norm: bool = False) -> LlamaConfig:
    return LlamaConfig(
        name="test-mini",
        vocab_size=64,
        n_ctx=128,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=64,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        qk_norm=qk_norm,
    )


def _mini_model(config: LlamaConfig, seed: int = 0, device: str = DEVICE):
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    d = config.d_model
    h = config.n_kv_heads * config.head_dim
    f = config.intermediate_size
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=device)

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(config.vocab_size, d),
        "model.norm.weight":         torch.ones(d, dtype=DTYPE, device=device),
    }
    for i in range(config.n_layer):
        p = f"model.layers.{i}."
        tensors |= {
            p + "input_layernorm.weight":         torch.ones(d, dtype=DTYPE, device=device),
            p + "post_attention_layernorm.weight": torch.ones(d, dtype=DTYPE, device=device),
            p + "self_attn.q_proj.weight":         rand(d, d),
            p + "self_attn.k_proj.weight":         rand(h, d),
            p + "self_attn.v_proj.weight":         rand(h, d),
            p + "self_attn.o_proj.weight":         rand(d, d),
            p + "mlp.gate_proj.weight":            rand(f, d),
            p + "mlp.up_proj.weight":              rand(f, d),
            p + "mlp.down_proj.weight":            rand(d, f),
        }
        if config.qk_norm:
            tensors[p + "self_attn.q_norm.weight"] = torch.ones(
                config.head_dim, dtype=DTYPE, device=device
            )
            tensors[p + "self_attn.k_norm.weight"] = torch.ones(
                config.head_dim, dtype=DTYPE, device=device
            )
    weights = LlamaWeights(tensors, config)
    return LlamaModel(weights, config)


# ---------------------------------------------------------------------------
# QK-norm architecture tests
# ---------------------------------------------------------------------------

def test_qk_norm_flag_default_false():
    """LlamaConfig defaults to qk_norm=False (backward compatible)."""
    cfg = LlamaConfig(
        name="x", vocab_size=100, n_ctx=64, d_model=32,
        n_layer=1, n_head=4, n_kv_heads=2, intermediate_size=64,
    )
    assert cfg.qk_norm is False


def test_qk_norm_output_shape_unchanged():
    """QK-norm must not change the output tensor shape."""
    cfg_no = _mini_config(qk_norm=False)
    cfg_qk = _mini_config(qk_norm=True)
    model_no = _mini_model(cfg_no, seed=1)
    model_qk = _mini_model(cfg_qk, seed=1)  # same seed → same non-norm weights

    ids = torch.randint(0, cfg_no.vocab_size, (1, 5), device=DEVICE)
    pos = torch.arange(5, device=DEVICE)
    with torch.no_grad():
        out_no = model_no.forward(ids, start_pos=0, position_ids=pos)
        out_qk = model_qk.forward(ids, start_pos=0, position_ids=pos)

    assert out_no.shape == out_qk.shape


def test_qk_norm_changes_output():
    """When qk_norm=True, logits differ from the qk_norm=False baseline.

    This verifies the norm path is actually executed (not dead code).
    The q_norm/k_norm gain initialised to ones with non-unit RMS input
    still changes the pre-softmax scale.  We use non-constant random
    projections so Q/K have non-unit row norms.
    """
    cfg_no = _mini_config(qk_norm=False)
    cfg_qk = _mini_config(qk_norm=True)
    # Same projection weights, different norm behaviour.
    model_no = _mini_model(cfg_no, seed=7)
    model_qk = _mini_model(cfg_qk, seed=7)

    ids = torch.randint(0, cfg_no.vocab_size, (1, 4), device=DEVICE)
    pos = torch.arange(4, device=DEVICE)
    with torch.no_grad():
        out_no = model_no.forward(ids, start_pos=0, position_ids=pos)
        out_qk = model_qk.forward(ids, start_pos=0, position_ids=pos)

    assert not torch.allclose(out_no, out_qk, atol=1e-4), (
        "qk_norm=True and qk_norm=False produced identical outputs — "
        "the norm path is likely not being applied."
    )


def test_qk_norm_greedy_deterministic():
    """Greedy generation with qk_norm=True is deterministic across two runs."""
    from engine.sampling import SamplingConfig, SamplingMode

    cfg = _mini_config(qk_norm=True)
    model = _mini_model(cfg, seed=42)
    greedy = SamplingConfig(mode=SamplingMode.GREEDY)

    ids = torch.randint(0, cfg.vocab_size, (1, 3), device=DEVICE)
    with torch.no_grad():
        out1 = model.generate(ids, max_new_tokens=6, sampling=greedy)
        out2 = model.generate(ids, max_new_tokens=6, sampling=greedy)

    assert torch.equal(out1, out2)


def test_qk_norm_cached_matches_uncached():
    """KV-cached prefill must produce logits within tolerance of uncached forward.

    Token-exact comparison is intentionally avoided: SDPA (used for prefill, T_q>1)
    and manual attention (used for single-token decode, T_q=1) accumulate floats in
    different orders, which can flip the argmax of random-weight models. Production
    engines test logit closeness instead of token identity.
    """
    from engine.kv_cache import LlamaStaticKVCache

    cfg = _mini_config(qk_norm=True)
    model = _mini_model(cfg, seed=3)

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)

    with torch.no_grad():
        ref_logits = model.forward(ids)[:, -1, :]            # (1, vocab) — uncached

    cache = LlamaStaticKVCache(cfg, batch=1, max_seq=ids.shape[1] + 2,
                                device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        cached_logits = model.forward(ids, cache=cache, start_pos=0)[:, -1, :]

    max_delta = (ref_logits - cached_logits).abs().max().item()
    assert torch.allclose(ref_logits, cached_logits, atol=0.05), (
        f"Prefill logits diverged with qk_norm: max|Δ| = {max_delta:.4f}"
    )


def test_qk_norm_weight_missing_raises(tmp_path):
    """load_llama_weights with qk_norm=True raises if q_norm weights are absent."""
    from engine.llama_weights import load_llama_weights
    import safetensors.torch

    cfg = _mini_config(qk_norm=True)

    # Build a minimal weight file WITHOUT q_norm / k_norm tensors.
    tensors = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.d_model),
        "model.norm.weight":         torch.ones(cfg.d_model),
    }
    for i in range(cfg.n_layer):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"]         = torch.ones(cfg.d_model)
        tensors[p + "post_attention_layernorm.weight"] = torch.ones(cfg.d_model)
        tensors[p + "self_attn.q_proj.weight"]         = torch.randn(cfg.d_model, cfg.d_model)
        tensors[p + "self_attn.k_proj.weight"]         = torch.randn(cfg.n_kv_heads * cfg.head_dim, cfg.d_model)
        tensors[p + "self_attn.v_proj.weight"]         = torch.randn(cfg.n_kv_heads * cfg.head_dim, cfg.d_model)
        tensors[p + "self_attn.o_proj.weight"]         = torch.randn(cfg.d_model, cfg.d_model)
        tensors[p + "mlp.gate_proj.weight"]            = torch.randn(cfg.intermediate_size, cfg.d_model)
        tensors[p + "mlp.up_proj.weight"]              = torch.randn(cfg.intermediate_size, cfg.d_model)
        tensors[p + "mlp.down_proj.weight"]            = torch.randn(cfg.d_model, cfg.intermediate_size)

    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))

    with pytest.raises(KeyError, match="qk_norm=True"):
        load_llama_weights(tmp_path, cfg, device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Qwen3 config constant tests
# ---------------------------------------------------------------------------

def test_qwen3_configs_have_qk_norm():
    """All shipped Qwen3 configs must have qk_norm=True."""
    for cfg in (QWEN3_0_6B, QWEN3_8B, QWEN3_14B):
        assert cfg.qk_norm is True, f"{cfg.name} should have qk_norm=True"


def test_qwen3_configs_rope_theta():
    """Qwen3 configs use rope_theta=1_000_000."""
    for cfg in (QWEN3_0_6B, QWEN3_8B, QWEN3_14B):
        assert cfg.rope_theta == 1_000_000.0, f"{cfg.name}: unexpected rope_theta"


def test_qwen3_configs_vocab_size():
    """Qwen3 configs use vocab_size=151_936."""
    for cfg in (QWEN3_0_6B, QWEN3_8B, QWEN3_14B):
        assert cfg.vocab_size == 151_936, f"{cfg.name}: unexpected vocab_size"


def test_qwen3_8b_head_dim():
    """Qwen3-8B: 4096 hidden / 32 heads = 128 head_dim."""
    assert QWEN3_8B.head_dim == 128


def test_qwen3_14b_head_dim():
    """Qwen3-14B: 5120 hidden / 40 heads = 128 head_dim."""
    assert QWEN3_14B.head_dim == 128


# ---------------------------------------------------------------------------
# QwenTokenizer tests
# ---------------------------------------------------------------------------

def _make_qwen_tiktoken_dir(tmp_path: Path) -> Path:
    """Write a minimal qwen.tiktoken + tokenizer_config.json to tmp_path.

    We borrow the GPT-2 mergeable_ranks (available via tiktoken without any
    download) as a stand-in for the Qwen vocab.  The tokenizer logic being
    tested is loading, encoding, and decoding — not the specific vocabulary.
    """
    import base64
    import tiktoken as _tiktoken

    gpt2 = _tiktoken.get_encoding("gpt2")
    ranks = gpt2._mergeable_ranks  # dict[bytes, int]

    lines = [
        base64.b64encode(tok).decode() + " " + str(rank)
        for tok, rank in sorted(ranks.items(), key=lambda kv: kv[1])
    ]
    (tmp_path / "qwen.tiktoken").write_text("\n".join(lines), encoding="utf-8")

    # Use GPT-2's EOS id as a dummy special token (safe to avoid rank conflicts).
    config = {
        "added_tokens_decoder": {
            "50256": {"content": "<|endoftext|>", "special": True},
        }
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_qwen_tokenizer_loads_from_tiktoken(tmp_path):
    """QwenTokenizer initialises without error from a qwen.tiktoken file."""
    from engine.qwen_tokenizer import QwenTokenizer
    d = _make_qwen_tiktoken_dir(tmp_path)
    tok = QwenTokenizer(d)
    assert len(tok) > 0


def test_qwen_tokenizer_encode_decode_roundtrip(tmp_path):
    """encode → decode roundtrip returns the original text (ASCII subset)."""
    from engine.qwen_tokenizer import QwenTokenizer
    d = _make_qwen_tiktoken_dir(tmp_path)
    tok = QwenTokenizer(d)
    text = "Hello, world! This is a test."
    ids = tok.encode(text, add_special_tokens=False)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    recovered = tok.decode(ids, skip_special_tokens=False)
    assert recovered == text


def test_qwen_tokenizer_special_token_ids(tmp_path):
    """Special token properties return integers equal to the Qwen3 defaults."""
    from engine.qwen_tokenizer import QwenTokenizer
    d = _make_qwen_tiktoken_dir(tmp_path)
    tok = QwenTokenizer(d)
    # Our dummy tokenizer_config.json only overrides <|endoftext|>;
    # the Qwen3 defaults are preserved for im_start / im_end.
    assert tok.im_start_id == 151_644
    assert tok.im_end_id == 151_645
    assert tok.eos_token_id == 151_645


def test_qwen_tokenizer_missing_dir_raises():
    """QwenTokenizer raises FileNotFoundError for a non-existent directory."""
    from engine.qwen_tokenizer import QwenTokenizer
    with pytest.raises(FileNotFoundError, match="No tokenizer found"):
        QwenTokenizer("/nonexistent/path/that/does/not/exist")


def test_qwen_tokenizer_skip_special_tokens(tmp_path):
    """skip_special_tokens=True removes special-token ids from decode output."""
    from engine.qwen_tokenizer import QwenTokenizer
    d = _make_qwen_tiktoken_dir(tmp_path)
    tok = QwenTokenizer(d)
    text = "Hello"
    ids = tok.encode(text, add_special_tokens=False)
    # Inject the special token at the end and verify it's stripped.
    special_id = 50256  # from our tokenizer_config.json
    ids_with_special = ids + [special_id]
    decoded_skip = tok.decode(ids_with_special, skip_special_tokens=True)
    decoded_keep = tok.decode(ids_with_special, skip_special_tokens=False)
    assert decoded_skip == "Hello"
    # With skip=False the special token is decoded as bytes (may appear as empty or replacement char).
    assert len(decoded_keep) >= len(decoded_skip)
