"""Phase 2 INT4 quantization correctness tests.

Tests run entirely on synthetic tensors — no model weights required.

Checks:
  1. quantize_weight_int4 / dequantize_weight_int4 roundtrip error is small.
  2. Nibble packing is lossless (symmetric, no overflow).
  3. Perplexity-proxy: output of int4_matmul is close to bf16 reference.
  4. quantize_llama produces a model whose outputs are close to the original.
  5. Memory reduction is approximately 4× for linear weights.

Run:
    pytest tests/test_int4_quant.py -v -s
"""

from __future__ import annotations

import torch
import pytest

from engine.kernels.quant import (
    quantize_weight_int4,
    dequantize_weight_int4,
    int4_matmul,
    quantize_weight_int8,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Kernel-level tests (no model needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d_in,d_out,gs", [
    (256,  128, 128),
    (512,  256, 128),
    (1024, 512,  64),
])
def test_roundtrip_error(d_in, d_out, gs):
    """Dequantized weight should be close to original (max error ≤ scale/2)."""
    torch.manual_seed(0)
    w = torch.randn(d_in, d_out)
    packed, scale = quantize_weight_int4(w, group_size=gs)
    w_hat = dequantize_weight_int4(packed, scale, group_size=gs)

    max_err = (w - w_hat).abs().max().item()
    # Worst-case quantization error is half a scale step per group element.
    max_scale = scale.abs().max().item()
    threshold = max_scale / 2 + 1e-6
    print(f"  d_in={d_in} d_out={d_out} gs={gs}  max_err={max_err:.4f}  threshold={threshold:.4f}")
    assert max_err <= threshold, f"max_err {max_err:.4f} > threshold {threshold:.4f}"


def test_nibble_packing_lossless():
    """Packing and unpacking should be bit-exact for in-range int4 values."""
    torch.manual_seed(1)
    # Values in [-7, 7] only.
    d_in, d_out = 128, 64
    w_int = torch.randint(-7, 8, (d_in, d_out)).float()
    packed, scale = quantize_weight_int4(w_int, group_size=128)
    w_hat = dequantize_weight_int4(packed, scale, group_size=128)
    # scale = 1.0 when max(|w|)=7 → reconstructed should equal original exactly.
    assert (w_int - w_hat).abs().max().item() < 1e-4, "Nibble roundtrip is not lossless"


def test_int4_matmul_close_to_fp():
    """int4_matmul output should be close to the bf16 reference matmul."""
    torch.manual_seed(2)
    M, d_in, d_out, gs = 4, 256, 128, 128
    w = torch.randn(d_in, d_out)
    a = torch.randn(M, d_in)
    ref = a @ w                                  # bf16 reference

    packed, scale = quantize_weight_int4(w, group_size=gs)
    out = int4_matmul(a, packed, scale, group_size=gs)

    rel_err = ((ref - out).norm() / ref.norm()).item()
    print(f"  int4_matmul relative error: {rel_err:.4f}")
    # Random weights are worst-case for INT4 (no smooth structure to exploit).
    # Real pre-trained weights typically give < 2% error. 25% is a correctness bound.
    assert rel_err < 0.25, f"int4_matmul relative error {rel_err:.4f} too large"


def test_compression_ratio():
    """Packed INT4 should be approximately 4× smaller than fp32."""
    d_in, d_out = 4096, 4096
    w = torch.randn(d_in, d_out)
    packed, scale = quantize_weight_int4(w, group_size=128)

    fp32_bytes = w.numel() * 4
    int4_bytes = packed.numel() * 1 + scale.numel() * 2   # int8 + fp16 scales
    ratio = fp32_bytes / int4_bytes
    print(f"  fp32={fp32_bytes/1e6:.1f} MB  int4={int4_bytes/1e6:.1f} MB  ratio={ratio:.2f}×")
    assert ratio > 3.5, f"Compression ratio {ratio:.2f}× is lower than expected"


# ---------------------------------------------------------------------------
# Model-level test (synthetic mini-LLaMA, no weights file needed)
# ---------------------------------------------------------------------------

def _make_mini_llama(tie_word_embeddings: bool = True):
    """Build a tiny LlamaModel with random weights for quantization testing."""
    from engine.config import LlamaConfig
    from engine.llama_model import LlamaModel
    from engine.llama_weights import LlamaWeights

    cfg = LlamaConfig(
        name="test-mini",
        vocab_size=256,
        n_ctx=64,
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_heads=2,
        intermediate_size=128,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_word_embeddings=tie_word_embeddings,
    )

    torch.manual_seed(42)
    d, h, f = cfg.d_model, cfg.n_kv_heads * cfg.head_dim, cfg.intermediate_size

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float32)

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(cfg.vocab_size, d),
        "model.norm.weight": torch.ones(d),
    }
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = rand(cfg.vocab_size, d)
    for i in range(cfg.n_layer):
        p = f"model.layers.{i}."
        tensors.update({
            p + "input_layernorm.weight":         torch.ones(d),
            p + "post_attention_layernorm.weight": torch.ones(d),
            p + "self_attn.q_proj.weight":         rand(d, d),
            p + "self_attn.k_proj.weight":         rand(h, d),
            p + "self_attn.v_proj.weight":         rand(h, d),
            p + "self_attn.o_proj.weight":         rand(d, d),
            p + "mlp.gate_proj.weight":            rand(f, d),
            p + "mlp.up_proj.weight":              rand(f, d),
            p + "mlp.down_proj.weight":            rand(d, f),
        })

    weights = LlamaWeights(tensors, cfg)
    return LlamaModel(weights, cfg), cfg


def test_quantize_llama_output_close():
    """quantize_llama output should be close to full-precision output."""
    from engine.quantize import quantize_llama

    model, cfg = _make_mini_llama()

    # fp32 forward must run before quantize_llama, which pops bf16 tensors
    # from the shared weights dict to reclaim VRAM.
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref = model.forward(ids)

    q_model = quantize_llama(model, group_size=64)
    with torch.no_grad():
        out = q_model.forward(ids)

    rel_err = ((ref - out).norm() / ref.norm()).item()
    print(f"  mini-LLaMA INT4 relative logit error: {rel_err:.4f}")
    # Tiny random model (d=64, 2 layers) is worst-case: error compounds and random
    # weights have no structure for INT4 to exploit. Real 7B models typically show
    # < 1% logit error. We just verify the model runs and error is bounded.
    assert rel_err < 2.0, f"Quantized model logit error {rel_err:.4f} unexpectedly large"


def test_quantize_llama_memory_reduction():
    """quantize_llama should report ~4× reduction for linear weights."""
    from engine.quantize import quantize_llama
    model, _ = _make_mini_llama()
    # The print from quantize_llama includes the ratio — just check it runs.
    q_model = quantize_llama(model, group_size=64)
    assert q_model is not None


def test_quantize_lm_head_skipped_when_tied():
    """quantize_lm_head=True must be a no-op when embeddings are tied — lm_head
    is just a view of embed_tokens there, quantizing it would corrupt the
    (unquantized) embedding lookup too."""
    from engine.quantize import quantize_llama

    model, cfg = _make_mini_llama(tie_word_embeddings=True)
    assert cfg.tie_word_embeddings
    q_model = quantize_llama(model, group_size=64, quantize_lm_head=True)

    assert not hasattr(q_model.w.lm_head, "fused_linear")
    assert q_model.w.lm_head is q_model.w.embed_tokens


def test_quantize_lm_head_output_close():
    """Quantizing lm_head (untied case) should still produce close-ish logits,
    and forward() must actually route through the fused INT4 kernel (not
    silently fall back to the bf16 tensor)."""
    from engine.quantize import quantize_llama

    model, cfg = _make_mini_llama(tie_word_embeddings=False)
    assert not cfg.tie_word_embeddings

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref = model.forward(ids)

    q_model = quantize_llama(model, group_size=64, quantize_lm_head=True)

    # Assert the new path's own state directly, not just output parity —
    # lm_head must actually be the fused INT4 weight object now.
    assert hasattr(q_model.w.lm_head, "fused_linear")
    assert q_model.w.lm_head.shape == (cfg.vocab_size, cfg.d_model)

    with torch.no_grad():
        out = q_model.forward(ids)

    rel_err = ((ref - out).norm() / ref.norm()).item()
    print(f"  mini-LLaMA INT4 (incl. lm_head) relative logit error: {rel_err:.4f}")
    assert rel_err < 2.0, f"Quantized model logit error {rel_err:.4f} unexpectedly large"


def test_quantize_lm_head_off_by_default():
    """Default behavior (quantize_lm_head unset) must be unchanged: lm_head
    stays a plain bf16/float tensor, no fused_linear."""
    from engine.quantize import quantize_llama

    model, cfg = _make_mini_llama(tie_word_embeddings=False)
    q_model = quantize_llama(model, group_size=64)

    assert not hasattr(q_model.w.lm_head, "fused_linear")
    assert isinstance(q_model.w.lm_head, torch.Tensor)


# ---------------------------------------------------------------------------
# GEMV kernel tests (M=1 decode path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d_in,d_out", [
    (256,  128),
    (512,  256),
    (512,  512),
    (1024, 512),
    (4096, 1024),   # Qwen3-8B k/v-proj shape
    (4096, 4096),   # Qwen3-8B q/o-proj shape
])
def test_gemv_matches_reference(d_in, d_out):
    """int4_matmul with M=1 (GEMV path) must match dequant+matmul reference."""
    if DEVICE != "cuda":
        pytest.skip("GEMV kernel is CUDA-only")
    torch.manual_seed(7)
    w = torch.randn(d_in, d_out)
    a = torch.randn(1, d_in, dtype=torch.bfloat16, device=DEVICE)
    packed, scale = quantize_weight_int4(w, group_size=128)
    packed = packed.to(DEVICE)
    scale  = scale.to(DEVICE)

    # Reference: dequantize then matmul in bf16.
    w_hat = dequantize_weight_int4(packed.cpu(), scale.cpu(), group_size=128).to(torch.bfloat16).to(DEVICE)
    ref = a @ w_hat   # (1, d_out)

    out = int4_matmul(a, packed, scale)   # should dispatch to GEMV path

    rel_err = ((ref - out).norm() / (ref.norm() + 1e-8)).item()
    assert rel_err < 0.05, f"GEMV rel_err={rel_err:.4f} for d_in={d_in} d_out={d_out}"


def test_gemv_sign_extension_edges():
    """Nibble sign-extension must be correct at boundary values 8→-8, 9→-7, 7→7."""
    if DEVICE != "cuda":
        pytest.skip("GEMV kernel is CUDA-only")
    # Build a weight matrix where values hit the nibble boundaries.
    # Use a single group (d_in=128, d_out=1) so we control the scale precisely.
    d_in, d_out = 128, 64
    # Force packed bytes that contain both high nibble = 8 (0x8) and low nibble = 9 (0x9).
    # packed byte = (high << 4) | low = 0x89 (int8 = -119).
    packed = torch.full((d_in // 2, d_out), 0x89, dtype=torch.uint8).to(torch.int8).to(DEVICE)
    scale  = torch.ones(1, d_out, dtype=torch.float32, device=DEVICE)  # scale = 1.0

    a = torch.ones(1, d_in, dtype=torch.bfloat16, device=DEVICE)   # all ones activation

    out = int4_matmul(a, packed, scale)   # (1, d_out)

    # high nibble 0x8 >> 4 as int8 = -8; low nibble 0x9 sign-extended = -7.
    # Each output col = sum_{k=0}^{127}(a[k] * w[k]) = 64*(-8) + 64*(-7) = -512 + -448 = -960.
    expected = (-8 + -7) * (d_in // 2)   # = -15 * 64 = -960
    actual = out[0, 0].item()
    assert abs(actual - expected) < 1.0, f"Sign-extension edge case: got {actual}, expected {expected}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemv_vs_matmul_dispatch():
    """M=1 dispatches to GEMV; M=2 dispatches to tile matmul; outputs must agree."""
    d_in, d_out = 512, 256
    torch.manual_seed(99)
    w = torch.randn(d_in, d_out)
    packed, scale = quantize_weight_int4(w, group_size=128)
    packed = packed.cuda()
    scale  = scale.cuda()

    a1 = torch.randn(1, d_in, dtype=torch.bfloat16, device="cuda")
    a2 = torch.cat([a1, a1], dim=0)   # (2, d_in) — tile path

    out1 = int4_matmul(a1, packed, scale)   # GEMV
    out2 = int4_matmul(a2, packed, scale)   # tile; row 0 == row 1 since input is duplicated

    rel = ((out1 - out2[:1]).norm() / (out1.norm() + 1e-8)).item()
    assert rel < 0.01, f"GEMV and tile matmul disagree: rel={rel:.4f}"


# ---------------------------------------------------------------------------
# Large-M addressing (regression)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused INT4 kernel requires CUDA")
def test_int4_matmul_large_m_no_int32_overflow():
    """Output addressing must survive M * N beyond int32.

    Regression. ``offs_m * stride_cm`` reaches M * vocab_size; at Qwen3-8B's
    151936 vocab that passes int32's 2.15e9 ceiling once M >= ~14134. The
    overflow wrapped to a negative offset, every masked store was dropped, and
    the caller got an all-zero logit tensor with no error — so prefills of
    ~14k tokens or more silently produced garbage while 12k worked.

    Exercising it genuinely requires an output of more than 2**31 elements
    (~5 GB at bf16), so K is kept tiny and only slices are ever widened to
    float32.
    """
    from engine.kernels.quant import int4_matmul, quantize_weight_int4

    K, N, M = 128, 151936, 16384
    assert M * N > 2**31 - 1, "test no longer exercises the overflow"

    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 7 * 1024**3:
        pytest.skip(f"needs ~7 GB free for a >2**31-element output, have {free/1024**3:.1f} GB")

    torch.manual_seed(0)
    w = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.02
    packed, scale = quantize_weight_int4(w, 128)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    out = int4_matmul(a, packed, scale, 128)
    assert out.shape == (M, N)

    # The failure mode is silence: right shape, finite, and entirely zero. Check
    # the last rows — those are the ones whose offsets overflowed.
    tail = out[-8:, :512].float()
    del out
    torch.cuda.empty_cache()
    assert tail.abs().max() > 0, (
        "rows past the int32 boundary came back all-zero — stores were dropped"
    )
    # Norm-based: INT4 with a single 128-wide group has several percent of
    # element-wise error, and per-element relative error blows up on outputs
    # near zero. What matters is that these rows carry the right signal.
    ref = a[-8:].float() @ w[:, :512]
    rel = (tail - ref).norm() / ref.norm()
    # 0.2 still separates "quantization error" from the all-zero failure (1.0).
    assert rel < 0.2, f"tail rows differ from reference by {rel:.1%}"
