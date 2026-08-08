"""Phase 5 correctness gates: Triton kernels vs PyTorch references within tolerance.

Each kernel must match its PyTorch reference numerically. Importing ``engine.kernels``
configures CPATH for Triton's JIT compiler (see engine/kernels/__init__.py), so these tests
also serve as the "Triton runs on this GPU" smoke test.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")

from engine.kernels.flash_attention import flash_attention, naive_attention  # noqa: E402
from engine.kernels.quant import int8_matmul, quantize_weight_int8  # noqa: E402
from engine.kernels.softmax import triton_softmax  # noqa: E402

DEVICE = "cuda"


@pytest.mark.parametrize("shape", [(128, 128), (256, 1024), (12 * 64, 768), (4096, 2048)])
def test_fused_softmax_matches_torch(shape) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    ref = torch.softmax(x, dim=-1)
    got = triton_softmax(x)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), (got - ref).abs().max().item()


def test_fused_softmax_rows_sum_to_one() -> None:
    x = torch.randn(1000, 512, device=DEVICE)
    got = triton_softmax(x)
    sums = got.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.parametrize("seq", [128, 512, 1024])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_matches_naive(seq: int, causal: bool) -> None:
    torch.manual_seed(0)
    b, h, d = 2, 12, 64
    q = torch.randn(b, h, seq, d, device=DEVICE, dtype=torch.float32)
    k = torch.randn(b, h, seq, d, device=DEVICE, dtype=torch.float32)
    v = torch.randn(b, h, seq, d, device=DEVICE, dtype=torch.float32)
    ref = naive_attention(q, k, v, causal=causal)
    got = flash_attention(q, k, v, causal=causal)
    max_err = (ref - got).abs().max().item()
    assert max_err < 1e-3, f"seq={seq} causal={causal}: max|Δ|={max_err:.2e}"


@pytest.mark.parametrize("shape", [(64, 768, 2304), (16, 1024, 1024)])
def test_int8_matmul_matches_dequant_reference(shape) -> None:
    """The kernel must compute exactly a @ (wq.float() * scale), within fp tolerance."""
    torch.manual_seed(0)
    m, k, n = shape
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    w = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
    wq, scale = quantize_weight_int8(w)
    ref = a @ (wq.float() * scale[None, :])              # exact dequant matmul
    got = int8_matmul(a, wq, scale)
    rel = (got - ref).abs().max() / ref.abs().max()
    assert rel < 1e-3, f"{shape}: rel err {rel.item():.2e}"


def test_int8_quantization_error_bounded() -> None:
    """Quantization error of A@W (the quality cost) stays small relative to the signal."""
    torch.manual_seed(0)
    a = torch.randn(32, 768, device=DEVICE)
    w = torch.randn(768, 2304, device=DEVICE) * 0.1      # GPT-2-ish weight scale
    wq, scale = quantize_weight_int8(w)
    exact = a @ w
    quant = a @ (wq.float() * scale[None, :])
    rel = (quant - exact).norm() / exact.norm()
    assert rel < 0.02, f"int8 relative error {rel.item():.4f} too high"
