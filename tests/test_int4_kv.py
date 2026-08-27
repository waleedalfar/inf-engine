"""INT4 (packed-nibble) KV storage.

The two kernels that touch the packed pool are written independently — the
write kernel packs across two strided halves of the source vector, the read
kernel unpacks per-channel from a byte loaded twice — so the thing worth
testing is that they agree on the layout, and that both agree with the torch
gather path. A layout disagreement here is exactly the class of bug that
produced plausible-looking wrong logits before (V addressed through K's
strides), and it is invisible to any test that only checks one side.

Inputs are deliberately *not* cloned into contiguity: the pool slices and the
projection outputs carry the strides the real engine hands the kernels.
"""

from __future__ import annotations

import math

import pytest
import torch

from engine.config import LlamaConfig
from engine.paged_cache import INT4, BlockManager, PagedLlamaKVCache

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEAD_DIM = 64
N_KV = 2
BS = 16


def _cfg(head_dim: int = HEAD_DIM) -> LlamaConfig:
    return LlamaConfig(
        name="int4-mini", vocab_size=64, n_ctx=4096, d_model=head_dim * 4,
        n_layer=1, n_head=4, n_kv_heads=N_KV, intermediate_size=128,
        head_dim_override=head_dim,
    )


def _cache(n_blocks: int, device: str, head_dim: int = HEAD_DIM,
           **kw) -> PagedLlamaKVCache:
    mgr = BlockManager(n_blocks, BS)
    return PagedLlamaKVCache(_cfg(head_dim), mgr, device, torch.bfloat16,
                             kv_dtype=INT4, **kw)


def _reference(x: torch.Tensor) -> torch.Tensor:
    """What INT4 round-trip must produce: per-(token, head) amax scaling to +/-7."""
    f = x.float()
    scale = (f.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-12)
    codes = torch.round(f / scale).clamp(-7, 7)
    return codes * scale


def test_pool_is_half_width_and_scales_are_not():
    c = _cache(8, "cpu")
    assert c.int4 and c.quantized and c.is_int and c.qmax == 7.0
    assert c.k_pool.dtype is torch.int8
    assert c.k_pool.shape[-1] == HEAD_DIM // 2
    assert c.k_scale.shape[-1] == BS          # one scale per (block, head, pos)
    # Half the bytes of an INT8 pool of the same capacity.
    c8 = PagedLlamaKVCache(_cfg(), BlockManager(8, BS), "cpu", torch.bfloat16,
                           kv_dtype=torch.int8)
    assert c.memory_bytes() * 2 == c8.memory_bytes()


def test_unpack_inverts_the_documented_nibble_layout():
    c = _cache(8, "cpu")
    # Byte j -> (channel 2j low nibble, channel 2j+1 high nibble), 4-bit two's
    # complement. 0x79 == (lo=9 -> -7, hi=7 -> 7).
    packed = torch.tensor([[0x79, 0x01]], dtype=torch.int8)
    got = c._unpack_int4(packed)
    assert got.tolist() == [[-7, 7, 1, 0]]

    # Every representable code survives the round trip through int8 storage.
    codes = torch.arange(-8, 8)
    lo, hi = torch.meshgrid(codes, codes, indexing="ij")
    raw = (lo & 0xF) | ((hi & 0xF) << 4)
    raw = torch.where(raw >= 128, raw - 256, raw).to(torch.int8).reshape(-1, 1)
    out = c._unpack_int4(raw)
    assert torch.equal(out[:, 0], lo.reshape(-1))
    assert torch.equal(out[:, 1], hi.reshape(-1))


@cuda
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
def test_write_then_gather_round_trips_within_int4_resolution(HEAD_DIM):
    """The write kernel packs; the torch gather path unpacks. They must agree.

    128 is the head_dim Qwen3 actually uses (both the 8B target and the 0.6B
    draft); 64 exercises a BLOCK_D2 that is not the tile width.
    """
    torch.manual_seed(0)
    c = _cache(16, "cuda", head_dim=HEAD_DIM)
    sid, n_tok = 0, 40
    c.allocate_sequence(sid, n_tok)
    c.begin_step([sid])

    # (A, n_kv, q_len, head_dim) as the projections produce it — a transpose of
    # the (A, q_len, n_kv, D) layout, so head_dim is contiguous but the token
    # and head axes are not. Kept un-cloned on purpose.
    k = torch.randn(1, n_tok, N_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = k.transpose(1, 2)
    v = torch.randn(1, n_tok, N_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = v.transpose(1, 2) * 3.0        # a different scale per tensor
    assert not k.is_contiguous() and not v.is_contiguous()

    k_all, v_all = c.extend(0, k, v)
    assert k_all.shape == (1, N_KV, n_tok, HEAD_DIM)

    for got, src in ((k_all, k), (v_all, v)):
        want = _reference(src)
        step = (src.float().abs().amax(dim=-1, keepdim=True) / 7.0)
        err = (got.float() - want).abs()
        # Tolerance is one quantization step, not a fixed epsilon: a handful of
        # elements land within an ULP of a .5 boundary, where the kernel (round
        # half away from zero, Triton's fp32 divide) and torch.round (half to
        # even) pick different codes. Measured 3 such elements in 5120, each off
        # by exactly one code, with the *scales* bit-identical. Anything worse
        # than one step is a real disagreement.
        # 1.05 not 1.0: the dequantized result is rounded to bf16, which pushes a
        # tie-broken element a shade past one full step (measured 1.0072).
        assert (err <= 1.05 * step).all(), (err / step).max()
        # And the bulk must be far tighter than that — a wrong nibble order or a
        # lost sign bit passes the bound above on some elements but not this.
        assert err.pow(2).mean().sqrt() < 0.1 * step.mean(), err.pow(2).mean().sqrt()


@cuda
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
def test_paged_kernel_reads_the_same_values_the_gather_path_does(HEAD_DIM):
    """Read kernel vs. write kernel: the layout contract, end to end.

    The gather path is the independent witness — it unpacks in torch, in a
    different order, from the same bytes. If the two kernels disagreed about
    which nibble holds which channel, attention would still return finite,
    plausible numbers; only this comparison catches it.
    """
    torch.manual_seed(1)
    n_tok, n_head = 70, 4
    n_blocks = math.ceil(n_tok / BS) + 2
    c = _cache(n_blocks, "cuda", head_dim=HEAD_DIM)
    sid = 0
    c.allocate_sequence(sid, n_tok)
    c.begin_step([sid])

    k = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_all, v_all = c.extend(0, k, v)          # writes AND returns the dequantized view

    bt, lens = c.build_static_buffers([sid], 1, n_blocks * BS, "cuda")
    q = torch.randn(1, n_head, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    got = c.paged_attend(0, q, bt, torch.tensor([n_tok], device="cuda"))

    # Reference SDPA over the *dequantized* history, GQA-expanded by hand.
    rep = n_head // N_KV
    kr = k_all.repeat_interleave(rep, dim=1).float()
    vr = v_all.repeat_interleave(rep, dim=1).float()
    att = (q.float() @ kr.transpose(-1, -2)) * (HEAD_DIM ** -0.5)
    ref = att.softmax(dim=-1) @ vr

    err = (got.float() - ref).abs().max() / ref.abs().max()
    assert err < 5e-2, err


@cuda
@pytest.mark.parametrize("n_splits", [1, 4])
def test_packed_attention_matches_unpacked_int8_closely(n_splits):
    """INT4 is coarser than INT8, but it is not *wrong*.

    Same inputs through both pools. The metric is cosine similarity, not a
    relative error bound, because the two are not interchangeable here and the
    difference was measured rather than assumed: on gaussian K/V, correct INT4
    lands at cos 0.9882 / relative RMS 0.155 against INT8 — 15% is simply what
    15 levels over +/-amax buys, and no error bound tight enough to be
    interesting would pass. Swapping the two nibbles' roles (the layout bug this
    file exists to catch) gives cos 0.027 / relative RMS 1.414. Cosine separates
    them by a factor of 36; relative RMS by 9.
    """
    torch.manual_seed(2)
    n_tok, n_head = 200, 4
    n_blocks = math.ceil(n_tok / BS) + 2
    q = torch.randn(1, n_head, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    outs = []
    for kv in (INT4, torch.int8):
        mgr = BlockManager(n_blocks, BS)
        c = PagedLlamaKVCache(_cfg(), mgr, "cuda", torch.bfloat16, kv_dtype=kv)
        c.allocate_sequence(0, n_tok)
        c.begin_step([0])
        c.extend(0, k, v)
        bt, _ = c.build_static_buffers([0], 1, n_blocks * BS, "cuda")
        outs.append(c.paged_attend(0, q, bt, torch.tensor([n_tok], device="cuda"),
                                   n_splits=n_splits).float())

    cos = torch.nn.functional.cosine_similarity(
        outs[0].flatten(), outs[1].flatten(), dim=0)
    assert cos > 0.95, cos


@cuda
def test_windowed_int4_skips_out_of_window_splits_correctly():
    """PACK4 and WINDOW are independent constexprs; check they compose."""
    torch.manual_seed(3)
    n_tok, n_head, window, sinks = 300, 4, 64, 4
    n_blocks = math.ceil(n_tok / BS) + 2
    mgr = BlockManager(n_blocks, BS)
    c = PagedLlamaKVCache(_cfg(), mgr, "cuda", torch.bfloat16, kv_dtype=INT4,
                          attn_window=window, attn_sinks=sinks)
    c.allocate_sequence(0, n_tok)
    c.begin_step([0])
    k = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, N_KV, n_tok, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_all, v_all = c.extend(0, k, v)

    bt, _ = c.build_static_buffers([0], 1, n_blocks * BS, "cuda")
    q = torch.randn(1, n_head, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    got = c.paged_attend(0, q, bt, torch.tensor([n_tok], device="cuda"), n_splits=4)

    rep = n_head // N_KV
    kr = k_all.repeat_interleave(rep, dim=1).float()
    vr = v_all.repeat_interleave(rep, dim=1).float()
    att = (q.float() @ kr.transpose(-1, -2)) * (HEAD_DIM ** -0.5)
    pos = torch.arange(n_tok, device="cuda")
    keep = (pos >= n_tok - window) | (pos < sinks)
    att = att.masked_fill(~keep, float("-inf"))
    ref = att.softmax(dim=-1) @ vr

    err = (got.float() - ref).abs().max() / ref.abs().max()
    assert err < 5e-2, err
