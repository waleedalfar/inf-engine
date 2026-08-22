"""Correctness tests for the paged flash-decoding attention kernel.

Validated against ``F.scaled_dot_product_attention`` with an explicit bool mask,
across the combinations the audit skill calls out as the ways kernels of this
shape silently go wrong:

  - ``start_pos > 0`` (an upper-left causal mask passes every start_pos==0 test)
  - ``kv_len`` not a multiple of the KV chunk size or the page size
  - splits that lie entirely past the true length (all-masked -> must not NaN)
  - block tables whose pages are shuffled, so a bug that assumes contiguous
    physical layout cannot pass
  - GQA groups > 1, where a bug can mix up which query heads share a KV head

Run:
    pytest tests/test_paged_attention_kernel.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)

if torch.cuda.is_available():
    from engine.kernels.paged_attention import paged_flash_attention

DEV = "cuda"
DTYPE = torch.bfloat16


def _build(B, n_head, n_kv, q_len, kv_len, D, page, shuffle=True, seed=0):
    """Random paged KV pool plus the contiguous (k, v) it represents."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_pages_per_seq = (kv_len + page - 1) // page
    total_pages = B * n_pages_per_seq + 3          # +3 so some pages are unused
    k_pool = torch.randn(total_pages, n_kv, page, D, device=DEV, dtype=DTYPE, generator=g)
    v_pool = torch.randn(total_pages, n_kv, page, D, device=DEV, dtype=DTYPE, generator=g)

    perm = torch.randperm(total_pages, generator=g, device=DEV)
    block_table = torch.empty(B, n_pages_per_seq, dtype=torch.int32, device=DEV)
    for b in range(B):
        pages = perm[b * n_pages_per_seq:(b + 1) * n_pages_per_seq]
        block_table[b] = pages if shuffle else torch.arange(
            b * n_pages_per_seq, (b + 1) * n_pages_per_seq, device=DEV, dtype=torch.int32)

    # Contiguous reference view of the same data.
    k_ref = torch.empty(B, n_kv, n_pages_per_seq * page, D, device=DEV, dtype=DTYPE)
    v_ref = torch.empty_like(k_ref)
    for b in range(B):
        for lb in range(n_pages_per_seq):
            phys = int(block_table[b, lb])
            k_ref[b, :, lb * page:(lb + 1) * page] = k_pool[phys].to(k_ref.dtype)
            v_ref[b, :, lb * page:(lb + 1) * page] = v_pool[phys].to(v_ref.dtype)

    q = torch.randn(B, n_head, q_len, D, device=DEV, dtype=DTYPE, generator=g)
    return q, k_pool, v_pool, block_table, k_ref, v_ref


def _reference(q, k_ref, v_ref, kv_len, q_len, n_rep):
    """SDPA with the explicit offset-causal mask, K/V expanded and truncated."""
    B, n_head, _, D = q.shape
    k = k_ref[:, :, :kv_len]
    v = v_ref[:, :, :kv_len]
    if n_rep > 1:
        b_, h_, t_, d_ = k.shape
        k = k[:, :, None].expand(b_, h_, n_rep, t_, d_).reshape(b_, h_ * n_rep, t_, d_)
        v = v[:, :, None].expand(b_, h_, n_rep, t_, d_).reshape(b_, h_ * n_rep, t_, d_)
    start = kv_len - q_len
    rows = torch.arange(q_len, device=DEV)
    cols = torch.arange(kv_len, device=DEV)
    mask = cols[None, :] <= (rows[:, None] + start)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])


@pytest.mark.parametrize("q_len", [1, 2, 5, 8])
@pytest.mark.parametrize("kv_len", [1, 7, 16, 63, 64, 129, 512, 1000])
def test_matches_sdpa_across_shapes(q_len, kv_len):
    if kv_len < q_len:
        pytest.skip("kv_len must cover the new tokens")
    B, n_head, n_kv, D, page = 2, 8, 2, 64, 16
    q, kp, vp, bt, kr, vr = _build(B, n_head, n_kv, q_len, kv_len, D, page, seed=kv_len)
    kv_lens = torch.full((B,), kv_len, dtype=torch.int32, device=DEV)

    got = paged_flash_attention(q, kp, vp, bt, kv_lens, page_size=page)
    want = _reference(q, kr, vr, kv_len, q_len, n_head // n_kv)

    assert torch.isfinite(got.float()).all(), "kernel produced NaN/Inf"
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("n_rep", [1, 2, 4, 8])
def test_gqa_group_mapping(n_rep):
    """Each query head must attend the KV head it actually shares.

    With random per-head K/V a wrong head mapping produces a completely
    different output, so this catches an off-by-one in rep/tok decomposition.
    """
    B, n_kv, q_len, kv_len, D, page = 1, 4, 3, 200, 64, 16
    n_head = n_kv * n_rep
    q, kp, vp, bt, kr, vr = _build(B, n_head, n_kv, q_len, kv_len, D, page, seed=n_rep)
    kv_lens = torch.full((B,), kv_len, dtype=torch.int32, device=DEV)
    got = paged_flash_attention(q, kp, vp, bt, kv_lens, page_size=page)
    want = _reference(q, kr, vr, kv_len, q_len, n_rep)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


def test_offset_causal_not_upper_left():
    """Direct probe: an upper-left mask would starve every row but the first.

    Uses one-hot values so the output reveals exactly which keys were attended
    (the diagnosis recipe from the causal-mask audit skill).
    """
    # kv_len < D so every position maps to a distinct one-hot channel and the
    # "beyond the bound" slice is never empty.
    B, n_head, n_kv, q_len, kv_len, D, page = 1, 1, 1, 4, 48, 64, 16
    kp = torch.zeros(8, n_kv, page, D, device=DEV, dtype=DTYPE)
    vp = torch.zeros(8, n_kv, page, D, device=DEV, dtype=DTYPE)
    # value[pos] = one-hot(pos % D); key[pos] = 0 so all admitted keys tie and
    # the output becomes the mean of the admitted positions' one-hots.
    for pos in range(kv_len):
        vp[pos // page, 0, pos % page, pos % D] = 1.0
    bt = torch.arange(kv_len // page, dtype=torch.int32, device=DEV)[None]
    q = torch.zeros(B, n_head, q_len, D, device=DEV, dtype=DTYPE)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=DEV)

    out = paged_flash_attention(q, kp, vp, bt, kv_lens, page_size=page).float()
    start = kv_len - q_len
    for t in range(q_len):
        n_admitted = start + t + 1
        # Uniform average over admitted positions -> each contributes 1/n.
        assert out[0, 0, t, :n_admitted].min() > 0, (
            f"row {t}: some of the {n_admitted} admitted keys got zero weight "
            "(upper-left masking would starve rows > 0)"
        )
        assert out[0, 0, t, n_admitted:].abs().max() < 1e-3, (
            f"row {t}: attended a key beyond its causal bound"
        )


def test_splits_past_true_length_do_not_nan():
    """Force many splits against a short history: most have zero admitted keys."""
    B, n_head, n_kv, q_len, kv_len, D, page = 1, 4, 2, 2, 20, 64, 16
    q, kp, vp, bt, kr, vr = _build(B, n_head, n_kv, q_len, 20, D, page, seed=3)
    # Give the block table far more pages than the sequence uses, as a padded
    # CUDA-graph bucket would.
    pad = torch.zeros(B, 60 - bt.shape[1], dtype=torch.int32, device=DEV)
    bt_pad = torch.cat([bt, pad], dim=1)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=DEV)

    got = paged_flash_attention(q, kp, vp, bt_pad, kv_lens, page_size=page, n_splits=32)
    want = _reference(q, kr, vr, kv_len, q_len, n_head // n_kv)
    assert torch.isfinite(got.float()).all(), "all-masked split produced NaN"
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("n_splits", [1, 2, 3, 8, 17, 32])
def test_split_count_does_not_change_result(n_splits):
    """The lse combine must make the answer independent of the split count."""
    B, n_head, n_kv, q_len, kv_len, D, page = 1, 8, 2, 5, 777, 64, 16
    q, kp, vp, bt, kr, vr = _build(B, n_head, n_kv, q_len, kv_len, D, page, seed=9)
    kv_lens = torch.full((B,), kv_len, dtype=torch.int32, device=DEV)
    got = paged_flash_attention(q, kp, vp, bt, kv_lens, page_size=page, n_splits=n_splits)
    want = _reference(q, kr, vr, kv_len, q_len, n_head // n_kv)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


def test_ragged_batch_lengths():
    """Sequences of different lengths in one batch each use their own bound."""
    B, n_head, n_kv, q_len, D, page = 3, 8, 2, 3, 64, 16
    lens = [37, 256, 129]
    q, kp, vp, bt, kr, vr = _build(B, n_head, n_kv, q_len, max(lens), D, page, seed=5)
    kv_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
    got = paged_flash_attention(q, kp, vp, bt, kv_lens, page_size=page)
    for b in range(B):
        want_b = _reference(q[b:b + 1], kr[b:b + 1], vr[b:b + 1],
                            lens[b], q_len, n_head // n_kv)
        torch.testing.assert_close(got[b:b + 1].float(), want_b.float(),
                                   rtol=2e-2, atol=2e-2)
