"""YaRN RoPE scaling for running past the trained context length.

Qwen3 is trained to 32768 and officially extends to 131072 with YaRN factor 4.
Without scaling, positions past the trained window get frequencies the model has
never seen and attention degrades sharply; plain interpolation avoids that but
damages the high-frequency dimensions used for local ordering. YaRN splits the
spectrum. These tests pin the properties that split must have.

Run:
    pytest tests/test_rope_scaling.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from engine.layers import _yarn_inv_freq, precompute_rope_freqs

HEAD_DIM = 128
THETA = 1_000_000.0
ORIG = 32_768


def test_factor_one_is_exactly_unscaled():
    """The default must reproduce the original tables bit-for-bit."""
    a = precompute_rope_freqs(HEAD_DIM, 512, THETA, "cpu")
    b = precompute_rope_freqs(HEAD_DIM, 512, THETA, "cpu",
                              rope_scaling_factor=1.0, rope_original_n_ctx=ORIG)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)


def test_scaling_requires_original_length():
    with pytest.raises(ValueError, match="rope_original_n_ctx"):
        precompute_rope_freqs(HEAD_DIM, 512, THETA, "cpu", rope_scaling_factor=4.0)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_high_frequencies_are_preserved(factor):
    """The fast-rotating dimensions must be extrapolated, not interpolated.

    This is what separates YaRN from plain position interpolation: dividing
    *every* frequency by the factor is what destroys local ordering.
    """
    base = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    scaled, _ = _yarn_inv_freq(HEAD_DIM, THETA, factor, ORIG, "cpu")
    # Dimension 0 is the fastest rotation — it should be untouched.
    torch.testing.assert_close(scaled[0], base[0], rtol=1e-6, atol=0)
    assert scaled[0] > base[0] / factor * 1.5, "fastest dimension was interpolated"


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_low_frequencies_are_interpolated(factor):
    """The slow dimensions — wavelength past the trained window — must be scaled."""
    base = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    scaled, _ = _yarn_inv_freq(HEAD_DIM, THETA, factor, ORIG, "cpu")
    torch.testing.assert_close(scaled[-1], base[-1] / factor, rtol=1e-5, atol=0)


def test_frequencies_lie_between_the_two_extremes():
    base = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    scaled, _ = _yarn_inv_freq(HEAD_DIM, THETA, 4.0, ORIG, "cpu")
    assert (scaled <= base + 1e-9).all(), "a frequency exceeded pure extrapolation"
    assert (scaled >= base / 4.0 - 1e-9).all(), "a frequency fell below pure interpolation"
    # And the blend is monotone in the dimension index.
    ratio = (scaled / base)
    assert (ratio.diff() <= 1e-6).all(), "extrapolation weight is not monotone"


def test_mscale_matches_yarn_formula():
    for factor in (2.0, 4.0, 8.0):
        _, mscale = _yarn_inv_freq(HEAD_DIM, THETA, factor, ORIG, "cpu")
        assert mscale == pytest.approx(0.1 * math.log(factor) + 1.0)


def test_tables_carry_the_attention_scale():
    cos_a, _ = precompute_rope_freqs(HEAD_DIM, 64, THETA, "cpu")
    cos_b, _ = precompute_rope_freqs(HEAD_DIM, 64, THETA, "cpu",
                                     rope_scaling_factor=4.0, rope_original_n_ctx=ORIG)
    # Position 0 has zero angle, so cos is 1 everywhere before mscale.
    assert cos_a[0].max() == pytest.approx(1.0)
    assert cos_b[0].max() == pytest.approx(0.1 * math.log(4.0) + 1.0)


def test_config_defaults_are_unscaled():
    from engine.config import QWEN3_8B
    assert QWEN3_8B.rope_scaling_factor == 1.0
    assert QWEN3_8B.rope_original_n_ctx is None
