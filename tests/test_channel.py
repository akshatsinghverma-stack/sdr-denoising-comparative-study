"""
Regression test for the add_multipath causal-convolution bug.

A multipath channel h=[h0, h1, h2, ...] models a *causal* tap-delay system,
r[n] = sum_k h[k]*s[n-k], where h[0] is the direct/zero-delay path. Using
mode='same' convolution (correct for symmetric matched filters, wrong here)
silently shifted the signal by ~1 sample for any non-centered kernel --
concretely, h=[1,0,0] (which must be an exact identity channel) returned the
signal shifted by one sample instead of unchanged. This was present for
every multipath profile actually used in Case Study 2, not just this edge
case, and was only caught by testing a "no ISI" (h=[1,0,0]) severity level
directly against the identity-channel expectation.
"""
import numpy as np

from src.channel import add_multipath


def test_identity_tap_is_true_identity():
    """h=[1,0,0] (a channel with only the direct path) must return the
    signal completely unchanged -- this is the exact check that caught the
    'same'-mode shift bug."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    out = add_multipath(signal, [1.0, 0.0, 0.0])
    assert np.allclose(out, signal), (
        "A multipath channel with only the direct (zero-delay) tap active must be a true "
        "identity -- any deviation indicates a convolution-alignment bug."
    )


def test_multipath_is_causal():
    """r[n] must depend only on s[0..n], never on future samples: perturbing
    a sample at the tail of the input must not change any earlier output."""
    rng = np.random.default_rng(1)
    h = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
    signal = rng.standard_normal(50) + 1j * rng.standard_normal(50)

    out1 = add_multipath(signal, h)
    signal2 = signal.copy()
    signal2[-1] += 5.0  # perturb only the very last sample
    out2 = add_multipath(signal2, h)

    # Everything except the last couple of (causally-affected) samples must be untouched.
    assert np.allclose(out1[:-len(h)], out2[:-len(h)])


def test_multipath_matches_manual_causal_convolution():
    rng = np.random.default_rng(2)
    h = np.array([1.0, 0.4 + 0.3j, -0.1 + 0.1j])
    signal = rng.standard_normal(20) + 1j * rng.standard_normal(20)
    out = add_multipath(signal, h)
    expected = np.convolve(signal, h, mode="full")[: len(signal)]
    assert np.allclose(out, expected)
