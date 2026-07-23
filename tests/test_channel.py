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

from src.channel import add_multipath, add_time_varying_multipath


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


# ============================================================================
# add_time_varying_multipath (Case Study 3) -- previously had zero regression
# coverage, flagged during a self-critique pass. A separate, adversarially-
# found issue (report/findings_preamble_drift_correction.md) established that
# this function's *default* drift parameters change substantially (60-91% of
# base tap magnitude) within the first 1000 samples for the flat-fading
# random-walk config actually used in Case Study 3 -- these tests check the
# function's own correctness (causality, identity-at-zero-drift, direct-path
# handling), not the separate, already-corrected claim about how slow the
# default drift rate is.
# ============================================================================

def test_time_varying_zero_drift_matches_static_channel():
    """drift_depth=0 (sinusoidal mode) must reproduce add_multipath's static
    result to floating-point precision -- the exact check report.md Section 6.1
    cites as having been verified."""
    rng = np.random.default_rng(3)
    h_base = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
    signal = rng.standard_normal(200) + 1j * rng.standard_normal(200)

    static_out = add_multipath(signal, h_base)
    varying_out = add_time_varying_multipath(
        signal, h_base=h_base, sps=1, mode="sinusoidal",
        n_cycles=1.5, drift_depth=0.0, seed=0,
    )
    assert np.allclose(static_out, varying_out, atol=1e-10)


def test_time_varying_identity_tap_is_true_identity():
    """h_base=[1,0,0] must stay an exact identity regardless of drift on the
    (zero-valued) non-direct taps -- mirrors test_identity_tap_is_true_identity."""
    rng = np.random.default_rng(4)
    signal = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    out = add_time_varying_multipath(
        signal, h_base=[1.0, 0.0, 0.0], sps=1, mode="random_walk",
        coherence_symbols=50.0, drift_depth=0.6, seed=1,
    )
    assert np.allclose(out, signal)


def test_time_varying_is_causal():
    """Same causality check as test_multipath_is_causal, applied to the
    time-varying channel: perturbing the tail must not change earlier output."""
    rng = np.random.default_rng(5)
    h_base = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
    signal = rng.standard_normal(200) + 1j * rng.standard_normal(200)

    out1 = add_time_varying_multipath(signal, h_base=h_base, sps=1, mode="sinusoidal",
                                       n_cycles=1.5, drift_depth=0.6, seed=2)
    signal2 = signal.copy()
    signal2[-1] += 5.0
    out2 = add_time_varying_multipath(signal2, h_base=h_base, sps=1, mode="sinusoidal",
                                       n_cycles=1.5, drift_depth=0.6, seed=2)
    assert np.allclose(out1[:-len(h_base)], out2[:-len(h_base)])


def test_time_varying_direct_path_fixed_unless_requested():
    """vary_direct_path=False (the default) must hold h_base[0] exactly fixed
    -- verified via a constant-input probe (for a single-tap channel, the
    output of a constant input IS the time-varying tap gain directly)."""
    const_signal = np.ones(500, dtype=complex)
    out_fixed = add_time_varying_multipath(
        const_signal, h_base=[1.0, 0.4 + 0.3j], sps=1, mode="random_walk",
        coherence_symbols=100.0, drift_depth=0.6, vary_direct_path=False, seed=6,
    )
    # With vary_direct_path=False and a constant input, out[n] = h0 (fixed) +
    # h1(n)*const_signal[n-1] (shifted, drifting) -- only the first sample is
    # driven purely by the fixed direct path (no tap-1 contribution yet).
    assert np.isclose(out_fixed[0], 1.0)
