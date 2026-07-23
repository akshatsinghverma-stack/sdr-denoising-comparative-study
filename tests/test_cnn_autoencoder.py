"""
test_cnn_autoencoder.py — Regression coverage for reconstruct_from_windows,
added after a code-correctness critique pass found it had ZERO test coverage
despite silently zero-filling any trailing samples not covered by a full
window (previously read by a hard-decision demodulator as a fixed,
deterministic bit regardless of what was transmitted -- see the `fallback`
parameter this test locks in, and report/findings_cnn_high_snr_floor.md's
correction).

cnn_autoencoder.py imports tensorflow at module level, which requirements-
test.txt (and CI) deliberately excludes to keep the regression suite fast and
lightweight -- so this file is skipped, not failed, when tensorflow isn't
installed (matches the project's existing "no test imports the CNN/Hybrid
modules" CI design; run it locally with the full requirements.txt to get
real coverage).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tf = pytest.importorskip("tensorflow")

from src.cnn_autoencoder import make_windows, reconstruct_from_windows


def test_uncovered_tail_exists_for_a_realistic_config():
    """Sanity-check the premise this whole test file exists to cover: with
    window_len=128/stride=64 and a signal length not a multiple of stride
    relative to window_len (Case Study 1's actual 25,000-sample diagnostic
    configuration), a real gap exists between the last window's coverage and
    the signal's end."""
    original_len, window_len, stride = 25_000, 128, 64
    last_start = ((original_len - window_len) // stride) * stride
    covered_end = last_start + window_len
    assert covered_end < original_len, "test premise requires an uncovered tail"
    assert original_len - covered_end == 40


def test_reconstruction_without_fallback_zero_fills_the_uncovered_tail():
    """Locks in the ORIGINAL (buggy) behavior when no fallback is supplied,
    so the fix below is a deliberate, tested opt-in via `fallback`, not a
    silent behavior change for any caller that doesn't pass it."""
    original_len, window_len, stride = 200, 128, 64
    num_windows = len(range(0, original_len - window_len + 1, stride))
    windows = np.full((num_windows, window_len, 2), 5.0)

    out = reconstruct_from_windows(windows, original_len, window_len, stride)
    last_start = ((original_len - window_len) // stride) * stride
    covered_end = last_start + window_len
    assert covered_end < original_len
    assert np.allclose(out[covered_end:], 0.0)
    assert np.allclose(out[:covered_end], 5.0)


def test_fallback_patches_the_uncovered_tail_instead_of_zero_filling():
    """The fix: passing `fallback` (the noisy input) fills exactly the
    uncovered tail positions with the fallback's own values, and leaves every
    covered position (the actual denoised prediction) untouched."""
    original_len, window_len, stride = 200, 128, 64
    num_windows = len(range(0, original_len - window_len + 1, stride))
    windows = np.full((num_windows, window_len, 2), 5.0)
    fallback = np.full((original_len, 2), -9.0)

    out = reconstruct_from_windows(windows, original_len, window_len, stride, fallback=fallback)
    last_start = ((original_len - window_len) // stride) * stride
    covered_end = last_start + window_len
    assert np.allclose(out[covered_end:], -9.0), "uncovered tail must come from fallback, not zero"
    assert np.allclose(out[:covered_end], 5.0), "covered positions must be unaffected by fallback"


def test_reconstruction_is_exact_when_tail_is_fully_covered():
    """When (original_len - window_len) IS a multiple of stride (e.g. Case
    Study 2's SPS=4 configuration, 400,000 samples), there is no uncovered
    tail at all, so fallback must have zero effect regardless of whether
    it's supplied -- confirms the bug is configuration-dependent, not
    universal, matching report.md's correction."""
    original_len, window_len, stride = 400_000, 128, 64
    assert (original_len - window_len) % stride == 0
    num_windows = len(range(0, original_len - window_len + 1, stride))
    windows = np.full((num_windows, window_len, 2), 3.0)
    fallback = np.full((original_len, 2), -1.0)

    out_no_fallback = reconstruct_from_windows(windows, original_len, window_len, stride)
    out_with_fallback = reconstruct_from_windows(windows, original_len, window_len, stride, fallback=fallback)
    assert np.allclose(out_no_fallback, 3.0)
    assert np.allclose(out_with_fallback, 3.0)


def test_make_windows_and_reconstruct_round_trip_on_covered_region():
    """End-to-end sanity check: windowing a real signal then reconstructing
    identity predictions (the windows unchanged) recovers the original
    signal exactly over the covered region, isolating this test from the
    tail-coverage question above."""
    rng = np.random.default_rng(0)
    original_len, window_len, stride = 500, 128, 64
    signal = rng.normal(size=(original_len, 2))
    windows = make_windows(signal, window_len, stride)
    out = reconstruct_from_windows(windows, original_len, window_len, stride, fallback=signal)
    last_start = ((original_len - window_len) // stride) * stride
    covered_end = last_start + window_len
    assert np.allclose(out[:covered_end], signal[:covered_end], atol=1e-10)
    assert np.allclose(out[covered_end:], signal[covered_end:], atol=1e-10)
