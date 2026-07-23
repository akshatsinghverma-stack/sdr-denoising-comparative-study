"""
Regression tests for metrics.py, including the rule-of-three BER reporting
fix (a bare BER=0.0 is misleading when it really means "no errors observed
in the bits we happened to test", not "proven perfect").
"""
import numpy as np
import pytest

from src.metrics import compute_ber, compute_ber_count


def test_compute_ber_and_count_are_consistent():
    rng = np.random.default_rng(0)
    tx_bits = rng.integers(0, 2, size=10_000)
    rx_bits = tx_bits.copy()
    flip_idx = rng.choice(10_000, size=37, replace=False)
    rx_bits[flip_idx] = 1 - rx_bits[flip_idx]

    ber = compute_ber(tx_bits, rx_bits)
    errors, n_bits = compute_ber_count(tx_bits, rx_bits)

    assert errors == 37
    assert n_bits == 10_000
    assert ber == pytest.approx(37 / 10_000)


def test_compute_ber_count_zero_errors():
    tx_bits = np.array([0, 1, 1, 0, 1])
    rx_bits = tx_bits.copy()
    errors, n_bits = compute_ber_count(tx_bits, rx_bits)
    assert errors == 0
    assert n_bits == 5

    # Rule-of-three upper bound, as used in the results pipeline.
    upper_bound_95 = 3.0 / n_bits
    assert upper_bound_95 == pytest.approx(0.6)


def test_compute_ber_count_handles_length_mismatch():
    tx_bits = np.array([0, 1, 1, 0, 1, 0])
    rx_bits = np.array([0, 1, 1, 0])  # shorter, e.g. windowing edge effect
    errors, n_bits = compute_ber_count(tx_bits, rx_bits)
    assert n_bits == 4
    assert errors == 0
