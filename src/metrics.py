"""
Metrics Module
===============
Functions for evaluating denoising quality:
  - Output SNR (dB)
  - Bit Error Rate (BER)
  - Mean Squared Error (MSE)
"""

import numpy as np


def compute_snr_db(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute the output SNR in dB.

    SNR_out = 10 * log10( P_signal / P_residual_noise )

    where P_residual_noise = mean(|clean - denoised|^2).
    """
    sig_power = np.mean(np.abs(clean) ** 2)
    noise_power = np.mean(np.abs(clean - denoised) ** 2)
    if noise_power < 1e-20:
        return 100.0  # effectively perfect
    return 10.0 * np.log10(sig_power / noise_power)


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean Squared Error between clean and denoised signals."""
    return float(np.mean(np.abs(clean - denoised) ** 2))


def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit Error Rate = fraction of bits that differ.

    Both arrays must be integer 0/1 and the same length.
    """
    errors, n_bits = compute_ber_count(tx_bits, rx_bits)
    return float(errors) / n_bits


def compute_ber_count(tx_bits: np.ndarray, rx_bits: np.ndarray):
    """Raw (error_count, n_bits) pair, needed to aggregate error counts
    across Monte Carlo trials (a mean-of-rates is not enough to tell whether
    zero observed errors means "perfect" or just "not enough bits tested")."""
    if len(tx_bits) != len(rx_bits):
        # If lengths differ (e.g. due to windowing edge effects), truncate
        min_len = min(len(tx_bits), len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
    errors = int(np.sum(tx_bits != rx_bits))
    return errors, len(tx_bits)
