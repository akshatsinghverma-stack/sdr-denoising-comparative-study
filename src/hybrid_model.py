"""
Hybrid LMS → CNN Cascade
==========================
Two-stage pipeline:
  1. LMS adaptive filter performs coarse noise removal.
  2. CNN autoencoder removes the *residual* noise left by LMS.

The CNN is retrained (or fine-tuned) on LMS-pre-cleaned signals so it learns
the residual noise pattern specific to LMS output, rather than raw AWGN.
"""

import numpy as np
from src.lms_filter import LMSFilter
from src.cnn_autoencoder import (
    prepare_dataset, train_autoencoder, denoise_signal,
)


def lms_pre_clean(clean_signals: list[np.ndarray],
                  noisy_signals: list[np.ndarray],
                  num_taps: int = 16, mu: float = 0.01,
                  training_length: int = 1000, modulation: str = "BPSK"):
    """Apply LMS to each noisy signal and return the LMS outputs.

    Returns list of LMS-denoised complex arrays (same lengths as inputs).
    """
    lms_outputs = []
    for clean, noisy in zip(clean_signals, noisy_signals):
        filt = LMSFilter(num_taps=num_taps, mu=mu)
        denoised, _ = filt.denoise(clean, noisy, training_length, modulation)
        lms_outputs.append(denoised)
    return lms_outputs


def train_hybrid_cnn(clean_signals: list[np.ndarray],
                     noisy_signals: list[np.ndarray],
                     num_taps: int = 16, mu: float = 0.01,
                     training_length: int = 1000, modulation: str = "BPSK",
                     window_len: int = 128, stride: int = 64,
                     epochs: int = 50, batch_size: int = 64,
                     verbose: int = 1):
    """Train the residual CNN on LMS-pre-cleaned data.

    Returns the trained Keras model + training history.
    """
    # Stage 1: LMS coarse pass
    lms_outputs = lms_pre_clean(clean_signals, noisy_signals, num_taps, mu, training_length, modulation)

    # Split base signals into train/val BEFORE windowing to prevent leakage
    # We'll put the first 80% of signals in train, last 20% in val.
    # If there are multiple signals (e.g., across SNRs), we split each signal 80/20.
    train_clean, val_clean = [], []
    train_lms, val_lms = [], []

    for c, l in zip(clean_signals, lms_outputs):
        split = int(0.8 * len(c))
        train_clean.append(c[:split])
        val_clean.append(c[split:])
        train_lms.append(l[:split])
        val_lms.append(l[split:])

    X_tr, Y_tr = prepare_dataset(train_clean, train_lms, window_len, stride)
    X_va, Y_va = prepare_dataset(val_clean, val_lms, window_len, stride)

    # Shuffle training windows (validation doesn't strictly need shuffling)
    idx = np.random.default_rng(0).permutation(len(X_tr))
    X_tr, Y_tr = X_tr[idx], Y_tr[idx]

    model, history = train_autoencoder(X_tr, Y_tr, X_va, Y_va,
                                       window_len=window_len,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=verbose)
    return model, history


def denoise_hybrid(lms_filter: LMSFilter, cnn_model,
                   clean: np.ndarray, noisy: np.ndarray,
                   window_len: int = 128, stride: int = 64,
                   training_length: int = 1000, modulation: str = "BPSK"):
    """Run the full hybrid pipeline on a single signal."""
    # Stage 1: LMS
    lms_out, _ = lms_filter.denoise(clean, noisy, training_length, modulation)
    # Stage 2: CNN residual
    denoised = denoise_signal(cnn_model, lms_out, window_len, stride)
    return denoised
