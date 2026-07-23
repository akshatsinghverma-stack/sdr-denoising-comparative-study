"""
1-D CNN Denoising Autoencoder
==============================
A small Conv1D encoder-decoder that learns to map noisy IQ windows to
clean IQ windows.  Designed for fast CPU training (< 10 min).

Architecture (corrected -- this previously described a Conv1DTranspose
decoder the code has never actually built; verify docstrings against the
real layer calls, not just at review time):
------------
Encoder:  Conv1D(32,7) -> Conv1D(16,5) -> Conv1D(8,3)
Decoder:  Conv1D(16,3) -> Conv1D(32,5) -> Conv1D(2,7)   -- all plain Conv1D
with 'same' padding, not Conv1DTranspose; there is no explicit upsampling
step because no layer here changes the sequence length.

All layers use 'same' padding so the output length equals input length,
avoiding the need for cropping / padding logic.

Input shape : (batch, window_len, 2)   — I and Q as two channels
Output shape: (batch, window_len, 2)
"""

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info/warning spam

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _iq_to_real(signal: np.ndarray) -> np.ndarray:
    """Convert complex IQ array to (N, 2) real array [I, Q]."""
    return np.stack([signal.real, signal.imag], axis=-1)


def _real_to_iq(arr: np.ndarray) -> np.ndarray:
    """Convert (N, 2) real array back to complex IQ."""
    return arr[:, 0] + 1j * arr[:, 1]


def make_windows(signal_2ch: np.ndarray, window_len: int = 128, stride: int = 64):
    """Slice a (N, 2) array into overlapping windows.

    Returns
    -------
    windows : np.ndarray, shape (num_windows, window_len, 2)
    """
    N = signal_2ch.shape[0]
    starts = range(0, N - window_len + 1, stride)
    windows = np.array([signal_2ch[s : s + window_len] for s in starts])
    return windows


def reconstruct_from_windows(windows: np.ndarray, original_len: int,
                             window_len: int = 128, stride: int = 64,
                             weighting: str = "uniform", fallback: np.ndarray = None):
    """Overlap-add reconstruction from windowed predictions.

    Parameters
    ----------
    windows : np.ndarray, shape (num_windows, window_len, 2)
    original_len : int
        Length of the original signal.
    weighting : {"uniform", "triangular"}
        "uniform" (default) reproduces the ORIGINAL behavior exactly (every
        sample in every overlapping window counted equally) -- every
        existing experiment in this project uses this default and is
        unaffected. "triangular" is the intervention named in
        report/findings_cnn_high_snr_floor.md to test whether the
        high-SNR error floor (found to cluster non-uniformly near one side
        of the window-overlap cycle) is a reconstruction-weighting artifact:
        it downweights each window's own edges relative to its center, so an
        interior sample's two overlapping predictions are blended favoring
        whichever window evaluated it closer to that window's own center.
    fallback : np.ndarray, shape (original_len, 2), optional
        Whenever (original_len - window_len) is not a multiple of stride,
        the trailing `(original_len - window_len) % stride` samples are
        never covered by any window (found via a code-correctness critique
        pass -- previously silently left at 0.0, which a hard-decision
        demodulator reads as a fixed, deterministic bit regardless of what
        was transmitted, rather than "no denoising applied here" as intended
        -- see report/findings_cnn_high_snr_floor.md's correction). If given,
        those uncovered tail positions are filled from `fallback` (typically
        the noisy input) instead of left at zero, matching the "edge samples
        pass through unmodified" convention already used by
        src/mmse_equalizer.py. If None, the tail is left at zero (only used
        by the diagnostic in experiments/test_cnn_overlap_weighting_intervention.py,
        which isolates the weighting scheme and is unaffected by this tail
        either way since neither weighting covers it).

    Returns
    -------
    signal : np.ndarray, shape (original_len, 2)
    """
    if weighting == "uniform":
        w = np.ones(window_len, dtype=np.float64)
    elif weighting == "triangular":
        center = (window_len - 1) / 2.0
        w = 1.0 - np.abs(np.arange(window_len) - center) / (center + 1.0)
        w = np.clip(w, 1e-3, 1.0)  # never fully zero out an edge sample
    else:
        raise ValueError(f"Unknown weighting {weighting!r}, expected 'uniform' or 'triangular'")

    output = np.zeros((original_len, 2), dtype=np.float64)
    counts = np.zeros(original_len, dtype=np.float64)

    for i, s in enumerate(range(0, original_len - window_len + 1, stride)):
        output[s : s + window_len] += windows[i] * w[:, None]
        counts[s : s + window_len] += w

    uncovered = counts == 0
    # Avoid division by zero for tail samples not covered by any window
    counts[uncovered] = 1.0
    output /= counts[:, None]
    if fallback is not None:
        output[uncovered] = fallback[uncovered]
    return output


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_autoencoder(window_len: int = 128, num_channels: int = 2):
    """Build the 1-D CNN denoising autoencoder.

    Small architecture (6,890 parameters, verified via model.count_params() --
    see report.md Section 8; an earlier "~15k" estimate here was never
    checked against the actual built model) for fast CPU training.
    """
    inp = keras.Input(shape=(window_len, num_channels), name="noisy_input")

    # --- Encoder ---
    x = layers.Conv1D(32, 7, activation="relu", padding="same", name="enc1")(inp)
    x = layers.Conv1D(16, 5, activation="relu", padding="same", name="enc2")(x)
    x = layers.Conv1D(8,  3, activation="relu", padding="same", name="enc3")(x)

    # --- Decoder ---
    x = layers.Conv1D(16, 3, activation="relu", padding="same", name="dec1")(x)
    x = layers.Conv1D(32, 5, activation="relu", padding="same", name="dec2")(x)
    x = layers.Conv1D(num_channels, 7, activation="linear", padding="same", name="dec_out")(x)

    model = keras.Model(inp, x, name="cnn_denoiser")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model


# ---------------------------------------------------------------------------
# Training & prediction API
# ---------------------------------------------------------------------------

def prepare_dataset(clean_signals: list[np.ndarray],
                    noisy_signals: list[np.ndarray],
                    window_len: int = 128, stride: int = 64):
    """Prepare windowed (X, Y) dataset from multiple signal pairs.

    Parameters
    ----------
    clean_signals : list of complex np.ndarray
    noisy_signals : list of complex np.ndarray  (same lengths, paired)

    Returns
    -------
    X : np.ndarray  — noisy windows
    Y : np.ndarray  — clean windows
    """
    X_parts, Y_parts = [], []
    for clean, noisy in zip(clean_signals, noisy_signals):
        c2 = _iq_to_real(clean)
        n2 = _iq_to_real(noisy)
        X_parts.append(make_windows(n2, window_len, stride))
        Y_parts.append(make_windows(c2, window_len, stride))
    return np.concatenate(X_parts), np.concatenate(Y_parts)


def train_autoencoder(X_train, Y_train, X_val, Y_val,
                      window_len: int = 128,
                      epochs: int = 50, batch_size: int = 64,
                      verbose: int = 1):
    """Build, train, and return the autoencoder + training history."""
    model = build_autoencoder(window_len=window_len)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def denoise_signal(model, noisy_signal: np.ndarray,
                   window_len: int = 128, stride: int = 64,
                   weighting: str = "uniform") -> np.ndarray:
    """Denoise a full-length complex signal using the trained autoencoder.

    weighting: passed through to reconstruct_from_windows (default "uniform"
    = original, unchanged behavior).

    Any trailing samples not covered by a full window (whenever
    original_len - window_len isn't a multiple of stride) are filled from
    the noisy input rather than left at zero -- see reconstruct_from_windows'
    `fallback` parameter docstring.

    Returns complex IQ array of the same length as input.
    """
    n2 = _iq_to_real(noisy_signal)
    original_len = n2.shape[0]
    windows = make_windows(n2, window_len, stride)

    pred_windows = model.predict(windows, verbose=0)
    reconstructed = reconstruct_from_windows(pred_windows, original_len,
                                             window_len, stride, weighting=weighting,
                                             fallback=n2)
    return _real_to_iq(reconstructed)
