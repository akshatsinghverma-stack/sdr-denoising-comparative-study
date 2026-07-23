#!/usr/bin/env python3
"""
test_cnn_overlap_weighting_intervention.py — The actual falsification test
named (but not performed) in report/findings_cnn_high_snr_floor.md.
=============================================================================
That diagnostic found the high-SNR CNN error floor strongly consistent with
a window/overlap-add reconstruction artifact (error positions cluster
non-uniformly by window phase, χ²=168.0, p=3.5e-36) and named the concrete
next step: change the overlap-add weighting scheme and check whether the
floor specifically shrinks. This performs that test.

A single plain-MSE CNN is trained once (Case Study 1's memoryless-AWGN
config, reduced scale for speed). Its window-level predictions are computed
ONCE per test signal, then reconstructed TWICE from the exact same
predictions -- once with the original uniform overlap-add weighting, once
with the new triangular weighting (src/cnn_autoencoder.py's
reconstruct_from_windows, weighting="triangular", added for this test and
defaulting to "uniform" everywhere else so no existing experiment is
affected). This isolates the reconstruction scheme as the only variable.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_multipath
from src.cnn_autoencoder import (prepare_dataset, train_autoencoder, make_windows,
                                   reconstruct_from_windows, _iq_to_real, _real_to_iq)
from src.metrics import compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

NUM_SYMBOLS = 25_000
SNR_LEVELS_TRAIN = [-10, -5, 0, 5, 10, 15, 20]
ANALYSIS_SNRS = [10, 15, 20]
SEED = 42
MC_TRIALS = 4
SPS = 1
MULTIPATH = None
WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def denoise_both_weightings(model, noisy_signal):
    """Predict window-level outputs ONCE, reconstruct twice (uniform vs
    triangular) from the identical predictions -- isolates reconstruction
    weighting as the only variable."""
    n2 = _iq_to_real(noisy_signal)
    original_len = n2.shape[0]
    windows = make_windows(n2, WINDOW_LEN, STRIDE)
    pred_windows = model.predict(windows, verbose=0)

    uniform = reconstruct_from_windows(pred_windows, original_len, WINDOW_LEN, STRIDE,
                                        weighting="uniform")
    triangular = reconstruct_from_windows(pred_windows, original_len, WINDOW_LEN, STRIDE,
                                           weighting="triangular")
    return _real_to_iq(uniform), _real_to_iq(triangular)


def main():
    print("=" * 78)
    print("  CNN Overlap-Weighting Intervention: does triangular weighting shrink the floor?")
    print("=" * 78)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for modulation in ["BPSK", "QPSK"]:
        print(f"\n{'-'*60}\n  {modulation}\n{'-'*60}")
        demod_fn = _get_demod(modulation)

        train_tx, _, h_rrc = _generate_signal(modulation, SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)
        train_noisy_signals = {snr: add_awgn(train_rx_clean, snr, seed=SEED + snr + 100)
                                for snr in SNR_LEVELS_TRAIN}
        clean_list = [train_tx] * len(SNR_LEVELS_TRAIN)
        noisy_list = [train_noisy_signals[snr] for snr in SNR_LEVELS_TRAIN]
        train_clean_split, val_clean_split, train_noisy_split, val_noisy_split = [], [], [], []
        for c, n in zip(clean_list, noisy_list):
            split = int(0.8 * len(c))
            train_clean_split.append(c[:split]); val_clean_split.append(c[split:])
            train_noisy_split.append(n[:split]); val_noisy_split.append(n[split:])

        X_tr, Y_tr = prepare_dataset(train_clean_split, train_noisy_split, WINDOW_LEN, STRIDE)
        X_va, Y_va = prepare_dataset(val_clean_split, val_noisy_split, WINDOW_LEN, STRIDE)
        idx = np.random.default_rng(0).permutation(len(X_tr))
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]

        model, _ = train_autoencoder(X_tr, Y_tr, X_va, Y_va, window_len=WINDOW_LEN,
                                      epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=0)
        print("  [CNN] trained")

        for trial in range(MC_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)

            for snr in ANALYSIS_SNRS:
                noisy = add_awgn(test_rx_clean, snr, seed=SEED + 9000 + trial * 131 + snr)
                out_uniform, out_triangular = denoise_both_weightings(model, noisy)

                syms_u = receiver_frontend(out_uniform, h_rrc, sps=SPS)
                syms_t = receiver_frontend(out_triangular, h_rrc, sps=SPS)
                n = min(len(test_bits), len(syms_u), len(syms_t))

                e_u, _ = compute_ber_count(test_bits[:n], demod_fn(syms_u)[:n])
                e_t, _ = compute_ber_count(test_bits[:n], demod_fn(syms_t)[:n])
                rows.append(dict(modulation=modulation, snr_db_in=snr, trial=trial,
                                  errors_uniform=e_u, errors_triangular=e_t, n_bits=n))

            print(f"    trial {trial+1}/{MC_TRIALS} done")

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_cnn_overlap_weighting_intervention.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS: total errors, uniform vs. triangular weighting (pooled across trials)")
    print("=" * 90)
    summary = df.groupby(["modulation", "snr_db_in"]).agg(
        total_errors_uniform=("errors_uniform", "sum"),
        total_errors_triangular=("errors_triangular", "sum"),
        total_bits=("n_bits", "sum"),
    )
    summary["pct_change"] = 100 * (summary["total_errors_triangular"] - summary["total_errors_uniform"]) / summary["total_errors_uniform"].replace(0, np.nan)
    print(summary.to_string())
    print(f"\n[OK] Wrote {out_path}")
    return df


if __name__ == "__main__":
    main()
