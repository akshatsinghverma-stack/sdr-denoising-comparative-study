#!/usr/bin/env python3
"""
diagnose_training_variance.py — How much of this project's reported BER
variance comes from test-time randomness vs. the CNN's own training draw?
=============================================================================
report.md's Named, Scoped-Out Future Work: "Monte Carlo trials in both case
studies vary the *test* noise and bit sequence; the CNN/LMS/Hybrid models
themselves were each trained exactly once per modulation per case study. The
variance contributed by the training draw itself ... is currently unknown
and would need training repeated across several seeds to quantify."

This script answers that directly for the CNN (Case Study 1's memoryless-AWGN
setup, reduced scale for speed): train N_TRAINING_DRAWS independently-seeded
CNNs (different weight initialization AND training-data shuffle order per
draw), evaluate every draw on the SAME fixed set of MC_TEST_TRIALS test
signals, and decompose the resulting BER grid into:
  - within-draw (test-time) variance -- what this project's existing MC
    trial methodology already measures and reports as +/-std.
  - between-draw (training-seed) variance -- the variance of each draw's
    OWN mean BER, i.e. how much the training process itself contributes,
    holding test-time randomness fixed.
If between-draw variance is small relative to within-draw variance, this
project's practice of training once and reporting test-time MC variance
alone is justified. If it is comparable or larger, the reported +/-std
values understate the true uncertainty in this project's BER estimates.
No existing module is modified.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_multipath
from src.cnn_autoencoder import prepare_dataset, train_autoencoder, denoise_signal
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

NUM_SYMBOLS = 15_000
SNR_LEVELS_TRAIN = [-10, -5, 0, 5, 10, 15, 20]  # full sweep, for training-data fidelity
ANALYSIS_SNRS = [0, 15]                          # representative moderate / high-SNR-floor points
SEED = 42
N_TRAINING_DRAWS = 5
MC_TEST_TRIALS = 4

SPS = 1
MULTIPATH = None
WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODULATIONS = ["BPSK", "QPSK"]


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def main():
    print("=" * 78)
    print("  Training-Variance Diagnostic: test-time MC variance vs. training-draw variance")
    print("=" * 78)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for modulation in MODULATIONS:
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")
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

        # Fixed test signals -- the SAME across every training draw, so any
        # BER difference between draws is attributable only to the training
        # process (weight init + shuffle order), not to different test data.
        test_signals = []
        for trial in range(MC_TEST_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            test_signals.append((test_tx, test_bits, test_rx_clean))

        for draw in range(N_TRAINING_DRAWS):
            # Vary BOTH the training-data shuffle order and TF's own weight-init
            # randomness per draw -- the two things "trained exactly once"
            # left unquantified.
            idx = np.random.default_rng(draw).permutation(len(X_tr))
            X_tr_shuffled, Y_tr_shuffled = X_tr[idx], Y_tr[idx]

            import tensorflow as tf
            tf.keras.utils.set_random_seed(1000 + draw)

            t0 = time.time()
            model, _ = train_autoencoder(X_tr_shuffled, Y_tr_shuffled, X_va, Y_va,
                                          window_len=WINDOW_LEN, epochs=CNN_EPOCHS,
                                          batch_size=CNN_BATCH, verbose=0)
            print(f"  [draw {draw}] trained in {time.time()-t0:.1f}s")

            for snr in ANALYSIS_SNRS:
                for trial, (test_tx, test_bits, test_rx_clean) in enumerate(test_signals):
                    noisy = add_awgn(test_rx_clean, snr, seed=SEED + 9000 + trial * 131 + snr)
                    cnn_out = denoise_signal(model, noisy, WINDOW_LEN, STRIDE)
                    cnn_syms = receiver_frontend(cnn_out, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(cnn_syms))
                    rows.append(dict(modulation=modulation, snr_db_in=snr, training_draw=draw,
                                      test_trial=trial, errors=e, n_bits=n, ber=e / n))

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_training_variance.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("  ANALYSIS: within-draw (test-time) vs. between-draw (training-seed) variance")
    print("=" * 90)
    summary_rows = []
    for modulation in MODULATIONS:
        for snr in ANALYSIS_SNRS:
            sub = df[(df.modulation == modulation) & (df.snr_db_in == snr)]
            per_draw_mean = sub.groupby("training_draw")["ber"].mean()
            within_draw_std = sub.groupby("training_draw")["ber"].std().mean()  # avg within-draw std
            between_draw_std = per_draw_mean.std()  # std of the per-draw means
            grand_mean = sub["ber"].mean()
            print(f"\n  {modulation} SNR={snr}dB:")
            print(f"    Per-draw mean BER: {per_draw_mean.values}")
            print(f"    Within-draw (test-time) std, averaged across draws: {within_draw_std:.3e}")
            print(f"    Between-draw (training-seed) std of draw-means:    {between_draw_std:.3e}")
            ratio = between_draw_std / within_draw_std if within_draw_std > 0 else np.nan
            print(f"    Ratio (between/within): {ratio:.2f}x")
            summary_rows.append(dict(modulation=modulation, snr_db_in=snr, grand_mean_ber=grand_mean,
                                      within_draw_std=within_draw_std, between_draw_std=between_draw_std,
                                      ratio_between_over_within=ratio))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "results_training_variance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Wrote {out_path}, {summary_path}")
    return df, summary_df


if __name__ == "__main__":
    main()
