#!/usr/bin/env python3
"""
diagnose_cnn_high_snr_floor.py — Root-causing the loss-independent, high-SNR
CNN error floor found in the boundary-aware-loss follow-up.
=============================================================================
report/findings_boundary_aware_loss.md (report.md Section 3.7) found that at
10-20dB, plain-MSE and boundary-hinge CNNs produce IDENTICAL error counts for
both modulations -- something loss-independent caps BER there, flagged as an
open question with two named candidate explanations, neither verified: (a) a
window/overlap-add reconstruction artifact at window boundaries, or (b) a
small population of structurally hard-to-denoise samples (extreme noise
realizations no reasonable model could fix). This script distinguishes
between them directly rather than leaving both as unverified guesses.

Method
------
Retrains both CNN variants (same architecture/procedure as
run_case1_boundary_loss.py, reduced MC trial count for speed -- this is a
diagnostic, not a case-study reproduction) and, in addition to aggregate BER,
records the exact bit/symbol POSITION of every error for both models. This
lets three questions be answered directly instead of inferred from counts
alone:

  1. Do MSE and Hinge make errors at the *same* positions (not just the same
     COUNT)? If yes, that is direct, strong evidence of a shared cause.
  2. Do error positions cluster at a specific phase of (symbol_index mod
     STRIDE)? The overlap-add reconstruction averages exactly two windows for
     every interior sample; a per-window boundary artifact would predict a
     specific, non-uniform phase pattern (tested empirically below, not
     assumed analytically, since the two overlapping windows' relative
     boundary-distance trades off in a non-obvious way).
  3. Is the raw per-sample noise magnitude at error positions unusually large
     compared to typical samples at the same SNR? This tests hypothesis (b)
     directly, independent of (a).

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
from src.cnn_boundary_aware import train_boundary_aware_autoencoder
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend, bits_to_symbols

NUM_SYMBOLS = 25_000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]   # full sweep for training-data fidelity
ANALYSIS_SNRS = [10, 15, 20]               # where the floor was observed
SEED = 42
MC_TRIALS = 4

SPS = 1
MULTIPATH = None
WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64
HINGE_MARGIN = 0.3
HINGE_WEIGHT = 1.0

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODULATIONS = ["BPSK", "QPSK"]


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def _symbol_positions_of_errors(true_bits, pred_bits, modulation):
    """Bit-level error mask -> underlying SYMBOL indices (BPSK: 1 bit/symbol,
    QPSK: 2 bits/symbol, so a wrong bit in either position marks that symbol)."""
    wrong = true_bits != pred_bits
    if modulation == "BPSK":
        return np.where(wrong)[0]
    # QPSK: bits interleaved [b0_s0, b1_s0, b0_s1, b1_s1, ...]
    wrong_sym = wrong[0::2] | wrong[1::2]
    return np.where(wrong_sym)[0]


def main():
    print("=" * 78)
    print("  CNN High-SNR Error-Floor Diagnostic")
    print("=" * 78)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    error_position_rows = []
    summary_rows = []

    for modulation in MODULATIONS:
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")
        demod_fn = _get_demod(modulation)

        train_tx, _, h_rrc = _generate_signal(modulation, SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)
        train_noisy_signals = {snr: add_awgn(train_rx_clean, snr, seed=SEED + snr + 100)
                                for snr in SNR_LEVELS}

        clean_list = [train_tx] * len(SNR_LEVELS)
        noisy_list = [train_noisy_signals[snr] for snr in SNR_LEVELS]
        train_clean_split, val_clean_split, train_noisy_split, val_noisy_split = [], [], [], []
        for c, n in zip(clean_list, noisy_list):
            split = int(0.8 * len(c))
            train_clean_split.append(c[:split]); val_clean_split.append(c[split:])
            train_noisy_split.append(n[:split]); val_noisy_split.append(n[split:])

        X_tr, Y_tr = prepare_dataset(train_clean_split, train_noisy_split, WINDOW_LEN, STRIDE)
        X_va, Y_va = prepare_dataset(val_clean_split, val_noisy_split, WINDOW_LEN, STRIDE)
        idx = np.random.default_rng(0).permutation(len(X_tr))
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
        print(f"  Training set: {X_tr.shape[0]} windows, Val set: {X_va.shape[0]} windows")

        t0 = time.time()
        cnn_mse_model, _ = train_autoencoder(X_tr, Y_tr, X_va, Y_va, window_len=WINDOW_LEN,
                                              epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=2)
        print(f"  [CNN-MSE] trained in {time.time()-t0:.1f}s")

        t0 = time.time()
        cnn_hinge_model, _ = train_boundary_aware_autoencoder(
            X_tr, Y_tr, X_va, Y_va, window_len=WINDOW_LEN, epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
            loss_kind="hinge", margin=HINGE_MARGIN, hinge_weight=HINGE_WEIGHT, verbose=2)
        print(f"  [CNN-Hinge] trained in {time.time()-t0:.1f}s")

        for trial in range(MC_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            test_ref_symbols = bits_to_symbols(test_bits, modulation)

            for snr in SNR_LEVELS:
                noisy = add_awgn(test_rx_clean, snr, seed=SEED + 9000 + trial * 131 + snr)
                syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                bits_noproc = demod_fn(syms_noproc)

                cnn_mse_out = denoise_signal(cnn_mse_model, noisy, WINDOW_LEN, STRIDE)
                cnn_mse_syms = receiver_frontend(cnn_mse_out, h_rrc, sps=SPS)
                bits_mse = demod_fn(cnn_mse_syms)

                cnn_hinge_out = denoise_signal(cnn_hinge_model, noisy, WINDOW_LEN, STRIDE)
                cnn_hinge_syms = receiver_frontend(cnn_hinge_out, h_rrc, sps=SPS)
                bits_hinge = demod_fn(cnn_hinge_syms)

                n = min(len(test_bits), len(bits_mse), len(bits_hinge), len(bits_noproc))

                e_mse, _ = compute_ber_count(test_bits[:n], bits_mse[:n])
                e_hinge, _ = compute_ber_count(test_bits[:n], bits_hinge[:n])
                summary_rows.append(dict(modulation=modulation, snr_db_in=snr, trial=trial,
                                          errors_mse=e_mse, errors_hinge=e_hinge, n_bits=n))

                if snr not in ANALYSIS_SNRS:
                    continue

                pos_mse = set(_symbol_positions_of_errors(test_bits[:n], bits_mse[:n], modulation).tolist())
                pos_hinge = set(_symbol_positions_of_errors(test_bits[:n], bits_hinge[:n], modulation).tolist())
                pos_noproc = set(_symbol_positions_of_errors(test_bits[:n], bits_noproc[:n], modulation).tolist())
                shared = pos_mse & pos_hinge

                # Per-sample noise magnitude (deviation of the raw noisy symbol
                # from the true noiseless post-channel reference) at every
                # error position, to test the "extreme noise realization" hypothesis.
                noise_mag = np.abs(syms_noproc[:n] - test_ref_symbols[:n])
                typical_noise_mag = np.median(noise_mag)

                for pos in pos_mse | pos_hinge:
                    error_position_rows.append(dict(
                        modulation=modulation, snr_db_in=snr, trial=trial, symbol_index=int(pos),
                        phase_mod_stride=int(pos % STRIDE),
                        in_mse=pos in pos_mse, in_hinge=pos in pos_hinge, in_both=pos in shared,
                        also_wrong_in_noproc=pos in pos_noproc,
                        noise_mag_at_error=float(noise_mag[pos]) if pos < len(noise_mag) else np.nan,
                        typical_noise_mag=float(typical_noise_mag),
                    ))

            print(f"    trial {trial+1}/{MC_TRIALS} done")

    err_df = pd.DataFrame(error_position_rows)
    err_path = TABLES_DIR / "results_cnn_high_snr_floor_positions.csv"
    err_df.to_csv(err_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "results_cnn_high_snr_floor_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("  ANALYSIS")
    print("=" * 90)

    if len(err_df) == 0:
        print("  [WARN] No errors recorded at the analyzed high-SNR levels -- floor may not "
              "have reproduced at this reduced trial count.")
    else:
        n_total = len(err_df.drop_duplicates(subset=["modulation", "snr_db_in", "trial", "symbol_index"]))
        n_shared = err_df["in_both"].sum()
        print(f"  Total distinct error positions (MSE or Hinge): {n_total}")
        print(f"  Positions where BOTH MSE and Hinge are wrong (same position, not just same count): "
              f"{n_shared} ({100*n_shared/max(n_total,1):.1f}%)")

        n_also_noproc = err_df["also_wrong_in_noproc"].sum()
        print(f"  Of those, also wrong for No-Processing (raw hard-decision): "
              f"{n_also_noproc} / {len(err_df)} ({100*n_also_noproc/len(err_df):.1f}%)")

        ratio = err_df["noise_mag_at_error"] / err_df["typical_noise_mag"]
        print(f"  Noise magnitude at error positions vs. this trial's typical (median) magnitude: "
              f"mean ratio = {ratio.mean():.2f}x, median ratio = {ratio.median():.2f}x")

        print(f"\n  Phase (symbol_index mod {STRIDE}) distribution of error positions:")
        phase_counts = err_df["phase_mod_stride"].value_counts().sort_index()
        # Bucket into quartiles of the stride to keep the printout readable.
        q = STRIDE // 4
        buckets = {
            f"[0,{q})   (near window start)": phase_counts[(phase_counts.index >= 0) & (phase_counts.index < q)].sum(),
            f"[{q},{2*q}) ": phase_counts[(phase_counts.index >= q) & (phase_counts.index < 2*q)].sum(),
            f"[{2*q},{3*q}) ": phase_counts[(phase_counts.index >= 2*q) & (phase_counts.index < 3*q)].sum(),
            f"[{3*q},{STRIDE}) (near window end)": phase_counts[(phase_counts.index >= 3*q) & (phase_counts.index < STRIDE)].sum(),
        }
        for k, v in buckets.items():
            print(f"    {k}: {v} ({100*v/len(err_df):.1f}%)")

    print(f"\n[OK] Wrote {err_path}, {summary_path}")
    return err_df, summary_df


if __name__ == "__main__":
    main()
