#!/usr/bin/env python3
"""
run_case1_boundary_loss.py — Does a boundary-aware CNN loss close the
"CNN loses to No-Processing" gap found in Case Study 1?
============================================================================
Case Study 1 (report.md Section 3.5) found that on a memoryless AWGN channel
(SPS=1, no pulse shaping, no ISI) every method tested -- LMS, NLMS, CNN,
Hybrid -- produces strictly worse BER than doing nothing at all, across every
SNR level and both modulations (56/56 comparisons), despite the CNN posting
large apparent SNR gains. The report's diagnosis for the CNN specifically:
it is trained with plain MSE, a smooth, symmetric, threshold-blind loss with
no explicit penalty for crossing the decision boundary -- so it can (and
empirically does) trade a small number of sign flips near zero for a larger
aggregate reduction in squared error elsewhere in the batch. That was an
observation, not a tested causal claim.

This script tests it directly: reproduce Case Study 1's memoryless-AWGN
setup (reduced symbol/trial count for speed -- this is a focused diagnostic,
not a full case-study reproduction), train BOTH a plain-MSE CNN (identical
architecture/procedure to the original Case Study 1 CNN) and a
boundary-aware CNN (same architecture, only the loss differs -- see
`src/cnn_boundary_aware.py`) on the same data, and compare both against
No-Processing across the SNR sweep.

In addition to BER, this script also computes a direct mechanistic
diagnostic: for every bit, did the CNN's output flip the hard decision
relative to the raw noisy sample, and if so, was that flip *harmful*
(broke a bit that raw hard-decision got right) or *beneficial* (fixed a bit
raw hard-decision got wrong)? This directly tests the "trades sign flips"
mechanism rather than just observing its aggregate BER consequence.
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

# ===========================================================================
# Configuration -- mirrors experiments/run_case1_no_isi.py (SPS=1, no pulse
# shaping / ISI, same WINDOW_LEN/STRIDE/CNN training approach, same SNR
# sweep and seed base), but with NUM_SYMBOLS and MC_TRIALS reduced for a
# focused diagnostic rather than a full case-study reproduction.
# ===========================================================================
NUM_SYMBOLS = 25_000          # vs 100,000 in the full Case Study 1
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
SEED = 42
MC_TRIALS = 5                 # vs 10 in the full Case Study 1

SPS = 1
MULTIPATH = None

WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64

# Boundary-hinge loss hyperparameters (see src/cnn_boundary_aware.py).
# margin ~ 0.3-0.5x typical constellation amplitude (BPSK: +-1, QPSK: each
# component +-0.707) is a reasonable "push away from the boundary" distance;
# hinge_weight=1.0 puts it on comparable footing with MSE once MSE has
# shrunk (high SNR), while MSE still dominates at low SNR where the noise
# floor itself is large -- i.e. the hinge term is designed to matter most
# exactly where plain MSE has the least intrinsic pressure to avoid a flip.
HINGE_MARGIN = 0.3
HINGE_WEIGHT = 1.0

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

MODULATIONS = ["BPSK", "QPSK"]


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    if modulation == "BPSK":
        return generate_bpsk(NUM_SYMBOLS, seed=seed, sps=SPS)
    return generate_qpsk(NUM_SYMBOLS, seed=seed, sps=SPS)


def main():
    print("=" * 78)
    print("  Boundary-Aware CNN Loss Diagnostic -- Case Study 1 (memoryless AWGN)")
    print(f"  ({NUM_SYMBOLS} symbols x {MC_TRIALS} MC trials, SPS={SPS})")
    print("=" * 78)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for modulation in MODULATIONS:
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")
        demod_fn = _get_demod(modulation)

        # --- 1. Training data: same construction as run_case1_no_isi.py ---
        train_tx, _, h_rrc = _generate_signal(modulation, seed=SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)

        train_noisy_signals = {}
        for snr in SNR_LEVELS:
            train_noisy_signals[snr] = add_awgn(train_rx_clean, snr, seed=SEED + snr + 100)

        clean_list = [train_tx] * len(SNR_LEVELS)
        noisy_list = [train_noisy_signals[snr] for snr in SNR_LEVELS]

        train_clean_split, val_clean_split = [], []
        train_noisy_split, val_noisy_split = [], []
        for c, n in zip(clean_list, noisy_list):
            split = int(0.8 * len(c))
            train_clean_split.append(c[:split])
            val_clean_split.append(c[split:])
            train_noisy_split.append(n[:split])
            val_noisy_split.append(n[split:])

        X_tr, Y_tr = prepare_dataset(train_clean_split, train_noisy_split, WINDOW_LEN, STRIDE)
        X_va, Y_va = prepare_dataset(val_clean_split, val_noisy_split, WINDOW_LEN, STRIDE)

        idx = np.random.default_rng(0).permutation(len(X_tr))
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]

        print(f"  Training set: {X_tr.shape[0]} windows, Val set: {X_va.shape[0]} windows")

        # --- 2. Train plain-MSE CNN (baseline, matches original Case 1 CNN) ---
        print("\n  [CNN-MSE] Training plain-MSE baseline...")
        t0 = time.time()
        cnn_mse_model, cnn_mse_history = train_autoencoder(
            X_tr, Y_tr, X_va, Y_va,
            window_len=WINDOW_LEN, epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=2
        )
        print(f"  [CNN-MSE] Trained in {time.time() - t0:.1f}s, "
              f"{len(cnn_mse_history.history['loss'])} epochs")

        # --- 3. Train boundary-aware (hinge) CNN ---
        print("\n  [CNN-Hinge] Training boundary-hinge-aware CNN "
              f"(margin={HINGE_MARGIN}, weight={HINGE_WEIGHT})...")
        t0 = time.time()
        cnn_hinge_model, cnn_hinge_history = train_boundary_aware_autoencoder(
            X_tr, Y_tr, X_va, Y_va,
            window_len=WINDOW_LEN, epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
            loss_kind="hinge", margin=HINGE_MARGIN, hinge_weight=HINGE_WEIGHT,
            verbose=2
        )
        print(f"  [CNN-Hinge] Trained in {time.time() - t0:.1f}s, "
              f"{len(cnn_hinge_history.history['loss'])} epochs")

        # --- 4. Monte Carlo evaluation across SNR levels ---
        methods = ["No Processing", "CNN (MSE)", "CNN (Boundary-Hinge)"]
        ber_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        snr_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        ber_error_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
        ber_nbits_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}

        # Mechanistic flip diagnostic: for each CNN variant, relative to the
        # raw hard-decision (No Processing) bits, how many bit-level flips
        # were harmful (broke a correct bit) vs beneficial (fixed a wrong
        # bit)? Pooled totals across all MC trials, per SNR.
        flip_harmful = {m: {snr: 0 for snr in SNR_LEVELS} for m in ["CNN (MSE)", "CNN (Boundary-Hinge)"]}
        flip_beneficial = {m: {snr: 0 for snr in SNR_LEVELS} for m in ["CNN (MSE)", "CNN (Boundary-Hinge)"]}

        print(f"\n  Evaluating across SNR levels: {SNR_LEVELS} ({MC_TRIALS} MC trials each)")
        for trial in range(MC_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, seed=SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            test_ref_symbols = bits_to_symbols(test_bits, modulation)

            for snr in SNR_LEVELS:
                noisy = add_awgn(test_rx_clean, snr, seed=SEED + 9000 + trial * 131 + snr)

                def _record(method, snr_out, err_count, n_bits):
                    ber_trials[method][snr].append(err_count / n_bits)
                    snr_trials[method][snr].append(snr_out)
                    ber_error_totals[method][snr] += err_count
                    ber_nbits_totals[method][snr] += n_bits

                # --- No Processing ---
                syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                snr_noproc = compute_snr_db(test_ref_symbols, syms_noproc)
                bits_noproc = demod_fn(syms_noproc)
                e, n = compute_ber_count(test_bits, bits_noproc)
                _record("No Processing", snr_noproc, e, n)
                correct_noproc = (bits_noproc[:n] == test_bits[:n])

                # --- CNN (MSE) ---
                cnn_mse_out = denoise_signal(cnn_mse_model, noisy, WINDOW_LEN, STRIDE)
                cnn_mse_syms = receiver_frontend(cnn_mse_out, h_rrc, sps=SPS)
                cnn_mse_snr = compute_snr_db(test_ref_symbols, cnn_mse_syms)
                bits_mse = demod_fn(cnn_mse_syms)
                e, n = compute_ber_count(test_bits, bits_mse)
                _record("CNN (MSE)", cnn_mse_snr, e, n)
                correct_mse = (bits_mse[:n] == test_bits[:n])
                flip_harmful["CNN (MSE)"][snr] += int(np.sum(correct_noproc[:n] & ~correct_mse))
                flip_beneficial["CNN (MSE)"][snr] += int(np.sum(~correct_noproc[:n] & correct_mse))

                # --- CNN (Boundary-Hinge) ---
                cnn_hinge_out = denoise_signal(cnn_hinge_model, noisy, WINDOW_LEN, STRIDE)
                cnn_hinge_syms = receiver_frontend(cnn_hinge_out, h_rrc, sps=SPS)
                cnn_hinge_snr = compute_snr_db(test_ref_symbols, cnn_hinge_syms)
                bits_hinge = demod_fn(cnn_hinge_syms)
                e, n = compute_ber_count(test_bits, bits_hinge)
                _record("CNN (Boundary-Hinge)", cnn_hinge_snr, e, n)
                correct_hinge = (bits_hinge[:n] == test_bits[:n])
                flip_harmful["CNN (Boundary-Hinge)"][snr] += int(np.sum(correct_noproc[:n] & ~correct_hinge))
                flip_beneficial["CNN (Boundary-Hinge)"][snr] += int(np.sum(~correct_noproc[:n] & correct_hinge))

            print(f"    trial {trial + 1}/{MC_TRIALS} done")

        # --- 5. Aggregate mean/std across MC trials, rule-of-three for zero-error cells ---
        for m in methods:
            for snr in SNR_LEVELS:
                bers = np.array(ber_trials[m][snr])
                snrs = np.array(snr_trials[m][snr])
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                ber_ub = (3.0 / total_bits) if total_errors == 0 else None

                row = {
                    "modulation": modulation, "snr_db_in": snr, "method": m,
                    "snr_db_out_mean": round(snrs.mean(), 3),
                    "snr_db_out_std": round(snrs.std(), 3),
                    "ber_mean": bers.mean(),
                    "ber_std": bers.std(),
                    "ber_total_errors": total_errors,
                    "ber_total_bits": total_bits,
                    "ber_95ci_upper_bound": ber_ub,
                    "n_trials": MC_TRIALS,
                }
                if m in flip_harmful:
                    row["flip_harmful_total"] = flip_harmful[m][snr]
                    row["flip_beneficial_total"] = flip_beneficial[m][snr]
                    row["flip_net"] = flip_beneficial[m][snr] - flip_harmful[m][snr]
                all_results.append(row)

            print(f"\n    [{m}] mean +/- std across {MC_TRIALS} trials:")
            for snr in SNR_LEVELS:
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                bers = np.array(ber_trials[m][snr])
                snrs = np.array(snr_trials[m][snr])
                if total_errors == 0:
                    ber_str = f"0 err / {total_bits:,} bits (95% UB ~= {3.0/total_bits:.2e})"
                else:
                    ber_str = f"{bers.mean():.3e}+/-{bers.std():.1e} ({total_errors} err / {total_bits:,} bits)"
                extra = ""
                if m in flip_harmful:
                    extra = (f"  [flips: harmful={flip_harmful[m][snr]}, "
                             f"beneficial={flip_beneficial[m][snr]}, "
                             f"net={flip_beneficial[m][snr]-flip_harmful[m][snr]}]")
                print(f"      SNR_in={snr:+3d}dB  SNR_out={snrs.mean():6.2f}+/-{snrs.std():.2f}dB  "
                      f"BER={ber_str}{extra}")

    # --- 6. Save results ---
    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_boundary_loss.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\n[OK] Results written to {csv_path}")
    return df


if __name__ == "__main__":
    main()
