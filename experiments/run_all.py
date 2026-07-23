#!/usr/bin/env python3
"""
run_all.py — Full Experiment Pipeline
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.cnn_autoencoder import prepare_dataset, train_autoencoder, denoise_signal
from src.hybrid_model import train_hybrid_cnn, denoise_hybrid
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import (
    demod_bpsk, demod_qpsk_fast,
    plot_ber_vs_snr, plot_snr_vs_snr, plot_constellation,
    plot_convergence, plot_training_loss,
    receiver_frontend, bits_to_symbols
)

# ===========================================================================
# Configuration
# ===========================================================================
# NOTE: rolled back to the simple baseline (no ISI/RRC, no RLS) so issues
# #1-#5 can be fixed and verified in the originally-scoped setting first.
# ISI/RRC pulse shaping and RLS are roadmap items #6/#7, to be re-added after.
NUM_SYMBOLS = 100_000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
SEED = 42

# Monte Carlo trials for the *test* evaluation only (training happens once).
# Each trial regenerates an independent bit sequence + noise realisation, so
# BER/SNR estimates are reported as mean +/- std across MC_TRIALS instead of
# a single point estimate. At 100,000 symbols x 10 trials, even a BER of 1e-4
# corresponds to ~100 observed bit errors -- enough to trust the estimate.
MC_TRIALS = 10

SPS = 1
MULTIPATH = None

WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000 * SPS  # 1000 symbols preamble

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

CONSTELLATION_SNR = 10  # dB

# ===========================================================================
# Helpers
# ===========================================================================

def _get_demod(modulation):
    if modulation == "BPSK":
        return demod_bpsk
    else:
        return demod_qpsk_fast

def _generate_signal(modulation, seed):
    if modulation == "BPSK":
        return generate_bpsk(NUM_SYMBOLS, seed=seed, sps=SPS)
    else:
        return generate_qpsk(NUM_SYMBOLS, seed=seed, sps=SPS)

# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    print("=" * 70)
    print(f"  SDR Denoising & Equalization - Experiment Pipeline "
          f"({NUM_SYMBOLS} symbols x {MC_TRIALS} MC trials)")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for modulation in ["BPSK", "QPSK"]:
        print(f"\n{'-' * 60}")
        print(f"  Modulation: {modulation}")
        print(f"{'-' * 60}")

        demod_fn = _get_demod(modulation)

        # 1. Generate training base signal (once; only the test side is
        #    Monte-Carlo'd -- CNN/Hybrid are trained once, not retrained per trial).
        train_tx, _, h_rrc = _generate_signal(modulation, seed=SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)

        # 2. Add AWGN at all SNR levels (training data)
        train_noisy_signals = {}
        for snr in SNR_LEVELS:
            train_noisy_signals[snr] = add_awgn(train_rx_clean, snr, seed=SEED + snr + 100)

        # 3. Train CNN autoencoder
        print("\n  [CNN] Preparing training data across all SNR levels...")
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

        print(f"  [CNN] Training set: {X_tr.shape[0]} windows, Val set: {X_va.shape[0]} windows")
        t0 = time.time()
        cnn_model, cnn_history = train_autoencoder(
            X_tr, Y_tr, X_va, Y_va,
            window_len=WINDOW_LEN, epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=1
        )
        cnn_train_time = time.time() - t0
        print(f"  [CNN] Training done in {cnn_train_time:.1f}s")
        plot_training_loss(cnn_history, str(FIGURES_DIR / f"cnn_training_loss_{modulation}.png"))

        # 4. Train Hybrid CNN (on LMS-pre-cleaned data)
        print("\n  [Hybrid] Training residual CNN on LMS-pre-cleaned data...")
        t0 = time.time()
        hybrid_model, hybrid_history = train_hybrid_cnn(
            clean_list, noisy_list,
            num_taps=TAPS, mu=LMS_MU,
            training_length=TRAINING_LENGTH, modulation=modulation,
            window_len=WINDOW_LEN, stride=STRIDE,
            epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=1
        )
        hybrid_train_time = time.time() - t0
        print(f"  [Hybrid] Training done in {hybrid_train_time:.1f}s")

        # 5. Monte Carlo evaluation across SNR levels
        methods = ["No Processing", "LMS", "NLMS", "CNN", "Hybrid"]
        # trial_metrics[method][snr] -> list of per-trial values
        ber_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        snr_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        runtime_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        # Raw (error_count, n_bits) totals across all trials -- needed to tell
        # "zero errors because it's actually perfect" apart from "zero errors
        # because we didn't test enough bits" (see rule-of-three below).
        ber_error_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
        ber_nbits_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
        convergence_lms, convergence_nlms = None, None
        constellation_data = {}

        print(f"\n  Evaluating across SNR levels: {SNR_LEVELS} ({MC_TRIALS} MC trials each)")
        for trial in range(MC_TRIALS):
            # Fresh, independent bit sequence + noise realisation per trial.
            test_tx, test_bits, _ = _generate_signal(modulation, seed=SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            test_ref_symbols = bits_to_symbols(test_bits, modulation)

            for snr in SNR_LEVELS:
                noisy = add_awgn(test_rx_clean, snr, seed=SEED + 9000 + trial * 131 + snr)

                def _record(method, snr_out, err_count, n_bits, runtime):
                    ber_trials[method][snr].append(err_count / n_bits)
                    snr_trials[method][snr].append(snr_out)
                    runtime_trials[method][snr].append(runtime)
                    ber_error_totals[method][snr] += err_count
                    ber_nbits_totals[method][snr] += n_bits

                # --- No Processing ---
                syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                snr_noproc = compute_snr_db(test_ref_symbols, syms_noproc)
                e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
                _record("No Processing", snr_noproc, e, n, 0.0)

                # --- LMS ---
                t0 = time.time()
                lms_filt = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                lms_out, lms_err = lms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                lms_syms = receiver_frontend(lms_out, h_rrc, sps=SPS)
                lms_time = time.time() - t0
                lms_snr = compute_snr_db(test_ref_symbols, lms_syms)
                e, n = compute_ber_count(test_bits, demod_fn(lms_syms))
                _record("LMS", lms_snr, e, n, lms_time)

                # --- NLMS ---
                t0 = time.time()
                nlms_filt = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                nlms_out, nlms_err = nlms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                nlms_syms = receiver_frontend(nlms_out, h_rrc, sps=SPS)
                nlms_time = time.time() - t0
                nlms_snr = compute_snr_db(test_ref_symbols, nlms_syms)
                e, n = compute_ber_count(test_bits, demod_fn(nlms_syms))
                _record("NLMS", nlms_snr, e, n, nlms_time)

                if trial == 0 and snr == CONSTELLATION_SNR:
                    convergence_lms, convergence_nlms = lms_err, nlms_err

                # --- CNN ---
                t0 = time.time()
                cnn_out = denoise_signal(cnn_model, noisy, WINDOW_LEN, STRIDE)
                cnn_syms = receiver_frontend(cnn_out, h_rrc, sps=SPS)
                cnn_time = time.time() - t0
                cnn_snr = compute_snr_db(test_ref_symbols, cnn_syms)
                e, n = compute_ber_count(test_bits, demod_fn(cnn_syms))
                _record("CNN", cnn_snr, e, n, cnn_time)

                # --- Hybrid ---
                t0 = time.time()
                hybrid_lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                hybrid_out = denoise_hybrid(hybrid_lms, hybrid_model, test_tx, noisy, WINDOW_LEN, STRIDE, training_length=TRAINING_LENGTH, modulation=modulation)
                hybrid_syms = receiver_frontend(hybrid_out, h_rrc, sps=SPS)
                hybrid_time = time.time() - t0
                hybrid_snr = compute_snr_db(test_ref_symbols, hybrid_syms)
                e, n = compute_ber_count(test_bits, demod_fn(hybrid_syms))
                _record("Hybrid", hybrid_snr, e, n, hybrid_time)

                if trial == 0 and snr == CONSTELLATION_SNR:
                    constellation_data = {
                        "No Processing": syms_noproc,
                        "LMS": lms_syms, "NLMS": nlms_syms,
                        "CNN": cnn_syms, "Hybrid": hybrid_syms,
                    }

            print(f"    trial {trial + 1}/{MC_TRIALS} done")

        # 6. Aggregate mean/std across MC trials
        ber_mean = {m: [] for m in methods}
        ber_std = {m: [] for m in methods}
        snr_mean = {m: [] for m in methods}
        snr_std = {m: [] for m in methods}

        for m in methods:
            for snr in SNR_LEVELS:
                bers = np.array(ber_trials[m][snr])
                snrs = np.array(snr_trials[m][snr])
                runtimes = np.array(runtime_trials[m][snr])
                ber_mean[m].append(bers.mean())
                ber_std[m].append(bers.std())
                snr_mean[m].append(snrs.mean())
                snr_std[m].append(snrs.std())

                # Rule-of-three: with zero errors observed across N bits, the
                # 95% upper confidence bound on the true BER is ~3/N (the
                # exact Clopper-Pearson bound for 0 successes). A bare "0.0"
                # is misleading -- it looks like a proven-perfect result when
                # it's really just "no errors in the bits we happened to test".
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                ber_ub = (3.0 / total_bits) if total_errors == 0 else None

                all_results.append({
                    "modulation": modulation, "snr_db_in": snr, "method": m,
                    "snr_db_out_mean": round(snrs.mean(), 3),
                    "snr_db_out_std": round(snrs.std(), 3),
                    "ber_mean": bers.mean(),
                    "ber_std": bers.std(),
                    "ber_total_errors": total_errors,
                    "ber_total_bits": total_bits,
                    "ber_95ci_upper_bound": ber_ub,
                    "runtime_sec_mean": round(runtimes.mean(), 4),
                    "n_trials": MC_TRIALS,
                })

            print(f"\n    [{m}] mean +/- std across {MC_TRIALS} trials:")
            for i, snr in enumerate(SNR_LEVELS):
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                if total_errors == 0:
                    ber_str = f"0 errors observed in {total_bits:,} bits (95% CI upper bound ~= {3.0/total_bits:.2e}, rule of three)"
                else:
                    ber_str = f"{ber_mean[m][i]:.3e}+/-{ber_std[m][i]:.1e} ({total_errors} errors / {total_bits:,} bits)"
                print(f"      SNR_in={snr:+3d}dB  SNR_out={snr_mean[m][i]:6.2f}+/-{snr_std[m][i]:.2f}dB  BER={ber_str}")

        # Generate plots (mean with error bars = std across MC trials)
        print(f"\n  Generating plots for {modulation}...")
        plot_ber_vs_snr(SNR_LEVELS, ber_mean, modulation, str(FIGURES_DIR / f"ber_vs_snr_{modulation}.png"), err_dict=ber_std)
        plot_snr_vs_snr(SNR_LEVELS, snr_mean, modulation, str(FIGURES_DIR / f"snr_vs_snr_{modulation}.png"), err_dict=snr_std)
        if constellation_data:
            plot_constellation(constellation_data, modulation, CONSTELLATION_SNR, str(FIGURES_DIR / f"constellation_{modulation}.png"))
        if convergence_lms is not None:
            plot_convergence(convergence_lms, convergence_nlms, str(FIGURES_DIR / f"convergence_{modulation}.png"))

    # Save results table
    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    print("\n[OK] Pipeline complete. Check results/ and report/ directories.")
    return df

if __name__ == "__main__":
    main()
