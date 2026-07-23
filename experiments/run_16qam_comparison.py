#!/usr/bin/env python3
"""
run_16qam_comparison.py — Higher-Order Modulation Follow-up to Case Study 2
=============================================================================
Case Study 2 (run_case2_isi.py) found that BPSK and QPSK experience the
*same* RRC-pulse-shaped + static-multipath ISI channel very differently:
BPSK's single binary decision boundary (180 degrees of margin) is forgiving
enough that classical equalization (LMS/NLMS) is often *worse* than doing
nothing, while QPSK's four-quadrant boundary (90 degrees of margin) is much
more sensitive to the same channel's phase/amplitude smearing, so
equalization wins by 1-3 orders of magnitude (report.md Section 4.6).

Hypothesis under test here: "decision-boundary crowding drives how much
equalization matters" -- so a higher-order modulation with an even more
crowded constellation (16-QAM: 16 points on a square grid, minimum distance
between neighbours far smaller than QPSK's quadrant boundary for the same
average power) should show an even larger equalization win than QPSK did.

This is a REDUCED-SCALE run (not a third full case study): fewer symbols,
fewer MC trials, and it mirrors run_case2_isi.py's exact channel, SNR
calibration step (critical -- this fixes a real bug documented in
report.md Section 4.2, and skipping it would silently reintroduce it), and
metric conventions (post-ISI reference SNR, rule-of-three for exact-zero
BER cells).

SNR range: 16-QAM needs substantially higher SNR than BPSK/QPSK to reach a
usable BER at all (minimum distance between neighbouring constellation
points is ~4.47x smaller than BPSK's for the same average power, since
d_min = 2/sqrt(10) here vs. BPSK's d_min = 2), so the BPSK/QPSK sweep
(-10..20dB) would show 16-QAM stuck near the 4/16-ary symbol-error floor
(~0.3-0.4 BER) for most of its range and never reach the interesting
regime. [0, 10, 20, 25, 30] dB is used instead: 0dB confirms 16-QAM is
still essentially unusable (sanity floor), 10dB is transitional, and
20-30dB is where 16-QAM BER actually falls through the range where
equalization can plausibly matter -- deliberately shifted/widened relative
to the BPSK/QPSK sweep rather than reusing it verbatim.
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_16qam
from src.channel import add_awgn, add_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.cnn_autoencoder import prepare_dataset, train_autoencoder, denoise_signal
from src.hybrid_model import train_hybrid_cnn, denoise_hybrid
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import (
    demod_16qam_fast,
    plot_ber_vs_snr, plot_snr_vs_snr, plot_constellation,
    plot_convergence, plot_training_loss,
    receiver_frontend
)

# ===========================================================================
# Configuration -- reduced scale, mirrors run_case2_isi.py's channel exactly
# ===========================================================================
MODULATION = "16QAM"
NUM_SYMBOLS = 25_000            # reduced from Case Study 2's 100,000
SNR_LEVELS = [0, 10, 20, 25, 30]  # shifted/widened -- see module docstring
SEED = 42
MC_TRIALS = 5                    # reduced from Case Study 2's 10

SPS = 4
MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]  # identical static 3-tap channel
                                              # used throughout Case Study 2

WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 50
CNN_BATCH = 64
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000 * SPS  # 1000-symbol preamble (same as Case Study 2)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "higher_order_modulation"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

CONSTELLATION_SNR = 20  # dB

# ===========================================================================
# Helpers
# ===========================================================================

def _generate_signal(seed):
    return generate_16qam(NUM_SYMBOLS, seed=seed, sps=SPS)


def _calibrate_snr_gain_db(rx_clean, h_rrc):
    """Empirical SNR-gain calibration -- IDENTICAL approach to
    run_case2_isi.py's _calibrate_snr_gain_db. This is not a cosmetic step:
    report.md Section 4.2 documents that assuming a pure 10*log10(sps)
    correction (rather than measuring the real, channel-specific gain
    empirically) leaves a residual ~0.25dB error once ISI is present, and
    that without any calibration at all, No-Processing's own SNR-out was
    off by +5.77dB from its nominal input SNR. The fix here is the same:
    inject a noise probe at a known SNR, measure what SNR is actually
    observed downstream of the matched filter, and use the gap as the
    calibration subtracted from every nominal SNR value in this sweep.
    """
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    probe_db = 0.0
    noisy_probe = add_awgn(rx_clean, probe_db, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    snr_out_probe = compute_snr_db(isi_ref, syms_probe)
    return snr_out_probe - probe_db


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    print("=" * 70)
    print(f"  16-QAM Higher-Order-Modulation Follow-up "
          f"({NUM_SYMBOLS} symbols x {MC_TRIALS} MC trials, SPS={SPS})")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    demod_fn = demod_16qam_fast

    train_tx, _, h_rrc = _generate_signal(seed=SEED)
    train_rx_clean = add_multipath(train_tx, MULTIPATH)

    # Genie-aided linear MMSE equalizer (same construction as Case Study 2;
    # see src/mmse_equalizer.py for the symbol-spaced-vs-fractionally-spaced
    # caveat -- this is a reference ceiling, not a directly competing method).
    mmse_channel_taps, mmse_peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)

    gain_db = _calibrate_snr_gain_db(train_rx_clean, h_rrc)
    print(f"  [Calibration] nominal-SNR-to-post-receiver-SNR gain = {gain_db:+.3f} dB "
          f"(subtracted from every add_awgn call so nominal SNR matches measured No-Processing SNR_out)")

    train_noisy_signals = {}
    for snr in SNR_LEVELS:
        train_noisy_signals[snr] = add_awgn(train_rx_clean, snr - gain_db, seed=SEED + snr + 100)

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
    plot_training_loss(cnn_history, str(FIGURES_DIR / f"cnn_training_loss_{MODULATION}.png"))

    print("\n  [Hybrid] Training residual CNN on LMS-pre-cleaned data...")
    t0 = time.time()
    hybrid_model, hybrid_history = train_hybrid_cnn(
        clean_list, noisy_list,
        num_taps=TAPS, mu=LMS_MU,
        training_length=TRAINING_LENGTH, modulation=MODULATION,
        window_len=WINDOW_LEN, stride=STRIDE,
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=1
    )
    hybrid_train_time = time.time() - t0
    print(f"  [Hybrid] Training done in {hybrid_train_time:.1f}s")

    methods = ["No Processing", "LMS", "NLMS", "CNN", "Hybrid", "MMSE (Genie)"]
    ber_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
    snr_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
    runtime_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
    ber_error_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
    ber_nbits_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
    convergence_lms, convergence_nlms = None, None
    constellation_data = {}

    print(f"\n  Evaluating across SNR levels: {SNR_LEVELS} ({MC_TRIALS} MC trials each)")
    for trial in range(MC_TRIALS):
        test_tx, test_bits, _ = _generate_signal(seed=SEED + 5000 + trial * 97)
        test_rx_clean = add_multipath(test_tx, MULTIPATH)

        # SNR reference: post-ISI, pre-noise (identical convention to Case
        # Study 2 -- see run_case2_isi.py comment for why the pre-ISI ideal
        # constellation would be the wrong reference here).
        isi_ref_symbols = receiver_frontend(test_rx_clean, h_rrc, sps=SPS)

        for snr in SNR_LEVELS:
            noisy = add_awgn(test_rx_clean, snr - gain_db, seed=SEED + 9000 + trial * 131 + snr)

            def _record(method, snr_out, err_count, n_bits, runtime):
                ber_trials[method][snr].append(err_count / n_bits)
                snr_trials[method][snr].append(snr_out)
                runtime_trials[method][snr].append(runtime)
                ber_error_totals[method][snr] += err_count
                ber_nbits_totals[method][snr] += n_bits

            # --- No Processing ---
            syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
            snr_noproc = compute_snr_db(isi_ref_symbols, syms_noproc)
            e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
            _record("No Processing", snr_noproc, e, n, 0.0)

            # --- MMSE (Genie) ---
            t0 = time.time()
            noise_var = np.mean(np.abs(isi_ref_symbols - syms_noproc) ** 2)
            mmse_w, mmse_delay = design_mmse_equalizer(
                mmse_channel_taps, mmse_peak_idx, noise_var, num_taps=TAPS
            )
            mmse_out = apply_mmse_equalizer(syms_noproc, mmse_w, mmse_delay)
            mmse_time = time.time() - t0
            mmse_snr = compute_snr_db(isi_ref_symbols, mmse_out)
            e, n = compute_ber_count(test_bits, demod_fn(mmse_out))
            _record("MMSE (Genie)", mmse_snr, e, n, mmse_time)

            # --- LMS ---
            t0 = time.time()
            lms_filt = LMSFilter(num_taps=TAPS, mu=LMS_MU)
            lms_out, lms_err = lms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=MODULATION)
            lms_syms = receiver_frontend(lms_out, h_rrc, sps=SPS)
            lms_time = time.time() - t0
            lms_snr = compute_snr_db(isi_ref_symbols, lms_syms)
            e, n = compute_ber_count(test_bits, demod_fn(lms_syms))
            _record("LMS", lms_snr, e, n, lms_time)

            # --- NLMS ---
            t0 = time.time()
            nlms_filt = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
            nlms_out, nlms_err = nlms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=MODULATION)
            nlms_syms = receiver_frontend(nlms_out, h_rrc, sps=SPS)
            nlms_time = time.time() - t0
            nlms_snr = compute_snr_db(isi_ref_symbols, nlms_syms)
            e, n = compute_ber_count(test_bits, demod_fn(nlms_syms))
            _record("NLMS", nlms_snr, e, n, nlms_time)

            if trial == 0 and snr == CONSTELLATION_SNR:
                convergence_lms, convergence_nlms = lms_err, nlms_err

            # --- CNN ---
            t0 = time.time()
            cnn_out = denoise_signal(cnn_model, noisy, WINDOW_LEN, STRIDE)
            cnn_syms = receiver_frontend(cnn_out, h_rrc, sps=SPS)
            cnn_time = time.time() - t0
            cnn_snr = compute_snr_db(isi_ref_symbols, cnn_syms)
            e, n = compute_ber_count(test_bits, demod_fn(cnn_syms))
            _record("CNN", cnn_snr, e, n, cnn_time)

            # --- Hybrid ---
            t0 = time.time()
            hybrid_lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
            hybrid_out = denoise_hybrid(hybrid_lms, hybrid_model, test_tx, noisy, WINDOW_LEN, STRIDE, training_length=TRAINING_LENGTH, modulation=MODULATION)
            hybrid_syms = receiver_frontend(hybrid_out, h_rrc, sps=SPS)
            hybrid_time = time.time() - t0
            hybrid_snr = compute_snr_db(isi_ref_symbols, hybrid_syms)
            e, n = compute_ber_count(test_bits, demod_fn(hybrid_syms))
            _record("Hybrid", hybrid_snr, e, n, hybrid_time)

            if trial == 0 and snr == CONSTELLATION_SNR:
                constellation_data = {
                    "No Processing": syms_noproc,
                    "LMS": lms_syms, "NLMS": nlms_syms,
                    "CNN": cnn_syms, "Hybrid": hybrid_syms,
                    "MMSE (Genie)": mmse_out,
                }

        print(f"    trial {trial + 1}/{MC_TRIALS} done")

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

            total_errors = ber_error_totals[m][snr]
            total_bits = ber_nbits_totals[m][snr]
            ber_ub = (3.0 / total_bits) if total_errors == 0 else None

            all_results.append({
                "modulation": MODULATION, "snr_db_in": snr, "method": m,
                "snr_db_out_vs_postisi_ref_mean": round(snrs.mean(), 3),
                "snr_db_out_vs_postisi_ref_std": round(snrs.std(), 3),
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

    print(f"\n  Generating plots for {MODULATION}...")
    plot_ber_vs_snr(SNR_LEVELS, ber_mean, MODULATION, str(FIGURES_DIR / f"ber_vs_snr_{MODULATION}.png"), err_dict=ber_std, show_theoretical=False)
    plot_snr_vs_snr(SNR_LEVELS, snr_mean, MODULATION, str(FIGURES_DIR / f"snr_vs_snr_{MODULATION}.png"), err_dict=snr_std)
    if constellation_data:
        plot_constellation(constellation_data, MODULATION, CONSTELLATION_SNR, str(FIGURES_DIR / f"constellation_{MODULATION}.png"))
    if convergence_lms is not None:
        plot_convergence(convergence_lms, convergence_nlms, str(FIGURES_DIR / f"convergence_{MODULATION}.png"))

    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_16qam.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY — 16-QAM Higher-Order-Modulation Follow-up")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    print("\n[OK] 16-QAM comparison pipeline complete.")
    return df

if __name__ == "__main__":
    main()
