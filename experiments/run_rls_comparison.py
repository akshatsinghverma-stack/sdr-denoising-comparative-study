#!/usr/bin/env python3
"""
run_rls_comparison.py — Does RLS's faster convergence show a real edge over
LMS/NLMS on Case Study 2's ISI channel?
=============================================================================
report.md's Named, Scoped-Out Future Work explicitly deferred RLS from Case
Study 2 "to keep the number of simultaneously-changing variables between Case
Study 1 and 2 to exactly one (the channel)." `src/rls_filter.py` existed but
was unused. Before using it, its decision-directed default was hardened to
match LMSFilter/NLMSFilter's now-established safe default (see that module's
docstring) -- it previously continued adapting past the preamble by default,
gated only by a loose `|d - y| <= 2.0` check, the same broken-gate pattern
Section 3.2 documented and fixed for LMS/NLMS.

This script reproduces Case Study 2's exact channel, symbol count, MC trial
count, and SNR sweep (full scale, not a reduced diagnostic -- RLS is cheap
enough per-sample that this is unnecessary), adding RLS alongside
No-Processing/LMS/NLMS/MMSE (Genie). CNN/Hybrid are intentionally excluded --
they are unchanged from Case Study 2 and add no new information to the
specific question here (how does a *classical* least-squares approach
compare to gradient-based adaptive filtering on a correlated-input/ISI
channel), and retraining them would cost far more compute for no new signal.
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
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.rls_filter import RLSFilter
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend, plot_ber_vs_snr

# Identical to Case Study 2 (report.md Section 4.1) -- full scale, since RLS
# is cheap enough per-sample not to need a reduced-diagnostic scope.
NUM_SYMBOLS = 100_000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
SEED = 42
MC_TRIALS = 10

SPS = 4
MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
RLS_LAMBDA = 0.99
TRAINING_LENGTH = 1000 * SPS

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "rls_comparison"
TABLES_DIR = RESULTS_DIR / "tables"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def _calibrate_snr_gain_db(rx_clean, h_rrc):
    """Same empirical calibration as run_case2_isi.py -- see that script for
    the full rationale (matched-filter coherent-combining gain + this
    channel's own gain, measured via a 0dB probe rather than derived
    analytically)."""
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    noisy_probe = add_awgn(rx_clean, 0.0, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    return compute_snr_db(isi_ref, syms_probe) - 0.0


def main():
    print("=" * 78)
    print(f"  RLS Comparison: does faster least-squares convergence beat LMS/NLMS on ISI?")
    print(f"  ({NUM_SYMBOLS} symbols x {MC_TRIALS} MC trials, SPS={SPS})")
    print("=" * 78)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    methods = ["No Processing", "LMS", "NLMS", "RLS", "MMSE (Genie)"]

    for modulation in ["BPSK", "QPSK"]:
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")
        demod_fn = _get_demod(modulation)

        train_tx, _, h_rrc = _generate_signal(modulation, SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)
        mmse_channel_taps, mmse_peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)
        gain_db = _calibrate_snr_gain_db(train_rx_clean, h_rrc)
        print(f"  [Calibration] nominal-to-post-receiver SNR gain = {gain_db:+.3f} dB")

        ber_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        snr_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        runtime_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        ber_error_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
        ber_nbits_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}

        for trial in range(MC_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            isi_ref_symbols = receiver_frontend(test_rx_clean, h_rrc, sps=SPS)

            for snr in SNR_LEVELS:
                noisy = add_awgn(test_rx_clean, snr - gain_db, seed=SEED + 9000 + trial * 131 + snr)

                def _record(method, snr_out, err_count, n_bits, runtime):
                    ber_trials[method][snr].append(err_count / n_bits)
                    snr_trials[method][snr].append(snr_out)
                    runtime_trials[method][snr].append(runtime)
                    ber_error_totals[method][snr] += err_count
                    ber_nbits_totals[method][snr] += n_bits

                syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                snr_noproc = compute_snr_db(isi_ref_symbols, syms_noproc)
                e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
                _record("No Processing", snr_noproc, e, n, 0.0)

                t0 = time.time()
                noise_var = np.mean(np.abs(isi_ref_symbols - syms_noproc) ** 2)
                mmse_w, mmse_delay = design_mmse_equalizer(mmse_channel_taps, mmse_peak_idx, noise_var, num_taps=TAPS)
                mmse_out = apply_mmse_equalizer(syms_noproc, mmse_w, mmse_delay)
                mmse_time = time.time() - t0
                mmse_snr = compute_snr_db(isi_ref_symbols, mmse_out)
                e, n = compute_ber_count(test_bits, demod_fn(mmse_out))
                _record("MMSE (Genie)", mmse_snr, e, n, mmse_time)

                t0 = time.time()
                lms_filt = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                lms_out, _ = lms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                lms_syms = receiver_frontend(lms_out, h_rrc, sps=SPS)
                lms_time = time.time() - t0
                lms_snr = compute_snr_db(isi_ref_symbols, lms_syms)
                e, n = compute_ber_count(test_bits, demod_fn(lms_syms))
                _record("LMS", lms_snr, e, n, lms_time)

                t0 = time.time()
                nlms_filt = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                nlms_out, _ = nlms_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                nlms_syms = receiver_frontend(nlms_out, h_rrc, sps=SPS)
                nlms_time = time.time() - t0
                nlms_snr = compute_snr_db(isi_ref_symbols, nlms_syms)
                e, n = compute_ber_count(test_bits, demod_fn(nlms_syms))
                _record("NLMS", nlms_snr, e, n, nlms_time)

                t0 = time.time()
                rls_filt = RLSFilter(num_taps=TAPS, lam=RLS_LAMBDA)
                rls_out, _ = rls_filt.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                rls_syms = receiver_frontend(rls_out, h_rrc, sps=SPS)
                rls_time = time.time() - t0
                rls_snr = compute_snr_db(isi_ref_symbols, rls_syms)
                e, n = compute_ber_count(test_bits, demod_fn(rls_syms))
                _record("RLS", rls_snr, e, n, rls_time)

            print(f"    trial {trial + 1}/{MC_TRIALS} done")

        ber_mean = {m: [] for m in methods}
        ber_std = {m: [] for m in methods}
        for m in methods:
            for snr in SNR_LEVELS:
                bers = np.array(ber_trials[m][snr])
                snrs = np.array(snr_trials[m][snr])
                runtimes = np.array(runtime_trials[m][snr])
                ber_mean[m].append(bers.mean())
                ber_std[m].append(bers.std())
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                ber_ub = (3.0 / total_bits) if total_errors == 0 else None
                all_results.append({
                    "modulation": modulation, "snr_db_in": snr, "method": m,
                    "snr_db_out_vs_postisi_ref_mean": round(snrs.mean(), 3),
                    "snr_db_out_vs_postisi_ref_std": round(snrs.std(), 3),
                    "ber_mean": bers.mean(), "ber_std": bers.std(),
                    "ber_total_errors": total_errors, "ber_total_bits": total_bits,
                    "ber_95ci_upper_bound": ber_ub,
                    "runtime_sec_mean": round(runtimes.mean(), 4),
                    "n_trials": MC_TRIALS,
                })
            print(f"\n    [{m}] mean +/- std across {MC_TRIALS} trials:")
            for i, snr in enumerate(SNR_LEVELS):
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                if total_errors == 0:
                    ber_str = f"0 err / {total_bits:,} (95% UB ~= {3.0/total_bits:.2e})"
                else:
                    ber_str = f"{ber_mean[m][i]:.3e}+/-{ber_std[m][i]:.1e} ({total_errors} err / {total_bits:,})"
                print(f"      SNR_in={snr:+3d}dB  BER={ber_str}")

        plot_ber_vs_snr(SNR_LEVELS, ber_mean, modulation, str(FIGURES_DIR / f"ber_vs_snr_{modulation}.png"),
                         err_dict=ber_std, show_theoretical=False)

    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_rls_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Wrote {csv_path}")
    return df


if __name__ == "__main__":
    main()
