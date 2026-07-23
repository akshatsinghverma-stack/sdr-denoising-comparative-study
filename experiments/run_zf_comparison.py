#!/usr/bin/env python3
"""
run_zf_comparison.py — Zero-Forcing (ZF) equalizer, a missing baseline found
by comparing this project against published equalization literature.
=============================================================================
A self-critique pass that searched comparable published work (an arXiv
comparative study of ZF/LMS/RLS adaptive equalizers) found this project had
a genie-aided MMSE equalizer but no Zero-Forcing equalizer -- ZF is one of
the two standard textbook linear-equalizer baselines (the other being MMSE),
and is arguably the more "canonical" one to compare against in classical
equalization literature. `src/mmse_equalizer.py` already had all the
machinery needed (measured channel taps, a Wiener-Hopf-style linear solve) --
ZF is exactly that same solve with the noise term dropped, so
`design_zf_equalizer` was added as a two-line wrapper (not a new module),
along with regression tests confirming it converges to MMSE as noise
vanishes and achieves near-perfect equalization at negligible noise.

This script reproduces Case Study 2's exact channel and adds ZF alongside
the existing methods (reusing No-Processing/MMSE-Genie's already-measured
channel taps and SNR calibration) to test the textbook prediction directly:
does ZF's noise-blind full channel inversion make it WORSE than MMSE at low
SNR (amplifying noise to force through a spectral dip) while matching MMSE
at high SNR (where there's little noise to amplify)?
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
from src.mmse_equalizer import (measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer,
                                  design_zf_equalizer, apply_zf_equalizer)
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

NUM_SYMBOLS = 100_000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
SEED = 42
MC_TRIALS = 10
SPS = 4
MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
TAPS = 16

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def _calibrate_snr_gain_db(rx_clean, h_rrc):
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    noisy_probe = add_awgn(rx_clean, 0.0, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    return compute_snr_db(isi_ref, syms_probe) - 0.0


def main():
    print("=" * 78)
    print("  ZF Comparison: does Zero-Forcing's noise-blind inversion cost more than it saves?")
    print("=" * 78)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    methods = ["No Processing", "MMSE (Genie)", "ZF (Genie)"]

    for modulation in ["BPSK", "QPSK"]:
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")
        demod_fn = _get_demod(modulation)

        train_tx, _, h_rrc = _generate_signal(modulation, SEED)
        train_rx_clean = add_multipath(train_tx, MULTIPATH)
        mmse_channel_taps, mmse_peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)
        gain_db = _calibrate_snr_gain_db(train_rx_clean, h_rrc)
        print(f"  [Calibration] nominal-to-post-receiver SNR gain = {gain_db:+.3f} dB")

        # ZF taps don't depend on noise -- design once per modulation.
        zf_w, zf_delay = design_zf_equalizer(mmse_channel_taps, mmse_peak_idx, num_taps=TAPS)

        ber_trials = {m: {snr: [] for snr in SNR_LEVELS} for m in methods}
        ber_error_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
        ber_nbits_totals = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}

        for trial in range(MC_TRIALS):
            test_tx, test_bits, _ = _generate_signal(modulation, SEED + 5000 + trial * 97)
            test_rx_clean = add_multipath(test_tx, MULTIPATH)
            isi_ref_symbols = receiver_frontend(test_rx_clean, h_rrc, sps=SPS)

            for snr in SNR_LEVELS:
                noisy = add_awgn(test_rx_clean, snr - gain_db, seed=SEED + 9000 + trial * 131 + snr)

                def _record(method, err_count, n_bits):
                    ber_trials[method][snr].append(err_count / n_bits)
                    ber_error_totals[method][snr] += err_count
                    ber_nbits_totals[method][snr] += n_bits

                syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
                _record("No Processing", e, n)

                noise_var = np.mean(np.abs(isi_ref_symbols - syms_noproc) ** 2)
                mmse_w, mmse_delay = design_mmse_equalizer(mmse_channel_taps, mmse_peak_idx, noise_var, num_taps=TAPS)
                mmse_out = apply_mmse_equalizer(syms_noproc, mmse_w, mmse_delay)
                e, n = compute_ber_count(test_bits, demod_fn(mmse_out))
                _record("MMSE (Genie)", e, n)

                zf_out = apply_zf_equalizer(syms_noproc, zf_w, zf_delay)
                e, n = compute_ber_count(test_bits, demod_fn(zf_out))
                _record("ZF (Genie)", e, n)

            print(f"    trial {trial + 1}/{MC_TRIALS} done")

        for m in methods:
            for snr in SNR_LEVELS:
                bers = np.array(ber_trials[m][snr])
                total_errors = ber_error_totals[m][snr]
                total_bits = ber_nbits_totals[m][snr]
                ber_ub = (3.0 / total_bits) if total_errors == 0 else None
                all_results.append({
                    "modulation": modulation, "snr_db_in": snr, "method": m,
                    "ber_mean": bers.mean(), "ber_std": bers.std(),
                    "ber_total_errors": total_errors, "ber_total_bits": total_bits,
                    "ber_95ci_upper_bound": ber_ub, "n_trials": MC_TRIALS,
                })
            print(f"\n    [{m}]:")
            for snr in SNR_LEVELS:
                e = ber_error_totals[m][snr]; n = ber_nbits_totals[m][snr]
                print(f"      SNR_in={snr:+3d}dB  BER={e}/{n} ({e/n:.2e})")

    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_zf_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Wrote {csv_path}")
    return df


if __name__ == "__main__":
    main()
