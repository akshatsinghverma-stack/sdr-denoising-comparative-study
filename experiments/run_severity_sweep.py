#!/usr/bin/env python3
"""
run_severity_sweep.py — ISI Severity Sweep (diagnostic follow-up)
===================================================================
Case Study 1 (no ISI) and Case Study 2 (one specific multipath channel)
gave opposite answers to "does equalization help BER" -- but that's only
two data points. This script scales the multipath channel's secondary/
tertiary tap magnitudes by a severity factor in [0, 1] (0 = no ISI, exactly
Case Study 1's channel; 1 = exactly Case Study 2's channel) and re-runs the
comparison at each severity level, to locate *where* the crossover from
"equalization hurts" to "equalization helps" actually happens, rather than
just observing that it happens somewhere between the two endpoints already
tested.

Scope note: this is a diagnostic sweep, not a third full case study -- it
intentionally uses fewer symbols, fewer Monte Carlo trials, and fewer SNR
points than Case Study 1/2 to keep runtime bounded across 5 severity levels.
Treat trends here as indicative, not as precise as the two main case studies.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.cnn_autoencoder import prepare_dataset, train_autoencoder, denoise_signal
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

# ===========================================================================
# Configuration (reduced scale -- see module docstring)
# ===========================================================================
NUM_SYMBOLS = 30_000
SNR_LEVELS = [-10, 0, 10, 20]
SEED = 42
MC_TRIALS = 5

SPS = 4
BASE_MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = no ISI, 1 = full Case Study 2 channel

WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 30  # fewer than the main studies' 50; this is a fast diagnostic
CNN_BATCH = 64
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000 * SPS

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "severity_sweep"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def _multipath_for_severity(severity):
    return [1.0, severity * BASE_MULTIPATH[1], severity * BASE_MULTIPATH[2]]


def _calibrate_snr_gain_db(rx_clean, h_rrc):
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    noisy_probe = add_awgn(rx_clean, 0.0, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    return compute_snr_db(isi_ref, syms_probe) - 0.0


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
        print(f"\n{'=' * 60}\n  Modulation: {modulation}\n{'=' * 60}")

        for severity in SEVERITIES:
            t_severity = time.time()
            multipath = _multipath_for_severity(severity)
            multipath_str = [f"{h.real:.3f}+{h.imag:.3f}j" for h in multipath]
            print(f"\n  --- severity={severity:.2f}  multipath={multipath_str} ---")

            train_tx, _, h_rrc = _generate_signal(modulation, seed=SEED)
            train_rx_clean = add_multipath(train_tx, multipath)
            gain_db = _calibrate_snr_gain_db(train_rx_clean, h_rrc)

            train_noisy = {snr: add_awgn(train_rx_clean, snr - gain_db, seed=SEED + snr + 100)
                           for snr in SNR_LEVELS}
            clean_list = [train_tx] * len(SNR_LEVELS)
            noisy_list = [train_noisy[snr] for snr in SNR_LEVELS]
            tr_c, va_c, tr_n, va_n = [], [], [], []
            for c, n in zip(clean_list, noisy_list):
                s = int(0.8 * len(c))
                tr_c.append(c[:s]); va_c.append(c[s:])
                tr_n.append(n[:s]); va_n.append(n[s:])
            X_tr, Y_tr = prepare_dataset(tr_c, tr_n, WINDOW_LEN, STRIDE)
            X_va, Y_va = prepare_dataset(va_c, va_n, WINDOW_LEN, STRIDE)
            idx = np.random.default_rng(0).permutation(len(X_tr))
            X_tr, Y_tr = X_tr[idx], Y_tr[idx]
            cnn_model, _ = train_autoencoder(X_tr, Y_tr, X_va, Y_va, window_len=WINDOW_LEN,
                                              epochs=CNN_EPOCHS, batch_size=CNN_BATCH, verbose=0)

            mmse_taps, mmse_peak = measure_channel_taps(h_rrc, multipath, sps=SPS, num_taps=21)

            methods = ["No Processing", "LMS", "NLMS", "CNN", "MMSE (Genie)"]
            ber_err = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
            ber_bits = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}

            for trial in range(MC_TRIALS):
                test_tx, test_bits, _ = _generate_signal(modulation, seed=SEED + 5000 + trial * 97)
                test_rx_clean = add_multipath(test_tx, multipath)
                isi_ref = receiver_frontend(test_rx_clean, h_rrc, sps=SPS)

                for snr in SNR_LEVELS:
                    noisy = add_awgn(test_rx_clean, snr - gain_db, seed=SEED + 9000 + trial * 131 + snr)

                    syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
                    ber_err["No Processing"][snr] += e; ber_bits["No Processing"][snr] += n

                    noise_var = np.mean(np.abs(isi_ref - syms_noproc) ** 2)
                    w, delay = design_mmse_equalizer(mmse_taps, mmse_peak, noise_var, num_taps=TAPS)
                    mmse_out = apply_mmse_equalizer(syms_noproc, w, delay)
                    e, n = compute_ber_count(test_bits, demod_fn(mmse_out))
                    ber_err["MMSE (Genie)"][snr] += e; ber_bits["MMSE (Genie)"][snr] += n

                    lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                    lms_out, _ = lms.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                    lms_syms = receiver_frontend(lms_out, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(lms_syms))
                    ber_err["LMS"][snr] += e; ber_bits["LMS"][snr] += n

                    nlms = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                    nlms_out, _ = nlms.denoise(test_tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
                    nlms_syms = receiver_frontend(nlms_out, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(nlms_syms))
                    ber_err["NLMS"][snr] += e; ber_bits["NLMS"][snr] += n

                    cnn_out = denoise_signal(cnn_model, noisy, WINDOW_LEN, STRIDE)
                    cnn_syms = receiver_frontend(cnn_out, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(cnn_syms))
                    ber_err["CNN"][snr] += e; ber_bits["CNN"][snr] += n

            for m in methods:
                for snr in SNR_LEVELS:
                    e, n = ber_err[m][snr], ber_bits[m][snr]
                    all_results.append({
                        "modulation": modulation, "severity": severity, "snr_db_in": snr,
                        "method": m, "ber": e / n, "errors": e, "bits": n,
                    })
            print(f"    done in {time.time() - t_severity:.1f}s")

    df = pd.DataFrame(all_results)
    df.to_csv(TABLES_DIR / "results_severity_sweep.csv", index=False)

    # Summary plot: BER improvement ratio (No-Proc / Method) vs severity,
    # at 10dB and 20dB where Case Study 2 showed the largest effect.
    for modulation in ["BPSK", "QPSK"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, snr in zip(axes, [10, 20]):
            sub = df[(df.modulation == modulation) & (df.snr_db_in == snr)]
            noproc = sub[sub.method == "No Processing"].set_index("severity")
            for m in ["LMS", "NLMS", "CNN", "MMSE (Genie)"]:
                m_sub = sub[sub.method == m].set_index("severity")
                ratios = []
                for s in SEVERITIES:
                    base_ber = noproc.loc[s, "errors"] / noproc.loc[s, "bits"]
                    m_ber = m_sub.loc[s, "errors"] / m_sub.loc[s, "bits"]
                    ratios.append((base_ber / m_ber) if m_ber > 0 else np.nan)
                ax.plot(SEVERITIES, ratios, marker="o", label=m, linewidth=2)
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1.5, label="No improvement")
            ax.set_yscale("log")
            ax.set_xlabel("ISI severity (0=none, 1=full Case Study 2 channel)")
            ax.set_ylabel("BER improvement ratio (NoProc/Method)")
            ax.set_title(f"{modulation} @ {snr}dB")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"severity_crossover_{modulation}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {FIGURES_DIR / f'severity_crossover_{modulation}.png'}")

    print("\n[OK] Severity sweep complete.")
    return df


if __name__ == "__main__":
    main()
