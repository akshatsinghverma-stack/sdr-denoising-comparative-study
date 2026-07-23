#!/usr/bin/env python3
"""
test_cnn_channel_generalization.py — Does the CNN actually generalize across
channels, or has it just memorized the one it was trained on?
=============================================================================
An outside-reviewer critique pass flagged a real, previously-undisclosed gap:
every CNN result in this project (Case Study 2, the severity sweep, Case
Study 4) trains and evaluates the CNN on the *exact same* known multipath
impulse response (`MULTIPATH = [1.0, 0.4+0.3j, -0.1+0.1j]` in
run_case2_isi.py, reused verbatim for training and every test trial). LMS,
NLMS, and the genie MMSE equalizer all either re-estimate the channel from
the preamble (LMS/NLMS) or are handed the true channel per-trial (MMSE) --
none of them get a free pass on "the same channel every time." Only the CNN
implicitly assumes train-time and test-time channels match, and that
assumption was never named as a limitation, let alone tested. If the CNN's
reported wins (up to 610x for 16-QAM, Section 7) partly reflect memorizing
the inverse of one specific channel rather than learning to equalize ISI in
general, that would matter directly for this project's own "deploy a CNN
whenever real multipath exists" recommendation (Section 9.5).

This script trains one CNN once on the Case Study 2 channel (the "matched"
condition, reproducing that channel exactly) and then evaluates it, with NO
retraining, on five held-out channels that differ in delay spread, tap
count, and tap phase/magnitude from the training channel -- not just a
severity-scaled version of the same three taps (that's what
run_severity_sweep.py already does, and it always trains and tests at
matched severity, so it never tests this gap either). LMS, NLMS, and MMSE
are run on the same held-out channels as a like-for-like reference: they
adapt/measure per-trial regardless of which channel is presented, so their
BER change from held-out channels reflects genuine channel difficulty, not
a train/test mismatch penalty specific to one method.

Scope note: this is a diagnostic, not a full case study re-run -- reduced
symbol count/trial count/epochs to keep runtime bounded across 6 channel
conditions x 2 modulations. Treat the resulting numbers as indicative of the
generalization gap's existence and rough size, not as precise as the main
case studies.
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
from src.cnn_autoencoder import prepare_dataset, train_autoencoder, denoise_signal
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

# ===========================================================================
# Configuration (reduced scale -- see module docstring)
# ===========================================================================
NUM_SYMBOLS = 30_000
SNR_LEVELS = [0, 10, 20]
SEED = 42
MC_TRIALS = 5

SPS = 4
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000 * SPS
WINDOW_LEN = 128
STRIDE = 64
CNN_EPOCHS = 30

# The channel the CNN is trained on -- identical to run_case2_isi.py's
# MULTIPATH, so this diagnostic's "matched" condition reproduces that
# project's actual training condition exactly, not an approximation of it.
TRAIN_CHANNEL = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]

# Held-out channels: deliberately vary delay spread (tap count) and tap
# phase/magnitude, not just an overall severity scaling of TRAIN_CHANNEL's
# own taps (run_severity_sweep.py already covers that axis). Total ISI
# energy (sum of |h[k]|^2 for k>0) is kept roughly comparable to
# TRAIN_CHANNEL's own ~0.26 so "held-out is just harder/easier overall"
# doesn't confound "held-out is a different shape."
HELD_OUT_CHANNELS = {
    "matched (train==test)": TRAIN_CHANNEL,
    "rotated phase (2-tap)": [1.0, -0.3 + 0.35j],
    "longer delay spread (4-tap)": [1.0, 0.28 - 0.18j, 0.18 + 0.12j, -0.14 + 0.05j],
    "sign-flipped taps (3-tap)": [1.0, -0.4 - 0.3j, 0.1 - 0.1j],
    "different phase, same taps count": [1.0, -0.15 + 0.42j, 0.22 - 0.15j],
    "single dominant reflection (2-tap)": [1.0, 0.5 - 0.15j],
}

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
    print("  CNN Channel Generalization: trained on ONE channel, tested on several")
    print("=" * 78)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
        print(f"\n{'-' * 60}\n  Modulation: {modulation}\n{'-' * 60}")

        # --- Train the CNN ONCE, only on TRAIN_CHANNEL, exactly like
        # run_case2_isi.py -- this model is never retrained below. ---
        train_tx, _, h_rrc = _generate_signal(modulation, seed=SEED)
        train_rx_clean = add_multipath(train_tx, TRAIN_CHANNEL)
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

        print("  [CNN] Training once on the matched (Case Study 2) channel...")
        t0 = time.time()
        cnn_model, _ = train_autoencoder(X_tr, Y_tr, X_va, Y_va, window_len=WINDOW_LEN,
                                          epochs=CNN_EPOCHS, batch_size=64, verbose=0)
        print(f"  [CNN] Training done in {time.time() - t0:.1f}s -- frozen for every channel below")

        methods = ["No Processing", "LMS", "NLMS", "CNN", "MMSE (Genie)"]

        for channel_name, multipath in HELD_OUT_CHANNELS.items():
            isi_energy = sum(abs(h) ** 2 for h in multipath[1:])
            print(f"\n  --- channel: {channel_name}  (ISI energy={isi_energy:.3f}) ---")

            test_gain_db = _calibrate_snr_gain_db(add_multipath(train_tx, multipath), h_rrc)
            mmse_taps, mmse_peak = measure_channel_taps(h_rrc, multipath, sps=SPS, num_taps=21)

            ber_err = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}
            ber_bits = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods}

            for trial in range(MC_TRIALS):
                test_tx, test_bits, _ = _generate_signal(modulation, seed=SEED + 5000 + trial * 97)
                test_rx_clean = add_multipath(test_tx, multipath)
                isi_ref = receiver_frontend(test_rx_clean, h_rrc, sps=SPS)

                for snr in SNR_LEVELS:
                    noisy = add_awgn(test_rx_clean, snr - test_gain_db, seed=SEED + 9000 + trial * 131 + snr)

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

                    # CNN: NEVER retrained -- same frozen model from the
                    # matched-channel training above, applied to whatever
                    # channel is currently being tested.
                    cnn_out = denoise_signal(cnn_model, noisy, WINDOW_LEN, STRIDE)
                    cnn_syms = receiver_frontend(cnn_out, h_rrc, sps=SPS)
                    e, n = compute_ber_count(test_bits, demod_fn(cnn_syms))
                    ber_err["CNN"][snr] += e; ber_bits["CNN"][snr] += n

            for m in methods:
                for snr in SNR_LEVELS:
                    e, n = ber_err[m][snr], ber_bits[m][snr]
                    all_results.append({
                        "modulation": modulation, "channel": channel_name,
                        "isi_energy": round(isi_energy, 4), "snr_db_in": snr,
                        "method": m, "ber": e / n, "errors": e, "bits": n,
                    })
            cnn_bers = [round(ber_err["CNN"][snr] / max(ber_bits["CNN"][snr], 1), 5) for snr in SNR_LEVELS]
            print(f"    CNN BER @ {SNR_LEVELS}dB: {cnn_bers}")

    df = pd.DataFrame(all_results)
    out_path = TABLES_DIR / "results_cnn_channel_generalization.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS: BER by channel, relative to the matched (train==test) condition")
    print("=" * 90)
    for modulation in ["BPSK", "QPSK"]:
        print(f"\n  {modulation}:")
        for snr in SNR_LEVELS:
            sub = df[(df.modulation == modulation) & (df.snr_db_in == snr)]
            matched = sub[sub.channel == "matched (train==test)"].set_index("method")
            print(f"    SNR={snr:+3d}dB")
            for m in ["No Processing", "LMS", "NLMS", "CNN", "MMSE (Genie)"]:
                matched_ber = matched.loc[m, "errors"] / max(matched.loc[m, "bits"], 1)
                degradations = []
                for ch in HELD_OUT_CHANNELS:
                    if ch == "matched (train==test)":
                        continue
                    row = sub[(sub.channel == ch) & (sub.method == m)].iloc[0]
                    held_out_ber = row["errors"] / max(row["bits"], 1)
                    ratio = (held_out_ber / matched_ber) if matched_ber > 0 else (
                        float("inf") if held_out_ber > 0 else 1.0)
                    degradations.append(ratio)
                print(f"      {m:16s} matched_ber={matched_ber:.4e}  "
                      f"held-out/matched ratios={[f'{r:.2f}x' for r in degradations]}")

    print(f"\n[OK] Wrote {out_path}")
    return df


if __name__ == "__main__":
    main()
