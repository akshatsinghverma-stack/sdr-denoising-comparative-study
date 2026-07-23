#!/usr/bin/env python3
"""
run_timevarying_channel.py — Does decision-directed tracking earn its keep
on a genuinely time-varying channel? (diagnostic, new)
=============================================================================
Every stability decision in this project (freezing LMS/NLMS weights after
the 1000-symbol preamble, report Section 3.2) was justified by "this channel
doesn't change over time, so there's nothing for decision-directed tracking
to do" -- a claim that, until now, was only ever checked on time-INVARIANT
channels (Case Study 1: none; Case Study 2: one static 3-tap multipath).
This script builds the one experiment named-but-never-run in the report's
"Future Work" section: a slowly time-varying multipath channel
(`src/channel.py:add_time_varying_multipath`), on which continued adaptation
might plausibly have an actual job to do.

For each Monte Carlo trial, LMS and NLMS are each run TWICE on the exact same
noisy signal:
  - "Frozen"    : enable_decision_directed=False (today's default).
  - "DD"        : enable_decision_directed=True.

IMPORTANT, and itself a finding (see report/findings_timevarying_channel.md):
the DD reliability gate (the global on/off switch described in report Section
3.2 -- comparing hard decisions on the preamble tail against the *clean*
reference) was written and only ever tested against Case Study 1's SPS=1
channel, where every clean sample IS exactly a constellation point. At SPS=4
(this project's pulse-shaped/ISI regime, used here as in Case Study 2), the
clean reference is a continuously-varying RRC-shaped waveform that equals an
exact constellation point at only 1 in `sps` samples -- so the gate's exact-
value mismatch check structurally reads as ~75-80% "mismatch" regardless of
SNR or channel behaviour, and default-gated DD (reliability_threshold=0.1)
never engages even once (verified below, `dd_engage_rate_default_gate`). This
is a latent incompatibility, not a channel-related result, and it means the
"DD" column in the main sweep below uses `reliability_threshold=1.0` (i.e.
the global gate is deliberately bypassed via its own public parameter,
`min_confidence` remains the only active safeguard) so this script can still
test the actual research question using the local, oversampling-agnostic
confidence gate. Both configurations are measured and reported.
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
from src.channel import add_awgn, add_time_varying_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.metrics import compute_snr_db, compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast, receiver_frontend

# ===========================================================================
# Configuration (diagnostic scale -- see module docstring / report Section 5
# scope note for precedent on reduced-scale diagnostic sweeps in this project)
# ===========================================================================
NUM_SYMBOLS = 25_000
SNR_LEVELS = [-5, 0, 5, 10, 15]
SEED = 42
MC_TRIALS = 5

SPS = 4
MULTIPATH_BASE = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]  # Case Study 2's static profile, reused as drift center
CHANNEL_MODES = ["sinusoidal", "random_walk"]
N_CYCLES = 1.5          # sinusoidal: 1.5 full oscillations across the whole test signal
DRIFT_DEPTH = 0.6       # both modes: +-60% fractional modulation of non-direct taps
COHERENCE_SYMBOLS = 4000.0  # random_walk: ~4x the preamble length -> "slow" by construction

TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000 * SPS  # 1000-symbol preamble, matching Case Study 1/2
MIN_CONFIDENCE = 0.3
RELIABILITY_THRESHOLD_DEFAULT = 0.1   # today's project default
RELIABILITY_THRESHOLD_BYPASS = 1.0    # bypasses the (SPS>1-incompatible) global gate -- see docstring

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "timevarying"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def _generate_signal(modulation, seed):
    gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
    return gen(NUM_SYMBOLS, seed=seed, sps=SPS)


def _calibrate_snr_gain_db(rx_clean, h_rrc):
    """Same empirical calibration technique as run_case2_isi.py /
    run_severity_sweep.py: measure the net nominal-SNR -> post-receiver-SNR
    gain once (matched-filter coherent-combining gain + this channel's own
    gain/loss) via a 0dB noise probe, rather than deriving it analytically."""
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    noisy_probe = add_awgn(rx_clean, 0.0, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    return compute_snr_db(isi_ref, syms_probe) - 0.0


def _make_channel(tx, mode, seed):
    return add_time_varying_multipath(
        tx, h_base=MULTIPATH_BASE, sps=SPS, mode=mode,
        n_cycles=N_CYCLES, drift_depth=DRIFT_DEPTH,
        coherence_symbols=COHERENCE_SYMBOLS, seed=seed,
    )


def _check_default_gate_dead(modulation, mode):
    """Verify (not assume) that the default reliability gate never engages
    at SPS=4, across a few SNRs/trials, before relying on the bypassed-gate
    variant for the main comparison. Returns the observed engagement rate."""
    engaged = 0
    total = 0
    for trial in range(3):
        tx, _, h_rrc = _generate_signal(modulation, seed=SEED + trial)
        rx = _make_channel(tx, mode, seed=SEED + trial)
        for snr in (-5, 5, 15):
            noisy = add_awgn(rx, snr, seed=SEED + 1000 + trial * 13 + snr)
            lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
            lms.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation,
                        enable_decision_directed=True,
                        reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                        min_confidence=MIN_CONFIDENCE)
            engaged += int(lms._dd_enabled)
            total += 1
    return engaged / total


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Time-Varying Channel Diagnostic: Frozen vs Decision-Directed LMS/NLMS")
    print("=" * 78)

    # --- Preliminary check: is the default reliability gate structurally
    # dead at SPS=4, as reasoned in the module docstring? Verify, don't assume.
    gate_rates = {}
    for modulation in ["BPSK", "QPSK"]:
        for mode in CHANNEL_MODES:
            rate = _check_default_gate_dead(modulation, mode)
            gate_rates[(modulation, mode)] = rate
            print(f"  [Gate check] {modulation} / {mode}: default-gate (threshold="
                  f"{RELIABILITY_THRESHOLD_DEFAULT}) DD engagement rate = {rate:.0%} "
                  f"over 9 (trial,SNR) probes")

    all_results = []       # aggregated per (modulation, mode, snr, method)
    trial_results = []     # per-trial detail, for divergence/stability analysis

    methods_adaptive = ["LMS-Frozen", "LMS-DD", "NLMS-Frozen", "NLMS-DD"]
    methods_all = ["No Processing"] + methods_adaptive

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
        for mode in CHANNEL_MODES:
            print(f"\n{'-' * 70}\n  {modulation} / channel mode = {mode}\n{'-' * 70}")

            # Calibration realization: fixed seed, same convention as run_case2_isi.py
            calib_tx, _, h_rrc = _generate_signal(modulation, seed=SEED)
            calib_rx = _make_channel(calib_tx, mode, seed=SEED)
            gain_db = _calibrate_snr_gain_db(calib_rx, h_rrc)
            print(f"  [Calibration] nominal-to-postreceiver SNR gain = {gain_db:+.3f} dB")

            ber_err = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods_all}
            ber_bits = {m: {snr: 0 for snr in SNR_LEVELS} for m in methods_all}
            snr_vals = {m: {snr: [] for snr in SNR_LEVELS} for m in methods_all}
            dd_engaged = {m: {snr: [] for snr in SNR_LEVELS} for m in ["LMS-DD", "NLMS-DD"]}

            for trial in range(MC_TRIALS):
                test_tx, test_bits, _ = _generate_signal(modulation, seed=SEED + 5000 + trial * 97)
                # Each trial gets its own channel-drift realization too (Monte
                # Carlo over channel + noise + bits jointly), seeded off trial.
                test_rx = _make_channel(test_tx, mode, seed=SEED + 6000 + trial * 53)
                isi_ref = receiver_frontend(test_rx, h_rrc, sps=SPS)

                for snr in SNR_LEVELS:
                    noisy = add_awgn(test_rx, snr - gain_db, seed=SEED + 9000 + trial * 131 + snr)

                    def _record(method, snr_out, err_count, n_bits):
                        ber_err[method][snr] += err_count
                        ber_bits[method][snr] += n_bits
                        snr_vals[method][snr].append(snr_out)
                        trial_results.append({
                            "modulation": modulation, "channel_mode": mode, "trial": trial,
                            "snr_db_in": snr, "method": method,
                            "snr_db_out_vs_postisi_ref": snr_out,
                            "ber": err_count / n_bits, "errors": err_count, "bits": n_bits,
                        })

                    # --- No Processing ---
                    syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
                    snr_noproc = compute_snr_db(isi_ref, syms_noproc)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_noproc))
                    _record("No Processing", snr_noproc, e, n)

                    # --- LMS: Frozen vs DD (gate bypassed) ---
                    lms_f = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                    out_f, _ = lms_f.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                              modulation=modulation, enable_decision_directed=False)
                    syms_f = receiver_frontend(out_f, h_rrc, sps=SPS)
                    snr_f = compute_snr_db(isi_ref, syms_f)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_f))
                    _record("LMS-Frozen", snr_f, e, n)

                    lms_dd = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                    out_dd, _ = lms_dd.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                                modulation=modulation, enable_decision_directed=True,
                                                reliability_threshold=RELIABILITY_THRESHOLD_BYPASS,
                                                min_confidence=MIN_CONFIDENCE)
                    syms_dd = receiver_frontend(out_dd, h_rrc, sps=SPS)
                    snr_dd = compute_snr_db(isi_ref, syms_dd)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_dd))
                    _record("LMS-DD", snr_dd, e, n)
                    dd_engaged["LMS-DD"][snr].append(int(lms_dd._dd_enabled))

                    # --- NLMS: Frozen vs DD (gate bypassed) ---
                    nlms_f = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                    out_f, _ = nlms_f.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                               modulation=modulation, enable_decision_directed=False)
                    syms_f = receiver_frontend(out_f, h_rrc, sps=SPS)
                    snr_f = compute_snr_db(isi_ref, syms_f)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_f))
                    _record("NLMS-Frozen", snr_f, e, n)

                    nlms_dd = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                    out_dd, _ = nlms_dd.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                                 modulation=modulation, enable_decision_directed=True,
                                                 reliability_threshold=RELIABILITY_THRESHOLD_BYPASS,
                                                 min_confidence=MIN_CONFIDENCE)
                    syms_dd = receiver_frontend(out_dd, h_rrc, sps=SPS)
                    snr_dd = compute_snr_db(isi_ref, syms_dd)
                    e, n = compute_ber_count(test_bits, demod_fn(syms_dd))
                    _record("NLMS-DD", snr_dd, e, n)
                    dd_engaged["NLMS-DD"][snr].append(int(nlms_dd._dd_enabled))

                print(f"    trial {trial + 1}/{MC_TRIALS} done")

            for m in methods_all:
                for snr in SNR_LEVELS:
                    e, n = ber_err[m][snr], ber_bits[m][snr]
                    snrs = np.array(snr_vals[m][snr])
                    ber_ub = (3.0 / n) if e == 0 else None
                    row = {
                        "modulation": modulation, "channel_mode": mode, "snr_db_in": snr, "method": m,
                        "snr_db_out_vs_postisi_ref_mean": round(snrs.mean(), 3),
                        "snr_db_out_vs_postisi_ref_std": round(snrs.std(), 3),
                        "ber_mean": e / n if n else float("nan"),
                        "ber_total_errors": e, "ber_total_bits": n,
                        "ber_95ci_upper_bound": ber_ub,
                        "n_trials": MC_TRIALS,
                    }
                    if m in dd_engaged:
                        row["dd_engaged_fraction"] = float(np.mean(dd_engaged[m][snr]))
                    all_results.append(row)

            print(f"\n    [{modulation}/{mode}] BER summary (errors/bits, mean SNR_out vs post-ISI ref):")
            for snr in SNR_LEVELS:
                parts = []
                for m in methods_all:
                    e, n = ber_err[m][snr], ber_bits[m][snr]
                    parts.append(f"{m}={e}/{n} ({e/n:.2e})")
                print(f"      SNR_in={snr:+3d}dB  " + "  ".join(parts))

    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_timevarying.csv"
    df.to_csv(csv_path, index=False)

    trial_df = pd.DataFrame(trial_results)
    trial_csv_path = TABLES_DIR / "results_timevarying_trials.csv"
    trial_df.to_csv(trial_csv_path, index=False)

    gate_df = pd.DataFrame(
        [{"modulation": k[0], "channel_mode": k[1], "default_gate_engagement_rate": v}
         for k, v in gate_rates.items()]
    )
    gate_csv_path = TABLES_DIR / "results_timevarying_gate_check.csv"
    gate_df.to_csv(gate_csv_path, index=False)

    # --- Figures: BER improvement ratio (Frozen BER / DD BER) vs SNR, per
    # modulation/channel mode -- >1 means DD beats Frozen.
    for modulation in ["BPSK", "QPSK"]:
        fig, axes = plt.subplots(1, len(CHANNEL_MODES), figsize=(6 * len(CHANNEL_MODES), 5))
        if len(CHANNEL_MODES) == 1:
            axes = [axes]
        for ax, mode in zip(axes, CHANNEL_MODES):
            sub = df[(df.modulation == modulation) & (df.channel_mode == mode)]
            for adaptive, frozen_name, dd_name in [("LMS", "LMS-Frozen", "LMS-DD"),
                                                    ("NLMS", "NLMS-Frozen", "NLMS-DD")]:
                f_sub = sub[sub.method == frozen_name].set_index("snr_db_in")
                d_sub = sub[sub.method == dd_name].set_index("snr_db_in")
                ratios = []
                for snr in SNR_LEVELS:
                    f_ber = f_sub.loc[snr, "ber_total_errors"] / f_sub.loc[snr, "ber_total_bits"]
                    d_ber = d_sub.loc[snr, "ber_total_errors"] / d_sub.loc[snr, "ber_total_bits"]
                    ratios.append((f_ber / d_ber) if d_ber > 0 else np.nan)
                ax.plot(SNR_LEVELS, ratios, marker="o", label=adaptive, linewidth=2)
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1.5, label="No difference")
            ax.set_yscale("log")
            ax.set_xlabel("Input SNR (dB)")
            ax.set_ylabel("BER improvement ratio (Frozen/DD)")
            ax.set_title(f"{modulation} — channel mode: {mode}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = FIGURES_DIR / f"frozen_vs_dd_{modulation}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY — Time-Varying Channel Diagnostic")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\n[OK] Wrote {csv_path}, {trial_csv_path}, {gate_csv_path}")
    return df


# ===========================================================================
# Part 2 — SPS=1 flat-fading control (isolates time-variation from the
# SPS>1 oversampling confound discovered in Part 1)
# ===========================================================================
# Part 1 (above) found that enable_decision_directed=True, at SPS=4, either
# (a) never engages at all (default reliability_threshold=0.1 -- the gate
# compares hard decisions against the continuous RRC-shaped clean reference,
# which equals an exact constellation point at only 1-in-sps samples, so it
# reads ~75-80% "mismatch" regardless of channel/SNR), or (b) if the gate is
# bypassed to force it to run, catastrophically fails (BER ~0.48, confirmed
# even on Case Study 2's STATIC channel -- see report/findings_timevarying_
# channel.md for the full root-cause) because the per-sample DD update rule
# treats every one of the 4x-oversampled samples as an independent symbol
# decision, corrupting the filter's pulse-shape reconstruction.
#
# That failure mode has nothing to do with time-variation -- it happens on a
# static channel too. To cleanly test the ORIGINAL Section 3.2 hypothesis
# ("a time-invariant channel gives DD nothing to track; a time-varying one
# might") in the one regime where DD's machinery was actually built and
# validated (Case Study 1's SPS=1, no pulse shaping, every sample IS a
# symbol), this control test applies a genuinely time-varying, complex
# (amplitude AND phase) FLAT-fading single-tap channel at SPS=1, with the
# DD reliability gate left at its DEFAULT (unmodified, un-bypassed) setting.
FLATFADE_NUM_SYMBOLS = 20_000
FLATFADE_SNR_LEVELS = [-5, 0, 5, 10, 15]
FLATFADE_MC_TRIALS = 5
FLATFADE_TRAINING_LENGTH = 1000  # symbols == samples at SPS=1
FLATFADE_COHERENCE_SYMBOLS = 4000.0  # same "slow relative to preamble" ratio as Part 1
FLATFADE_DRIFT_DEPTH = 0.4


def run_flatfade_sps1_control():
    print("\n" + "=" * 78)
    print("  Part 2 (control): SPS=1 flat-fading channel, DEFAULT (un-bypassed) DD gate")
    print("=" * 78)

    methods_all = ["No Processing", "LMS-Frozen", "LMS-DD", "NLMS-Frozen", "NLMS-DD"]
    all_results = []
    trial_results = []

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
        print(f"\n{'-' * 60}\n  {modulation} (SPS=1, flat fading)\n{'-' * 60}")

        ber_err = {m: {snr: 0 for snr in FLATFADE_SNR_LEVELS} for m in methods_all}
        ber_bits = {m: {snr: 0 for snr in FLATFADE_SNR_LEVELS} for m in methods_all}
        dd_engaged = {m: {snr: [] for snr in FLATFADE_SNR_LEVELS} for m in ["LMS-DD", "NLMS-DD"]}

        for trial in range(FLATFADE_MC_TRIALS):
            gen = generate_bpsk if modulation == "BPSK" else generate_qpsk
            test_tx, test_bits, _ = gen(FLATFADE_NUM_SYMBOLS, seed=SEED + 5000 + trial * 97, sps=1)
            test_rx = add_time_varying_multipath(
                test_tx, h_base=[1.0], sps=1, mode="random_walk",
                coherence_symbols=FLATFADE_COHERENCE_SYMBOLS, drift_depth=FLATFADE_DRIFT_DEPTH,
                vary_direct_path=True, seed=SEED + 6000 + trial * 53,
            )

            for snr in FLATFADE_SNR_LEVELS:
                noisy = add_awgn(test_rx, snr, seed=SEED + 9000 + trial * 131 + snr)

                def _record(method, err_count, n_bits):
                    ber_err[method][snr] += err_count
                    ber_bits[method][snr] += n_bits
                    trial_results.append({
                        "modulation": modulation, "trial": trial, "snr_db_in": snr,
                        "method": method, "ber": err_count / n_bits,
                        "errors": err_count, "bits": n_bits,
                    })

                e, n = compute_ber_count(test_bits, demod_fn(noisy))
                _record("No Processing", e, n)

                lms_f = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                out_f, _ = lms_f.denoise(test_tx, noisy, training_length=FLATFADE_TRAINING_LENGTH,
                                          modulation=modulation, enable_decision_directed=False)
                e, n = compute_ber_count(test_bits, demod_fn(out_f))
                _record("LMS-Frozen", e, n)

                lms_dd = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                out_dd, _ = lms_dd.denoise(test_tx, noisy, training_length=FLATFADE_TRAINING_LENGTH,
                                            modulation=modulation, enable_decision_directed=True,
                                            reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                                            min_confidence=MIN_CONFIDENCE)
                e, n = compute_ber_count(test_bits, demod_fn(out_dd))
                _record("LMS-DD", e, n)
                dd_engaged["LMS-DD"][snr].append(int(lms_dd._dd_enabled))

                nlms_f = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_f, _ = nlms_f.denoise(test_tx, noisy, training_length=FLATFADE_TRAINING_LENGTH,
                                           modulation=modulation, enable_decision_directed=False)
                e, n = compute_ber_count(test_bits, demod_fn(out_f))
                _record("NLMS-Frozen", e, n)

                nlms_dd = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_dd, _ = nlms_dd.denoise(test_tx, noisy, training_length=FLATFADE_TRAINING_LENGTH,
                                             modulation=modulation, enable_decision_directed=True,
                                             reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                                             min_confidence=MIN_CONFIDENCE)
                e, n = compute_ber_count(test_bits, demod_fn(out_dd))
                _record("NLMS-DD", e, n)
                dd_engaged["NLMS-DD"][snr].append(int(nlms_dd._dd_enabled))

            print(f"    trial {trial + 1}/{FLATFADE_MC_TRIALS} done")

        for m in methods_all:
            for snr in FLATFADE_SNR_LEVELS:
                e, n = ber_err[m][snr], ber_bits[m][snr]
                row = {
                    "modulation": modulation, "snr_db_in": snr, "method": m,
                    "ber_mean": e / n if n else float("nan"),
                    "ber_total_errors": e, "ber_total_bits": n,
                    "ber_95ci_upper_bound": (3.0 / n) if e == 0 else None,
                    "n_trials": FLATFADE_MC_TRIALS,
                }
                if m in dd_engaged:
                    row["dd_engaged_fraction"] = float(np.mean(dd_engaged[m][snr]))
                all_results.append(row)

        print(f"\n    [{modulation}] BER summary (errors/bits):")
        for snr in FLATFADE_SNR_LEVELS:
            parts = [f"{m}={ber_err[m][snr]}/{ber_bits[m][snr]} ({ber_err[m][snr]/ber_bits[m][snr]:.2e})"
                     for m in methods_all]
            print(f"      SNR_in={snr:+3d}dB  " + "  ".join(parts))

    df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "results_timevarying_flatfade_sps1.csv"
    df.to_csv(csv_path, index=False)
    trial_df = pd.DataFrame(trial_results)
    trial_df.to_csv(TABLES_DIR / "results_timevarying_flatfade_sps1_trials.csv", index=False)

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY — SPS=1 Flat-Fading Control")
    print("=" * 90)
    print(df.to_string(index=False))
    print(f"\n[OK] Wrote {csv_path}")
    return df


if __name__ == "__main__":
    main()
    run_flatfade_sps1_control()
