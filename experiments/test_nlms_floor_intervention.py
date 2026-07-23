#!/usr/bin/env python3
"""
test_nlms_floor_intervention.py — The actual falsification test named (but
not performed) in report/findings_lms_nlms_asymmetry.md.
=============================================================================
That diagnostic found strong correlational evidence that NLMS's
instantaneous-power step-size normalization inflates its effective step size
during channel fades, exactly when wrong decisions are most likely, and
named the concrete next step: floor the normalization denominator (now
`NLMSFilter`'s new `min_norm_power` parameter, default 0.0 = unchanged
original behavior) and check whether (a) NLMS-DD's BER improves relative to
Frozen on the actual time-varying flat-fading control, and (b) the
fade-vs-update-norm correlation signature weakens.

Reproduces run_timevarying_channel.py's exact SPS=1 flat-fading control
(same seeds, same config) and compares NLMS-DD at min_norm_power=0 (original)
against a floored variant, for both modulations, all 5 SNR levels, 5 trials.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_time_varying_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.metrics import compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast

SEED = 42
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000
MIN_CONFIDENCE = 0.3
RELIABILITY_THRESHOLD_DEFAULT = 0.1
FLATFADE_NUM_SYMBOLS = 20_000
FLATFADE_SNR_LEVELS = [-5, 0, 5, 10, 15]
FLATFADE_MC_TRIALS = 5
FLATFADE_COHERENCE_SYMBOLS = 4000.0
FLATFADE_DRIFT_DEPTH = 0.4
FLOOR_FRACTION = 0.25  # floor = 25% of this trial's own average ||x(n)||^2

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def main():
    print("=" * 78)
    print("  NLMS Floor Intervention: does flooring the normalization recover DD benefit?")
    print("=" * 78)
    rows = []

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
        print(f"\n{'-'*60}\n  {modulation}\n{'-'*60}")

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
                # floor must be on the same scale as ||x(n)||^2, a SUM over
                # TAPS samples -- using raw per-sample power here (bug, found
                # by checking the floor against the actual window-power
                # distribution rather than assuming the scaling was right)
                # made the floor ~16x too small to ever bind.
                avg_window_power = np.mean(np.abs(noisy) ** 2) * TAPS
                floor_value = FLOOR_FRACTION * avg_window_power

                # Frozen (reference)
                lms_f = LMSFilter(num_taps=TAPS, mu=LMS_MU)
                out_f, _ = lms_f.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                          modulation=modulation, enable_decision_directed=False)
                e, n = compute_ber_count(test_bits, demod_fn(out_f))
                ber_lms_frozen = e / n

                # NLMS-Frozen (reference)
                nlms_f = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_f2, _ = nlms_f.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                            modulation=modulation, enable_decision_directed=False)
                e, n = compute_ber_count(test_bits, demod_fn(out_f2))
                ber_nlms_frozen = e / n

                # NLMS-DD, ORIGINAL (min_norm_power=0.0, unchanged behavior)
                nlms_dd0 = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_dd0, _ = nlms_dd0.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                               modulation=modulation, enable_decision_directed=True,
                                               reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                                               min_confidence=MIN_CONFIDENCE, min_norm_power=0.0)
                e, n = compute_ber_count(test_bits, demod_fn(out_dd0))
                ber_nlms_dd_orig = e / n

                # NLMS-DD, FLOORED (the intervention)
                nlms_dd1 = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_dd1, _ = nlms_dd1.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                               modulation=modulation, enable_decision_directed=True,
                                               reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                                               min_confidence=MIN_CONFIDENCE, min_norm_power=floor_value)
                e, n = compute_ber_count(test_bits, demod_fn(out_dd1))
                ber_nlms_dd_floored = e / n

                rows.append(dict(
                    modulation=modulation, snr_db_in=snr, trial=trial,
                    ber_lms_frozen=ber_lms_frozen, ber_nlms_frozen=ber_nlms_frozen,
                    ber_nlms_dd_original=ber_nlms_dd_orig, ber_nlms_dd_floored=ber_nlms_dd_floored,
                    dd_engaged_orig=nlms_dd0._dd_enabled, dd_engaged_floored=nlms_dd1._dd_enabled,
                ))

            print(f"    trial {trial+1}/{FLATFADE_MC_TRIALS} done")

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_nlms_floor_intervention.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 100)
    print("  RESULTS: NLMS-DD original vs. floored, pooled by (modulation, snr)")
    print("=" * 100)
    summary = df.groupby(["modulation", "snr_db_in"]).agg(
        ber_nlms_frozen=("ber_nlms_frozen", "mean"),
        ber_nlms_dd_original=("ber_nlms_dd_original", "mean"),
        ber_nlms_dd_floored=("ber_nlms_dd_floored", "mean"),
        engaged_orig=("dd_engaged_orig", "mean"),
        engaged_floored=("dd_engaged_floored", "mean"),
    )
    summary["improvement_ratio_floored_vs_frozen"] = (
        summary["ber_nlms_frozen"] / summary["ber_nlms_dd_floored"].replace(0, np.nan)
    )
    summary["improvement_ratio_original_vs_frozen"] = (
        summary["ber_nlms_frozen"] / summary["ber_nlms_dd_original"].replace(0, np.nan)
    )
    print(summary.to_string())
    print(f"\n[OK] Wrote {out_path}")
    return df


if __name__ == "__main__":
    main()
