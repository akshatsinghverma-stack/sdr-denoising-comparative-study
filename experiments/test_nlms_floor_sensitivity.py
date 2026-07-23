#!/usr/bin/env python3
"""
test_nlms_floor_sensitivity.py — Is the NLMS normalization-floor intervention
robust to the floor fraction chosen, or was 25% an unvaried, cherry-picked
value?
=============================================================================
report/findings_lms_nlms_asymmetry.md's intervention used a single floor
fraction (25% of each trial's own mean windowed power) without checking
whether a different fraction would change the qualitative conclusion -- a
gap flagged during a self-critique pass. This sweeps 10%, 25%, and 50% on
the same signals used in the original intervention (identical seeds) and
reports the pooled Frozen/DD improvement ratio for each, so the original
25% choice can be judged against a real sensitivity check rather than left
as an arbitrary single data point.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn, add_time_varying_multipath
from src.nlms_filter import NLMSFilter
from src.metrics import compute_ber_count
from src.utils import demod_bpsk, demod_qpsk_fast

SEED = 42
TAPS = 16
NLMS_MU = 0.5
TRAINING_LENGTH = 1000
MIN_CONFIDENCE = 0.3
RELIABILITY_THRESHOLD_DEFAULT = 0.1
FLATFADE_NUM_SYMBOLS = 20_000
FLATFADE_SNR_LEVELS = [10, 15]  # the two SNRs where the original test showed the clearest effect
FLATFADE_MC_TRIALS = 5
FLATFADE_COHERENCE_SYMBOLS = 4000.0
FLATFADE_DRIFT_DEPTH = 0.4
FLOOR_FRACTIONS = [0.10, 0.25, 0.50]

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _get_demod(modulation):
    return demod_bpsk if modulation == "BPSK" else demod_qpsk_fast


def main():
    print("=" * 78)
    print("  NLMS Floor-Fraction Sensitivity: is 25% a robust choice, or cherry-picked?")
    print("=" * 78)
    rows = []

    for modulation in ["BPSK", "QPSK"]:
        demod_fn = _get_demod(modulation)
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
                avg_window_power = np.mean(np.abs(noisy) ** 2) * TAPS

                nlms_f = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                out_f, _ = nlms_f.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                           modulation=modulation, enable_decision_directed=False)
                e, n = compute_ber_count(test_bits, demod_fn(out_f))
                ber_frozen = e / n

                for frac in FLOOR_FRACTIONS:
                    floor_value = frac * avg_window_power
                    nlms_dd = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
                    out_dd, _ = nlms_dd.denoise(test_tx, noisy, training_length=TRAINING_LENGTH,
                                                 modulation=modulation, enable_decision_directed=True,
                                                 reliability_threshold=RELIABILITY_THRESHOLD_DEFAULT,
                                                 min_confidence=MIN_CONFIDENCE, min_norm_power=floor_value)
                    e, n = compute_ber_count(test_bits, demod_fn(out_dd))
                    ber_dd = e / n
                    rows.append(dict(modulation=modulation, snr_db_in=snr, trial=trial,
                                      floor_fraction=frac, ber_frozen=ber_frozen, ber_dd=ber_dd,
                                      dd_engaged=nlms_dd._dd_enabled))

            print(f"    {modulation} trial {trial+1}/{FLATFADE_MC_TRIALS} done")

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_nlms_floor_sensitivity.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("  RESULTS: pooled Frozen/DD improvement ratio by floor fraction")
    print("=" * 90)
    summary = df.groupby(["modulation", "snr_db_in", "floor_fraction"]).apply(
        lambda g: pd.Series({
            "frozen_total_ber": g["ber_frozen"].mean(),
            "dd_total_ber": g["ber_dd"].mean(),
            "improvement_ratio": g["ber_frozen"].mean() / g["ber_dd"].mean() if g["ber_dd"].mean() > 0 else np.nan,
        }), include_groups=False
    )
    print(summary.to_string())
    print(f"\n[OK] Wrote {out_path}")
    return df


if __name__ == "__main__":
    main()
