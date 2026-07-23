#!/usr/bin/env python3
"""
compute_cost_analysis.py -- Analytical + measured compute-cost comparison
============================================================================
The main pipelines (run_case1_no_isi.py, run_case2_isi.py) already report a
`runtime_sec_mean` column in results/tables/results_case{1,2}*.csv, but that
number is a wall-clock measurement of *this specific Python/NumPy/TensorFlow
implementation* -- it mixes real algorithmic cost with interpreter loop
overhead, TensorFlow graph-dispatch overhead, etc. It answers "how long did
this code take on this machine today", not "how many arithmetic operations
does this method fundamentally require per output sample" -- which is the
question that actually matters when deciding whether a method can run at
line-rate on a specific piece of real-time hardware (an FPGA, a DSP core, a
microcontroller, a full CPU/GPU).

This script builds that second, hardware-portable number from first
principles (multiply-accumulate operations per output sample, derived
analytically from each method's actual structure -- shown, not asserted),
places it side by side with a fresh wall-clock benchmark measured directly on
this machine, and cross-references both against the existing
`runtime_sec_mean` columns to explain where the two diverge and why.

It does NOT modify any existing module, script, or result file -- it only
imports and calls existing functions/classes, benchmarks them, and writes a
new report file (report/findings_compute_cost.md).

Run:
    ./venv/Scripts/python.exe experiments/compute_cost_analysis.py
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from src.signal_gen import generate_bpsk
from src.channel import add_awgn, add_multipath
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.cnn_autoencoder import build_autoencoder, denoise_signal
from src.hybrid_model import denoise_hybrid
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer
from src.utils import receiver_frontend

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

# Shared config, matched to the two main pipelines so the benchmark reflects
# real operating points, not made-up sizes.
TAPS = 16
WINDOW_LEN = 128
STRIDE = 64
NUM_SYMBOLS = 100_000
LMS_MU = 0.01
NLMS_MU = 0.5
N_BENCH_REPEATS = 7  # repeated timing runs per method for a mean+std estimate

MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]  # same channel as Case Study 2


# ===========================================================================
# 1. Analytical MAC-count derivation
# ===========================================================================
#
# Convention used throughout: 1 complex multiply = 4 real multiplies + 2 real
# adds; following the task's own approximation we count this as "~4 real
# MACs per complex MAC" (folding the 2 adds into the same order-of-magnitude
# bucket as the 4 multiplies -- the standard back-of-envelope DSP costing
# convention; a more pedantic count of 4 mults + 2 adds = 6 real ops changes
# the headline multiplier by <50%, not the conclusion).
REAL_MACS_PER_COMPLEX_MAC = 4


def lms_nlms_mmse_macs_per_sample(num_taps: int = TAPS):
    """LMS / NLMS / genie-MMSE steady-state (deployed, frozen-filter) cost.

    All three apply an FIR filter of the same structural form to produce one
    output sample:

        y[n] = sum_{k=0}^{taps-1} w[k] * x[n-k]        (LMS/NLMS, see
                                                          lms_filter.py L164-166,
                                                          199-200)
        s_hat[n] = sum_i conj(w[i]) * y[n+delay-i]      (MMSE, see
                                                          mmse_equalizer.py L128-134)

    Both are a length-`taps` complex dot product: exactly `taps` complex
    multiply-accumulates per output sample. This is the cost that matters for
    a *deployed* system, because this project's own methodology (report.md
    Section 3.2) freezes LMS/NLMS weights after the preamble and never
    updates them again -- the per-sample production cost in steady state is
    the FIR apply only. The one-time preamble/adaptation cost (like the
    genie-MMSE's one-time O(taps^2)-O(taps^3) design-matrix solve, see
    mmse_equalizer.py `design_mmse_equalizer`) is a start-up cost amortized
    over the entire signal and is reported separately below, not folded into
    the per-sample number, exactly as the task specifies for MMSE.

    NLMS additionally computes one extra real division per sample even in a
    from-scratch idealized implementation, because its update rule needs
    ||x(n)||^2 in the denominator -- noted here as an extra *operation*, not
    an extra MAC, since one division per sample is cheap in MAC-count terms
    but NOT cheap in cycle-count terms on hardware without a hardware divide
    unit (see Section 3 discrepancy discussion).
    """
    complex_macs = num_taps            # one complex MAC per tap
    real_macs = complex_macs * REAL_MACS_PER_COMPLEX_MAC
    return {
        "complex_macs_per_sample": complex_macs,
        "real_macs_per_sample": real_macs,
    }


def conv1d_macs(kernel_size, in_ch, out_ch, output_length):
    """Standard MAC-count formula for one Conv1D layer, 'same' padding,
    stride 1: each of the `output_length` output positions, for each of the
    `out_ch` output channels, requires `kernel_size * in_ch` multiply-
    accumulates (one per input channel per kernel tap)."""
    return kernel_size * in_ch * out_ch * output_length


def cnn_macs_per_window(window_len: int = WINDOW_LEN):
    """CNN autoencoder MAC count, derived layer-by-layer from the exact
    architecture in src/cnn_autoencoder.py `build_autoencoder` (all layers
    'same'-padded, stride 1, so every layer's output length == window_len).

    Layer                    kernel  in_ch  out_ch
    enc1  Conv1D(32,7)          7      2     32
    enc2  Conv1D(16,5)          5     32     16
    enc3  Conv1D(8,3)           3     16      8
    dec1  Conv1D(16,3)          3      8     16
    dec2  Conv1D(32,5)          5     16     32
    dec_out Conv1D(2,7)         7     32      2

    These are REAL-valued convolutions over 2 channels (I, Q stacked as
    separate real channels -- see cnn_autoencoder.py `_iq_to_real`), so no
    complex-MAC 4x factor applies here (unlike LMS/NLMS/MMSE, which are
    genuinely complex-valued).
    """
    layers = [
        ("enc1", 7, 2, 32),
        ("enc2", 5, 32, 16),
        ("enc3", 3, 16, 8),
        ("dec1", 3, 8, 16),
        ("dec2", 5, 16, 32),
        ("dec_out", 7, 32, 2),
    ]
    per_layer = []
    total = 0
    for name, k, cin, cout in layers:
        macs = conv1d_macs(k, cin, cout, window_len)
        per_layer.append((name, k, cin, cout, macs))
        total += macs
    return per_layer, total


def cnn_params(window_len: int = WINDOW_LEN):
    """Analytical parameter count (kernel weights + bias per Conv1D layer),
    cross-checked below against model.count_params() from an actual built
    Keras model -- must match exactly if the architecture description above
    is faithful to the code."""
    layers = [
        ("enc1", 7, 2, 32),
        ("enc2", 5, 32, 16),
        ("enc3", 3, 16, 8),
        ("dec1", 3, 8, 16),
        ("dec2", 5, 16, 32),
        ("dec_out", 7, 32, 2),
    ]
    total = 0
    per_layer = []
    for name, k, cin, cout in layers:
        p = k * cin * cout + cout  # kernel weights + bias
        per_layer.append((name, p))
        total += p
    return per_layer, total


# ===========================================================================
# 2. Wall-clock benchmarking on THIS machine (CPU only -- this project trains
#    and runs its CNNs on CPU throughout, no GPU, per report.md Section 2/8)
# ===========================================================================

def _time_repeated(fn, n_repeats=N_BENCH_REPEATS):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return times.mean(), times.std()


def benchmark_case(sps: int, num_symbols: int, label: str):
    """Benchmark every method on a signal sized/shaped like one of the two
    main case studies (sps=1 <-> Case Study 1, sps=4 + multipath <-> Case
    Study 2), so the benchmark is directly comparable to that case's
    existing runtime_sec_mean column."""
    print(f"\n--- Benchmarking on a {label} signal (sps={sps}, {num_symbols} symbols) ---")
    tx, bits, h_rrc = generate_bpsk(num_symbols, seed=123, sps=sps)
    training_length = 1000 * sps

    if sps > 1:
        rx_clean = add_multipath(tx, MULTIPATH)
    else:
        rx_clean = tx
    noisy = add_awgn(rx_clean, 5.0, seed=456)
    n_samples = len(noisy)

    results = {}

    # --- LMS ---
    def run_lms():
        f = LMSFilter(num_taps=TAPS, mu=LMS_MU)
        f.denoise(tx, noisy, training_length=training_length, modulation="BPSK")
    mean_t, std_t = _time_repeated(run_lms)
    results["LMS"] = (mean_t, std_t, n_samples)
    print(f"  LMS   : {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over {n_samples} samples "
          f"-> {mean_t/n_samples*1e9:.1f} ns/sample")

    # --- NLMS ---
    def run_nlms():
        f = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
        f.denoise(tx, noisy, training_length=training_length, modulation="BPSK")
    mean_t, std_t = _time_repeated(run_nlms)
    results["NLMS"] = (mean_t, std_t, n_samples)
    print(f"  NLMS  : {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over {n_samples} samples "
          f"-> {mean_t/n_samples*1e9:.1f} ns/sample")

    # --- CNN (untrained model -- inference cost is identical regardless of
    #     trained vs. random weights; same architecture, same FLOPs) ---
    cnn_model = build_autoencoder(window_len=WINDOW_LEN)

    def run_cnn():
        denoise_signal(cnn_model, noisy, WINDOW_LEN, STRIDE)
    mean_t, std_t = _time_repeated(run_cnn, n_repeats=5)
    results["CNN"] = (mean_t, std_t, n_samples)
    print(f"  CNN   : {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over {n_samples} samples "
          f"-> {mean_t/n_samples*1e9:.1f} ns/sample")

    # --- Hybrid (LMS -> CNN in sequence) ---
    def run_hybrid():
        f = LMSFilter(num_taps=TAPS, mu=LMS_MU)
        denoise_hybrid(f, cnn_model, tx, noisy, WINDOW_LEN, STRIDE,
                        training_length=training_length, modulation="BPSK")
    mean_t, std_t = _time_repeated(run_hybrid, n_repeats=5)
    results["Hybrid"] = (mean_t, std_t, n_samples)
    print(f"  Hybrid: {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over {n_samples} samples "
          f"-> {mean_t/n_samples*1e9:.1f} ns/sample")

    # --- MMSE (Genie) -- only meaningful with a real multipath channel ---
    if sps > 1:
        mmse_channel_taps, mmse_peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=sps, num_taps=21)
        syms_noproc = receiver_frontend(noisy, h_rrc, sps=sps)
        isi_ref = receiver_frontend(rx_clean, h_rrc, sps=sps)
        noise_var = np.mean(np.abs(isi_ref - syms_noproc) ** 2)

        def run_mmse_design_and_apply():
            w, delay = design_mmse_equalizer(mmse_channel_taps, mmse_peak_idx, noise_var, num_taps=TAPS)
            apply_mmse_equalizer(syms_noproc, w, delay)
        mean_t, std_t = _time_repeated(run_mmse_design_and_apply)
        results["MMSE (Genie) [design+apply]"] = (mean_t, std_t, len(syms_noproc))
        print(f"  MMSE (design+apply): {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over "
              f"{len(syms_noproc)} symbols -> {mean_t/len(syms_noproc)*1e9:.1f} ns/symbol")

        # apply-only (the actual steady-state / deployed cost, design is a
        # one-time amortized setup cost per the task's own MMSE guidance)
        w, delay = design_mmse_equalizer(mmse_channel_taps, mmse_peak_idx, noise_var, num_taps=TAPS)

        def run_mmse_apply_only():
            apply_mmse_equalizer(syms_noproc, w, delay)
        mean_t, std_t = _time_repeated(run_mmse_apply_only)
        results["MMSE (Genie) [apply only]"] = (mean_t, std_t, len(syms_noproc))
        print(f"  MMSE (apply only)  : {mean_t*1e3:.3f} +/- {std_t*1e3:.3f} ms/call over "
              f"{len(syms_noproc)} symbols -> {mean_t/len(syms_noproc)*1e9:.1f} ns/symbol")

    return results, n_samples


# ===========================================================================
# 3. Cross-reference against the existing runtime_sec_mean columns
# ===========================================================================

def load_existing_runtime(case_csv: Path, n_samples_per_call: int):
    df = pd.read_csv(case_csv)
    grp = df.groupby("method")["runtime_sec_mean"].mean()
    per_sample_ns = (grp / n_samples_per_call) * 1e9
    return grp, per_sample_ns


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("  Compute-Cost Analysis: MACs/sample and measured latency, per method")
    print("=" * 78)

    # --- Analytical MACs ---
    print("\n[1] Analytical MAC-count derivation")
    lms_macs = lms_nlms_mmse_macs_per_sample(TAPS)
    print(f"  LMS / NLMS / MMSE FIR-apply: {TAPS} complex MACs/sample "
          f"x {REAL_MACS_PER_COMPLEX_MAC} real MACs/complex MAC = "
          f"{lms_macs['real_macs_per_sample']} real MACs/sample")

    cnn_layers, cnn_total_macs = cnn_macs_per_window(WINDOW_LEN)
    print(f"\n  CNN layer-by-layer MACs per {WINDOW_LEN}-sample window "
          f"(kernel_size x in_ch x out_ch x output_length, output_length={WINDOW_LEN} "
          f"for every layer since padding='same', stride=1):")
    for name, k, cin, cout, macs in cnn_layers:
        print(f"    {name:8s}: {k} x {cin:2d} x {cout:2d} x {WINDOW_LEN} = {macs:,} MACs")
    print(f"    TOTAL per window: {cnn_total_macs:,} real MACs")
    cnn_macs_per_sample_window_norm = cnn_total_macs / WINDOW_LEN
    cnn_macs_per_sample_stride_norm = cnn_total_macs / STRIDE
    print(f"  -> per-sample (divide by window_len={WINDOW_LEN}): "
          f"{cnn_macs_per_sample_window_norm:,.1f} real MACs/sample")
    print(f"  -> per-*unique-advancing*-sample (divide by stride={STRIDE}, "
          f"since windows overlap by (window_len-stride)/window_len = "
          f"{(WINDOW_LEN-STRIDE)/WINDOW_LEN:.0%} and every overlapping window is fully "
          f"recomputed): {cnn_macs_per_sample_stride_norm:,.1f} real MACs/sample")

    cnn_layers_p, cnn_total_params = cnn_params(WINDOW_LEN)
    print(f"\n  CNN analytical param count: {cnn_total_params:,} "
          f"(kernel weights + bias per layer, summed)")

    # Cross-check against the real Keras model
    model = build_autoencoder(window_len=WINDOW_LEN)
    keras_params = model.count_params()
    print(f"  Keras model.count_params(): {keras_params:,}  "
          f"({'MATCHES' if keras_params == cnn_total_params else 'MISMATCH'} analytical count)")

    hybrid_macs_window = lms_macs["real_macs_per_sample"] + cnn_macs_per_sample_window_norm
    hybrid_macs_stride = lms_macs["real_macs_per_sample"] + cnn_macs_per_sample_stride_norm
    print(f"\n  Hybrid = LMS + CNN (sequential): "
          f"{lms_macs['real_macs_per_sample']} + {cnn_macs_per_sample_window_norm:,.1f} = "
          f"{hybrid_macs_window:,.1f} real MACs/sample (window-normalized)")

    # --- Wall-clock benchmarks ---
    print("\n[2] Wall-clock benchmarks on this machine (CPU only, no GPU)")
    case1_results, case1_n = benchmark_case(sps=1, num_symbols=NUM_SYMBOLS, label="Case-Study-1-like (no ISI)")
    case2_results, case2_n = benchmark_case(sps=4, num_symbols=NUM_SYMBOLS, label="Case-Study-2-like (RRC+multipath ISI)")

    # --- Cross-reference against existing runtime_sec_mean ---
    print("\n[3] Cross-referencing against existing results_case*.csv runtime_sec_mean")
    case1_csv = TABLES_DIR / "results_case1_no_isi.csv"
    case2_csv = TABLES_DIR / "results_case2_isi.csv"
    existing1_abs, existing1_ns = load_existing_runtime(case1_csv, case1_n)
    existing2_abs, existing2_ns = load_existing_runtime(case2_csv, case2_n)
    print("\n  Case 1 CSV runtime_sec_mean (avg over modulation/SNR), per-sample ns:")
    for m in existing1_abs.index:
        print(f"    {m:15s}: {existing1_abs[m]:.4f}s/call -> {existing1_ns[m]:.1f} ns/sample")
    print("\n  Case 2 CSV runtime_sec_mean (avg over modulation/SNR), per-sample ns:")
    for m in existing2_abs.index:
        print(f"    {m:15s}: {existing2_abs[m]:.4f}s/call -> {existing2_ns[m]:.1f} ns/sample")

    # Persist everything needed for the write-up as a small results dict/DF,
    # printed to stdout in full (captured by the caller / redirected to file)
    # -- also stash a compact CSV-like summary for reproducibility.
    summary_rows = []
    for method, taps_or_params, macs_window, macs_stride in [
        ("LMS", TAPS, lms_macs["real_macs_per_sample"], lms_macs["real_macs_per_sample"]),
        ("NLMS", TAPS, lms_macs["real_macs_per_sample"], lms_macs["real_macs_per_sample"]),
        ("MMSE (Genie)", TAPS, lms_macs["real_macs_per_sample"], lms_macs["real_macs_per_sample"]),
        ("CNN", cnn_total_params, cnn_macs_per_sample_window_norm, cnn_macs_per_sample_stride_norm),
        ("Hybrid", TAPS + cnn_total_params, hybrid_macs_window, hybrid_macs_stride),
    ]:
        row = {
            "method": method,
            "taps_or_params": taps_or_params,
            "macs_per_sample_window_norm": round(macs_window, 1),
            "macs_per_sample_stride_norm": round(macs_stride, 1),
            "case1_latency_ns_per_sample_measured": None,
            "case2_latency_ns_per_sample_measured": None,
        }
        lookup_key = "MMSE (Genie) [apply only]" if method == "MMSE (Genie)" else method
        c1 = case1_results.get(lookup_key)
        if c1:
            row["case1_latency_ns_per_sample_measured"] = round(c1[0] / c1[2] * 1e9, 1)
        c2 = case2_results.get(lookup_key)
        if c2:
            row["case2_latency_ns_per_sample_measured"] = round(c2[0] / c2[2] * 1e9, 1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    base_lms = summary_df.loc[summary_df.method == "LMS", "macs_per_sample_window_norm"].iloc[0]
    summary_df["relative_macs_vs_lms"] = (summary_df["macs_per_sample_window_norm"] / base_lms).round(1)
    print("\n[4] Final summary table (also see report/findings_compute_cost.md)")
    print(summary_df.to_string(index=False))

    out_csv = TABLES_DIR / "compute_cost_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote {out_csv}")
    print("[OK] Compute-cost analysis complete.")

    return {
        "lms_macs": lms_macs,
        "cnn_layers": cnn_layers,
        "cnn_total_macs": cnn_total_macs,
        "cnn_total_params": cnn_total_params,
        "case1_results": case1_results,
        "case2_results": case2_results,
        "existing1_abs": existing1_abs,
        "existing1_ns": existing1_ns,
        "existing2_abs": existing2_abs,
        "existing2_ns": existing2_ns,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    main()
