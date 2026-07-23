# SDR Denoising Project

**Comparative Study of Classical Adaptive Filtering and Deep Learning Based
Denoising Techniques for SDR Communication Signals**

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the regression test suite (~25s) -- codifies every bug found in this project
pytest tests/

# Case Study 1: memoryless AWGN channel, no ISI (~8 min on CPU)
python experiments/run_case1_no_isi.py

# Case Study 2: RRC pulse shaping + multipath ISI channel, incl. genie MMSE (~35 min on CPU)
python experiments/run_case2_isi.py

# ISI severity sweep: locates the crossover between Case 1 and Case 2 (~10 min on CPU)
python experiments/run_severity_sweep.py
```

`experiments/run_all.py` is currently identical to `run_case1_no_isi.py` (kept
for backward compatibility). See `report/report.md` for why this project is
organized as two case studies plus a connecting sweep rather than one
pipeline, and Section 6 of that report for the reasoning that ties them
together.

## Project Structure

```
sdr_denoising_project/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── signal_gen.py         # BPSK/QPSK signal generators (with optional RRC pulse shaping)
│   ├── channel.py            # AWGN + multipath (ISI) channel models (causal convolution)
│   ├── lms_filter.py         # LMS adaptive filter (preamble + optional decision-directed mode)
│   ├── nlms_filter.py        # NLMS adaptive filter (same interface as LMS)
│   ├── rls_filter.py         # RLS adaptive filter (implemented, not yet wired into either case study)
│   ├── cnn_autoencoder.py    # 1-D CNN denoising autoencoder
│   ├── hybrid_model.py       # Hybrid LMS → CNN cascade
│   ├── mmse_equalizer.py     # Genie-aided linear MMSE equalizer (upper-bound reference, Case Study 2)
│   ├── metrics.py            # SNR, BER (+ raw error/bit counts), MSE calculations
│   └── utils.py              # Demapping & plotting helpers (with error-bar support)
├── experiments/
│   ├── run_case1_no_isi.py   # Case Study 1 pipeline
│   ├── run_case2_isi.py      # Case Study 2 pipeline (incl. genie MMSE)
│   ├── run_severity_sweep.py # ISI severity sweep (diagnostic, connects the two case studies)
│   └── run_all.py            # Alias of run_case1_no_isi.py
├── tests/                    # Regression test suite (35 tests, ~25s) -- see below
├── results/
│   ├── figures/
│   │   ├── case1_no_isi/     # Case Study 1 plots
│   │   ├── case2_isi/        # Case Study 2 plots
│   │   └── severity_sweep/   # Crossover plots
│   └── tables/
│       ├── results_case1_no_isi.csv
│       ├── results_case2_isi.csv
│       └── results_severity_sweep.csv
└── report/
    └── report.md             # Full report — both case studies, the severity sweep, and the MMSE bound
```

## Methods Compared

| # | Method | Description |
|---|--------|-------------|
| 1 | **LMS**    | Least-Mean-Squares adaptive filter (16 taps, μ=0.01, preamble-trained then frozen) |
| 2 | **NLMS**   | Normalised LMS (16 taps, μ=0.5, ε=1e-6) |
| 3 | **CNN**    | 1-D Convolutional Denoising Autoencoder (~15k params) |
| 4 | **Hybrid** | LMS (coarse) → retrained CNN (residual fine-tuning) |
| 5 | **MMSE (Genie)** | Closed-form linear MMSE equalizer given the *known* channel and noise level (Case Study 2 only — an upper-bound reference, not a competing adaptive method) |
| — | **No Processing** | Raw noisy signal → hard-decision demod (falsifiability baseline) |

## Modulations & SNR Range

- **BPSK** and **QPSK** (Gray-coded), 100,000 symbols per trial, 10 Monte Carlo
  trials per (modulation, SNR, method) — results reported as mean ± std, with
  a rule-of-three 95% upper bound reported for any cell with zero observed
  errors (a bare `0.0` would misleadingly read as "proven perfect").
- AWGN channel, SNR sweep: −10, −5, 0, +5, +10, +15, +20 dB.

## Outputs

- `results/tables/results_case1_no_isi.csv`, `results_case2_isi.csv`, `results_severity_sweep.csv`
- `results/figures/case1_no_isi/`, `case2_isi/`, `severity_sweep/` — BER curves
  (with error bars; theoretical AWGN reference only on Case 1, deliberately
  removed from Case 2 since it isn't a valid bound there), SNR curves,
  constellations, convergence, loss, and the severity-crossover plots
- `report/report.md` — full report: methodology, both case studies' results,
  the severity sweep, the genie MMSE bound, the key findings connecting them,
  and honest limitations/future work

## Key Findings

1. **SNR improvement does not imply BER improvement**, and whether adaptive
   filtering/denoising helps BER at all depends on whether the channel has
   structure to exploit. On a memoryless AWGN channel (Case Study 1), the raw
   received sample is already a sufficient statistic for symbol detection, so
   every method — LMS, NLMS, CNN, Hybrid — produces *strictly worse* BER than
   doing nothing, despite large apparent SNR gains. Add real inter-symbol
   interference (Case Study 2) and QPSK's result reverses sharply (up to
   ~470x fewer errors with CNN); BPSK, whose decision boundary is far more
   forgiving of this channel's distortion, barely benefits and classical
   filters are frequently *worse* than doing nothing even with real ISI
   present — the same channel, radically different outcomes, purely a
   function of decision-boundary geometry.
2. **The ISI severity sweep turns that into an actual crossover curve**
   (rather than two disconnected snapshots): QPSK at 10dB goes from
   "equalization hurts" (LMS 0.52x at zero ISI) through breakeven (~severity
   0.5) to "equalization helps substantially" (2.68x at full severity).
3. **The SNR-vs-BER disconnect recurs three times, via three different
   mechanisms** — verified each time via held-out generalization testing and
   direct decision-margin/phase inspection, never assumed: boundary-blind MSE
   training (Case Study 1), structured non-Gaussian residual dragging down an
   average SNR metric (Case Study 2 CNN), and phase-structured residual error
   from a linear equalizer's noise-enhancement/residual-ISI tradeoff (the
   genie MMSE bound, which a nonlinear CNN demonstrably exceeds at high SNR).
4. **Three real bugs were found and fixed** during this project (an LMS
   step-size/divergence bug, a receiver matched-filter gain bug, an SNR
   calibration bug, and a multipath-convolution causality bug — four, not
   three, if counted individually; see report Section 4.2 for the full
   account), every one caught by checking a value against what it must
   equal by definition rather than accepting an odd number as a quirk. A
   35-test regression suite (`tests/`) now codifies all of them.

See `report/report.md` for the full mechanism and verification behind each.

## Assumptions & Limitations

1. **LMS/NLMS use one fixed μ across the whole SNR sweep.** This is the direct
   cause of a small (~0.2-0.3dB), quantifiable SNR shortfall at high SNR in
   Case Study 1 (matches the textbook LMS misadjustment formula almost
   exactly) — per-SNR μ tuning is the natural next step.
2. **RLS is implemented (`src/rls_filter.py`) but not yet wired into either
   case study**, to keep exactly one variable (the channel) different between
   Case Study 1 and 2. Named explicitly as future work, along with MLSE/Viterbi
   equalization, a genuinely time-varying channel, and multiple independent
   training draws — see report Section 8.
3. **Single static multipath channel** in Case Study 2's main run — partially
   addressed by the severity sweep (varies magnitude, not delay spread or tap count).
4. **Small CNN** (~15k parameters, 3 encoder + 3 decoder Conv1D layers) —
   deeper/wider architectures were not explored.
5. **CPU-only training**, TensorFlow with EarlyStopping.
6. **The theoretical Q-function BER curve** plotted on BER-vs-SNR figures is
   only valid for Case Study 1 (memoryless AWGN); it has been removed
   (not just captioned) from Case Study 2's figures, since it is not a bound
   there and a figure viewed without its caption could otherwise mislead.
7. **The genie MMSE equalizer is a linear, symbol-spaced reference bound**,
   not a universal ceiling — see report Section 4.7 before citing it as
   "the best possible."

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib, scikit-learn, tensorflow, pandas, pytest, python-docx

## License

Academic project — not licensed for redistribution.
