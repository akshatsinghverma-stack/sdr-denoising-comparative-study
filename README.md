# SDR Denoising Project

[![tests](https://github.com/akshatsinghverma-stack/sdr-denoising-comparative-study/actions/workflows/tests.yml/badge.svg)](https://github.com/akshatsinghverma-stack/sdr-denoising-comparative-study/actions/workflows/tests.yml)

**Comparative Study of Classical Adaptive Filtering and Deep Learning Based
Denoising Techniques for SDR Communication Signals**

Compares five denoising/equalization approaches — LMS, NLMS, RLS, a 1-D CNN
autoencoder, and a Hybrid LMS→CNN cascade — against a No-Processing baseline
and a genie-aided MMSE reference, for BPSK/QPSK/16-QAM signals under AWGN,
multipath ISI, and time-varying fading. What began as two controlled case
studies grew, through follow-up questions each result raised, into four case
studies plus several connecting/diagnostic analyses — every addition was
triggered by a specific gap or untested claim the previous result left open.
See `report/report.md` for the full technical writeup, or
`report/SDR_Denoising_Project_Report.docx` for a formatted version with
figures, tables, and a table of contents.

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

# Case Study 3: time-varying channel -- does decision-directed tracking earn its keep?
python experiments/run_timevarying_channel.py

# Case Study 4: 16-QAM -- does decision-boundary crowding keep sharpening the effect?
python experiments/run_16qam_comparison.py

# Diagnostics / connecting analyses (each answers one specific open question):
python experiments/run_case1_boundary_loss.py          # boundary-aware CNN loss vs. plain MSE
python experiments/compute_cost_analysis.py             # MACs/sample + wall-clock benchmark
python experiments/run_rls_comparison.py                # RLS vs. LMS/NLMS on ISI
python experiments/diagnose_lms_nlms_asymmetry.py       # why NLMS doesn't benefit from DD tracking
python experiments/diagnose_cnn_high_snr_floor.py       # root-causing the high-SNR CNN error floor
python experiments/diagnose_training_variance.py        # test-time vs. training-draw BER variance
```

`experiments/run_all.py` is currently identical to `run_case1_no_isi.py` (kept
for backward compatibility).

## Project Structure

```
sdr_denoising_project/
├── requirements.txt              # Full project dependencies (incl. TensorFlow)
├── requirements-test.txt         # Lightweight deps for tests/ only (no TensorFlow -- CI uses this)
├── README.md                     # This file
├── .github/workflows/tests.yml   # CI: runs the regression suite on every push
├── src/
│   ├── signal_gen.py             # BPSK/QPSK/16-QAM/8-PSK generators (Gray-coded, optional RRC pulse shaping)
│   ├── channel.py                # AWGN, static multipath, and time-varying multipath channel models
│   ├── lms_filter.py             # LMS adaptive filter (preamble + optional decision-directed mode)
│   ├── nlms_filter.py            # NLMS adaptive filter (same interface as LMS)
│   ├── rls_filter.py             # RLS adaptive filter (frozen-by-default, matching LMS/NLMS's safe default)
│   ├── cnn_autoencoder.py        # 1-D CNN denoising autoencoder (plain MSE loss)
│   ├── cnn_boundary_aware.py     # Same architecture, boundary-hinge loss variant
│   ├── hybrid_model.py           # Hybrid LMS → CNN cascade
│   ├── mmse_equalizer.py         # Genie-aided linear MMSE equalizer (upper-bound reference)
│   ├── metrics.py                # SNR, BER (+ raw error/bit counts), MSE calculations
│   └── utils.py                  # Demapping, receiver frontend, and plotting helpers
├── experiments/
│   ├── run_case1_no_isi.py             # Case Study 1: memoryless AWGN
│   ├── run_case2_isi.py                # Case Study 2: RRC + multipath ISI, incl. genie MMSE
│   ├── run_severity_sweep.py           # ISI severity sweep (connects Case Studies 1 & 2)
│   ├── run_timevarying_channel.py      # Case Study 3: time-varying channel, DD tracking
│   ├── run_16qam_comparison.py         # Case Study 4: 16-QAM
│   ├── run_case1_boundary_loss.py      # Boundary-aware CNN loss follow-up
│   ├── compute_cost_analysis.py        # MACs/sample + wall-clock compute-cost analysis
│   ├── run_rls_comparison.py           # RLS vs. LMS/NLMS/MMSE on Case Study 2's channel
│   ├── diagnose_lms_nlms_asymmetry.py  # Root-causes LMS-vs-NLMS DD-tracking asymmetry
│   ├── diagnose_cnn_high_snr_floor.py  # Root-causes the loss-independent high-SNR CNN floor
│   ├── diagnose_training_variance.py   # Test-time vs. training-draw BER variance
│   └── run_all.py                      # Alias of run_case1_no_isi.py
├── tests/                        # Regression test suite (~50 tests, ~25s) -- see below
├── results/
│   ├── figures/                  # One subdirectory per case study/analysis
│   └── tables/                   # One or more CSVs per case study/analysis
└── report/
    ├── report.md                          # Full technical report (all case studies + analyses)
    ├── findings_*.md                       # Standalone deep-dive write-ups for each diagnostic
    ├── build_word_report.py                # Generates the formatted Word report
    └── SDR_Denoising_Project_Report.docx   # Formatted report: TOC, figures, tables, captions
```

## Methods Compared

| # | Method | Description |
|---|--------|-------------|
| 1 | **LMS**    | Least-Mean-Squares adaptive filter (16 taps, μ=0.01, preamble-trained then frozen) |
| 2 | **NLMS**   | Normalised LMS (16 taps, μ=0.5, ε=1e-6) |
| 3 | **RLS**    | Recursive Least Squares (16 taps, λ=0.99) — added as a third classical baseline on Case Study 2's channel |
| 4 | **CNN**    | 1-D Convolutional Denoising Autoencoder (6,890 parameters, verified via `model.count_params()`) |
| 5 | **Hybrid** | LMS (coarse) → retrained CNN (residual fine-tuning) |
| 6 | **MMSE (Genie)** | Closed-form linear MMSE equalizer given the *known* channel and noise level — an upper-bound reference, not a competing adaptive method |
| — | **No Processing** | Raw noisy signal → hard-decision demod (falsifiability baseline every other row is judged against) |

## Modulations & Channels

- **BPSK**, **QPSK** (Gray-coded), and **16-QAM** (Case Study 4), 100,000
  symbols/trial for the two main case studies (reduced to 15,000-30,000 for
  diagnostic-scoped follow-ups), 10 Monte Carlo trials (5 for diagnostics) per
  (modulation, SNR, method) — results reported as mean ± std, with a
  rule-of-three 95% upper bound for any cell with zero observed errors.
- **AWGN** (all case studies), **static multipath ISI** (Case Studies 2/4,
  the severity sweep), and **time-varying multipath / flat fading** (Case
  Study 3, sinusoidal and random-walk drift models).

## Key Findings

1. **SNR improvement does not imply BER improvement**, and whether any
   method helps at all depends jointly on whether the channel has structure
   to exploit and how sensitive the specific modulation's decision geometry
   is to the specific distortion that structure introduces. On a memoryless
   AWGN channel (Case Study 1), the raw received sample is already a
   sufficient statistic for symbol detection, so every method — LMS, NLMS,
   CNN, Hybrid — produces *strictly worse* BER than doing nothing, despite
   large apparent SNR gains (verified directly: the converged LMS filter's
   tap weights match the theoretical Wiener shrinkage factor).
2. **Add real inter-symbol interference (Case Study 2) and the result can
   reverse sharply** — QPSK wins by up to ~470x with CNN — but BPSK barely
   benefits and classical filters are frequently *worse* than doing nothing
   even with ISI present. The ISI severity sweep turns this into an actual
   crossover curve rather than two disconnected snapshots.
3. **A genuinely time-varying channel (Case Study 3) partially refutes this
   project's own most load-bearing design decision**: LMS decision-directed
   tracking earns back a real, substantial BER improvement once the channel
   actually moves — refuting the "nothing to track on a static channel"
   justification as a general principle, even though "freeze by default"
   remains the right default today for an unrelated reason (a decision-
   directed implementation bug at SPS>1). NLMS does *not* show the same
   benefit — root-caused (not just observed) via an instrumented diagnostic:
   NLMS's instantaneous-power step-size normalization measurably inflates
   its effective step size during channel fades, in 15/15 trials tested,
   exactly when wrong decisions are most likely; LMS's fixed step size does
   the opposite in 16/16 trials.
4. **16-QAM (Case Study 4) confirms decision-boundary crowding keeps
   sharpening the effect** — equalization wins by up to 610x, No-Processing
   develops a hard, un-closeable BER floor that doesn't exist for BPSK/QPSK,
   and the genie MMSE's linear-equalizer ceiling turns from a curiosity into
   an outright regression below doing nothing.
5. **Two previously-open questions were resolved with the same instrumented-
   diagnostic-plus-significance-test methodology**: a boundary-aware CNN
   loss confirmed "MSE is boundary-blind" as a real, partial, modulation-
   dependent cause of Case Study 1's headline result; and a separate,
   loss-independent high-SNR CNN error floor was root-caused to the CNN's
   window/overlap-add reconstruction (81.1% same-position overlap between
   loss functions, χ²=168.0 p=3.5×10⁻³⁶ non-uniform clustering by window
   phase), ruling out "unfixable noise" as the primary cause.
6. **Compute cost and BER cost do not move together.** CNN needs ~106-212x
   more raw arithmetic than LMS/NLMS/MMSE per sample, yet is *faster* in
   wall-clock terms on this project's CPU-only stack — an implementation-
   efficiency artifact, not an algorithmic one; the analytical MAC gap
   reasserts itself as a genuine microcontroller-vs-applications-processor
   deployment question.
7. **Six real bugs were found and fixed** during this project (LMS
   step-size/divergence, a receiver matched-filter gain bug, an SNR
   calibration bug, a multipath-convolution causality bug, a
   decision-directed reliability gate that was silently dead code at
   SPS>1, and — found via a deliberate adversarial self-critique pass late
   in the project — an asymmetric RRC pulse-shaping filter), every one
   caught by checking a value against what it must equal by definition. A
   55-test regression suite (`tests/`), run automatically on every push via
   GitHub Actions, now codifies all of them.
8. **A four-agent adversarial self-critique pass** (statistical rigor, DSP
   correctness, code quality, and a skeptical outside reader, each briefed
   independently) surfaced and led to fixes for: the RRC filter bug above,
   a genuine reporting error in this project's own LMS-vs-NLMS diagnostic
   (undefined ratios were being counted as "not inflated"), several
   "confirmed"/"root-caused" claims that were correlational rather than
   interventional evidence and have been reworded accordingly, an
   uncorrected-multiple-comparisons gap in Case Study 2's significance
   testing, and a channel-model design claim ("looks static during the
   preamble") that was asserted but never measured and turned out to be
   false for the actual parameters used (see
   `report/findings_preamble_drift_correction.md`).

See `report/report.md` for the full mechanism and verification behind each
finding, and the `report/findings_*.md` files for the deep-dive write-up
behind each diagnostic.

## Assumptions & Limitations

1. **LMS/NLMS/RLS use one fixed step size/forgetting-factor across the whole
   SNR sweep.** This is the direct cause of a small, quantifiable SNR
   shortfall at high SNR in Case Study 1 (matches the textbook LMS
   misadjustment formula almost exactly) — per-SNR tuning is a natural next
   step.
2. **Single static multipath channel** in Case Study 2's main run — partially
   addressed by the severity sweep (varies magnitude, not delay spread or tap
   count).
3. **Small CNN** (6,890 parameters, 3 encoder + 3 decoder Conv1D layers) —
   deeper/wider architectures were not explored.
4. **CPU-only training**, TensorFlow with EarlyStopping.
5. **The theoretical Q-function BER curve** plotted on BER-vs-SNR figures is
   only valid for Case Study 1 (memoryless AWGN); it has been removed
   (not just captioned) from every ISI-channel figure, since it is not a
   bound there and a figure viewed without its caption could otherwise
   mislead.
6. **The genie MMSE equalizer is a linear, symbol-spaced reference bound**,
   not a universal ceiling.
7. **The decision-directed update rule is not yet fixed for SPS>1** — it
   remains off by default everywhere in this project, which is why this
   limitation hasn't affected any reported result, but it blocks safely
   testing DD tracking on this project's actual (pulse-shaped) ISI channels.
8. Named, scoped-out future work (not attempted this round, each for a
   specific stated reason): MLSE/Viterbi equalization, a full 8-PSK pipeline
   run, cycle-accurate hardware validation of the compute-cost analysis, and
   fixing the SPS>1 decision-directed update rule itself — see
   `report/report.md` Section 11 for the complete list and reasoning.

## Requirements

- Python 3.10+
- Full project: numpy, scipy, matplotlib, scikit-learn, tensorflow, pandas, pytest, python-docx
- Tests only (`requirements-test.txt`, used by CI): numpy, scipy, matplotlib, scikit-learn, pandas, pytest — no TensorFlow, since no test imports the CNN/Hybrid modules

## License

Academic project — not licensed for redistribution.
