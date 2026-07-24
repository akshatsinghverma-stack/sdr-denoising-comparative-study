# SDR Denoising Project

[![tests](https://github.com/akshatsinghverma-stack/sdr-denoising-comparative-study/actions/workflows/tests.yml/badge.svg)](https://github.com/akshatsinghverma-stack/sdr-denoising-comparative-study/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: academic use](https://img.shields.io/badge/license-academic--use-lightgrey)](#license)

**Comparative Study of Classical Adaptive Filtering and Deep Learning Based
Denoising Techniques for SDR Communication Signals**

Compares five denoising/equalization approaches — LMS, NLMS, RLS, a 1-D CNN
autoencoder, and a Hybrid LMS→CNN cascade — against a No-Processing baseline
and a genie-aided MMSE/Zero-Forcing reference, for BPSK/QPSK/16-QAM signals
under AWGN, multipath ISI, and time-varying fading. What began as two
controlled case studies grew, through follow-up questions each result
raised, into four case studies plus several connecting/diagnostic analyses —
every addition was triggered by a specific gap or untested claim the
previous result left open, and two full rounds of adversarial self-critique
(four independent reviewer agents each) found and fixed real bugs rather
than just polishing prose.

📄 **[Full technical report](report/report.md)** · 📝 **[Formatted Word report](report/SDR_Denoising_Project_Report.docx)** (abstract, TOC, figures, tables, references) · 🔍 **[Deep-dive findings](report/)** (one file per diagnostic)

**Author:** [Akshat Singh Verma](https://github.com/akshatsinghverma-stack) — B.Tech, Artificial Intelligence & Machine Learning, UPES

## Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methods Compared](#methods-compared)
- [Modulations & Channels](#modulations--channels)
- [Key Findings](#key-findings)
- [Assumptions & Limitations](#assumptions--limitations)
- [Requirements](#requirements)

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
python experiments/run_zf_comparison.py                 # Zero-Forcing vs. MMSE on ISI
python experiments/diagnose_lms_nlms_asymmetry.py       # why NLMS doesn't benefit from DD tracking
python experiments/test_nlms_floor_intervention.py      # falsification test for the NLMS mechanism above
python experiments/test_nlms_floor_sensitivity.py       # sensitivity sweep for that intervention's floor value
python experiments/diagnose_cnn_high_snr_floor.py       # root-causing the high-SNR CNN error floor
python experiments/test_cnn_overlap_weighting_intervention.py  # tests a fix for the floor above (negative result)
python experiments/test_cnn_channel_generalization.py   # does the CNN generalize across channels, or memorize one?
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
│   ├── run_zf_comparison.py            # Zero-Forcing vs. MMSE on Case Study 2's channel
│   ├── diagnose_lms_nlms_asymmetry.py  # Root-causes LMS-vs-NLMS DD-tracking asymmetry
│   ├── test_nlms_floor_intervention.py # Falsification test for the NLMS mechanism above
│   ├── test_nlms_floor_sensitivity.py  # Sensitivity sweep for the floor value used above
│   ├── diagnose_cnn_high_snr_floor.py  # Root-causes the loss-independent high-SNR CNN floor
│   ├── test_cnn_overlap_weighting_intervention.py  # Tests a fix for the floor above (negative result)
│   ├── test_cnn_channel_generalization.py  # Does the CNN generalize across channels, or memorize one?
│   ├── diagnose_training_variance.py   # Test-time vs. training-draw BER variance
│   └── run_all.py                      # Alias of run_case1_no_isi.py
├── tests/                        # Regression test suite (68 tests, ~2.5min) -- see below
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
   CNN, Hybrid — is *never better* than doing nothing (strictly worse in
   52/56 comparisons, tied in the remaining 4 — CNN specifically, at the
   two highest SNRs, once its own reconstruction bug was fixed, see finding
   #10), despite large apparent SNR gains (verified directly: the converged
   LMS filter's tap weights match the theoretical Wiener shrinkage factor).
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
   sharpening the effect** — equalization wins by up to 610x, and
   No-Processing develops a hard, un-closeable BER floor that doesn't exist
   for BPSK/QPSK. (An earlier version of this finding also claimed the genie
   MMSE ceiling turns into an outright regression here — that was this
   project's eighth real bug, corrected: see finding #12 below. The
   properly-computed genie MMSE bound stays close to optimal at every
   modulation tested, not a regression at any of them.)
5. **Two previously-open questions were resolved with the same instrumented-
   diagnostic-plus-significance-test methodology**: a boundary-aware CNN
   loss confirmed "MSE is boundary-blind" as a real, partial, modulation-
   dependent cause of Case Study 1's headline result; and a separate,
   loss-independent high-SNR CNN error floor was *first* attributed (on
   strong-looking correlational grounds — 81.1% same-position overlap
   between loss functions, χ²=168.0 p=3.5×10⁻³⁶ non-uniform clustering by
   window phase) to the CNN's window/overlap-add reconstruction — **this
   specific conclusion was later found to be wrong; see finding #10 below
   for the correction.**
6. **Compute cost and BER cost do not move together.** CNN needs ~106-212x
   more raw arithmetic than LMS/NLMS/MMSE per sample, yet is *faster* in
   wall-clock terms on this project's CPU-only stack — an implementation-
   efficiency artifact, not an algorithmic one; the analytical MAC gap
   reasserts itself as a genuine microcontroller-vs-applications-processor
   deployment question.
7. **Eight real bugs were found and fixed** during this project (LMS
   step-size/divergence, a receiver matched-filter gain bug, an SNR
   calibration bug, a multipath-convolution causality bug, a
   decision-directed reliability gate that was silently dead code at
   SPS>1, an asymmetric RRC pulse-shaping filter, a CNN window-
   reconstruction bug that silently zero-filled any trailing samples no
   window covered, read by a hard-decision demodulator as a fixed, wrong-
   half-the-time bit, and — found via a final code-correctness critique
   pass — a conjugate error in the genie MMSE/ZF equalizer's autocorrelation
   computation, `np.correlate` silently returning the conjugate of the value
   its own Wiener-Hopf formula needed), every one caught by checking a value
   against what it must equal by definition. A 68-test regression suite
   (`tests/`), run automatically on every push via GitHub Actions, now
   codifies all of them.
8. **A four-agent adversarial self-critique pass** (statistical rigor, DSP
   correctness, code quality, and a skeptical outside reader, each briefed
   independently) surfaced and led to fixes for: the RRC filter bug above,
   a genuine reporting error in this project's own LMS-vs-NLMS diagnostic
   (undefined ratios were being counted as "not inflated"), an
   uncorrected-multiple-comparisons gap in Case Study 2's significance
   testing (now corrected with both Bonferroni and Benjamini-Hochberg), and
   a channel-model design claim ("looks static during the preamble") that
   was asserted but never measured and turned out to be false for the
   actual parameters used (see
   [report/findings_preamble_drift_correction.md](report/findings_preamble_drift_correction.md)).
9. **Two named interventions were then actually run, not left as
   correlational guesses** — with one honest negative result and one real
   correction. Flooring NLMS's normalization denominator measurably closed
   its decision-directed disadvantage on the time-varying channel,
   confirming that mechanism by direct test. Reweighting the CNN's
   overlap-add reconstruction did **not** close the high-SNR error floor —
   reported as a negative result rather than omitted (and, per finding #10
   below, this result turned out to be correct for a different reason than
   originally thought: it couldn't have worked, since the floor it was
   targeting was mostly a bug reweighting has no power to fix). Separately, RLS
   (Section 4.8 in [report/report.md](report/report.md)) was re-evaluated after adding
   Polyak/Ruppert tail-averaging (the same hardening LMS/NLMS already had):
   the corrected result reverses this project's own first-pass finding —
   RLS now matches or beats LMS/NLMS at *every* SNR level tested, not just
   at high SNR — after an initial comparison was caught using a stale
   results file and corrected. A missing standard baseline, Zero-Forcing
   (ZF), was also added after comparing against published equalization
   literature rather than re-reading this project's own code.
10. **A code-correctness critique pass found that the "confirmed" high-SNR
    CNN error floor (finding #9) was itself mostly a bug, not a genuine
    reconstruction artifact.** `reconstruct_from_windows` silently
    zero-filled trailing samples no window covered; fixing it collapsed
    BPSK's floor (a constant 83/100,000 errors at 10-20dB) to 0-1/100,000,
    and QPSK's floor at 15-20dB (161/200,000) to 0/200,000. The statistically
    significant positional-clustering evidence (χ²=168.0, p=3.5×10⁻³⁶) that
    originally supported the reconstruction-artifact conclusion recomputes to
    χ²≈0.74 (p≈0.86, indistinguishable from uniform) on the corrected error
    population — the original result was manufactured by the bug. A smaller,
    genuine residual floor survives only at QPSK 10dB, now better explained
    by high-noise-magnitude samples than by reconstruction position. Case
    Study 1's own headline results table was also re-run with the fix: CNN's
    floor at BPSK/QPSK 15-20dB (previously 1.51e-4/1.71e-4) is now exactly 0,
    tying No-Processing's already-zero count rather than falling short of it
    — see finding #1's corrected wording above. See
    [report/findings_cnn_high_snr_floor.md](report/findings_cnn_high_snr_floor.md).
11. **An outside-reviewer critique pass — deliberately unfamiliar with this
    project's history, briefed to find the single biggest weakness a viva
    examiner would raise — found a gap none of the prior self-critique
    rounds had named**: every CNN result trains and tests on the exact same
    known channel, while LMS/NLMS/MMSE/ZF all re-estimate or are handed the
    channel per-trial regardless. Testing this directly confirmed a real,
    substantial effect: a CNN frozen after training on one channel degrades
    up to 129.71x on a held-out channel with a different tap-phase structure
    (QPSK, 10dB), while LMS and NLMS on the identical held-out channel
    degrade only 1.17x and 1.41x. This revises, rather than overturns, the
    project's practical recommendation — the CNN's advantage is real when
    training and deployment channels match, but shouldn't be assumed to
    survive a channel that drifts from that training distribution. See
    [report/findings_cnn_channel_generalization.md](report/findings_cnn_channel_generalization.md).
12. **A final code-correctness critique pass found an eighth real bug that had produced a wrong
    conclusion backed by a real, honest, three-part investigation, not just a correlational
    guess.** `design_mmse_equalizer` (`src/mmse_equalizer.py`) computed its autocorrelation via
    `np.correlate`, which silently returns the complex conjugate of the value its own documented
    Wiener-Hopf formula requires — negligible for BPSK, severe for QPSK/16-QAM. This was the actual
    cause of two previously-reported findings: "QPSK genie MMSE gets worse at high SNR" (Case Study
    2) and "16-QAM genie MMSE regresses below doing nothing" (Case Study 4) — both previously
    investigated with real phase-bias, tap-count, and frequency-response checks that were each
    individually correct but collectively missed the one place the bug lived. Fixed and verified
    against a from-scratch ground-truth Wiener-Hopf solution; re-running Case Study 2, the severity
    sweep, Case Study 4, and the ZF comparison at full scale reverses both findings completely —
    genie MMSE now ties or beats every method at high SNR for QPSK and 16-QAM, exactly what a genie
    with perfect channel and noise knowledge should do.

See [report/report.md](report/report.md) for the full mechanism and verification behind each
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
   [report/report.md](report/report.md) Section 11 for the complete list and reasoning.

## Requirements

- Python 3.10+
- Full project: numpy, scipy, matplotlib, scikit-learn, tensorflow, pandas, pytest, python-docx
- Tests only (`requirements-test.txt`, used by CI): numpy, scipy, matplotlib, scikit-learn, pandas, pytest — no TensorFlow, since no test imports the CNN/Hybrid modules

## License

Academic project — not licensed for redistribution.
