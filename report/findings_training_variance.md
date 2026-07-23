# Findings: Is Training the CNN Once Per Case Study a Real Limitation?

**Status**: quantifies a previously-named-but-unmeasured limitation. report.md's Named,
Scoped-Out Future Work: "Monte Carlo trials vary the *test* noise and bit sequence; the
CNN/LMS/Hybrid models themselves were each trained exactly once per modulation per case study.
The variance contributed by the training draw itself ... is currently unknown." New file only:
`experiments/diagnose_training_variance.py`. No existing module modified.

## Method

Reproduces Case Study 1's memoryless-AWGN setup (reduced scale for speed: 15,000 symbols,
BPSK/QPSK, SNR ∈ {0, 15}dB representative of the moderate and high-SNR-floor regimes). Trains
**5 independently-seeded CNNs** per modulation (different training-data shuffle order and
TensorFlow weight-initialization seed per draw), then evaluates **every** draw on the exact
**same 4 fixed test signals** — so any BER difference between draws is attributable only to the
training process itself, not to different test data. This decomposes the result into:

- **Within-draw variance**: BER spread across the 4 test trials for one fixed trained model —
  what this project's existing Monte Carlo methodology already measures and reports as ±std.
- **Between-draw variance**: spread of each draw's own mean BER across the 5 independent training
  runs — the previously-unmeasured quantity.

## Results

| | Per-draw mean BER (5 draws) | Within-draw std | Between-draw std | Ratio (between/within) |
|---|---|---|---|---|
| BPSK, 0dB | 0.0820, 0.0825, 0.0825, 0.0830, 0.0832 | 3.29e-3 | 4.51e-4 | **0.14x** |
| BPSK, 15dB | 0.000817 (all 5 draws identical) | 6.38e-5 | 0.0 | **0.00x** |
| QPSK, 0dB | 0.1611, 0.1609, 0.1609, 0.1615, 0.1622 | 1.68e-3 | 5.39e-4 | **0.32x** |
| QPSK, 15dB | 0.000833, 0.000833, 0.000858, 0.000833, 0.000833 | 9.60e-5 | 1.12e-5 | **0.12x** |

**In all four conditions tested, between-draw (training-seed) variance is smaller than
within-draw (test-time) variance — by 3x to effectively infinity.** At 15dB, BPSK's five
independently-trained models produced the *exact same* mean BER to 6 decimal places; QPSK's five
were within 3% of each other. At 0dB, the noisier regime, the ratio is largest (0.32x for QPSK)
but still well under 1 — training-draw variance never dominates or even matches test-time
variance in any condition tested.

## Conclusion

**This project's practice of training the CNN once per modulation per case study, and reporting
only test-time Monte Carlo variance, is justified — not an underestimate of true uncertainty.**
The training process (weight initialization, data shuffle order) is a substantially smaller
source of BER variation than ordinary test-time noise/bit-sequence randomness, for this
architecture and training procedure. This is also consistent with, and reinforces, the separate
high-SNR-floor finding (`report/findings_cnn_high_snr_floor.md`): at 15dB, independently-trained
models converge to nearly identical behavior, consistent with that floor being a structural
property of the window/overlap-add reconstruction architecture rather than something training
randomness could shift. This does not extend the finding to Hybrid or to Case Study 2's ISI
channel (out of scope for this diagnostic's reduced runtime budget), which remain untested for
training-draw variance specifically.
