# Findings: Higher-Order Modulation (16-QAM) — Does Decision-Boundary Crowding Sharpen the Equalization Story?

**Status**: diagnostic follow-up testing whether the BPSK-vs-QPSK decision-boundary-crowding
hypothesis (report.md Section 4.6-4.7) generalizes to a more crowded constellation. New files
only: `src/signal_gen.py` (`generate_16qam`, `generate_8psk`, additions), `src/utils.py`
(`demod_16qam_fast`, `demod_8psk_fast`, additions), `experiments/run_16qam_comparison.py`,
`results/tables/results_16qam.csv`, `results/figures/higher_order_modulation/`. No existing
function was modified.

## Hypothesis being tested

Report.md Section 4.6 found BPSK (180° decision margin) barely benefits from equalization on
Case Study 2's multipath channel, while QPSK (90° margin) benefits by 1-3 orders of magnitude —
attributed to decision-boundary crowding. 16-QAM (16 tightly-packed constellation points, much
smaller minimum distance between adjacent points) should sharpen this further if the hypothesis
is right.

## Implementation and self-test

`generate_16qam`: Gray-coded square 16-QAM (4 bits/symbol, I/Q levels {-3,-1,1,3}/√10 for unit
average power), following the exact pattern of `generate_bpsk`/`generate_qpsk`. `demod_16qam_fast`:
vectorized nearest-point decision, mirroring `demod_qpsk_fast`'s style. `generate_8psk`/
`demod_8psk_fast` were also built as a bonus (self-tested but not run through the full pipeline
this round).

**Self-test caught a real bug before any pipeline run**: a zero-noise, zero-ISI round-trip
(generate → pulse-shape → matched-filter → demod, no channel/noise) gave exactly 0 bit errors for
16-QAM (20,000 bits) confirming correctness, and constellation average power measured ≈1.0 as
expected. The same test on the bonus 8-PSK implementation initially failed an explicit Gray-code
adjacency check (adjacent constellation points must differ by exactly 1 bit) — the first mapping
used an encode step where a decode step was needed. Fixed and reverified: all 8 adjacent
transitions now differ by exactly 1 bit. This is exactly the kind of check this project's history
(report.md Section 4.2) treats as mandatory before trusting a new signal path.

## Experiment

`experiments/run_16qam_comparison.py`: 25,000 symbols, 5 Monte Carlo trials, Case Study 2's exact
multipath channel (`[1.0, 0.4+0.3j, -0.1+0.1j]`, sps=4 RRC pulse shaping) and the same empirical
SNR-gain calibration procedure used throughout Case Study 2 (not skipped). **SNR sweep shifted to
[0, 10, 20, 25, 30]dB** rather than reusing BPSK/QPSK's [-10..20]dB range — 16-QAM's tight minimum
distance means it needs substantially more SNR headroom before ISI effects are even visible above
the noise floor; the original range would have been almost entirely uninformative (all methods
near 50% BER).

## Results: BER improvement ratio (NoProc BER / Method BER), verified against raw CSV counts

| Method | BPSK best (report §4.6) | QPSK best (report §4.6) | 16-QAM (this run) |
|---|---|---|---|
| LMS | never wins (≤0.7x anywhere) | 31.5x (15dB) | break-even at 0dB (0.88x), **480x at 25dB** |
| NLMS | never wins | 43.0x (15dB) | break-even at 0dB (0.88x), **610x at 25dB** |
| CNN | 1.8x (10dB, best case) | 473x (15dB) | **557x at 25dB** |

Exact counts at 25dB (500,000 bits per cell): No-Processing 38,408 errors (BER 0.0768); LMS 80
errors (0.00016, 480x better); NLMS 63 errors (0.000126, 610x better); CNN 69 errors (0.000138,
557x better).

At low-to-moderate SNR (0-10dB), LMS/NLMS remain break-even-or-slightly-worse for 16-QAM too
(0.86-0.91x at 0dB) — the same qualitative pattern seen for BPSK throughout this project, before
enough SNR headroom exists for equalization to pay off.

## Two findings beyond the headline ratio

1. **No-Processing hits a genuinely new, hard BER floor**: 0.0755-0.0814 from 20-30dB, never
   improving further with more SNR — a floor not seen for BPSK or QPSK in this project. Deterministic
   ISI alone is now enough to permanently cross 16-QAM's tight decision margins regardless of how
   clean the noise gets; there is no SNR high enough to rescue the raw, unequalized receiver.
2. **The genie MMSE linear equalizer becomes actively worse than doing nothing at high SNR**:
   0.71-0.74x at 20-30dB (109,654/500,000 errors at 20dB vs. No-Processing's 40,713) — a sharper,
   more dramatic version of report.md Section 4.7's linear-equalizer-ceiling finding. With 16
   crowded constellation points, the same noise-enhancement/residual-ISI tradeoff that only
   modestly hurt QPSK's MMSE bound at high SNR is severe enough here to make the "perfect channel
   knowledge, linear-only" equalizer a net loss relative to raw hard-decision demod.

## Conclusion

**The decision-boundary-crowding hypothesis is confirmed, and more dramatically than the
BPSK→QPSK comparison alone predicted** — but the mechanism is richer than just "a bigger
improvement ratio." Crowding does two additional things: it creates a hard, un-closeable BER floor
for No-Processing that doesn't exist for BPSK/QPSK, and it turns the linear-equalizer ceiling
(previously a modest QPSK-specific curiosity) into an outright regression below doing nothing.
Both are consistent with, and extend, this project's core finding that a channel's practical
impact depends jointly on the channel's structure and the modulation's decision geometry — 16-QAM
is simply the sharpest lens on that relationship tested so far.
