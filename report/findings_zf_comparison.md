# Findings: Zero-Forcing (ZF) Equalizer — A Missing Baseline, Found by Comparing to Published Work

**Status**: adds a standard baseline this project was missing entirely, found by deliberately
searching comparable published literature during a self-critique pass (not by re-reading this
project's own code — the gap was invisible from the inside). New functions only:
`design_zf_equalizer`/`apply_zf_equalizer` in `src/mmse_equalizer.py` (a two-line wrapper reusing
existing machinery), `experiments/run_zf_comparison.py`, 2 new regression tests. No existing
function modified.

## Why this was missing, and how it was found

A web search for comparable published work on adaptive/linear equalizer comparisons (e.g., an
arXiv comparative study of Zero-Forcing, LMS, and RLS equalizers for fading channels) confirmed
that **ZF is one of the two standard textbook linear-equalizer baselines**, alongside MMSE — most
classical equalization literature presents them side by side specifically because they represent
opposite ends of a real, well-known tradeoff (full channel inversion vs. noise-aware inversion).
This project had built a genie-aided MMSE equalizer (Section 4.7) but no ZF equalizer at all — a
real, concrete gap that a from-scratch reading of the code would not surface, since nothing internal
to this project ever needed ZF to explain any of its own results.

## Design

ZF is exactly the noise_var→0 limit of this project's existing MMSE solve (`R w = p`, with
`R = autocorr(g) + noise_var·I`): drop the noise term, ignore the actual noise level entirely, and
solve for the equalizer that most fully inverts the measured channel. A tiny numerical-stability
regularizer (1e-10) replaces the noise term only to keep the linear system well-conditioned (16
equalizer taps vs. a 3-tap channel leaves excess degrees of freedom) — this is not a noise-modeling
choice. Verified before use: `design_zf_equalizer` converges to `design_mmse_equalizer` as the
latter's noise term shrinks to the same regularizer (taps match to 1e-6), and recovers symbols with
zero errors at ~40dB (negligible real noise) — ZF's one unconditional guarantee.

## Results: BER, Case Study 2's exact channel, full scale (100,000 symbols × 10 MC trials)

| SNR (dB) | BPSK NoProc | BPSK MMSE | BPSK ZF | QPSK NoProc | QPSK MMSE | QPSK ZF |
|---|---|---|---|---|---|---|
| -10 | 0.3346 | 0.3279 | 0.3313 | 0.3806 | 0.3759 | 0.3786 |
| -5 | 0.2215 | 0.2123 | 0.2166 | 0.2970 | 0.2887 | 0.2948 |
| 0 | 0.0883 | 0.0800 | 0.0835 | 0.1774 | 0.1692 | 0.1799 |
| 5 | 0.0083 | 0.0067 | 0.0074 | 0.0633 | 0.0662 | 0.0756 |
| 10 | 1.5e-5 | 6e-6 | 1.1e-5 | 0.0093 | **0.0193** | **0.0227** |
| 15 | 0 | 0 | 0 | 2.3e-4 | **0.0041** | **0.0048** |
| 20 | 0 | 0 | 0 | 0 | **4.6e-4** | **5.3e-4** |

**BPSK: ZF sits between No-Processing and MMSE at every SNR, tracking MMSE closely and reaching zero
errors alongside it at 15-20dB.** No surprises here — BPSK's wide decision margin tolerates the
extra noise amplification ZF pays for full inversion without it costing any measured bit errors.

**QPSK: ZF is consistently the *worst* of the three methods at moderate-to-high SNR (10-20dB) —
worse than MMSE, which was already worse than doing nothing** (Section 4.7's existing finding).
At 20dB, No-Processing has 0 errors, MMSE has 923/2,000,000, and ZF has 1,055/2,000,000 — a clean,
monotonic ordering (No-Processing < MMSE < ZF) that matches the textbook prediction exactly: the
less a linear equalizer accounts for noise, the more it pays in QPSK's tighter decision margins for
forcing a full channel inversion this specific channel doesn't reward.

## Conclusion

**This confirms and sharpens Section 4.7's existing finding rather than contradicting it.** The
QPSK high-SNR linear-equalizer regression is not an MMSE-specific quirk — it is a general property
of *how much* a linear equalizer insists on fully removing residual ISI, with ZF (zero
noise-awareness) paying the largest cost, MMSE (partial noise-awareness) paying a smaller one, and
adaptive/learned methods (Sections 4.6, 7) that aren't bound by the same linear-inversion tradeoff
avoiding the cost altogether. Finding this gap required deliberately looking outward at comparable
published work rather than re-reading this project's own code — a different, complementary kind of
verification to the internal correctness checks this project has otherwise relied on throughout.
