# Findings: Zero-Forcing (ZF) Equalizer — A Missing Baseline, Found by Comparing to Published Work

**Status**: adds a standard baseline this project was missing entirely, found by deliberately
searching comparable published literature during a self-critique pass (not by re-reading this
project's own code — the gap was invisible from the inside). New functions only:
`design_zf_equalizer`/`apply_zf_equalizer` in `src/mmse_equalizer.py` (a two-line wrapper reusing
existing machinery), `experiments/run_zf_comparison.py`, 2 new regression tests. No existing
function modified.

## Correction (found via a later code-correctness critique pass)

**The QPSK results and conclusion below were computed with a real bug in `design_mmse_equalizer`
(this project's eighth): `np.correlate(g, g)` silently returns the complex conjugate of the
autocorrelation value the function's own Wiener-Hopf formula requires, negligible for BPSK but
severe for QPSK. This made the original "MMSE already worse than doing nothing, ZF worse still"
finding a bug, not a real linear-equalizer tradeoff.** Fixed via `return np.conj(c_full[idx])`,
verified against a from-scratch ground-truth Wiener-Hopf solve, and this experiment re-run at full
scale with the fix. The table and QPSK conclusion below are the corrected, post-fix numbers; the
original (wrong) numbers and reasoning are preserved in the version history for transparency about
what was concluded and why before the bug was found.

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

## Results: BER, Case Study 2's exact channel, full scale (100,000 symbols × 10 MC trials), post-fix

| SNR (dB) | BPSK NoProc | BPSK MMSE | BPSK ZF | QPSK NoProc | QPSK MMSE | QPSK ZF |
|---|---|---|---|---|---|---|
| -10 | 0.3346 | 0.3281 | 0.3316 | 0.3806 | 0.3759 | 0.3781 |
| -5 | 0.2215 | 0.2124 | 0.2170 | 0.2970 | 0.2879 | 0.2909 |
| 0 | 0.0883 | 0.0801 | 0.0834 | 0.1774 | 0.1606 | 0.1633 |
| 5 | 0.0083 | 0.0064 | 0.0068 | 0.0633 | 0.0399 | 0.0407 |
| 10 | 1.5e-5 | 4e-6 | 5e-6 | 0.0093 | **9.6e-4** | **9.7e-4** |
| 15 | 0 | 0 | 0 | 2.3e-4 | **0** | **0** |
| 20 | 0 | 0 | 0 | 0 | **0** | **0** |

**BPSK: unchanged by the fix, as expected — ZF sits between No-Processing and MMSE at every SNR,
tracking MMSE closely and reaching zero errors alongside it at 15-20dB.** BPSK's wide decision
margin tolerates the extra noise amplification ZF pays for full inversion without it costing any
measured bit errors, before or after the fix (real symbols mostly mask a conjugate error).

**QPSK: with the fix, MMSE and ZF both now beat No-Processing convincingly at every SNR, reversing
the pre-fix picture completely.** At 20dB, all three tie at 0 errors. At 15dB, No-Processing still
has 466/2,000,000 errors while MMSE and ZF both reach exactly 0. At 10dB — now the clearest
remaining SNR to compare the two genie equalizers — MMSE has 1,923/2,000,000 errors and ZF has
1,942/2,000,000, both far below No-Processing's 18,692: **ZF is still the (very slightly) worse of
the two genie equalizers, exactly the textbook prediction, just at a realistic magnitude rather than
the bug-inflated one originally reported** (previously, this same experiment's own pre-fix run:
MMSE 38,582/2,000,000, ZF 45,373/2,000,000 at the same SNR — both roughly 20-40x too high, and both
worse than No-Processing's 18,692, which a genie should never be).

## Conclusion

**The ZF-vs-MMSE ordering (ZF slightly worse, both genies close to optimal) is real and survives the
fix — but the original headline ("QPSK linear-equalizer regression below doing nothing") does not.**
That was this project's eighth real bug (`src/mmse_equalizer.py`'s autocorrelation conjugate error,
corrected in Section 4.7), not a property of linear equalization. What remains, once computed
correctly, is the textbook-expected, much smaller effect: a linear equalizer that ignores noise
entirely (ZF) pays a small extra cost relative to one that accounts for it (MMSE), and both
comfortably beat not equalizing at all. Finding the *missing baseline* (ZF itself) still required
deliberately looking outward at published work rather than re-reading this project's own code; but
trusting the *numbers* that comparison produced required the same internal-correctness discipline
(checking a result against its own definitional identity) this project has relied on throughout —
outward-looking and inward-looking verification are complementary, and this finding needed both.
