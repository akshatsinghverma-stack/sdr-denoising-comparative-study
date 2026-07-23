# Findings: Does RLS's Faster Convergence Show a Real Edge Over LMS/NLMS on ISI?

**Status**: executes report.md's Named, Scoped-Out Future Work item — RLS was deliberately
excluded from Case Study 2's original run to keep exactly one variable (the channel) different
between Case Studies 1 and 2. Before this first real use, `src/rls_filter.py`'s decision-directed
default was hardened to match LMSFilter/NLMSFilter's established safe default (it previously
defaulted to unguarded decision-directed continuation, gated only by a loose `|d-y|<=2.0` check —
the same broken-gate pattern Section 3.2 documented for LMS/NLMS). New files:
`experiments/run_rls_comparison.py`, `tests/test_rls_stability.py` (16 new regression tests). No
existing module modified beyond `src/rls_filter.py`'s default-safety fix.

## Setup

Reproduces Case Study 2's exact channel, symbol count (100,000), Monte Carlo trial count (10), and
SNR sweep (full scale — RLS is cheap enough per-sample, ~0.6-2.1s per 400,000-sample signal, not to
need a reduced-diagnostic scope), adding RLS (λ=0.99, 16 taps, frozen-after-preamble) alongside
No-Processing, LMS, NLMS, and the genie MMSE bound. CNN/Hybrid are intentionally excluded — unchanged
from Case Study 2 and not relevant to this specific classical-filter question.

## Results: BER ratio (LMS or NLMS BER / RLS BER — greater than 1 means RLS wins)

| SNR (dB) | BPSK: LMS/RLS | BPSK: NLMS/RLS | QPSK: LMS/RLS | QPSK: NLMS/RLS |
|---|---|---|---|---|
| -10 | 0.88x | 0.89x | 0.91x | 0.92x |
| -5 | 0.85x | 0.85x | 0.91x | 0.91x |
| 0 | 0.89x | 0.90x | 0.90x | 0.90x |
| 5 | 0.92x | 0.95x | 0.88x | 0.89x |
| 10 | **1.27x** | **1.54x** | 0.84x | 0.90x |
| 15 | **∞ (0 err)** | **∞ (0 err)** | **∞ (0 err)** | **∞ (0 err)** |
| 20 | **∞ (0 err)** | **∞ (0 err)** | **∞ (0 err)** | **∞ (0 err)** |

**This is not the simple "RLS wins because it solves the exact least-squares problem instead of
following a noisy stochastic gradient" result the future-work item hypothesized.** Instead, there
is a real, consistent SNR-dependent crossover: **RLS is measurably** ***worse*** **than both LMS
and NLMS at low-to-moderate SNR** (ratios 0.84-0.95x from -10dB up to 5-10dB, i.e. RLS has 5-19%
*more* errors in this range, for both modulations) **but becomes strictly better at high SNR** —
BPSK crosses over at 10dB (RLS: 198 errors vs. LMS's 252, NLMS's 304) and reaches zero errors at
15-20dB where LMS/NLMS still show single-digit residual errors; QPSK's crossover lands one step
later (RLS still slightly behind at 10dB, 0.84-0.90x) but also reaches zero errors at 15-20dB.

## Why this is plausible, not asserted as fully explained

This project's LMS/NLMS were extensively hardened for this exact regime (Section 3.2): a
conservative, clamped step size and Polyak/Ruppert tail-averaging over the preamble trajectory,
specifically to reduce single-realization training variance. `RLSFilter` has no equivalent
tail-averaging, and its λ=0.99 forgetting factor and initialization (`P = I/δ`) were not tuned for
this project's specific preamble length or SNR range — a plausible, but not yet verified, explanation
for the low-SNR shortfall is that RLS's inverse-correlation-matrix recursion is more sensitive to
noisy early estimates during the preamble than LMS/NLMS's simpler update, an effect a per-SNR λ
(paralleling this project's existing per-SNR-μ future-work item for LMS/NLMS) or a tail-averaged RLS
weight estimate might close. This is named as a specific follow-up rather than root-caused here, to
keep this diagnostic scoped to establishing *whether* RLS helps rather than re-opening filter design.

**Compute cost**: RLS is cheaper per-sample than NLMS in this implementation (0.60-0.63s per
1,000,000-symbol BPSK signal vs. NLMS's 1.5-2.2s; 1.95-2.10s vs. NLMS's 3.5-3.7s for QPSK) — despite
its per-sample complexity being O(taps²) vs. LMS/NLMS's O(taps), consistent with this project's
compute-cost finding (Section 8) that wall-clock cost and algorithmic complexity do not always move
together in this project's software stack.

## Conclusion

**RLS earns its keep at high SNR, not across the board.** At 15-20dB, where this channel's residual
ISI is the dominant remaining error source (not noise), RLS's higher-precision, per-step-optimal
solution reaches zero measured errors where LMS/NLMS still show a small residual misadjustment
floor — a real, measurable advantage matching the theoretical motivation for trying it. But at
low-to-moderate SNR, where noise (not ISI) dominates, RLS is consistently, measurably *worse* than
the already-hardened LMS/NLMS — the opposite of what "RLS should show a real edge on this
correlated-input channel" alone would predict, and a reminder that an algorithm's textbook
theoretical advantage (faster/exact convergence) does not automatically transfer to every operating
regime without the same hardening (tail-averaging, tuned forgetting factor) applied elsewhere in this
project.
