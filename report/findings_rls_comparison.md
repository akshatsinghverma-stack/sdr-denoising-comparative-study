# Findings: Does RLS's Faster Convergence Show a Real Edge Over LMS/NLMS on ISI?

**Status**: executes report.md's Named, Scoped-Out Future Work item — RLS was deliberately
excluded from Case Study 2's original run to keep exactly one variable (the channel) different
between Case Studies 1 and 2. Before this first real use, `src/rls_filter.py`'s decision-directed
default was hardened to match LMSFilter/NLMSFilter's established safe default (it previously
defaulted to unguarded decision-directed continuation, gated only by a loose `|d-y|<=2.0` check —
the same broken-gate pattern Section 3.2 documented for LMS/NLMS). New files:
`experiments/run_rls_comparison.py`, `tests/test_rls_stability.py` (16 new regression tests). No
existing module modified beyond `src/rls_filter.py`'s default-safety fix and, later, its
tail-averaging fix (see below).

**Correction notice**: this file originally reported RLS as "measurably worse than LMS/NLMS at
low-to-moderate SNR." That conclusion was correct for RLS *without* Polyak/Ruppert tail-averaging,
but a first attempt to test the effect of adding it was compared against a stale results file from
an unrelated earlier run (a real process-management mistake, caught by checking the file's
modification timestamp against the actual run's completion time before trusting the comparison,
rather than after) — the corrected, verified re-run below completely changes the conclusion.

## Setup

Reproduces Case Study 2's exact channel, symbol count (100,000), Monte Carlo trial count (10), and
SNR sweep (full scale — RLS is cheap enough per-sample, ~0.6-2.1s per 400,000-sample signal, not to
need a reduced-diagnostic scope), adding RLS (λ=0.99, 16 taps) alongside No-Processing, LMS, NLMS,
and the genie MMSE bound. CNN/Hybrid are intentionally excluded — unchanged from Case Study 2 and
not relevant to this specific classical-filter question.

## First pass: RLS without tail-averaging — measurably worse at low-to-moderate SNR

BER ratio (LMS or NLMS BER / RLS BER, >1 = RLS wins), RLS frozen on the raw final preamble iterate
(no tail-averaging, matching LMS/NLMS's pre-hardening design):

| SNR (dB) | BPSK: LMS/RLS | BPSK: NLMS/RLS | QPSK: LMS/RLS | QPSK: NLMS/RLS |
|---|---|---|---|---|
| -10 to 5 | 0.85-0.92x (RLS worse) | 0.85-0.95x (RLS worse) | 0.88-0.91x (RLS worse) | 0.89-0.92x (RLS worse) |
| 10 | 1.27x (RLS wins) | 1.54x (RLS wins) | 0.84x (RLS still worse) | 0.90x (RLS still worse) |
| 15-20 | ∞ (RLS: 0 err) | ∞ (RLS: 0 err) | ∞ (RLS: 0 err) | ∞ (RLS: 0 err) |

This matched neither "RLS always wins" nor "RLS always loses" — a genuine SNR-dependent crossover,
with RLS behind at low-to-moderate SNR and ahead only from 10-20dB.

## The actual intervention: adding tail-averaging, verified correctly this time

LMS/NLMS both average the last ~70% of their preamble weight trajectory (Polyak/Ruppert averaging)
specifically to cut single-realization training variance (Section 3.2); `RLSFilter` originally had
no equivalent, freezing on the raw final iterate. This was implemented identically to LMS/NLMS's
convention and the full comparison re-run — **this time with the modification timestamp of the
output file checked against the run's actual completion time before trusting any comparison**,
after the stale-file mistake above.

**Result: tail-averaging produces a real, substantial, monotonically-growing improvement at every
single SNR level tested — RLS now matches or beats LMS/NLMS across the entire SNR range, not just
at high SNR.**

| SNR (dB) | RLS errors, before | RLS errors, after tail-averaging | Improvement | LMS/RLS(after) | NLMS/RLS(after) |
|---|---|---|---|---|---|
| BPSK -10 | 405,908 | 355,427 | 1.14x | 1.01x | 1.01x |
| BPSK -5 | 301,200 | 250,848 | 1.20x | 1.02x | 1.02x |
| BPSK 0 | 136,225 | 113,564 | 1.20x | 1.07x | 1.07x |
| BPSK 5 | 25,202 | 16,611 | 1.52x | 1.41x | 1.45x |
| BPSK 10 | 198 | 81 | 2.44x | 2.96x | 3.79x |
| BPSK 15/20 | 0 | 0 | — | ∞ | ∞ |
| QPSK -10 | 867,057 | 788,298 | 1.10x | 1.01x | 1.01x |
| QPSK -5 | 706,102 | 632,816 | 1.12x | 1.01x | 1.01x |
| QPSK 0 | 436,240 | 380,709 | 1.15x | 1.03x | 1.03x |
| QPSK 5 | 143,117 | 113,821 | 1.26x | 1.11x | 1.12x |
| QPSK 10 | 8,093 | 4,371 | 1.85x | 1.55x | 1.66x |
| QPSK 15/20 | 0 | 0 | — | ∞ | ∞ |

(All figures are raw error counts out of 1,000,000 bits for BPSK / 2,000,000 for QPSK, 10 Monte
Carlo trials at 100,000 symbols each — differences of this size, e.g. 198→81 or 8093→4371, are far
beyond any plausible trial-to-trial noise at this scale, not a marginal or ambiguous result.)

**LMS/RLS(after) and NLMS/RLS(after) are ≥1.0 at every SNR level** — RLS with tail-averaging is
never worse than LMS/NLMS anywhere in the tested range, and is a clear, growing winner from 5dB
upward. The improvement itself grows with SNR (1.10-1.20x at low SNR, up to 2.44-3.79x at 10dB)
because tail-averaging's benefit is specifically variance reduction, and variance matters most when
the residual training noise is a larger fraction of the total error budget — exactly the regime
Section 3.2 documented for LMS/NLMS's own tail-averaging benefit.

## Why the earlier "zero effect" claim was wrong, as a process lesson

The first attempt at this test compared the freshly-run "after" results file against a "before"
snapshot, and found them bit-for-bit identical — leading directly to a published (and wrong)
"tail-averaging is ruled out" conclusion. The actual cause: an earlier, unrelated background
process (from initial RLS integration testing, before tail-averaging existed) was still running and
overwrote the same output file moments before the real post-fix run finished, and the comparison
was made before checking whether the file being read was actually produced by the intended run. The
fix was procedural, not statistical: check the output file's modification timestamp against the
launching run's actual completion time before trusting any before/after comparison, every time —
now applied, and the corrected numbers above reflect the real, verified effect.

## Compute cost

RLS remains cheaper per-sample than NLMS in this implementation (0.60-0.63s per 1,000,000-symbol
BPSK signal vs. NLMS's 1.5-2.2s; 1.95-2.10s vs. NLMS's 3.5-3.7s for QPSK) despite its per-sample
complexity being O(taps²) vs. LMS/NLMS's O(taps) — consistent with this project's compute-cost
finding (Section 8) that wall-clock cost and algorithmic complexity do not always move together in
this project's software stack.

## Conclusion

**Once given the same hardening LMS/NLMS already had (tail-averaging), RLS matches or beats both at
every tested SNR level, confirming the original theoretical motivation for trying it at all** — "ISI
is exactly the regime where RLS's faster/more precise convergence over LMS/NLMS should show a real
edge, since RLS solves the exact least-squares problem each step rather than following a noisy
stochastic gradient" turns out to be correct, but only once RLS is given a fair comparison against
an *equally* hardened LMS/NLMS. The unhardened RLS's low-SNR shortfall was not a fundamental property
of the algorithm — it was an implementation gap, now closed and verified, not merely asserted to be
closed. This is a genuinely different, stronger conclusion than this project's first pass reached,
and the correction stands as its own lesson: an intervention's result is only as trustworthy as the
process used to measure it, and that process needs the same "verify before trusting" discipline this
project has applied to every other claim.
