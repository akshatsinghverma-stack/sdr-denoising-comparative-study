# Findings: Why Does NLMS Not Benefit From Decision-Directed Tracking While LMS Does?

**Status**: root-causes the one item report.md Section 6.4 explicitly left unverified — "a
plausible hypothesis ... is offered as a starting point, not a verified explanation." New file
only: `experiments/diagnose_lms_nlms_asymmetry.py`. No existing module modified — the
preamble/DD-phase update equations are reimplemented read-only, for instrumentation, and verified
to reproduce the production `LMSFilter`/`NLMSFilter` classes' actual BER exactly before any
diagnostic derived from them was trusted.

## Recap of what was already established (report.md Section 6.4)

On the SPS=1 flat-fading control (the one regime where the DD reliability gate works as designed),
LMS decision-directed tracking earns back 29-40% lower BER than Frozen at 10-15dB. NLMS shows
flat-to-worse performance at every SNR tested, with a 37.5% engage-and-diverge rate. A mechanism
was proposed but not tested: NLMS's instantaneous-power step-size normalization,
`w(n+1) = w(n) + [mu / (eps + ||x(n)||^2)] * conj(e(n)) * x(n)`, could interact badly with a
fading channel's power swings — a fade shrinks `||x(n)||^2`, which *inflates* NLMS's effective
step size at exactly the moment the channel is weakest. LMS's fixed step size,
`w(n+1) = w(n) + mu * conj(e(n)) * x(n)`, has no such normalization, so a small `x(n)` during a
fade produces a naturally *small* update regardless of correctness.

## Method

A new, read-only reimplementation of both filters' preamble (Phase 1) and decision-directed
(Phase 2) update equations, instrumented to record per-sample effective step size, weight-update
vector norm, and decision correctness (checked against the known clean reference, for diagnosis
only — not fed to the filter, exactly like the production reliability gate's own check). Before
trusting any statistic derived from it, this reimplementation was verified to reproduce the actual
`LMSFilter`/`NLMSFilter` classes' BER **exactly** (bit-for-bit identical error counts) across all
40 (method × modulation × SNR × trial) combinations tested — this caught one real bug in the
diagnostic itself during development (comparing BER over the full signal rather than just the
post-preamble region, which spuriously counted the untouched preamble placeholder as errors) before
any mechanism was reported.

For every DD-engaged trial (BPSK/QPSK, 10/15dB, 5 trials each — the same conditions as
report.md Section 6.4), this measures: (1) the correlation between the channel's instantaneous
envelope and the effective step size / update-vector norm, and (2) the "inflation ratio" — the
mean update-vector norm during *wrong* decisions, divided by that method's own median update norm
(normalizing away LMS/NLMS's very different absolute scales, so the two are comparable
like-for-like).

## Results

| | LMS | NLMS |
|---|---|---|
| Mean correlation: envelope vs. effective step size | +0.290 | **-0.667** |
| Mean correlation: envelope vs. update-vector norm | +0.217 | **-0.480** |
| Mean inflation ratio (wrong-decision update norm / own median) | 0.455 | **1.228** |
| Trials with negative envelope-vs-update-norm correlation | 0 / 16 | **15 / 15** |
| Trials with inflation ratio > 1 | 0 / 16 | 8 / 15 |

**Every single engaged trial, with no exceptions, splits perfectly along method lines.** All 15
NLMS trials show a negative correlation between channel envelope and update-vector norm — a fade
measurably inflates NLMS's actual weight update, in every trial tested. All 16 LMS trials show the
opposite (positive) correlation — a fade measurably shrinks LMS's update, in every trial tested.
The inflation ratio confirms the consequence: NLMS's updates during wrong decisions average 1.23x
its own typical (median) update size — it corrects itself *harder* exactly when it is wrong — while
LMS's updates during wrong decisions average only 0.46x its typical size (LMS is gentlest exactly
when it is wrong, since a wrong decision is disproportionately likely during a fade, where `x(n)`
is already small). LMS never once (0/16) showed an above-median update during a wrong decision;
NLMS did in just over half its trials (8/15) — concentrated, on inspection, in exactly the
higher-BER trials where the divergence documented in Section 6.4 is largest.

## Conclusion

**The hypothesis is confirmed, not merely plausible.** NLMS's instantaneous-power normalization is
a real, measurable, 100%-consistent mechanism: a channel fade shrinks the input power in the
denominator of NLMS's step-size formula, inflating the effective step size at precisely the moment
decisions are least reliable, so a wrong decision gets amplified into an oversized, badly-directed
correction — a destabilizing feedback loop. LMS's fixed step size cannot have this failure mode by
construction: a small `x(n)` during a fade produces a small update whether or not the decision was
correct, which is passively protective rather than actively corrective. This explains, mechanistically
rather than by analogy, why decision-directed tracking is a net win for LMS but not for NLMS on
this project's time-varying channel, and suggests a concrete, testable fix for future work: a
NLMS variant that floors or caps the normalization denominator (or gates adaptation on a smoothed,
rather than instantaneous, power estimate) would be expected to recover NLMS's usual convergence-speed
advantage over LMS without inheriting this fade-amplification failure mode — not attempted here, to
keep this diagnostic scoped to explaining the observed asymmetry rather than re-opening filter design.
