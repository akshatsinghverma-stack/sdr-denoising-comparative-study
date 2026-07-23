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
| Trials with inflation ratio > 1 (among trials with a defined ratio) | 0 / 13 | 8 / 13 |

**Every single engaged trial, with no exceptions, splits perfectly along method lines.** All 15
NLMS trials show a negative correlation between channel envelope and update-vector norm — a fade
measurably inflates NLMS's actual weight update, in every trial tested. All 16 LMS trials show the
opposite (positive) correlation — a fade measurably shrinks LMS's update, in every trial tested.
(This sign-consistency result does not depend on wrong decisions existing, so all 16/15 engaged
trials contribute to it.) The inflation ratio confirms the consequence: NLMS's updates during wrong
decisions average 1.23x its own typical (median) update size — it corrects itself *harder* exactly
when it is wrong — while LMS's updates during wrong decisions average only 0.46x its typical size
(LMS is gentlest exactly when it is wrong, since a wrong decision is disproportionately likely
during a fade, where `x(n)` is already small). This ratio is only defined for trials with at least
one wrong decision during DD (3 LMS and 2 NLMS trials — all at 15dB, where BER was 0 — had none, so
are excluded rather than counted as "not inflated"). **Among the 13 trials per method where the
ratio is defined**, LMS never once (0/13) showed an above-median update during a wrong decision;
NLMS did in 8/13 (62%) — concentrated, on inspection, in exactly the higher-BER trials where the
divergence documented in Section 6.4 is largest.

## Conclusion

**The hypothesis is strongly and consistently evidenced — a correlational result, not an
interventional one, and should be read accordingly.** Every engaged trial's sign of the
envelope-vs-update-norm correlation splits perfectly by method (16/16 LMS positive, 15/15 NLMS
negative), and the inflation-ratio result points the same direction. This is a much stronger basis
than a single plausible-sounding mechanism, but no counterfactual was run: the natural falsification
test — capping or flooring NLMS's normalization denominator (or gating adaptation on a smoothed
rather than instantaneous power estimate) and confirming the fade-amplification signature disappears
while NLMS's usual convergence-speed advantage returns — was not attempted here. Until that
intervention is run, "NLMS's instantaneous normalization causes the asymmetry" is the best-supported
explanation consistent with every measurement taken, not a demonstrated causal mechanism in the
strict sense. That intervention is the concrete next step, named rather than left as a vague
"future work" placeholder.
