# Correction: The "Channel Looks Static During the Preamble" Premise Was Never Actually Measured — And Is False for the Default Config

**Status**: a correction surfaced by a deliberate, adversarial self-critique pass (four independent
reviewer agents were asked to find real problems in this project, one specifically tasked with
DSP/channel-model correctness). This is not a new experiment — it is a fix to an unverified claim
in `src/channel.py`'s docstring and report.md Section 6.1, following the exact same "measure it,
don't assume it" standard this project has applied to its other four bugs. New file only (this
one); `src/channel.py`'s docstring corrected in place (no functional code changed).

## The claim, and why it was wrong

`add_time_varying_multipath`'s docstring, and report.md Section 6.1, both asserted: "the drift
rate is deliberately chosen so the channel looks static across the 1000-symbol preamble ... but
has moved substantially by the time thousands of symbols have elapsed." This claim was **never
actually measured against the real function output** — the only verification on record was that a
*zero-drift* call reproduces the static channel exactly (true, but a different and much weaker
claim than "the actual nonzero-drift default looks static").

**Measured directly** (a constant-input probe through the real `add_time_varying_multipath`
function — for a single-tap channel, feeding in an all-ones signal makes the output *equal* the
tap's time-varying gain, letting it be read out directly, no reimplementation needed):

- **The random-walk config actually used for Case Study 3's headline flat-fading control**
  (`coherence_symbols=4000`, `drift_depth=0.4`, `sps=1`, the exact parameters
  `run_flatfade_sps1_control()` uses): the single tap's magnitude swings by **60.5% to 90.7% of its
  base value within the first 1000 samples alone** — the entire preamble window — across the 5
  seeds tested. This is not "negligible."
- **The sinusoidal-mode config** (`n_cycles=1.5`, `drift_depth=0.6`, `sps=4`, used in Case Study
  3's Part 1/Finding 1-2): the non-direct taps drift by **±7% to ±22% of their base magnitude
  within the first 4000-sample (1000-symbol) preamble window** across 10 seeds tested.

Both figures were cross-verified two independent ways (a standalone reimplementation of the
drift-generation math, and — for the random-walk case — the actual production function via the
constant-input probe above) and agree exactly.

## What this does and does not change

**Does not change**: the headline, *measured* comparison in Section 6.4 — "LMS decision-directed
tracking earns back its keep: pooled BER drops 29-40% relative to Frozen at 10-15dB" — remains a
valid, fair comparison. Frozen and DD are evaluated on the *exact same* channel/noise realization in
every trial; whatever the channel actually does, both methods experience it identically, so the
BER difference between them is real and correctly attributed to DD's continued adaptation, not to
an artifact of the drift-rate claim being wrong.

**Does change**: the *causal story* for why DD has "an actual job to do." The original framing was
"the channel is static while LMS trains during the preamble, then starts moving — so DD tracks
purely post-preamble drift." That framing is false: the channel is already substantially different
by the *end* of the preamble than at its *start*. This means:
1. The "Frozen" baseline itself is not a clean, single-point estimate of a stationary channel — it
   is a Polyak/Ruppert-averaged fit to a channel that already moved considerably during the
   estimation window. Some of DD's reported advantage over Frozen may be DD correcting for this
   in-preamble blur, not purely for post-preamble drift, and the report did not previously
   distinguish between these two contributions.
2. The design principle "the drift rate was deliberately chosen to keep the preamble clean" is not
   an accurate description of what the code default actually does — it should instead be described
   as "a continuously time-varying channel from the very start of the signal, which both Frozen and
   DD must contend with equally, with DD's continued tracking measurably outperforming Frozen
   despite (or because of) this."

## Conclusion

**This is a real correction to this project's own stated methodology, not a reversal of its
finding.** The BER numbers behind Section 6.4's headline claim stand — they were measured on real
data, and both compared methods see the same channel. What was wrong was an *unverified design
claim* about that channel's behavior during the preamble, asserted by reasoning about the drift
parameters rather than measured from the function's actual output — precisely the failure mode
this project's other four bugs were all caught by avoiding. Report.md Section 6.1 and this
module's docstring have been corrected to state the measured drift magnitude rather than the
original, false "looks static" claim.
