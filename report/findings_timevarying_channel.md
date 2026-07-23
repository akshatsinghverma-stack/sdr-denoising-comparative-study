# Findings: Does Decision-Directed Tracking Earn Its Keep on a Time-Varying Channel?

**Status**: diagnostic follow-up to report.md Section 3.2 / Section 8's named future work item
("A genuinely time-varying channel... was never run"). New files only: `src/channel.py`
(`add_time_varying_multipath`, additions), `experiments/run_timevarying_channel.py`,
`results/tables/results_timevarying*.csv`, `results/figures/timevarying/`. No existing file's
behavior was changed.

## Headline answer

**It is more complicated than "yes" or "no," and the complication is itself the main finding.**

1. At **SPS=4** (this project's actual pulse-shaped/ISI regime -- Case Study 2, severity sweep),
   decision-directed mode **cannot be meaningfully tested with today's code as configured**: its
   safety gate never engages (0% of trials, every modulation, every channel-drift mode tested),
   and if that gate is deliberately bypassed to force it to run anyway, it is **catastrophic** --
   10x to 10,000x worse BER than frozen weights, at every SNR from -5 to 15dB -- for a reason
   that turns out to have **nothing to do with time-variation** (confirmed with a static-channel
   control, see Finding 2).
2. At **SPS=1** (Case Study 1's regime, where the gate was actually built and validated), tested
   against a genuinely time-varying flat-fading channel, the picture the report's Section 3.2
   hypothesized finally gets a real test: **LMS decision-directed tracking does earn back its
   keep** at moderate-to-high SNR (pooled BER reduced by 29-40% vs. frozen at 10-15dB, both
   modulations) -- refuting the blanket claim that continued adaptation is "pure downside risk"
   once the channel genuinely varies. **NLMS does not show the same benefit** (net BER wash-to-worse
   at every SNR tested) and reproduces the original engage-and-diverge risk at almost exactly the
   report's documented rate (37.5% of engaged trials here vs. the report's "~30-40%").

Net effect on the original design decision: **freezing-by-default remains the right call for
this codebase today**, but for a different and more urgent reason than the one on record -- see
Conclusion.

---

## 1. Channel model built

`src/channel.py:add_time_varying_multipath(signal, h_base, sps, mode, n_cycles, drift_depth,
coherence_symbols, vary_direct_path, seed)` -- a causal tap-delay channel
(`r[n] = sum_k h_k(n) * s[n-k]`) whose non-direct taps drift smoothly over the signal's duration,
reusing Case Study 2's exact static profile `[1.0, 0.4+0.3j, -0.1+0.1j]` as the drift center. Two
drift models, chosen to bracket "deterministic, repeatable" vs. "stochastic, more realistic":

- `mode="sinusoidal"`: each non-direct tap oscillates as
  `h_base[k]*(1 + drift_depth*sin(2*pi*n_cycles*n/N + phi_k))`, independent random phase `phi_k`
  per tap (reflectors don't fade in lockstep). Default `n_cycles=1.5` over the whole test signal.
- `mode="random_walk"`: each non-direct tap follows a band-limited (AR(1)-smoothed complex
  Gaussian) stochastic drift, parameterized by `coherence_symbols` (decorrelation time).

**Rate-of-drift justification** (why this is "slow fading relative to the preamble," not a
different, unfair problem): with a 20,000-25,000 symbol test signal and a 1,000-symbol preamble,
`n_cycles=1.5` (sinusoidal) gives a period of ~13,300-16,700 symbols -- 13-17x the preamble length,
so the channel looks essentially static across the preamble itself (the frozen filter's original
training assumption is still valid at t=0) but has swung through a large fraction of its drift
range by the time thousands of symbols have elapsed post-preamble -- giving continued adaptation an
actual, plausible job, without requiring symbol-by-symbol tracking no equalizer could achieve
anyway. `coherence_symbols=4000` (random-walk mode) targets the same ratio (4x the preamble
length). The direct/zero-delay tap is held fixed by default (`vary_direct_path=False`) -- a stable
LOS path with fading reflectors on top is the standard physical picture for this kind of channel --
with `vary_direct_path=True` available for the single-tap flat-fading control test (Section 5).

Verified before use: `h_base=[1,0,0]` reproduces the identity channel exactly in both modes (no
spurious shift, same discipline as the `add_multipath` causality regression); `drift_depth=0`
reproduces the static `add_multipath` output to within 1e-15 (floating point noise); nonzero drift
produces genuine, time-increasing deviation from the static reference.

## 2. Experiment design

`experiments/run_timevarying_channel.py`, Part 1: BPSK/QPSK, SPS=4, RRC pulse shaping (Case Study
2's setup), 25,000 symbols, 5 SNR points (-5/0/5/10/15dB), 5 MC trials (bits + noise + channel-drift
realization all re-randomized per trial), both channel-drift modes, LMS/NLMS (16 taps, same
`mu` as the main case studies) each run twice per trial: `enable_decision_directed=False`
("Frozen," today's default) and `=True` ("DD"). BER pooled as raw (errors, bits) counts across
trials, matching this project's Monte Carlo convention; a companion `_trials.csv` keeps per-trial
detail for the divergence analysis below.

## 3. Finding 1 -- the reliability gate is dead code at SPS>1

`enable_decision_directed=True`'s global safety gate (report Section 3.2) accepts continuation only
if hard decisions on the preamble tail match the **clean reference** with mismatch rate at or below
`reliability_threshold` (default 0.1). That check was built and tested only against Case Study 1's
SPS=1 channel, where every clean sample **is** exactly a constellation point
(`tx[990:1010].real = [-1, 1, 1, 1, ...]`, verified directly). At SPS=4, the clean reference is a
continuously-varying RRC-shaped waveform that equals an exact constellation point at only 1-in-4
samples (verified: `tx[3980:4010].real = [-0.81, -1.19, -1.41, -1.41, -1.21, ..., 1.25]`, plausible
in-between values everywhere else) -- so the exact-value mismatch check structurally reads as
~75-80% "mismatch" **regardless of SNR, channel behavior, or decision quality**, and the gate never
opens: measured engagement rate **0/9 (trial x SNR) probes, for all 4 (modulation x channel-mode)
combinations tested** (`results/tables/results_timevarying_gate_check.csv`). This was invisible
until now because no existing script in this project (Case Study 2, severity sweep) ever calls
`enable_decision_directed=True` -- the default gate silently protects by never doing anything, on
any SPS>1 signal, whether or not the channel varies.

To still test the actual research question, Part 1's "DD" column uses `reliability_threshold=1.0`
(bypasses the broken global gate via its own public parameter; the local, oversampling-agnostic
confidence gate `min_confidence=0.3` remains the only active safeguard -- confirmed to pass ~84% of
post-preamble samples in a spot check, i.e. it is a real, partial gate, not a rubber stamp).

## 4. Finding 2 -- bypassed-gate DD is catastrophic at SPS=4, and it is NOT about time-variation

With the gate bypassed, DD is worse than Frozen at every single SNR level, both modulations, both
channel-drift modes (BER improvement ratio Frozen/DD, computed from pooled error/bit counts,
`results/tables/results_timevarying.csv`):

| | -5dB | 0dB | 5dB | 10dB | 15dB |
|---|---|---|---|---|---|
| BPSK sinusoidal - LMS | 0.53x | 0.26x | 0.05x | 0.001x | 0.0001x |
| BPSK random_walk - LMS | 0.57x | 0.32x | 0.11x | 0.022x | 0.011x |
| QPSK sinusoidal - LMS | 0.66x | 0.41x | 0.15x | 0.015x | 0.0002x |
| QPSK random_walk - LMS | 0.69x | 0.48x | 0.25x | 0.11x | 0.067x |

(NLMS: nearly identical to LMS in every cell, within 1-2%.) At 15dB, sinusoidal-mode DD is roughly
5,000-10,000x worse than frozen; even the gentler random-walk mode is 15-90x worse at high SNR.
DD's pooled BER sits at 0.475-0.495 for every method/SNR/mode combination tested -- indistinguishable
from random guessing (0.5).

**This looked catastrophic enough to demand root-causing rather than reporting at face value**
(per this project's own standard, report Section 4.2/4.7). Two checks:

- **Weight-vector inspection**: at BPSK 15dB, Frozen's 16-tap weight vector has its energy
  concentrated at tap 0 (`|w[0]|=0.564`, decaying to `|w[15]|=0.009`) -- the expected shape for a
  filter keying mostly off the current sample. DD's final weight vector has the opposite shape:
  energy concentrated at tap 15 (`|w[15]|=0.583`, `|w[0]|=0.022`) -- DD converged to an entirely
  different, delay-shifted filter structure.
- **Static-channel control** (isolates "is this about time-variation" from "is this about
  oversampling"): running the identical bypassed-gate DD test on Case Study 2's original static
  `add_multipath` channel (BPSK, 15dB) gives Frozen BER = 4e-5 and DD BER = 0.470 -- the same
  catastrophic failure, on a channel with zero time variation.

**Root cause**: DD's phase-2 loop (both `LMSFilter` and `NLMSFilter`) makes a hard decision and
conditionally adapts on every sample, a design that is only correct when every sample is a symbol
(SPS=1). At SPS=4, 3 of every 4 samples are RRC-shaped transitional values, not decision points --
forcing them toward a constellation corner corrupts the filter's job of reconstructing the pulse
shape needed for correct symbol-center sampling, and it does so identically whether the channel
drifts or not. **This is a pre-existing SPS>1 incompatibility in the DD update rule itself, not a
channel-time-variation result** -- it would have shown up the moment anyone tried
`enable_decision_directed=True` on Case Study 2's own static channel, had the (also-broken)
reliability gate not been silently preventing that from ever happening. Fixing it would require
changing the per-sample adaptation loop in `src/lms_filter.py`/`src/nlms_filter.py` to only decide
and adapt at the correct symbol-spaced phase -- out of scope here (existing-file edits are excluded
from this diagnostic's scope) and left as a concrete, named next step.

## 5. Finding 3 -- the clean test: SPS=1, flat fading, unmodified gate

Because Finding 2 makes the SPS=4 test unable to isolate the time-variation question, a control
was added: `run_flatfade_sps1_control()` in the same script, SPS=1 (no pulse shaping, matching Case
Study 1's regime where the gate is not broken), 20,000 symbols, 5 MC trials, a genuinely
time-varying single-tap, complex (amplitude AND phase) flat-fading channel
(`add_time_varying_multipath(h_base=[1.0], mode="random_walk", vary_direct_path=True,
coherence_symbols=4000, drift_depth=0.4)` -- phase excursions up to +/-90 degrees, amplitude 0.3-3x,
verified by direct inspection), with the DD gate left at its default, unmodified
`reliability_threshold=0.1`.

**The gate behaves exactly as designed here**: 0% engagement at -5/0dB (correctly refuses to adapt
when the preamble itself wasn't trustworthy -- DD output is bit-identical to Frozen, ratio exactly
1.000, confirmed), rising to 20-100% engagement at 5-15dB as the preamble becomes more reliable
(`results/tables/results_timevarying_flatfade_sps1.csv`).

BER improvement ratio (Frozen/DD, pooled counts, >1 = DD wins), at the SNRs where the gate engages:

| | 5dB | 10dB | 15dB | gate engagement (10dB / 15dB) |
|---|---|---|---|---|
| BPSK - LMS | 0.81x (worse) | 1.41x | 1.50x | 80% / 100% |
| BPSK - NLMS | 0.66x (worse) | 0.77x (worse) | 0.72x (worse) | 80% / 80% |
| QPSK - LMS | 0.81x (worse) | 1.41x | 1.66x | 60% / 80% |
| QPSK - NLMS | 1.00x (no change) | 0.96x (worse) | 1.12x | 60% / 80% |

**LMS decision-directed tracking earns back its keep**: at 10-15dB, pooled BER drops 29-40%
relative to Frozen, for both modulations, and DD's absolute BER at these SNRs (0.061-0.133) beats
even No-Processing (0.074-0.151) -- a real, practically-meaningful win directly attributable to
tracking the channel's drift past the preamble. **NLMS does not show the same benefit** -- worse or
flat at every SNR tested; this asymmetry was noted but not fully root-caused (a plausible
hypothesis -- NLMS's instantaneous-power step-size normalization interacting poorly with a fading
channel's power swings during DD adaptation -- is offered as a starting point for follow-up, not a
verified explanation, consistent with this project's practice of flagging open questions rather
than forcing a premature explanation, cf. report Section 5's un-chased MMSE observation).

**Divergence risk, quantified concretely** (matching report Section 3.2's own framing): of the
30 (modulation x SNR x trial) cells tested per algorithm across SNR in {5,10,15}dB, the gate
engaged (output differs from Frozen) in 19/30 cells for LMS and 16/30 for NLMS. Of those engaged
cells: LMS-DD improved on Frozen in 14/19 (74%) and diverged badly (BER greater than 1.5x Frozen)
in 2/19 (10.5%); NLMS-DD improved in 7/16 (44%) and diverged in 6/16 (37.5% -- matching the
report's originally-documented "~30-40% of trials" almost exactly, now reproduced on a genuinely
time-varying channel rather than a static one). The worst single observed divergence: QPSK 5dB,
one trial, LMS-DD BER 0.447 vs. that same trial's Frozen BER 0.176 (a 2.5x regression) -- a real
instance of "engage-and-diverge," bounded (no run reached random-guessing BER ~0.5, unlike the
SPS=4 catastrophe in Finding 2) but concretely present, confirming the original concern is real
and not eliminated just because the channel now varies.

## 6. Conclusion: refutes or upholds Section 3.2?

**Both, depending on which claim you mean.**

- The narrow, mechanical claim actually driving today's default ("freeze weights after the
  preamble") remains correct as a matter of practice for this codebase -- not primarily because of
  anything about time-variation, but because Finding 2 shows the DD implementation itself is broken
  for every SPS>1 signal this project actually uses (Case Study 2, severity sweep), on any channel,
  static or not. This is a more urgent reason to keep freezing as the default than the one on
  record, and it should be documented as a known limitation rather than left to be rediscovered.
- The broader, stated justification for that default -- "on a channel with no time variation, there
  is nothing to track... continued adaptation is pure downside risk" -- is refuted as a general
  principle by Finding 3: on a genuinely time-varying channel, in the one regime where DD's
  machinery actually functions as designed, LMS decision-directed tracking earns back a real,
  substantial BER improvement (29-40%) at moderate-to-high SNR. Continued adaptation is not
  inherently pure downside once the channel actually moves -- it depends on the channel varying,
  the SNR being high enough for the gate to trust the preamble, and (unexpectedly) the specific
  algorithm: LMS benefited here, NLMS did not.
- The original concern behind the caution -- engage-and-diverge risk at borderline SNR -- is
  reproduced, not eliminated, on the time-varying channel (Section 5's 37.5% NLMS divergence rate,
  and LMS's own 0.81x regression at 5dB for both modulations). A time-varying channel gives
  tracking something to gain, but does not remove the risk of losing.

**Bottom line**: this diagnostic does not license turning on `enable_decision_directed=True` by
default for this project's actual (SPS=4) experiments -- it cannot even be exercised safely there
today (Finding 2). But it does mean the report's stated reasoning for why freezing is safe was
broader than what was actually verified, and should be narrowed: freezing is the right default for
this codebase today, primarily because DD's implementation needs a symbol-rate-aware fix before
SPS>1 channels can use it at all, not because time-variation would never make continuation
worthwhile.

## 7. Files produced

- `src/channel.py`: `add_time_varying_multipath()` (new function, additions only -- `add_multipath`
  untouched).
- `experiments/run_timevarying_channel.py`: Part 1 (SPS=4 main sweep + gate-dead-code check) and
  `run_flatfade_sps1_control()` (Part 2, SPS=1 control).
- `results/tables/results_timevarying.csv`, `results_timevarying_trials.csv`,
  `results_timevarying_gate_check.csv`, `results_timevarying_flatfade_sps1.csv`,
  `results_timevarying_flatfade_sps1_trials.csv`.
- `results/figures/timevarying/frozen_vs_dd_{BPSK,QPSK}.png`.

## 8. Named follow-up (in the spirit of report Section 8)

- **Fix the DD update rule for SPS>1**: gate the per-sample decision/adaptation to only fire at the
  correct symbol-spaced phase (or operate DD entirely at symbol rate, post-matched-filter, rather
  than on the 4x-oversampled samples). Requires editing `src/lms_filter.py`/`src/nlms_filter.py`,
  out of scope here.
- **Root-cause the LMS-vs-NLMS asymmetry** found in Section 5 (why NLMS's normalized step size
  fails to benefit from tracking where LMS's fixed step size does) -- flagged, not resolved.
- **Re-run Finding 3's flat-fading test at SPS=4 once the above fix exists**, to get the answer to
  the originally intended question (does tracking help on this project's actual ISI channel) that
  Finding 2 shows is currently unanswerable.
