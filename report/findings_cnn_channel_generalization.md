# Findings: Does the CNN Generalize Across Channels, or Memorize the One It Was Trained On?

**Status**: root-causes a gap flagged by an outside-reviewer critique pass —
deliberately unfamiliar with this project's internal history, briefed only
to find the single biggest weakness a skeptical viva examiner would raise —
that none of the prior self-critique rounds had named. New file only:
`experiments/test_cnn_channel_generalization.py`. No existing module's
behavior changed (`src/cnn_autoencoder.py`'s public interface is unaffected;
this is a new experiment, not a fix).

## The gap

Every CNN result reported elsewhere in this project — Case Study 2, the ISI
severity sweep (Section 5), Case Study 4's 16-QAM comparison — trains and
evaluates the CNN on the *exact same* known multipath impulse response:
`MULTIPATH = [1.0, 0.4+0.3j, -0.1+0.1j]` in `experiments/run_case2_isi.py`,
reused verbatim to build both the training signal and every test-trial
signal. LMS and NLMS do not get this same free pass — they re-estimate the
channel from the preamble every trial, so their reported BER already
reflects not knowing the channel in advance. The genie-aided MMSE and
Zero-Forcing equalizers (Sections 4.7, 4.9) are handed the *true* channel
every trial via `measure_channel_taps`, so they too are evaluated fairly
regardless of which channel is presented. Only the CNN implicitly assumes
train-time and test-time channels match — an assumption that had never been
named, let alone tested, despite feeding directly into this report's central
recommendation ("deploy a CNN whenever real multipath exists," Section 9.5).

Note this is a different question from the ISI severity sweep (Section 5),
which varies the *magnitude* of the same three taps and always trains and
tests a CNN at matched severity — it never asks what happens when the
deployment channel's *shape* (delay spread, tap count, phase) differs from
what the CNN was shown during training.

## Method

One CNN is trained once per modulation on the matched (Case Study 2) channel
— same architecture, same procedure as `run_case2_isi.py`, reduced scale
(30,000 symbols, 5 MC trials, 3 SNR points, 30 epochs) since this is a
diagnostic, not a full case-study reproduction. That single, frozen model is
then evaluated — with **no retraining** — on five held-out channels chosen
to differ in delay spread (2-tap, 3-tap, and 4-tap variants), tap count, and
tap phase/magnitude from the training channel, not just an overall severity
scaling of the same three taps:

| Channel | Taps | ISI energy |
|---|---|---|
| matched (train==test) | `[1.0, 0.4+0.3j, -0.1+0.1j]` | 0.270 |
| rotated phase (2-tap) | `[1.0, -0.3+0.35j]` | 0.212 |
| longer delay spread (4-tap) | `[1.0, 0.28-0.18j, 0.18+0.12j, -0.14+0.05j]` | 0.180 |
| sign-flipped taps (3-tap) | `[1.0, -0.4-0.3j, 0.1-0.1j]` | 0.270 |
| different phase, same tap count | `[1.0, -0.15+0.42j, 0.22-0.15j]` | 0.270 |
| single dominant reflection (2-tap) | `[1.0, 0.5-0.15j]` | 0.273 |

ISI energy (sum of `|h[k]|^2` for `k>0`) is kept comparable across channels
so "held-out is just harder/easier overall" doesn't confound "held-out is a
different shape." LMS, NLMS, and genie MMSE are run on the identical
held-out channels as a like-for-like reference: they adapt/measure per-trial
regardless of which channel is presented, so their BER change from held-out
channels reflects genuine channel difficulty, not a train/test mismatch
penalty specific to one method (`results/tables/results_cnn_channel_generalization.csv`).

## Results

**Confirmed: the CNN's advantage is substantially channel-specific, not a
general equalization capability, while LMS/NLMS/MMSE stay comparatively
stable across the same channel changes.**

At QPSK 10dB — the SNR where Case Study 2 shows the CNN's largest advantage
over classical filters — moving from the matched channel to the held-out
channel with sign-flipped tap phases degrades CNN BER by **129.71x**
relative to its own matched-channel performance (1.65×10⁻³ → 2.14×10⁻¹,
close to random-guessing territory), while LMS and NLMS on that *identical*
held-out channel degrade by only 1.17x and 1.41x. That channel is
objectively somewhat harder for every method — even No-Processing degrades
6.19x there relative to its own matched-channel baseline — but catastrophically
so only for the frozen CNN:

| Method | Matched BER | Degradation ratio, 5 held-out channels (QPSK, 10dB) |
|---|---|---|
| No Processing | 9.26×10⁻³ | 4.41x, 0.15x, 6.19x, 1.07x, 0.24x |
| LMS | 3.44×10⁻³ | 0.72x, 0.80x, 1.17x, 0.67x, 0.88x |
| NLMS | 3.82×10⁻³ | 0.88x, 0.81x, 1.41x, 0.77x, 0.92x |
| **CNN** | **1.65×10⁻³** | **5.16x, 7.06x, 129.71x, 1.22x, 13.21x** |
| MMSE (Genie) | 1.86×10⁻²  | 4.72x, 0.11x, 6.93x, 0.08x, 0.06x |

CNN's degradation ratios (1.22x-129.71x) are, for four of the five held-out
channels, far above LMS/NLMS's tight band (0.67x-1.41x) — the exception is
"different phase, same taps count" (1.22x), where the CNN is barely hurt at
all, showing the effect depends on which specific way a channel differs, not
just that it differs. The worst case (129.71x, sign-flipped taps) is what
matters for a deployment-robustness argument, but the range is real, not a
single cherry-picked outlier presented as if it were typical. (MMSE's own
ratios vary substantially too, but for a different, expected reason: it is re-measuring
the true channel every time, so its variation reflects each channel's own
noise-enhancement/spectral-null properties, not a train/test mismatch — the
comparison that matters here is CNN vs. LMS/NLMS, which share the same
"doesn't know the true channel" starting position.)

**The effect is present but smaller at BPSK**, consistent with this
project's repeated finding that BPSK's wide 180° decision margin is less
sensitive to the mechanisms QPSK is sensitive to (Sections 3.7, 4.6, 4.7, 7):
CNN's worst-case degradation at BPSK 10dB is 2.08x (sign-flipped-taps
channel again) vs. LMS/NLMS staying within 0.98-1.10x. **The effect vanishes
at 20dB**, where every method reaches near-zero BER on every channel
regardless — there's no headroom left for a generalization gap to show up in.

## Conclusion

**This does not overturn Case Study 2's headline finding** — the CNN still
substantially outperforms LMS/NLMS whenever train and test channels match,
which is the condition every prior case study in this project actually
tested. But it means that advantage has been measured under an unrealistic
best case: a receiver that already knows, implicitly, exactly which channel
it will face, encoded into the CNN's weights during training. No real
deployment can guarantee this. Report.md Section 9.5's practical
recommendation is revised accordingly: a CNN deployed on a channel that
drifts even modestly from its training distribution should not be assumed
to retain its reported advantage without either retraining on the
deployment channel or a measured robustness margin like the one established
here.

**Open**: whether retraining the CNN on a small sample of the deployment
channel (rather than requiring an exact match, or accepting the current
all-or-nothing generalization gap) recovers most of the lost advantage — a
fast, practically-relevant follow-up this diagnostic's machinery already
supports, since it already trains and evaluates the CNN on arbitrary
channels.
