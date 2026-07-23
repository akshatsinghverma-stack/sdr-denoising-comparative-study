# Findings: What Causes the Loss-Independent, High-SNR CNN Error Floor?

**Status**: root-causes the open question report.md Section 3.7 (and
`report/findings_boundary_aware_loss.md`) explicitly left unresolved: "a
third, loss-independent error floor appears at 10-20dB for both modulations
... plausibly a window/overlap-add reconstruction artifact ... or a small
population of structurally hard-to-denoise samples ... not root-caused in
this diagnostic." New file only: `experiments/diagnose_cnn_high_snr_floor.py`.
No existing module modified.

## Correction (found via a later code-correctness critique pass)

**Everything below this notice, up to "Correction: re-running with the fix,"
describes the original analysis and its conclusion — that the floor was a
window/overlap-add reconstruction artifact. That conclusion was wrong.** A
subsequent independent code-correctness review (an agent briefed to search
`src/` directly for undiscovered bugs, not to re-read this document's
conclusions) found that `reconstruct_from_windows` (`src/cnn_autoencoder.py`)
silently left any trailing samples not covered by a full window at their
initialized value of exactly `0.0` — for this diagnostic's 25,000-symbol,
SPS=1 configuration, the last 40 samples of every processed signal, every
trial (32 samples for the main case studies' 100,000-symbol configuration).
A hard-decision demodulator reads a raw `0.0` as a fixed, deterministic bit
(BPSK's `real >= 0` rule always decides bit 1) regardless of what was
transmitted — a constant, SNR-independent error source sitting exactly in
the SNR range where a "floor" above the vanishing noise-driven error rate
would be visible. This is this project's **seventh real bug**.

Fixed by giving `reconstruct_from_windows` a `fallback` array (the noisy
input) to fill genuinely uncovered positions instead of leaving them at
zero — matching the "edge samples pass through unmodified" convention
`src/mmse_equalizer.py` already used elsewhere. `denoise_signal` now passes
this by default. Five new regression tests (`tests/test_cnn_autoencoder.py`)
lock in both the tail-coverage gap's existence (previously zero test
coverage on this function) and the fix's behavior.

### Correction: re-running with the fix

Re-running `experiments/diagnose_cnn_high_snr_floor.py` unchanged except for
the fix gives a dramatically different picture:

| | BPSK 10dB | BPSK 15dB | BPSK 20dB | QPSK 10dB | QPSK 15dB | QPSK 20dB |
|---|---|---|---|---|---|---|
| Errors, before fix (MSE) | 83/100,000 | 83/100,000 | 83/100,000 | 365/200,000 | 161/200,000 | 161/200,000 |
| Errors, after fix (MSE) | 1/100,000 | 0/100,000 | 0/100,000 | 218/200,000 | 0/200,000 | 0/200,000 |

**BPSK's "floor" — a constant 83 errors at every one of 10/15/20dB, the
single clearest signature of a floor in the whole diagnostic — collapses
almost entirely.** QPSK's floor at 15-20dB also collapses completely to
zero. Recomputing the phase-clustering statistic (finding 2 below) on the
post-fix error population gives quartile counts of 77/81/75/85 (out of
318 total) — **χ²≈0.74, p≈0.86**, statistically indistinguishable from
uniform, versus the original χ²=168.0, p=3.5×10⁻³⁶. The original clustering
result was itself manufactured by the bug: the always-wrong, zero-filled
tail sits at a small, fixed range of `symbol_index mod 64` values (fixed
by the signal's length, not by anything about denoising), which is exactly
what would fabricate spurious non-uniform clustering out of a handful of
deterministic tail errors sitting at the same position every trial.

**One genuine, smaller residual floor survives the fix, at QPSK 10dB only**
(365→218/200,000, a real ~40% reduction, not full elimination). Re-examining
just this remaining population now points to hypothesis 2 (below), not
hypothesis 1: 43.7% of remaining error positions are also wrong for
No-Processing (up from 15.1% pre-fix), and the noise magnitude at these
positions averages 2.83x the trial's typical level (up from 1.39-1.64x) —
consistent with a small population of genuinely hard noise realizations,
not a positional reconstruction artifact.

**This also retroactively explains the negative intervention result below**:
the triangular overlap-add weighting was tested and found not to shrink the
floor. That is exactly what should happen if the floor is dominated by a
zero-filled tail no window covers at all — reweighting only redistributes
influence among windows that already cover a sample, and structurally cannot
touch a position with zero coverage. The intervention's negative result was
correct; the diagnosis of what it was testing against was not.

The analysis below is preserved as originally written, for transparency
about what the first-pass (wrong) conclusion was and how it was reached —
it should now be read as "strong correlational evidence for a mistaken
hypothesis," a useful cautionary example in its own right (see report.md
Conclusion #9), not as a currently-endorsed explanation.

---

## The two candidate explanations being distinguished

1. **Window/overlap-add reconstruction artifact**: the CNN operates on
   128-sample windows (stride 64, i.e. 50% overlap) and every interior sample
   is reconstructed by averaging two overlapping windows' predictions. If
   Conv1D-with-'same'-padding predictions are systematically less accurate
   near a window's own edges (regardless of loss function, since this is an
   architectural/receptive-field property), the floor would be a positional
   artifact.
2. **Structurally hard-to-denoise samples**: a small population of
   unusually extreme noise realizations that push a symbol across its
   decision boundary hard enough that no reasonable model — MSE- or
   hinge-trained — could recover it.

## Method

Retrained both CNN variants (identical architecture/procedure to
`run_case1_boundary_loss.py`, 4 MC trials for speed) and, in addition to
aggregate BER, recorded the exact symbol position of every error at 10/15/20dB
for both models, cross-referenced against: whether No-Processing (raw
hard-decision) was *also* wrong at that position, the raw per-sample noise
magnitude at that position relative to the trial's typical level, and the
position's phase modulo the window stride (64).

## Results

**1. MSE and Hinge overwhelmingly fail at the *same* positions, not just the
same count.** Of 903 distinct error positions pooled across both models, 732
(81.1%) are shared — far more overlap than chance would produce given how
sparse these errors are (well under 1% of all symbols). Broken down: BPSK is
98.8-100% shared at all three SNRs; QPSK is 100% shared at 15-20dB but only
59.2% at 10dB (10dB is the edge of the originally-reported 10-20dB floor
range, where ordinary residual-noise errors likely still mix in alongside
floor errors). **This alone is strong evidence the two loss functions share
a common failure cause**, not two independently-arrived-at coincidences.

**2. Error positions cluster non-uniformly within the window-overlap cycle —
confirmed statistically, not just observed.** Binning by `symbol_index mod
64` into quartiles: [0,16)=317, [16,32)=306, [32,48)=205, [48,64)=75 —
monotonically declining from window-start to window-end, the opposite of the
~226-per-bucket uniform expectation. A chi-square goodness-of-fit test against
the uniform null gives **χ²=168.0, p=3.5×10⁻³⁶** — this pattern is not
noise. This directly confirms hypothesis 1: whatever the overlap-add
reconstruction does, it does it unevenly across the window-local position,
concentrated toward one side.

**3. Only 15.1% of floor-error positions were also wrong for No-Processing.**
If these were "structurally hard" samples that no method could fix, raw
hard-decision (which fails purely as a function of the noise realization at
that exact sample, independent of any denoising) should fail there too, at a
much higher rate. Instead, **84.9% of the CNN's floor errors occur exactly
where the raw, unprocessed signal was already correct** — the CNN (whichever
loss) is actively introducing a new error, not merely failing to fix an
already-lost cause.

**4. Noise magnitude at error positions is only modestly elevated, not
extreme.** Mean ratio to the trial's typical (median) noise magnitude is
1.64x (median 1.39x) — a mild elevation, far short of the large outlier
ratios (3-5x+) a genuine noise-tail explanation would predict.

## Intervention: testing the proposed fix — a negative result, reported honestly

The named next step was to change the overlap-add weighting (downweighting each window's own
least-reliable edge) and check whether the floor specifically shrinks. This was implemented
(`reconstruct_from_windows(..., weighting="triangular")`, `src/cnn_autoencoder.py` — default
`"uniform"` reproduces the original behavior exactly, so no existing experiment is affected) and
tested directly: a single trained CNN's window-level predictions were computed once, then
reconstructed twice — once with the original uniform weighting, once with triangular — isolating
the reconstruction scheme as the only variable (`experiments/test_cnn_overlap_weighting_intervention.py`).

**Result: triangular weighting did not shrink the floor.** BPSK error counts were identical at
every SNR tested (84/84, 83/83, 83/83 at 10/15/20dB); QPSK was unchanged at 15-20dB (0/0 both) and
very slightly *worse*, not better, at 10dB (98 errors uniform vs. 100 triangular, out of 100,000
bits) — the opposite direction of the predicted effect, though small enough to plausibly be noise
at this scale.

## Conclusion (original, pre-correction — see the correction notice at the top of this file)

**The positional clustering finding stands — it is real, statistically significant, and not an
artifact of chance — but the specific proposed fix (reweighting the overlap-add) does not close the
gap, so the underlying mechanism is not as simple as "give less weight to each window's unreliable
edge."** This is a genuine negative result, reported as such rather than omitted: the four
correlational lines of evidence (shared error positions between loss functions, significant phase
clustering, low overlap with No-Processing's own errors, only mild noise elevation) remain valid
observations about *where* the floor's errors occur, but the tested intervention shows the cause is
not simply "the reconstruction under-trusts/over-trusts specific window positions in a way a
different weighting can fix." A plausible remaining explanation — not yet tested — is that the
degradation is baked into the window predictions *themselves* near their edges (a genuine
receptive-field/boundary effect inside the Conv1D layers), not just in how those predictions get
blended afterward, which reweighting cannot fix because it only changes how already-degraded
predictions are combined, not the predictions themselves. This remains a specific, evidenced,
partially-open question rather than either a fully solved one or an untested guess: the *where*
is confirmed; the *why*, and a working fix, are not.

**As the correction notice at the top of this file explains, this conclusion turned out to be
wrong**: the clustering was manufactured by a zero-fill bug in `reconstruct_from_windows`, not a
property of the reconstruction scheme. See "Correction: re-running with the fix" above for what
actually explains the floor.
