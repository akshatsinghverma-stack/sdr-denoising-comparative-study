# Findings: What Causes the Loss-Independent, High-SNR CNN Error Floor?

**Status**: root-causes the open question report.md Section 3.7 (and
`report/findings_boundary_aware_loss.md`) explicitly left unresolved: "a
third, loss-independent error floor appears at 10-20dB for both modulations
... plausibly a window/overlap-add reconstruction artifact ... or a small
population of structurally hard-to-denoise samples ... not root-caused in
this diagnostic." New file only: `experiments/diagnose_cnn_high_snr_floor.py`.
No existing module modified.

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

## Conclusion

**The window/overlap-add reconstruction artifact (hypothesis 1) is strongly and consistently
evidenced as the dominant cause; the "structurally hard sample" explanation (hypothesis 2) is not
supported as the primary driver — but this is observational, not interventional, evidence, and
should be described accordingly.** All four lines of evidence point the same direction: the two
loss functions fail at the same positions (shared architectural cause), those positions have a
highly significant, non-uniform phase pattern within the window-overlap cycle (a positional
signature, not a noise-magnitude signature), the raw signal was usually already fine there (ruling
out "unfixable" noise), and the noise present is only mildly, not extremely, elevated. This is much
stronger than a single plausible-sounding hypothesis, but no counterfactual was run — changing the
overlap-add scheme (a triangular/cosine weighting, or a smaller stride) and confirming the floor
specifically shrinks in the predicted way was not attempted here, only proposed. Until that
intervention is run, this remains the best-supported explanation consistent with every measurement
taken, not a demonstrated causal mechanism in the strict sense. This turns the report's
"not root-caused, two guesses offered" status into a specific, evidenced (though not yet
interventionally confirmed) mechanism: the CNN's window-based inference has a genuine,
loss-independent reconstruction weak spot tied to position within its
own 128-sample window, most concentrated toward one side of the overlap
cycle — a concrete, testable target for future work (e.g., a triangular/
cosine overlap-add weighting that downweights each window's least-reliable
edge instead of averaging all overlapping predictions equally, or a smaller
stride to reduce how much of each window's low-confidence region ends up
carrying full weight in the final reconstruction).
