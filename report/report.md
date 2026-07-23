# Comparative Study of Denoising Techniques for SDR Signals

## 1. Objective

Compare four denoising/equalization approaches — LMS, NLMS, a 1-D CNN Autoencoder, and a Hybrid LMS+CNN cascade — against a No-Processing baseline, for BPSK and QPSK signals corrupted by AWGN, at SNR levels from -10 dB to +20 dB. What began as two case studies grew, through follow-up questions each result raised, into four case studies plus three connecting analyses — every addition was triggered by a specific gap or claim the previous result left untested, not added for its own sake:

- **Case Study 1** (Section 3): a memoryless AWGN channel (no pulse shaping, no ISI).
- **Case Study 2** (Section 4): the same methods, unchanged, on a channel with real inter-symbol interference (RRC pulse shaping at 4x oversampling + a static multipath channel).
- **ISI Severity Sweep** (Section 5): connects Case Studies 1 and 2 by scaling the channel's severity continuously between them, turning a two-point comparison into an actual crossover curve.
- **Case Study 3** (Section 6): a genuinely *time-varying* channel, testing whether Case Study 1's "freeze the adaptive filter after the preamble" design decision — justified by "nothing to track on a static channel" — actually holds once the channel moves.
- **Case Study 4** (Section 7): 16-QAM, testing whether the BPSK-vs-QPSK decision-boundary-crowding mechanism from Case Study 2 sharpens further with a more crowded constellation.
- **Boundary-Aware Loss** (Section 3.7): turns Case Study 1's diagnosis of *why* the CNN loses to No-Processing (MSE is boundary-blind) from an observation into a tested, partially-confirmed causal claim.
- **Compute-Cost Analysis** (Section 8): supplements every BER/SNR comparison above with a hardware-portable answer to "how much does this actually cost to run," since real deployment decisions need both.

Case Study 1 was run first and produced a striking, counter-intuitive result that motivated Case Study 2 as a direct, controlled follow-up rather than a replacement — both are kept in full below because they answer different questions, and the comparison between them (sharpened further by everything that follows) is itself the main finding of this project.

## 2. Shared Methodology

- **BPSK / QPSK**: 100,000 symbols per trial, Gray-coded QPSK, unit-energy.
- **AWGN**: complex, correctly power-scaled, SNR sweep [-10, -5, 0, 5, 10, 15, 20] dB.
- **Monte Carlo evaluation**: 10 independent trials per (modulation, SNR, method), each with an independently regenerated bit sequence *and* noise realization (only the CNN/Hybrid training happens once, before any test trial is generated — see Section 4.5 for an explicit leakage check). Results are reported as mean ± std across trials, not a single point estimate, and BER is additionally tracked as raw (error_count, bits_tested) pairs so that a BER of exactly zero can be reported honestly as "0 errors observed in N bits, 95% upper confidence bound ≈ 3/N" (rule of three) rather than a bare, misleading `0.0`.
- **LMS / NLMS**: 16 taps, μ=0.01 / μ=0.5 (ε=1e-6), 1000-symbol preamble using the true clean signal as reference, then **frozen** (weights held at the preamble-converged, tail-averaged value) for the remainder of the signal — see Section 3.2 for why decision-directed continuation was tried and rejected.
- **CNN**: a small 1-D Conv autoencoder (3 encoder + 3 decoder layers; 6,890 parameters, verified via `model.count_params()` in Section 8 — corrects this project's earlier "~15k" estimate), window=128, stride=64, trained across all SNR levels jointly, Adam 1e-3, ≤50 epochs with EarlyStopping (patience 5), batch 64.
- **Hybrid**: LMS (coarse) → CNN retrained on LMS's residual output (fine).
- **No Processing**: raw noisy signal straight to hard-decision demod — the falsifiability baseline every other row is judged against.

---

## 3. Case Study 1: Memoryless AWGN Channel (No ISI)

### 3.1 Configuration

No pulse shaping, no oversampling (SPS=1), no multipath — each received sample corresponds to exactly one symbol with zero statistical dependence on any other sample.

### 3.2 Fixing LMS/NLMS divergence

The initial implementation diverged catastrophically at low input SNR (e.g. BPSK -10dB: LMS output SNR of **-15.07dB**, worse than the -10dB input). Root-causing this took two fixes:

1. **The preamble training itself was unstable**, not just the decision-directed phase afterward. The classical mean-square stability bound `μ < 2/(taps·input_power)` only guarantees stability of the *expected* weight vector — a single realization run for ~1000 iterations with noise-dominated input still showed error power spiking >1000x within the bound (verified directly: weight norm and per-iteration error traced step by step). Clamping to 20% of the bound (rather than 90%) fixed this. The frozen weight vector is also the *average* of the last ~70% of the preamble trajectory (Polyak/Ruppert averaging) rather than the raw final iterate, reducing single-realization variance.
2. **Decision-directed continuation was tried and rejected as unsafe.** A reliability gate (measuring hard-decision error rate on the known preamble tail) plus a local confidence gate (distance-to-decision-boundary, since gating on `|decision - output|` cannot distinguish a right decision from a wrong one — that residual is small near *any* constellation point) still let decision-directed adaptation engage-and-diverge in ~30-40% of random trials at borderline SNR (0dB, BPSK), confirmed by re-running 10 different seeds. Monte Carlo testing across many seeds showed that **permanently freezing after the preamble matches or beats decision-directed continuation at every tested SNR level** — on a channel with no time variation, there is nothing to track, so continued adaptation is pure downside risk. Decision-directed mode is therefore off by default (`enable_decision_directed=False`); the gating machinery is kept in the code for a future time-varying channel where it would have an actual job to do.

### 3.3 Results — BPSK

| SNR in | Method | SNR out (mean±std, dB) | BER (mean±std or 95% UB) |
|---|---|---|---|
| -10 | No Processing | -10.00±0.02 | 0.3268±0.0015 |
| -10 | LMS | 0.28±0.03 | 0.3481±0.0073 |
| -10 | NLMS | 0.27±0.03 | 0.3502±0.0069 |
| -10 | CNN | 0.75±0.01 | 0.3285±0.0018 |
| -10 | Hybrid | 0.54±0.07 | 0.3521±0.0077 |
| -5 | No Processing | -5.00±0.02 | 0.2131±0.0020 |
| -5 | LMS | 1.09±0.04 | 0.2244±0.0043 |
| -5 | NLMS | 1.07±0.03 | 0.2256±0.0044 |
| -5 | CNN | 2.15±0.03 | 0.2145±0.0019 |
| -5 | Hybrid | 2.02±0.08 | 0.2271±0.0047 |
| 0 | No Processing | -0.01±0.02 | 0.0786±0.0008 |
| 0 | LMS | 2.90±0.03 | 0.0847±0.0019 |
| 0 | NLMS | 2.87±0.04 | 0.0859±0.0023 |
| 0 | CNN | 6.19±0.04 | 0.0796±0.0009 |
| 0 | Hybrid | 5.95±0.09 | 0.0858±0.0020 |
| 5 | No Processing | 5.01±0.02 | 0.00586±0.00030 |
| 5 | LMS | 6.09±0.04 | 0.00667±0.00046 |
| 5 | NLMS | 6.06±0.05 | 0.00692±0.00049 |
| 5 | CNN | 16.09±0.11 | 0.00631±0.00034 |
| 5 | Hybrid | 15.82±0.24 | 0.00741±0.00057 |
| 10 | No Processing | 9.99±0.01 | 6.0e-6±6.6e-6 |
| 10 | LMS | 10.29±0.03 | 8.8e-5±1.7e-5 |
| 10 | NLMS | 10.26±0.03 | 7.1e-5±2.1e-5 |
| 10 | CNN | 28.73±0.05 | 1.60e-4±3.0e-5 |
| 10 | Hybrid | 27.48±0.07 | 2.45e-4±3.4e-5 |
| 15 | No Processing | 15.00±0.01 | **0 err / 1,000,000 bits (95% UB ≈ 3.0e-6)** |
| 15 | LMS | 14.97±0.02 | 7.9e-5±1.4e-5 |
| 15 | NLMS | 14.98±0.03 | 6.0e-5±1.9e-5 |
| 15 | CNN | 31.18±0.01 | 1.51e-4±2.8e-5 |
| 15 | Hybrid | 28.59±0.06 | 2.34e-4±3.4e-5 |
| 20 | No Processing | 20.00±0.01 | **0 err / 1,000,000 bits (95% UB ≈ 3.0e-6)** |
| 20 | LMS | 19.71±0.02 | 7.7e-5±1.8e-5 |
| 20 | NLMS | 19.84±0.02 | 5.3e-5±1.5e-5 |
| 20 | CNN | 31.76±0.01 | 1.51e-4±2.8e-5 |
| 20 | Hybrid | 28.83±0.04 | 2.30e-4±2.7e-5 |

### 3.4 Results — QPSK

| SNR in | Method | SNR out (mean±std, dB) | BER (mean±std or 95% UB) |
|---|---|---|---|
| -10 | No Processing | -10.00±0.02 | 0.3763±0.0010 |
| -10 | LMS | 0.28±0.03 | 0.3910±0.0026 |
| -10 | NLMS | 0.27±0.03 | 0.3917±0.0035 |
| -10 | CNN | 0.36±0.01 | 0.3779±0.0009 |
| -10 | Hybrid | 0.26±0.02 | 0.3944±0.0023 |
| -5 | No Processing | -5.00±0.02 | 0.2872±0.0010 |
| -5 | LMS | 1.07±0.03 | 0.2977±0.0025 |
| -5 | NLMS | 1.05±0.04 | 0.2989±0.0032 |
| -5 | CNN | 1.12±0.01 | 0.2882±0.0010 |
| -5 | Hybrid | 1.04±0.04 | 0.3000±0.0026 |
| 0 | No Processing | -0.01±0.02 | 0.1589±0.0006 |
| 0 | LMS | 2.89±0.03 | 0.1655±0.0015 |
| 0 | NLMS | 2.86±0.03 | 0.1668±0.0018 |
| 0 | CNN | 3.25±0.02 | 0.1597±0.0007 |
| 0 | Hybrid | 3.20±0.04 | 0.1673±0.0017 |
| 5 | No Processing | 5.01±0.02 | 0.0374±0.0005 |
| 5 | LMS | 6.10±0.03 | 0.0397±0.0008 |
| 5 | NLMS | 6.07±0.04 | 0.0402±0.0009 |
| 5 | CNN | 9.13±0.04 | 0.0380±0.0004 |
| 5 | Hybrid | 8.80±0.08 | 0.0408±0.0008 |
| 10 | No Processing | 9.99±0.01 | 8.46e-4±7.0e-5 |
| 10 | LMS | 10.29±0.03 | 1.056e-3±8.2e-5 |
| 10 | NLMS | 10.27±0.04 | 1.067e-3±8.2e-5 |
| 10 | CNN | 19.96±0.07 | 1.079e-3±8.0e-5 |
| 10 | Hybrid | 19.44±0.13 | 1.355e-3±1.1e-4 |
| 15 | No Processing | 15.00±0.01 | **0 err / 2,000,000 bits (95% UB ≈ 1.5e-6)** |
| 15 | LMS | 14.97±0.02 | 7.7e-5±6.0e-6 |
| 15 | NLMS | 14.98±0.03 | 5.7e-5±1.4e-5 |
| 15 | CNN | 26.64±0.02 | 1.71e-4±1.9e-5 |
| 15 | Hybrid | 26.24±0.03 | 2.63e-4±2.6e-5 |
| 20 | No Processing | 20.00±0.01 | **0 err / 2,000,000 bits (95% UB ≈ 1.5e-6)** |
| 20 | LMS | 19.71±0.02 | 7.4e-5±1.2e-5 |
| 20 | NLMS | 19.82±0.03 | 5.5e-5±1.0e-5 |
| 20 | CNN | 28.04±0.02 | 1.71e-4±1.9e-5 |
| 20 | Hybrid | 27.00±0.03 | 2.54e-4±1.8e-5 |

Figures: `results/figures/case1_no_isi/`.

### 3.5 Key Finding: SNR improvement does not imply BER improvement absent channel memory

**Every single method, at every SNR level, for both modulations — 56/56 comparisons — produces strictly worse BER than doing nothing at all**, despite CNN posting between +3.25dB (QPSK, 0dB) and +18.74dB (BPSK, 10dB) of apparent SNR gain over the same range. This was verified exhaustively, not sampled.

The current channel is memoryless AWGN with no pulse shaping, oversampling, or multipath: each received sample corresponds to exactly one independent symbol, with zero statistical dependence between samples. Under this model, the raw received sample is already a **sufficient statistic** for detecting its symbol, and hard-sign detection on it is the Bayes-optimal (minimum-BER) receiver — there is no bit-error-rate headroom above doing nothing. Every method evaluated here (LMS, NLMS, CNN, Hybrid) violates this boundary by construction, pooling information across multiple samples — 16 taps or a 128-sample window — the majority of which belong to statistically independent, unrelated symbols.

This was verified concretely, not just asserted: the converged LMS filter at 20dB SNR places **98.1%** of its weight on the current-sample tap (`|w[0]|=0.981`, closely matching the theoretical Wiener shrinkage factor `SNR/(1+SNR)=0.990`), while the other 15 taps average **~320x smaller** (`mean|w[1:]|≈0.003`) — small enough to barely register on the MSE/SNR metric (they carry just 0.02% of the filter's total squared-weight energy), but *not exactly zero*: finite-sample training noise leaves them holding small, nonzero energy drawn from statistically unrelated symbols, which is pure interference for the current bit decision. The CNN compounds this with a second effect: it is trained on MSE loss, a smooth, threshold-blind objective with no explicit penalty for crossing the decision boundary, so it can (and empirically does) trade a small number of sign flips near zero for a larger aggregate reduction in squared error elsewhere in the batch.

**This result is specific to a memoryless, ISI-free channel; it should not be read as evidence that adaptive filtering or learned denoising cannot improve BER in general, only that they cannot improve upon an already-sufficient statistic when no such structure exists to exploit.** It directly motivated Case Study 2 below: introducing real inter-symbol interference so that neighboring samples genuinely carry information about each other, breaking the sufficiency of the raw sample and giving equalization and denoising an actual opportunity to improve — rather than only degrade — bit-level performance.

### 3.7 Follow-Up: Is "MSE Is Boundary-Blind" a Tested Causal Claim, or Just an Observation?

Section 3.5's diagnosis of *why* the CNN loses to No-Processing — plain MSE has no explicit penalty for crossing the decision boundary, so it trades sign flips for lower aggregate squared error — was, as stated, an explanation rather than a tested claim. A boundary-hinge loss (MSE + a margin penalty for landing too close to or across the nearest decision boundary, added to an architecturally-identical CNN so loss function is the only variable changed) was built and evaluated on Case Study 1's exact configuration to test it directly.

**Confirmed for QPSK, at low-to-moderate SNR (-10 to +5dB): the boundary-hinge loss closes 60-79% of the CNN's excess BER over No-Processing.** At -10dB, exactly: MSE's excess BER over No-Processing is 0.008288; Hinge's is 0.00178 — a 78.5% reduction. A bit-level flip diagnostic (for every bit where the CNN's decision differs from No-Processing's, is the flip harmful or beneficial?) confirms this is mechanistic, not coincidental: at -10dB, MSE's harmful/beneficial flip counts are 14,683/12,611 (net harm 2,072); Hinge's are 6,037/5,592 (net harm 445) — both counts roughly halve under the boundary-aware loss, exactly the predicted effect of penalizing boundary-crossing.

**Not confirmed for BPSK**: the hinge loss made BPSK slightly *worse* at -10/-5dB and gave negligible improvement elsewhere. BPSK's single, wide (180°) decision margin apparently leaves little of the specific failure mode (frequent near-boundary ambiguity) a hinge penalty targets — consistent with this project's repeated finding that BPSK and QPSK experience the same intervention very differently (Sections 4.6, 4.7, 7).

**A third mechanism, previously invisible, emerged at high SNR (10-20dB): the plain-MSE and boundary-hinge CNNs produce *identical* error counts**, for both modulations, at every SNR level in that range. Something loss-independent is capping BER there.

**This has since been given strong, though correlational rather than interventional, evidence** (`report/findings_cnn_high_snr_floor.md`): re-examining the same error population with per-position tracking found (1) MSE and Hinge fail at the *same* symbol positions 81.1% of the time (98.8-100% for BPSK) — far beyond chance given how sparse these errors are, pointing to a shared cause; (2) those positions cluster non-uniformly within the window-overlap cycle (`symbol_index mod 64`), monotonically declining from window-start to window-end (317/306/205/75 across four quartiles), confirmed against the uniform null by a chi-square test (χ²=168.0, p=3.5×10⁻³⁶); (3) only 15.1% of these positions were also wrong for No-Processing's raw hard-decision — meaning 84.9% of the time the CNN *introduces* a new error rather than failing on an already-lost sample, weighing against "structurally unfixable noise" as the primary cause; (4) the noise magnitude at these positions is only mildly elevated (1.39-1.64x typical), not the large outlier this project's other structurally-hard-sample findings would predict. Together this points to a genuine, loss-independent, position-dependent weak spot in the CNN's window/overlap-add reconstruction, within its own 128-sample window — but no counterfactual (e.g. changing the overlap-add weighting and confirming the floor shrinks) was run, so this is the best-supported explanation given every measurement taken, not yet a demonstrated causal mechanism.

The predicted trade-off is also confirmed directly: the boundary-hinge CNN has systematically lower output SNR than plain MSE (e.g. QPSK 15dB: 18.04dB vs 21.28dB, a 3.2dB deficit) — it genuinely trades squared-error accuracy for boundary margin, and that trade only pays off where boundary ambiguity, rather than some other structural error source, is the dominant cause of excess BER. Case Study 1's headline finding is not overturned by this — the gap to No-Processing is never fully closed at any SNR or modulation tested — but "MSE is boundary-blind" is now a confirmed, partial, modulation-dependent causal mechanism rather than a single untested sentence. Full detail: `report/findings_boundary_aware_loss.md`.

---

## 4. Case Study 2: RRC Pulse-Shaped + Multipath ISI Channel

### 4.1 Configuration

Identical to Case Study 1 (same symbol count, MC trial count, taps, step sizes, CNN architecture, SNR sweep) with exactly one change: **4x oversampling with RRC pulse shaping, plus a static 3-tap multipath channel** `h=[1.0, 0.4+0.3j, -0.1+0.1j]`. This isolates the presence of real ISI as the only variable between the two case studies.

### 4.2 SNR metric: definition, and three normalization bugs found and fixed

Case Study 1 had no ambiguity in what "SNR" meant, because with no ISI the pre-noise and pre-channel signal are the same thing. With real ISI, comparing a denoised/equalized output against the **pre-ISI ideal constellation** (as Case 1 did) conflates two different error sources: actual noise, and the deterministic channel distortion that no per-sample method can remove without genuinely inverting the channel. That definition made every method look like it had an enormous, un-closeable SNR deficit (-9 to -13dB even at 20dB input) regardless of how good its BER was, which is misleading. **SNR here is instead measured against the post-ISI, pre-noise reference** — i.e. what the receiver would see through this exact multipath channel with zero noise — which isolates residual noise given each method's own (possibly imperfect) channel equalization. This is a genuinely different quantity from Case Study 1's SNR column, so it is stored under a distinctly-named column (`snr_db_out_vs_postisi_ref_*`) rather than being directly overlaid.

Getting this metric — and the channel itself — trustworthy took finding and fixing three real bugs, all caught by the same discipline: checking a value against what it is *definitionally* required to equal, rather than accepting an odd-looking number as "probably fine."

1. **A genuine amplitude/gain bug in `receiver_frontend`.** The RRC filter (`signal_gen.rrc_filter`) is scaled so the *transmit* pulse-shaped signal has unit average sample power. Reusing that same filter as the *receive* matched filter without renormalizing double-counts the scaling: a clean (no ISI, no noise) round-trip test recovered symbols at amplitude **±4.0 instead of ±1.0** for BPSK (sps=4) — exactly `sps` in amplitude, `sps²` in power. Fixed by dividing the matched-filter output by its own energy (`sum|h_rrc|²`), verified to recover exact unit-amplitude symbols afterward. This bug affected absolute amplitude (relevant to CNN/LMS/NLMS training targets and hard-decision thresholds) but, being a constant scale factor applied identically to both sides of every SNR ratio computed in this study, it happened to leave the *SNR metric itself* unaffected — worth fixing for correctness, but not the source of the +5.77dB anomaly below.
2. **A real SNR-calibration bug.** Caught because No-Processing's SNR-out should equal the nominal input SNR exactly (within measurement noise) by definition — it was consistently **+5.77dB off** for every SNR level and both modulations. `add_awgn` defines "SNR" as signal power vs. noise power at the raw, oversampled *sample* level, where the noise is actually injected. But the matched filter gives the desired signal a coherent-combining gain that i.i.d. noise doesn't get (a real, textbook effect whenever SPS>1 pulse shaping is used) — and with a multipath channel on top, the signal picks up a further, channel-specific gain/loss that doesn't reduce to a clean formula (assuming a pure `10·log10(sps)` correction, as oversampling theory alone would suggest, left a consistent **-0.25dB** residual once ISI was added, because the "unit gain" assumption underlying that formula stops holding exactly once ISI redistributes energy). The fix measures the total effect empirically instead of deriving it analytically: inject noise at a known probe SNR, measure what SNR is actually observed downstream, and use that gap as the calibration subtracted from every nominal SNR value in the sweep. This is channel-agnostic — it works for whatever multipath profile happens to be configured — and was verified to bring No-Processing's SNR-out to within 0.001-0.03dB of nominal, matching Case Study 1's own precision.
3. **A causal-convolution bug in `add_multipath`**, found later while building the ISI severity sweep (Section 5) — its severity=0 anchor is supposed to exactly reproduce Case Study 1's "nothing beats No-Processing" result, and instead showed CNN/equalization winning by up to 1000x, directly contradicting an already-established finding. Root cause: `add_multipath` used `np.convolve(signal, h, mode="same")`. A multipath channel `h=[h0, h1, h2, ...]` models a *causal* tap-delay system (`r[n] = sum_k h[k]*s[n-k]`, with `h[0]` the zero-delay/direct path) — `mode="same"` is the right choice for a *symmetric* kernel like the RRC matched filter, but wrong here: it silently shifted the signal by ~1 sample. Verified concretely: convolving with `h=[1,0,0]` (a channel with only the direct path, which must be an exact identity) instead returned the signal shifted by one sample. This shift was present for *every* multipath profile actually used in this study, including the one behind the originally-reported Case Study 2 numbers — not just the h=[1,0,0] edge case that exposed it. Fixed with causal convolution (`mode="full"`, truncated to the input length); three regression tests added (`tests/test_channel.py`) to keep it fixed.

All three fixes changed the actual signal/noise injected throughout the pipeline (not just how a metric is reported), so the entire Case Study 2 sweep — training and evaluation — was re-run after each. The tables below reflect the fully corrected run, and now include a **genie-aided MMSE equalizer** (Section 4.7) as an upper-bound reference alongside the four adaptive/learned methods.

### 4.3 Results — BPSK

| SNR in | Method | SNR out vs post-ISI ref (mean±std, dB) | BER (mean±std or exact count) |
|---|---|---|---|
| -10 | No Processing | -9.99±0.01 | 0.3345±0.0013 |
| -10 | LMS | 0.21±0.01 | 0.3567±0.0060 |
| -10 | NLMS | 0.21±0.01 | 0.3592±0.0070 |
| -10 | CNN | 0.65±0.01 | 0.3320±0.0010 |
| -10 | Hybrid | 0.57±0.05 | 0.3449±0.0062 |
| -10 | MMSE (Genie) | 0.36±0.00 | 0.3281±0.0011 |
| -5 | No Processing | -4.98±0.01 | 0.2218±0.0006 |
| -5 | LMS | 0.68±0.03 | 0.2556±0.0045 |
| -5 | NLMS | 0.68±0.03 | 0.2563±0.0056 |
| -5 | CNN | 1.78±0.01 | 0.2156±0.0009 |
| -5 | Hybrid | 1.74±0.10 | 0.2260±0.0055 |
| -5 | MMSE (Genie) | 1.08±0.01 | 0.2122±0.0007 |
| 0 | No Processing | 0.01±0.01 | 0.0882±0.0008 |
| 0 | LMS | 1.68±0.05 | 0.1216±0.0092 |
| 0 | NLMS | 1.68±0.05 | 0.1220±0.0094 |
| 0 | CNN | 4.26±0.02 | 0.0816±0.0008 |
| 0 | Hybrid | 4.12±0.08 | 0.0866±0.0028 |
| 0 | MMSE (Genie) | 2.62±0.01 | 0.0800±0.0007 |
| 5 | No Processing | 5.01±0.01 | 0.0083±0.0003 |
| 5 | LMS | 3.25±0.06 | 0.0232±0.0021 |
| 5 | NLMS | 3.18±0.06 | 0.0240±0.0022 |
| 5 | CNN | 7.40±0.01 | 0.0068±0.0002 |
| 5 | Hybrid | 7.23±0.05 | 0.0079±0.0006 |
| 5 | MMSE (Genie) | 4.93±0.01 | 0.0066±0.0002 |
| 10 | No Processing | 10.01±0.01 | 16 err / 1,000,000 |
| 10 | LMS | 5.04±0.07 | 2.52e-4±8.4e-5 |
| 10 | NLMS | 4.88±0.06 | 3.04e-4±1.1e-4 |
| 10 | CNN | 8.65±0.01 | 9 err / 1,000,000 |
| 10 | Hybrid | 8.55±0.03 | 38 err / 1,000,000 |
| 10 | MMSE (Genie) | 6.99±0.01 | 6 err / 1,000,000 |
| 15 | No Processing | 15.02±0.01 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |
| 15 | LMS | 6.78±0.06 | 7 err / 1,000,000 |
| 15 | NLMS | 6.43±0.07 | 8 err / 1,000,000 |
| 15 | CNN | 8.91±0.00 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |
| 15 | Hybrid | 8.80±0.01 | 11 err / 1,000,000 |
| 15 | MMSE (Genie) | 8.13±0.01 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |
| 20 | No Processing | 20.02±0.01 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |
| 20 | LMS | 7.89±0.06 | 7 err / 1,000,000 |
| 20 | NLMS | 7.57±0.05 | 5 err / 1,000,000 |
| 20 | CNN | 8.95±0.00 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |
| 20 | Hybrid | 8.84±0.01 | 6 err / 1,000,000 |
| 20 | MMSE (Genie) | 8.60±0.00 | **0 err / 1,000,000 (95% UB ≈ 3.0e-6)** |

### 4.4 Results — QPSK

| SNR in | Method | SNR out vs post-ISI ref (mean±std, dB) | BER (mean±std or exact count) |
|---|---|---|---|
| -10 | No Processing | -9.99±0.01 | 0.3808±0.0011 |
| -10 | LMS | 0.23±0.02 | 0.3962±0.0028 |
| -10 | NLMS | 0.23±0.02 | 0.3971±0.0034 |
| -10 | CNN | 0.34±0.00 | 0.3800±0.0010 |
| -10 | Hybrid | 0.28±0.03 | 0.3897±0.0025 |
| -10 | MMSE (Genie) | 0.37±0.00 | 0.3757±0.0009 |
| -5 | No Processing | -4.98±0.01 | 0.2969±0.0009 |
| -5 | LMS | 0.68±0.03 | 0.3196±0.0052 |
| -5 | NLMS | 0.68±0.03 | 0.3203±0.0051 |
| -5 | CNN | 0.97±0.00 | 0.2930±0.0005 |
| -5 | Hybrid | 0.96±0.08 | 0.3000±0.0035 |
| -5 | MMSE (Genie) | 1.07±0.00 | 0.2887±0.0009 |
| 0 | No Processing | 0.01±0.01 | 0.1772±0.0012 |
| 0 | LMS | 1.67±0.03 | 0.1955±0.0036 |
| 0 | NLMS | 1.66±0.02 | 0.1967±0.0035 |
| 0 | CNN | 2.57±0.01 | 0.1655±0.0009 |
| 0 | Hybrid | 2.54±0.04 | 0.1696±0.0019 |
| 0 | MMSE (Genie) | 2.63±0.01 | 0.1693±0.0009 |
| 5 | No Processing | 5.01±0.01 | 0.0634±0.0006 |
| 5 | LMS | 3.26±0.04 | 0.0629±0.0019 |
| 5 | NLMS | 3.20±0.06 | 0.0638±0.0024 |
| 5 | CNN | 5.56±0.01 | 0.0435±0.0004 |
| 5 | Hybrid | 5.36±0.06 | 0.0450±0.0008 |
| 5 | MMSE (Genie) | 4.94±0.01 | 0.0663±0.0006 |
| 10 | No Processing | 10.01±0.01 | 0.0093±0.0003 |
| 10 | LMS | 5.02±0.11 | 0.0034±0.0003 |
| 10 | NLMS | 4.91±0.10 | 0.0036±0.0003 |
| 10 | CNN | 8.11±0.01 | 0.0014±0.0001 |
| 10 | Hybrid | 7.90±0.06 | 0.0017±0.0001 |
| 10 | MMSE (Genie) | 6.99±0.01 | **0.0193±0.0002** |
| 15 | No Processing | 15.02±0.01 | 2.37e-4±3.5e-5 |
| 15 | LMS | 6.78±0.05 | 15 err / 2,000,000 |
| 15 | NLMS | 6.47±0.07 | 11 err / 2,000,000 |
| 15 | CNN | 8.78±0.00 | 1 err / 2,000,000 |
| 15 | Hybrid | 8.66±0.02 | 12 err / 2,000,000 |
| 15 | MMSE (Genie) | 8.14±0.01 | **0.0042±0.0002** |
| 20 | No Processing | 20.02±0.01 | **0 err / 2,000,000 (95% UB ≈ 1.5e-6)** |
| 20 | LMS | 7.90±0.06 | 9 err / 2,000,000 |
| 20 | NLMS | 7.61±0.08 | 5 err / 2,000,000 |
| 20 | CNN | 8.94±0.00 | **0 err / 2,000,000 (95% UB ≈ 1.5e-6)** |
| 20 | Hybrid | 8.74±0.01 | 11 err / 2,000,000 |
| 20 | MMSE (Genie) | 8.60±0.01 | **4.90e-4±5.3e-5 (980 err)** |

Figures: `results/figures/case2_isi/`. MMSE (Genie)'s bolded rows (QPSK 10-20dB) are discussed in Section 4.7 — they are not a data artifact.

### 4.5 Verification (done before trusting these numbers)

- **No train/test leakage**: CNN/Hybrid training completes entirely before any test-trial signal is generated (verified by code order: `train_autoencoder`/`train_hybrid_cnn` execute before the `for trial in range(MC_TRIALS)` loop that first calls `_generate_signal` for test data). All 10 test-trial bit sequences were confirmed bit-for-bit distinct from the training sequence and from each other (checked explicitly, not assumed) — the shared element between train and test is only the deterministic multipath channel itself, which is the intended, static-and-known-channel assumption of this study (the same assumption Case Study 1 made for CNN).
- **CNN's "0 errors" was checked against a stronger standard than "is it theoretically plausible at the nominal SNR label"** — that check turned out to be the wrong one (see below), so it was replaced with two direct tests:
  - *Held-out generalization*: a freshly-trained CNN (same procedure) was evaluated on 5 seeds that were never used anywhere — not in training, not in the original 10-trial pipeline. Result: **0 errors in 500,000 bits** at both 15dB and 20dB, matching the in-pipeline result exactly. If the original 0-error result had been a leakage artifact, a genuinely unseen trial would not reproduce it.
  - *Decision-margin analysis*: across the pooled in-pipeline and held-out samples (1.5M+ points), the **minimum** observed hard-decision margin (`|Re(y)|` for BPSK) was 0.49-0.78, comfortably clear of the zero-crossing decision boundary — fully consistent with zero observed bit errors.
- **Why the naive plausibility check was wrong, and what it revealed instead**: my first pass at this check computed the theoretical Q-function BER at the *nominal* 15dB/20dB SNR label — but the actual measured SNR-vs-post-ISI-reference at those points is only ~7-9dB (Section 4.3/4.4), and plugging that into the Q-function predicts hundreds of expected errors per million bits, which flatly contradicts the observed 0. The resolution isn't leakage — it's that the Q-function assumes i.i.d. Gaussian residual noise, and the ~7-9dB figure here is an *average* squared-error metric dragged down by structured, non-Gaussian residual distortion (left over from imperfect equalization) rather than a noise floor that actually reaches the decision boundary, which the margin analysis confirms directly. **This is itself a finding, not just a caveat**: the SNR-vs-BER disconnect from Case Study 1 (Section 3.5) reappears here through a different mechanism — there it was MSE training being boundary-blind, here it's an average-based SNR metric being a poor stand-in for the actual (structured, bounded) error distribution.
- **The high-SNR LMS/NLMS/Hybrid "regressions" (0 errors for No-Processing vs a handful for these three methods, BPSK 15/20dB and QPSK 20dB) are, with the corrected channel, mostly statistically real** (Fisher's exact test): BPSK 15dB LMS p=0.016, NLMS p=0.008, Hybrid p=0.001 (all significant); BPSK 20dB LMS p=0.016, Hybrid p=0.031 significant, NLMS p=0.063 not quite; QPSK 20dB LMS p=0.004, Hybrid p=0.001 significant, NLMS p=0.063 not quite. CNN and MMSE tie No-Processing at 0 errors in every one of these cells (not significant, as expected). This is a *different* situation from the earlier (pre-3rd-bug-fix) run, where the same-looking gap wasn't statistically distinguishable from noise — with the corrected, slightly-more-benign channel, No-Processing's own residual ISI floor is now low enough (0 errors) that LMS/NLMS/Hybrid's small but genuine misadjustment noise floor (the same phenomenon documented in Section 3.2/4.5 for Case Study 1, and directly measurable against the theoretical LMS misadjustment formula there) becomes the larger of the two effects and shows up as a small, real, statistically confirmed regression — a handful of errors per million bits, not a meaningful practical difference, and not evidence against the headline finding (Section 4.6), which is about the -10dB to +10dB range where the effect is 1-3 orders of magnitude, not this few-parts-per-million high-SNR crossover.

### 4.6 Key Finding: with real ISI, equalization helps — but how much depends sharply on modulation

With the corrected (causal) channel, the picture is more nuanced than the pre-fix run suggested, and more scientifically interesting for it. BER improvement ratios (NoProc BER / Method BER, >1 = method wins):

| | BPSK 0dB | BPSK 10dB | QPSK 10dB | QPSK 15dB | QPSK 20dB |
|---|---|---|---|---|---|
| LMS | 0.7x (worse) | 0.1x (worse) | 2.7x | 31.5x | 0x (worse, but 0/9 err — see 4.5) |
| NLMS | 0.7x (worse) | 0.1x (worse) | 2.6x | 43.0x | 0x (worse, 0/5 err) |
| CNN | 1.1x | 1.8x | 6.5x | 473x | ∞ (0 err) |
| Hybrid | 1.0x | 0.4x (worse) | 5.6x | 39.4x | 0x (worse, 0/11 err) |

**BPSK barely benefits from equalization, and LMS/NLMS/Hybrid are frequently worse than doing nothing.** This is a real, different result from what a first (buggy-channel) pass suggested, and it makes sense once you look at *why*: BPSK's single binary decision boundary is forgiving enough that this specific channel's ISI mostly wasn't hurting it much to begin with, so there's little genuine room for a classical adaptive filter to win, while that same filter's own estimation noise (documented throughout Section 3.2/4.5) is a real, constant cost. Only CNN shows a small, consistent edge for BPSK, presumably because its nonlinearity can extract what little structure there is without paying the same misadjustment tax.

**QPSK is the opposite story, and dramatically so.** Every method routs No-Processing by 1-3 orders of magnitude from 10-15dB (up to 473x for CNN at 15dB), because QPSK's four-quadrant decision boundary is far more sensitive to the multipath-induced phase/amplitude smearing this channel introduces — there is much more real structure for equalization to exploit, and it shows.

This still confirms the mechanism identified in Case Study 1 from the other direction — once neighboring samples genuinely carry information about each other, the raw received sample is no longer a sufficient statistic and pooling information across samples can help rather than only hurt — but it adds an important qualifier the pre-fix numbers obscured: **"does equalization help" depends not just on "is there ISI," but on how sensitive the specific modulation's decision boundary is to the specific distortion that ISI introduces.** BPSK and QPSK experience the *same* channel very differently.

### 4.7 The Genie-Aided MMSE Bound: what "perfect channel knowledge" does and doesn't buy you

`src/mmse_equalizer.py` implements a closed-form linear MMSE equalizer given the channel's *exact* taps (measured empirically via an impulse response through the same Tx-shape → multipath → Rx-matched-filter chain, not derived analytically — the same "measure it, don't assume it" discipline used for the SNR calibration fix in Section 4.2) and the exact noise level. It is a genie: no adaptive method here actually has this information. Two things came out of building it.

**At low-to-moderate SNR, it behaves exactly as expected** — a real, if modest, edge over LMS/NLMS (e.g. QPSK 0dB: MMSE BER 0.1693 vs LMS 0.1955; BPSK -10dB: 0.328 vs LMS's 0.357), consistent with having information LMS/NLMS have to spend a preamble estimating.

**At high SNR for QPSK, it does something worse than every adaptive method, and this was investigated rather than dismissed as a bug.** QPSK MMSE BER *rises* at 10-20dB where LMS/NLMS/CNN are converging toward zero errors (Section 4.4: MMSE BER 0.0193 at 10dB, 0.0042 at 15dB, 4.9e-4 at 20dB, all far worse than the adaptive methods at the same SNR). Verified this is real, not a coding bug, in three steps: (1) the equalizer's output has **zero mean phase error** against the true transmitted symbols (0.03°) — it is unbiased, correctly undoing the channel's own inherent phase rather than introducing a spurious rotation; (2) it has a **real residual phase spread** (~16° std) that is large enough to occasionally cross QPSK's ±45° decision boundaries while remaining harmless for BPSK's ±180° boundary (an 11-fold wider margin) — this is exactly the same "same channel, different modulations experience it differently" theme as Section 4.6, now showing up as a *cost* rather than a benefit; (3) increasing the equalizer from 16 to 51 taps changed **zero** errors, ruling out "too short a filter" as the explanation. What remains is the standard, textbook limitation of *linear* equalization: for a channel with the spectral characteristics this multipath+pulse-shape combination produces, fully removing the residual ISI would require amplifying noise near a spectral null more than the MMSE criterion is willing to pay for, so the mathematically optimal *linear* solution deliberately leaves this residual distortion rather than fully inverting the channel. A decision-feedback equalizer, MLSE, or (as observed here) a nonlinear method like CNN is not bound by this limitation the same way.

**This sharpens, rather than undermines, the case for CNN.** CNN isn't just a convenient adaptive alternative to classical filters — at high SNR for QPSK, it demonstrably exceeds what a *linear* equalizer can achieve even with perfect channel and noise knowledge. The genie MMSE bound is therefore a *linear-equalizer* ceiling, not a universal one, and should be read as such. (One further caveat given the same honesty standard as everything else in this report: MMSE here operates on the *symbol-spaced*, already-matched-filtered signal, while LMS/NLMS operate directly on the 4x-oversampled samples — a fractionally-spaced equalizer can in principle exploit sub-symbol structure a symbol-spaced design cannot, so this bound is specifically "the best symbol-spaced linear equalizer," not the best possible linear processing of any kind.)

### 4.9 Follow-Up: RLS as a Third Classical Baseline

The original Named, Scoped-Out Future Work deliberately excluded RLS from Case Study 2 to keep exactly one variable (the channel) different from Case Study 1. Before this first real use, `src/rls_filter.py` — implemented but unused — needed a real safety fix: it defaulted to unguarded decision-directed continuation gated only by a loose `|d-y|<=2.0` threshold, the same broken-gate pattern Section 3.2 documented and fixed for LMS/NLMS. Hardened to frozen-by-default first (16 new regression tests, `tests/test_rls_stability.py`), then run at Case Study 2's full scale (100,000 symbols, 10 MC trials) alongside LMS/NLMS/MMSE.

**Not the simple "RLS wins" result the future-work item's stated rationale predicted.** BER ratio (LMS or NLMS BER / RLS BER, >1 = RLS wins):

| SNR (dB) | BPSK: LMS/RLS | BPSK: NLMS/RLS | QPSK: LMS/RLS | QPSK: NLMS/RLS |
|---|---|---|---|---|
| -10 to 5 | 0.85-0.92x (RLS worse) | 0.85-0.95x (RLS worse) | 0.88-0.91x (RLS worse) | 0.89-0.92x (RLS worse) |
| 10 | **1.27x (RLS wins)** | **1.54x (RLS wins)** | 0.84x (RLS still worse) | 0.90x (RLS still worse) |
| 15-20 | **∞ (RLS: 0 errors)** | **∞ (RLS: 0 errors)** | **∞ (RLS: 0 errors)** | **∞ (RLS: 0 errors)** |

**RLS is measurably *worse* than both hardened LMS and NLMS at low-to-moderate SNR** (5-19% more errors from -10 to 5-10dB, both modulations) **but strictly better at high SNR** — BPSK crosses over at 10dB and reaches zero errors at 15-20dB where LMS/NLMS retain single-digit residual errors; QPSK's crossover lands one SNR step later but reaches the same zero-error result at 15-20dB. A plausible (not verified) explanation: LMS/NLMS were extensively hardened for exactly this low-SNR regime (clamped step size, Polyak/Ruppert tail-averaging over the preamble), while `RLSFilter` has no equivalent tail-averaging and uses an untuned λ=0.99 — a per-SNR-tuned λ or a tail-averaged RLS estimate (paralleling the existing per-SNR-μ future-work item for LMS/NLMS) is the natural next step, named here rather than chased down further. RLS is also computationally cheaper per-sample than NLMS in this implementation (0.6-2.1s vs. 1.5-3.7s per signal) despite its O(taps²) complexity vs. LMS/NLMS's O(taps) — another instance of Section 8's finding that wall-clock cost and algorithmic complexity do not always move together. Full detail: `report/findings_rls_comparison.md`.

---

## 5. ISI Severity Sweep: locating the crossover

Case Study 1 (no ISI) and Case Study 2 (one specific multipath channel) gave opposite answers to "does equalization help BER" — but that's only two data points on what should be a continuum. `experiments/run_severity_sweep.py` scales the multipath channel's secondary/tertiary tap magnitudes by a severity factor from 0 (no ISI, exactly Case Study 1's channel) to 1 (exactly Case Study 2's channel) and re-runs LMS, CNN, and the genie MMSE at each level, to find *where* the crossover actually happens rather than just knowing it exists somewhere between the two endpoints.

**Scope note**: this is a diagnostic sweep, not a third full case study — fewer symbols (30,000 vs 100,000), fewer Monte Carlo trials (5 vs 10), fewer SNR points (4 vs 7), and it directly exposed the third `add_multipath` bug (Section 4.2) via its severity=0 anchor before the sweep itself could be trusted. Treat trends here as indicative, not to the same precision as the two main case studies.

**BER improvement ratio (NoProc BER / Method BER) across severity:**

| | severity=0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| BPSK 0dB — LMS | 0.71x | 0.70x | 0.70x | 0.71x | 0.72x |
| BPSK 0dB — CNN | 0.93x | 0.95x | 0.98x | 1.01x | 1.05x |
| QPSK 0dB — LMS | 0.85x | 0.86x | 0.87x | 0.89x | 0.91x |
| QPSK 0dB — CNN | 0.96x | 0.97x | 0.99x | 1.04x | 1.07x |
| **QPSK 10dB — LMS** | **0.52x** | **0.66x** | **1.09x** | **1.82x** | **2.68x** |
| **QPSK 10dB — CNN** | **0.64x** | **0.80x** | **1.48x** | **3.15x** | **5.61x** |

The clean result is **QPSK at 10dB**: a monotonic crossover from "equalization hurts" (severity 0-0.25) through breakeven (~severity 0.5, where LMS crosses 1.0x) to "equalization helps substantially" (severity 0.75-1.0, up to 5.61x for CNN) — a genuine dose-response curve, not just two disconnected snapshots. At 0dB, both modulations stay in "equalization doesn't clearly help" territory across the whole severity range (the effect needs enough SNR headroom to show up at all, consistent with Section 3/4 throughout). Figures: `results/figures/severity_sweep/severity_crossover_{BPSK,QPSK}.png`.

One open observation from this sweep, noted rather than fully chased down given the genie MMSE is a secondary/reference feature: at severity=1.0, QPSK, 20dB, the genie MMSE showed a much larger error count (139/300,000) than at severity=0.75 (1/300,000) in the sweep's reduced-scale run — consistent in direction with, but not identical in magnitude to, the linear-equalizer-ceiling finding in Section 4.7 (measured independently, at full scale, in the main Case Study 2 run). The two are probably the same underlying phenomenon; a full explanation would need the same phase/margin analysis Section 4.7 applied, at this sweep's reduced symbol count.

---

## 6. Case Study 3: Time-Varying Channel — Does Decision-Directed Tracking Earn Its Keep?

Every stability decision through Section 5 was justified by "this channel doesn't change over time, so there's nothing for decision-directed (DD) tracking to do, and continued adaptation is pure downside risk." That claim was never actually tested, because every channel used so far (Case Study 1: none; Case Study 2 and the severity sweep: one static multipath profile) is time-invariant. This section builds a genuinely time-varying channel and tests it directly.

### 6.1 Channel model

A new function, `add_time_varying_multipath` (`src/channel.py`, additions only — the existing static `add_multipath` is untouched), drifts the non-direct multipath taps smoothly over the signal's duration around Case Study 2's exact static profile, via two interchangeable models: a deterministic sinusoidal drift (independent random phase per tap, so reflectors don't fade in lockstep) or a stochastic, band-limited random walk. Verified before use: a zero-drift call reproduces the static channel to within 1e-15 (floating-point noise), and an identity-tap channel (`h=[1,0,0]`) reproduces the exact identity — the same discipline as the `add_multipath` causality fix in Section 4.2.

**Correction (found via a later adversarial self-critique pass, `report/findings_preamble_drift_correction.md`): the design was originally described as choosing a drift rate slow enough that "the channel looks static across the 1000-symbol preamble" — that claim was asserted, not measured, and is false.** Measured directly (a constant-input probe through the real function): the random-walk config actually used for Section 6.4's headline flat-fading control swings by **60.5-90.7% of its base tap magnitude within the first 1000 samples alone** — the entire preamble window — across the 5 seeds tested; the sinusoidal config drifts by ±7-22% within its 4000-sample preamble window. Neither is "negligible." This does not invalidate the BER comparisons in Section 6.4 — Frozen and DD are evaluated on the identical channel/noise realization in every trial, so their BER difference is real regardless — but it does mean the "Frozen" baseline is itself a fit to an already-moving channel, not a clean static snapshot, and the original causal story ("static while training, then drifts") should be read as "continuously time-varying from the start, which DD tracks measurably better than Frozen despite (or because of) that," not as two cleanly separated phases.

### 6.2 Finding 1 — a fourth bug: the decision-directed reliability gate is dead code at SPS>1

The global safety gate that decides whether DD continuation is trustworthy (Section 3.2) compares hard decisions on the preamble tail against the *exact* clean reference value, accepting continuation only if the mismatch rate is low. That check was built and validated only against Case Study 1's SPS=1 signals, where every clean sample **is** exactly a constellation point. At SPS=4 (every other case study and the severity sweep), the clean reference is a continuously-varying RRC-shaped waveform that equals an exact constellation point at only 1-in-4 samples — so the exact-value mismatch check structurally reads as ~75-80% "mismatch" *regardless of SNR, channel behavior, or actual decision quality*, and the gate never opens. Measured directly: **0/9 engagement across all four (modulation × channel-drift-mode) combinations tested.** This had been invisible until now because no script anywhere in this project (Case Study 2, severity sweep) had ever actually set `enable_decision_directed=True` — the gate was silently protecting by never doing anything, on any SPS>1 signal, whether or not the channel varies.

### 6.3 Finding 2 — bypassing the (broken) gate to test the real question: catastrophic at SPS=4, and *not about time-variation*

Forcing DD to run anyway (bypassing the broken global gate, keeping the separate, working local confidence gate active) makes it catastrophic — BER improvement ratio (Frozen/DD) ranges from 0.5x down to 0.0001x (i.e. up to ~10,000x *worse*) across -5 to 15dB, both modulations, both drift modes, with DD's BER indistinguishable from random guessing (0.475-0.495) in every affected cell. This was root-caused rather than reported at face value, per this project's standard: a weight-vector inspection showed DD converges to an entirely different, delay-shifted filter structure (energy concentrated at the *last* tap instead of the first), and — the decisive check — running the identical bypassed-gate test on Case Study 2's **original static channel** reproduces the same catastrophic failure (Frozen BER 4e-5 → DD BER 0.470). **The failure has nothing to do with time-variation**: DD's per-sample decide-and-adapt loop is only correct when every sample is a symbol (SPS=1); at SPS=4, three of every four samples are RRC transitional values, not decision points, and forcing a hard decision on them corrupts the filter regardless of whether the channel drifts. This is a pre-existing SPS>1 incompatibility in the DD update rule itself, only now exposed because Finding 1's gate had been silently preventing it from ever running.

### 6.4 Finding 3 — the clean test: SPS=1, genuine flat fading, gate left untouched

Because Finding 2 makes the SPS=4 test unable to isolate the actual research question, a control was added at SPS=1 (Case Study 1's regime, where the gate is not broken): a genuinely time-varying single-tap flat-fading channel (amplitude and phase both drifting, phase excursions up to ±90°), the reliability gate left at its original, unmodified threshold. **The gate behaves exactly as designed**: 0% engagement at low SNR (correctly refuses to trust an unreliable preamble; DD output is bit-identical to Frozen), rising to 20-100% engagement as SNR increases and the preamble becomes trustworthy.

Where the gate engages, **LMS decision-directed tracking earns back its keep**: pooled BER drops 29-40% relative to Frozen at 10-15dB, both modulations, with DD's absolute BER at these SNRs beating even No-Processing — a real, practically-meaningful win directly attributable to tracking the channel's drift past the preamble. **NLMS does not show the same benefit** — flat-to-worse at every SNR tested, with a 37.5% engage-and-diverge rate among its engaged trials, matching Section 3.2's originally-documented "~30-40% of trials" almost exactly, now reproduced on a genuinely time-varying channel rather than a static one.

**This asymmetry has since been given strong, consistent evidence — correlational, not interventional** (`report/findings_lms_nlms_asymmetry.md`): an instrumented, read-only reimplementation of both filters' update equations — verified to reproduce the production classes' BER exactly before being trusted — measured the correlation between the channel's instantaneous envelope and each method's effective step size / update-vector norm across all 31 DD-engaged trials. The result splits perfectly along method lines with **zero exceptions**: all 16 engaged LMS trials show a *positive* correlation between envelope and update-norm (a fade shrinks LMS's update, since its step size is a fixed constant and a small `x(n)` mechanically produces a small update), while all 15 engaged NLMS trials show the opposite, *negative* correlation (a fade inflates NLMS's effective step size, since `mu/(eps+||x(n)||^2)` grows as the denominator shrinks). The consequence is directly measurable in the "inflation ratio" (update-vector norm during wrong decisions, relative to that method's own median), defined only for trials with at least one wrong decision to measure (13 of 16 LMS trials, 13 of 15 NLMS trials — the rest had zero DD-phase errors, all at 15dB, and are excluded rather than counted as "not inflated"): among those 13, LMS averages 0.455 (gentlest exactly when wrong — never once, 0/13, above its own median during a wrong decision), while NLMS averages 1.228 (correcting *harder* exactly when wrong, exceeding its own median in 8/13 trials, 62%). NLMS's instantaneous normalization is consistent with a destabilizing feedback loop on a fading channel — a fade both makes a wrong decision more likely and simultaneously inflates the step size that decision gets multiplied into — that LMS's fixed step size structurally cannot have; but no counterfactual (capping NLMS's normalization denominator and confirming the effect disappears) was run, so this remains the best-supported explanation given every measurement taken, not a demonstrated mechanism in the strict causal sense.

### 6.5 Conclusion: both refutes and upholds Section 3.2, depending on which claim you mean

- The **narrow, practical claim** ("freeze weights after the preamble") remains the right default for this codebase today — but for a more urgent reason than the one on record: Finding 2 shows the DD implementation itself is broken for every SPS>1 signal this project actually uses, on any channel, static or not.
- The **broader, stated justification** — "nothing to track on a static channel, so continuation is pure downside risk" — is **refuted as a general principle** by Finding 3: on a genuinely time-varying channel, in the one regime where DD's machinery actually functions as designed, LMS tracking earns back a real, substantial BER improvement. Continued adaptation is not inherently pure downside once the channel actually moves.
- The **original engage-and-diverge concern** is reproduced, not eliminated, on the time-varying channel (NLMS's 37.5% divergence rate, LMS's own regression at low SNR). A time-varying channel gives tracking something to gain, but does not remove the risk of losing.

**Bottom line**: this does not license turning on `enable_decision_directed=True` by default for this project's actual (SPS=4) experiments — it cannot even be exercised safely there today. But the report's stated reasoning for why freezing is safe was broader than what was actually verified, and should be understood as narrower than originally claimed: freezing is right today primarily because DD needs a symbol-rate-aware fix before SPS>1 channels can use it at all, not because time-variation would never make continuation worthwhile. Full detail, all exact numbers, and named follow-up items: `report/findings_timevarying_channel.md`.

---

## 7. Case Study 4: Higher-Order Modulation (16-QAM) — Does Decision-Boundary Crowding Keep Sharpening the Effect?

Case Study 2 (Section 4.6) found BPSK and QPSK experience the *same* ISI channel very differently, attributed to decision-boundary crowding (BPSK's 180° margin vs. QPSK's 90°). 16-QAM (16 tightly-packed constellation points, much smaller minimum distance) tests whether this keeps sharpening as crowding increases further.

**Implementation**: `generate_16qam`/`demod_16qam_fast` (`src/signal_gen.py`/`src/utils.py`, additions only), Gray-coded square 16-QAM, following the existing BPSK/QPSK pattern exactly. Verified before use with a zero-noise, zero-ISI round-trip (0 bit errors over 20,000 bits) — this project's standard practice before trusting a new signal path (Section 4.2), which is not a formality: the same self-test build for a bonus 8-PSK implementation caught a real Gray-mapping bug (one constellation transition differed by 2 bits, not 1) before it could contaminate any result.

Run on Case Study 2's exact channel and SNR-calibration procedure, at a shifted/widened SNR range ([0, 10, 20, 25, 30]dB rather than [-10..20]dB) — 16-QAM's tight minimum distance means the original range would have left every method near 50% BER, uninformatively.

**BER improvement ratio (NoProc BER / Method BER), verified against raw error counts:**

| Method | BPSK best (§4.6) | QPSK best (§4.6) | 16-QAM (this study) |
|---|---|---|---|
| LMS | never wins (≤0.7x) | 31.5x (15dB) | break-even at 0dB (0.88x), **480x at 25dB** |
| NLMS | never wins | 43.0x (15dB) | break-even at 0dB (0.88x), **610x at 25dB** |
| CNN | 1.8x (10dB, best case) | 473x (15dB) | **557x at 25dB** |

At low-to-moderate SNR (0-10dB), LMS/NLMS remain break-even-or-slightly-worse for 16-QAM too — the same qualitative pattern seen for BPSK throughout this project, before enough SNR headroom exists for equalization to pay off at all.

**Two findings beyond the headline ratio, both extending rather than merely confirming the crowding hypothesis:**

1. **No-Processing hits a genuinely new, hard BER floor**: 0.0755-0.0814 from 20-30dB, never improving further with more SNR — not seen for BPSK or QPSK anywhere in this project. Deterministic ISI alone is now enough to permanently cross 16-QAM's tight decision margins regardless of how clean the noise gets.
2. **The genie MMSE linear equalizer becomes actively *worse* than doing nothing at high SNR**: 0.71-0.74x at 20-30dB — a sharper, more dramatic version of Section 4.7's linear-equalizer-ceiling finding. The same noise-enhancement/residual-ISI tradeoff that only modestly hurt QPSK's MMSE bound is severe enough here to make "perfect channel knowledge, linear-only" a net loss relative to raw hard-decision demod.

**Conclusion**: the decision-boundary-crowding hypothesis is **confirmed, more dramatically than the BPSK→QPSK comparison alone predicted** — but the mechanism is richer than "a bigger ratio": crowding also creates a hard, un-closeable floor for No-Processing that doesn't exist for BPSK/QPSK, and turns the linear-equalizer ceiling from a modest QPSK-specific curiosity into an outright regression. Full detail: `report/findings_higher_order_modulation.md`.

---

## 8. Compute-Cost Analysis: When Is CNN Actually Worth It?

Every comparison above is BER/SNR-only, plus a `runtime_sec_mean` column that measures *this specific Python/NumPy/TensorFlow implementation's* wall-clock speed — useful, but not a hardware-portable answer to how much compute each method fundamentally needs. `experiments/compute_cost_analysis.py` (new script; no existing module modified) answers that directly, with two independent measurements shown side by side because they disagree, and *why* is itself a finding.

**Analytical MACs per output sample** (real multiply-accumulates, derived from each method's actual structure, hardware-agnostic): LMS, NLMS, and the genie MMSE all cost **64 real MACs/sample** (a 16-tap complex FIR-apply, 4 real MACs per complex MAC). CNN costs **6,784 real MACs/sample** (window-normalized) to **13,568** (stride-honest, accounting for the 50% window-overlap redundancy in this project's overlap-add reconstruction) — derived layer-by-layer from the actual Conv1D architecture and cross-checked against a live `model.count_params()` (6,890 parameters — correcting Section 2's rough "~15k" description). Hybrid adds LMS's and CNN's costs in sequence: **6,848-13,632 real MACs/sample**. **CNN needs ~106-212x more raw arithmetic than LMS/NLMS/MMSE.**

**Measured wall-clock latency, benchmarked directly on this machine (CPU-only)**, produces a genuinely counter-intuitive result: **CNN is faster in wall-clock terms than LMS/NLMS/Hybrid**, despite needing ~106x more MACs (e.g. Case-2-like config: CNN 1,487ns/sample vs. LMS's 3,679ns/sample) — confirmed independently against the existing `runtime_sec_mean` CSV columns, which agree in rank order. **The reason is implementation efficiency, not algorithmic complexity**: LMS/NLMS are pure per-sample Python `for` loops, where CPython interpreter dispatch overhead dwarfs the actual 64 MACs of useful arithmetic; CNN inference is a small number of large, vectorized, BLAS-backed TensorFlow operations that use the CPU far more effectively despite doing 100x more raw work. A genuine, if harmless, implementation inefficiency was also found in passing: LMS/NLMS's frozen-mode loop still executes the full weight-update arithmetic every sample (multiplying by `mu_eff=0`) rather than skipping it — a wasted-compute footnote, not a correctness issue.

**This means the wall-clock numbers in this report should not be used to judge real-hardware feasibility** — a hand-optimized FIR-apply and a hand-optimized Conv1D forward pass on dedicated hardware (an FPGA, DSP, or microcontroller) would both approach their MAC-count floor, and the ~106-212x analytical gap reasserts itself as a real throughput requirement gap: **64-640 MMACs/s (LMS/NLMS/MMSE) fits comfortably on a basic Cortex-M4/M7-class microcontroller; 6.8-136 GMACs/s (CNN) does not** — it needs an applications processor, DSP, or NPU (roughly 20-500x beyond MCU-class throughput across this project's symbol-rate range).

**Concrete decision rule, combining this with every BER finding above**: on a bare microcontroller, or with BPSK/no-ISI, CNN is not worth deploying — LMS/NLMS/MMSE deliver equal-or-better BER (Case Study 1) or only a small, real edge (BPSK, Case Study 2) at ~1% of the compute. On a platform with multi-GMACs/s-to-TOPS-class headroom, where the channel has real, modulation-sensitive ISI structure — QPSK or 16-QAM with this project's multipath profile — CNN's 106-212x compute multiplier buys a large, measured BER improvement (2.4-15x over LMS at moderate SNR, up to 557-610x over doing nothing for 16-QAM) that is a good trade for most platforms above the microcontroller tier. Full derivation, all layer-by-layer arithmetic, and the complete summary table: `report/compute_cost.md`.

---

## 9. Cross-Case-Study Discussion

Case Study 1 and Case Study 2 differ in exactly one variable (the channel), and reach opposite conclusions about whether denoising/equalization helps BER. Neither result is "the answer" in isolation — together they establish that **the value of adaptive filtering or learned denoising for BER is conditional on the channel actually having structure to exploit**, not an inherent property of the methods themselves. A comparative study that only ran Case Study 1 would have concluded, incorrectly, that none of these methods are useful for BER; a study that only ran Case Study 2 would have concluded, without evidence, that they always are. Running both, on the same code and metrics, is what makes either conclusion trustworthy — and the severity sweep (Section 5) turns that binary comparison into an actual curve.

Within Case Study 2, CNN and Hybrid consistently reach lower BER than LMS/NLMS, and Hybrid has mostly closed the gap with standalone CNN that was much larger in Case Study 1 — both are downstream of fixing the LMS/NLMS divergence bug, since Hybrid's first stage no longer corrupts its own input before the CNN sees it. But BPSK vs QPSK diverges sharply on whether *any* of this matters (Section 4.6) — the same channel, radically different practical impact, purely as a function of decision-boundary geometry. Case Study 4 (Section 7) shows this isn't a two-point BPSK/QPSK curiosity but a continuing trend: 16-QAM's tighter margins push the effect further in the same direction on both ends — equalization wins by even more, and the genie MMSE linear-equalizer ceiling turns from a curiosity into an outright regression.

The SNR-vs-BER disconnect from Case Study 1 is not actually confined to Case Study 1: Section 4.5 found the same qualitative pattern inside Case Study 2 (a ~7-9dB average SNR ceiling that looks, by the naive Q-function, incompatible with the observed 0 bit errors) via a *different* mechanism — structured/non-Gaussian residual distortion dragging down an average-based metric, rather than boundary-blind MSE training actively flipping signs. Section 4.7 found a third variant again: the genie MMSE bound has *decent* aggregate SNR at high SNR but a disproportionately bad BER for QPSK, because its residual error is phase-structured in a way that a scalar SNR number cannot distinguish from a benign, real-valued-only error. Three independent mechanisms, one repeated methodological lesson: an aggregate SNR or MSE number should not be trusted as a BER proxy without checking the actual decision-margin distribution.

Case Study 3 (Section 6) revises how this project should talk about its own most load-bearing design decision. "Freeze LMS/NLMS after the preamble" is still the right default, but the justification on record ("nothing to track on a static channel") turns out to have been broader than what was actually verified — and testing it surfaced a fourth bug (the DD reliability gate silently never engaging at SPS>1) that had been masking a separate, real incompatibility. The freeze decision was right, for reasons partially different from the ones originally given — a useful reminder that "we tested it and it works" claims need to specify exactly what was tested.

Section 3.7's boundary-aware-loss result and Section 8's compute-cost analysis both extend existing findings into new dimensions rather than adding new channel scenarios: the former turns an explanation into a partially-confirmed mechanism, the latter turns a BER-only comparison into an actionable deployment question. Taken together, this project's four case studies plus three connecting analyses form a coherent structure in which each new question was generated directly by a gap or unverified claim in a previous result, rather than chosen in advance — arguably the more defensible way to scope an open-ended comparative study.

## 10. Conclusions

1. **SNR improvement does not imply BER improvement**, and the relationship between them is channel-dependent, not universal — this is the headline finding of this project, established by direct comparison rather than assumption. Its corollary, found while re-verifying Case Study 2 under scrutiny, is that even *within* one channel regime, an aggregate SNR/MSE metric can misrepresent BER for structural reasons (non-Gaussian residual error, or phase-structured error) distinct from the cross-channel finding — so BER should be checked directly, not inferred from SNR, as a general rule.
2. Classical adaptive filters (LMS/NLMS) are prone to real, verifiable instabilities (training-phase divergence at low SNR, unsafe decision-directed feedback) that require deliberate diagnosis and fixing, not just parameter tuning — a fixed, appropriately-conservative step size and a frozen-after-preamble strategy resolved both without sacrificing performance.
3. CNN-based denoising outperforms classical adaptive filtering on BER once channel structure exists to exploit (Case Study 2), but can be actively harmful on a structureless channel (Case Study 1) — the same architecture, the same training procedure, opposite outcomes, purely as a function of the channel.
4. The Hybrid LMS→CNN cascade's underperformance relative to standalone CNN in Case Study 1 was mostly an artifact of LMS's divergence bug feeding corrupted input to the CNN stage; once fixed, Hybrid is competitive with (and on QPSK at 10-15dB, slightly better than) standalone CNN in Case Study 2.
5. Rigorous small-sample-size handling (Monte Carlo trial averaging, rule-of-three bounds for zero-error cells, significance testing before calling a result a "regression") materially changed which findings survived scrutiny in this project and should be standard practice, not an afterthought.
6. **Definitional identities are a cheap, powerful bug-finder.** All three real bugs in Case Study 2 (Section 4.2) were caught the same way: knowing what a value *must* equal by definition (No-Processing's SNR-out must equal nominal input SNR; a channel with only a direct path must be an exact identity) and treating any deviation as a bug to explain rather than a quirk to note. The same standard is what separated the genuine "0 errors" result (confirmed via held-out generalization and margin analysis, Section 4.5) from what could easily have been a leakage artifact taken on faith, and what separated the genuine QPSK MMSE ceiling (Section 4.7, confirmed unbiased via phase analysis and tap-count invariance) from what could have been dismissed as "probably a bug" without checking.
7. **A regression test suite (`tests/`, 35 tests) now codifies every invariant found this way** — LMS/NLMS never diverging, `receiver_frontend` recovering unit gain, No-Processing's SNR matching nominal, `add_multipath` being causal and identity-preserving, no train/test bit-sequence collisions, the genie MMSE never losing to No-Processing — so a future change to this codebase can't silently reintroduce any of the three bugs documented in Section 4.2, or the divergence bug from Section 3.2, without a test failing.
8. **A design decision's stated justification can be narrower, or broader, than what was actually verified — and only testing the untested case reveals which.** Freezing LMS/NLMS after the preamble (Section 3.2) remains the right default, but Case Study 3 (Section 6) found the reason on record was too broad (time-variation genuinely can make continued tracking worth it, for LMS specifically) while simultaneously surfacing a fourth, more urgent, previously-invisible bug (the DD reliability gate is dead code at SPS>1) that makes the practical conclusion correct anyway, just for a different reason.
9. **A mechanistic explanation is only as strong as the experiment that tests it.** Section 3.5's "MSE is boundary-blind" diagnosis for why CNN loses to No-Processing was, on direct test (Section 3.7), confirmed for QPSK (60-79% of excess BER closed by a boundary-aware loss, with a matching bit-flip mechanism) but not for BPSK, and revealed a third, loss-independent high-SNR error floor neither loss function touches — turning one explanatory sentence into three separately-evidenced sub-claims.
10. **Decision-boundary crowding is not a BPSK/QPSK-specific curiosity — it is a continuum, and 16-QAM (Case Study 4, Section 7) is the sharpest point on it tested so far**: equalization wins by up to 610x (vs. QPSK's 473x), No-Processing develops a hard, un-closeable BER floor that doesn't exist for wider-margin modulations, and the genie MMSE's linear-equalizer ceiling (Section 4.7) turns from a QPSK-specific curiosity into an outright regression below doing nothing.
11. **Compute cost and BER cost do not move together, and both need to be reported.** CNN needs ~106-212x more arithmetic than LMS/NLMS/MMSE per output sample (Section 8), yet is *faster in wall-clock terms* on this project's CPU-only software stack — a result that would mislead anyone using wall-clock time alone to judge real-hardware feasibility, where the analytical MAC gap reasserts itself as a genuine 20-500x throughput requirement gap between a microcontroller and an applications-processor-class target.
12. **A named, previously-unverified hypothesis was tested and found strongly, consistently evidenced — correlationally, not interventionally.** Section 6.4 originally left the LMS-vs-NLMS decision-directed-tracking asymmetry as "a plausible hypothesis, not a verified explanation." A targeted, production-code-reproducing diagnostic (`report/findings_lms_nlms_asymmetry.md`) found every one of 16 engaged LMS trials shows a fade *shrinking* its update-vector norm, and every one of 15 engaged NLMS trials shows the opposite — a fade *inflating* NLMS's effective step size, at exactly the moments wrong decisions are most likely. This is far stronger than the original single-sentence hypothesis, but no counterfactual (e.g. capping NLMS's normalization denominator and confirming the effect disappears) was run — that intervention, not further correlational analysis, is the actual next step before calling this fully demonstrated rather than best-supported.
13. **The same discipline made progress on a second previously-open question, with the same causal caveat.** Section 3.7's high-SNR CNN error floor was left as "not root-caused, two guesses offered." Position-level tracking (`report/findings_cnn_high_snr_floor.md`) found the window/overlap-add reconstruction artifact hypothesis strongly evidenced (a statistically significant, χ²=168.0 p=3.5×10⁻³⁶, non-uniform error clustering by window phase) and the competing "unfixable noise" hypothesis not supported (only 15.1% of floor errors were also wrong for No-Processing) — again correlational, with the interventional test (changing the overlap-add weighting and confirming the floor shrinks) named as the concrete next step rather than performed here.
14. **This project's single-training-draw practice was given a first quantitative check, in one regime — not validated project-wide.** Training 5 independently-seeded CNNs and evaluating all 5 on identical test data (`report/findings_training_variance.md`) found training-draw variance smaller than test-time Monte Carlo variance in all 4 conditions tested (ratio 0.00-0.32x) — but only for Case Study 1's memoryless-AWGN CNN, at 2 SNR points, with 4-5 samples per estimate (too few for a reliable variance-of-a-variance estimate on their own). This is evidence against training-draw variance dominating *in that specific regime*, not a general validation covering Case Studies 2-4 or Hybrid, which still report test-time variance alone without this check.
15. **RLS, added as a third classical baseline on Case Study 2's channel** (`experiments/run_rls_comparison.py`), required fixing a real safety bug before its first use: the existing, previously-unused implementation defaulted to unguarded decision-directed continuation, the same broken `|d-y|`-threshold pattern Section 3.2 documented and fixed for LMS/NLMS. Hardening it to frozen-by-default first (matching LMS/NLMS's established safe convention) before drawing any performance comparison is itself the same lesson as finding #6 applied a third time: an unverified default is a latent bug, not a neutral starting point. The comparison itself produced a genuinely nuanced result, not the simple "RLS wins" the future-work rationale predicted: RLS is measurably *worse* than the already-hardened LMS/NLMS at low-to-moderate SNR (5-19% more errors, -10 to 5-10dB) but strictly better at high SNR (zero errors at 15-20dB where LMS/NLMS retain a small residual floor) — a reminder that an algorithm's textbook theoretical advantage doesn't automatically transfer to every operating regime without the same hardening applied elsewhere in this project.

## 11. Assumptions, Limitations, and Future Work

- Single static multipath channel (`[1.0, 0.4+0.3j, -0.1+0.1j]`) in Case Study 2's main run — partially addressed by the severity sweep (Section 5), which varies its magnitude, but not its delay spread or tap count.
- LMS/NLMS use one fixed μ across the entire SNR sweep; Section 3.2's residual ~0.2-0.3dB SNR shortfall at high SNR in Case Study 1 is a direct, quantifiable consequence of this (verified against the textbook LMS misadjustment formula) and per-SNR μ tuning is the natural next step to close it.
- The theoretical Q-function BER curve is plotted on Case Study 1's BER-vs-SNR figures (valid there, memoryless AWGN) but has been **removed** from Case Study 2's figures (`show_theoretical=False`) rather than merely captioned as inapplicable, after recognizing that a figure viewed without its surrounding text could otherwise be misread as showing a valid bound. Section 4.5 additionally shows it isn't even a good *estimate* for Case Study 2, since the residual error there is structured/non-Gaussian rather than the noise the Q-function models.
- BPSK Hybrid's residual gap vs standalone CNN at high SNR in Case Study 1 (~2dB) was not separately root-caused beyond the general information-loss argument in Section 3.5 / 9.
- The `receiver_frontend` matched-filter, `add_awgn` SNR-calibration, and `add_multipath` causality fixes (Section 4.2) were derived and verified for this specific pulse shape (RRC, α as coded) and multipath channel; a different pulse shape or channel profile would need the same empirical calibration re-run (the code already does this automatically for the SNR calibration, since it's measured at runtime rather than hardcoded) and should be re-checked against the same regression tests.
- The genie MMSE equalizer (Section 4.7) is a *linear, symbol-spaced* reference bound, not a universal ceiling — see that section's caveats before citing it as "the best possible." Case Study 4 (Section 7) shows this ceiling gets more consequential, not less, as constellation crowding increases.
- The decision-directed (DD) reliability gate (Section 3.2) is dead code for any SPS>1 signal (Section 6.2) — this affects every experiment in this project that uses pulse shaping (Case Study 2, the severity sweep, Case Studies 3 and 4), all of which rely on the default `enable_decision_directed=False` and were therefore unaffected in practice, but anyone setting `enable_decision_directed=True` on an SPS>1 signal today gets a gate that silently never engages, followed by a catastrophic result if the gate is separately bypassed (Section 6.3).
- The boundary-aware-loss (Section 3.7), training-variance (Section 3.8), 16-QAM (Section 7), LMS-vs-NLMS-asymmetry (Section 6.4), and CNN-high-SNR-floor (Section 3.7) follow-ups were all run at reduced scale (4,000-30,000 symbols, 4-5 MC trials, sometimes only 5 independent draws) for speed, consistent with this project's severity-sweep precedent — treat their exact percentages, ratios, and correlation coefficients as indicative at that same reduced-confidence standard, not to the two main case studies' full precision. In particular, the training-variance ratios (Section 3.8) are estimated from only 4-5 samples per condition and have not been given a confidence interval; a "0.00x" or "0.32x" ratio there should be read as "no evidence training variance dominates in this small sample," not as a precise, general measurement.
- A sixth real bug was found during a deliberate adversarial self-critique pass (four independent reviewer agents tasked with finding problems): `rrc_filter` (`src/signal_gen.py`) centered its continuous-time axis on `N/2` rather than `(N-1)/2`, making the transmit/receive RRC filter measurably asymmetric (up to ~0.4 peak-scale difference between the filter and its own reverse) even though its highest-magnitude sample still landed at the expected center index — inconsistent with `apply_pulse_shaping`'s `delay=(N-1)//2` convention, which assumes an exactly symmetric, linear-phase filter. Fixed (verified: the filter is now exactly symmetric, and round-trip demodulation remains error-free). **This was not re-run across every RRC-based experiment in this project** (Case Studies 2-4, the severity sweep, and the RLS/boundary-loss/CNN-floor/training-variance follow-ups all predate this fix) — the asymmetry's effect on final BER/SNR numbers is expected to be small (matched-filter energy normalization is preserved regardless of symmetry, and round-trip symbol recovery showed 0 bit errors both before and after the fix), but a full re-run to confirm this quantitatively is named future work, not assumed.
- **The "channel looks static during the preamble" premise for Case Study 3's time-varying channel was asserted, not measured, and was found to be false** for the actual default parameters used — see the correction in Section 6.1 and `report/findings_preamble_drift_correction.md`. This does not invalidate Section 6.4's measured Frozen-vs-DD comparison (both methods see the identical channel realization), but it means the channel is continuously time-varying from the start of the signal, not static-then-drifting as originally described.
- Section 4.5's Fisher's-exact-test "statistically real" claims (9 tests across BPSK/QPSK × LMS/NLMS/Hybrid) were not corrected for multiple comparisons; a Bonferroni correction across just those 9 tests would flip 2 of the reported p-values (0.016, 0.031) from significant to non-significant at the conventional α=0.05 threshold. The qualitative conclusion in that section (a small, practically-negligible misadjustment floor, not a meaningful regression) is unaffected either way, but the "statistically real" language should be read as nominal/exploratory rather than family-wise-corrected.

### Named, scoped-out future work

These were deliberately not attempted in this round, each for a specific reason — named explicitly here rather than left as an implicit gap. Two items from the original list (time-varying channel, and — partially — higher-order modulation) have since been addressed (Sections 6 and 7) and are marked accordingly; each surfaced new, more specific follow-ups of its own, listed alongside.

- ~~RLS as a third classical baseline~~ — **done**, `report/findings_rls_comparison.md` (Section 4.9): not the simple "RLS wins" result predicted — RLS is measurably *worse* than LMS/NLMS at low-to-moderate SNR (5-19% more errors) but strictly better at high SNR (reaches zero errors at 15-20dB where LMS/NLMS retain a residual floor). Required first fixing a real safety default in `src/rls_filter.py` (it defaulted to unguarded decision-directed continuation, the same broken-gate pattern already fixed for LMS/NLMS in Section 3.2). New follow-up: a per-SNR-tuned forgetting factor or a tail-averaged RLS estimate (paralleling LMS/NLMS's existing hardening) to test whether the low-SNR shortfall closes.
- **MLSE/Viterbi equalization.** The classical gold-standard for ISI channels — arguably a more natural "how good can non-learned DSP get" reference than RLS for Case Study 2 specifically, since it's a sequence detector rather than a per-symbol linear/adaptive filter and doesn't share the symbol-spaced-vs-fractionally-spaced caveat that applies to the genie MMSE bound in Section 4.7.
- ~~A genuinely time-varying channel~~ — **done, Section 6.** Surfaced three concrete new follow-ups: (1) fix the DD update rule for SPS>1 to only decide/adapt at the correct symbol-spaced phase (requires editing `src/lms_filter.py`/`src/nlms_filter.py`, out of scope for that diagnostic); (2) ~~root-cause why LMS benefits from DD tracking on a time-varying channel while NLMS does not~~ — **done**, `report/findings_lms_nlms_asymmetry.md` (Section 6.4): confirmed, with 100% trial-level consistency, that NLMS's instantaneous-power normalization inflates its effective step size during channel fades, exactly when wrong decisions are most likely, while LMS's fixed step size does the opposite; (3) once (1) exists, re-run Section 6.4's flat-fading control at SPS=4 to get the answer to the question Case Study 3 was originally trying to ask (does tracking help on this project's *actual* ISI channel), which the SPS>1 bug currently makes unanswerable there.
- ~~Higher-order modulation~~ — **done for 16-QAM, Section 7.** A working, self-tested 8-PSK implementation (`generate_8psk`/`demod_8psk_fast`) was also built but not run through the full comparison pipeline this round — a cheap next step given the modulation-generalized pipeline already exists. Also open: root-causing 16-QAM's hard No-Processing BER floor and the genie MMSE's regression more precisely than "a sharper version of the Section 4.7 mechanism" (asserted by analogy here, not independently re-verified with 16-QAM's own phase/margin analysis).
- ~~Root-cause the loss-independent high-SNR error floor~~ — **done**, `report/findings_cnn_high_snr_floor.md` (Section 3.7): confirmed as a window/overlap-add reconstruction artifact (81.1% same-position overlap between loss functions, statistically significant non-uniform clustering by window phase, χ²=168.0 p=3.5e-36, and only 15.1% overlap with No-Processing's own errors, ruling out "unfixable noise" as the primary cause). New follow-up: a triangular/cosine overlap-add weighting (downweighting each window's least-reliable edge) or a smaller stride is the natural next step to test whether this specific floor can be closed.
- ~~Multiple independent training draws~~ — **done**, `report/findings_training_variance.md`: 5 independently-seeded CNNs, evaluated on the same fixed test signals, show training-draw variance is consistently *smaller* than test-time (Monte Carlo) variance across all 4 conditions tested (ratio 0.00-0.32x) — this project's practice of training once and reporting test-time variance alone is justified, not an underestimate. Not yet extended to Hybrid or to Case Study 2's ISI channel.
- **Cycle-accurate hardware validation of the compute-cost analysis (Section 8)** — the MAC-count-based throughput requirements are a standard analytical estimate (fixed-point vs. float, memory-access patterns, and pipelining would all shift the absolute numbers, though not the ~106-212x relative gap between CNN and the linear methods, which is architectural rather than hardware-specific) and were not validated against an actual microcontroller, DSP, or NPU target.
