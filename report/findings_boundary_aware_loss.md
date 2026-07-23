# Findings: Does a Boundary-Aware Loss Confirm "MSE Is Boundary-Blind" as the Causal Mechanism?

**Status**: diagnostic follow-up making report.md Section 3.6's explanation testable. New files
only: `src/cnn_boundary_aware.py`, `experiments/run_case1_boundary_loss.py`,
`results/tables/results_boundary_loss.csv`. No existing function was modified — the plain-MSE CNN
here is architecturally identical to the original, so loss function is the only variable changed.

## Hypothesis being tested

Report.md Section 3.6 diagnosed why the CNN loses to No-Processing in Case Study 1's memoryless
AWGN channel: it's trained with plain MSE, a smooth, symmetric, threshold-blind objective with no
explicit penalty for crossing the decision boundary, so it trades a small number of sign flips for
a larger aggregate reduction in squared error. That was an observation/explanation, not a tested
causal claim. If the diagnosis is right, a loss function that *does* penalize crossing the boundary
should partially close the CNN-vs-No-Processing gap.

## Implementation

`src/cnn_boundary_aware.py`: a boundary-hinge loss — MSE plus `relu(margin − sign(y_true)·y_pred)`
per I/Q component (masked so BPSK's always-zero Q-channel contributes no spurious gradient) —
added to an architecturally identical CNN autoencoder, so loss function is the only independent
variable between "CNN (MSE)" and "CNN (Boundary-Hinge)" below. `experiments/run_case1_boundary_loss.py`
mirrors Case Study 1's configuration (SPS=1, memoryless AWGN, WINDOW_LEN=128, STRIDE=64, full SNR
sweep, SEED=42), reduced to 25,000 symbols × 5 MC trials for speed, and additionally tracks a
bit-level flip diagnostic: for every bit where the CNN's decision differs from No-Processing's, is
that flip harmful (No-Processing was right, CNN is wrong) or beneficial (the reverse)?

## Results (BER, verified against raw error/bit counts in results_boundary_loss.csv)

| | BPSK -10dB | BPSK 10-20dB | QPSK -10dB | QPSK 15-20dB |
|---|---|---|---|---|
| No Processing | 0.3258 | ~0 (rule-of-3 UB 2.4e-5) | 0.3755 | ~0 (rule-of-3 UB 1.2e-5) |
| CNN (MSE) | 0.3294 | 8.32e-4 (103-104 err/125,000) | 0.3838 | 7.88e-4 (197 err/250,000) |
| CNN (Boundary-Hinge) | 0.3329 (worse) | 8.32e-4 (identical) | 0.3773 (**78.5% of excess BER closed**) | 7.88e-4 (identical) |

**QPSK excess-BER-over-No-Processing, exact**: MSE excess = 0.383828 − 0.37554 = 0.008288.
Hinge excess = 0.37732 − 0.37554 = 0.00178. Reduction = 1 − (0.00178/0.008288) = **78.5%**.

## Findings

1. **QPSK, low-to-moderate SNR (-10 to +5dB): confirmed.** Boundary-hinge cuts the excess BER over
   No-Processing by 60-79% across this range. The flip diagnostic explains why directly: at
   -10dB, MSE's harmful/beneficial flip counts are 14,683/12,611 (net harm 2,072); Hinge's are
   6,037/5,592 (net harm 445) — both flip counts roughly halve under the boundary-aware loss, and
   net harm shrinks by more than half. This is a mechanistic confirmation, not just a BER-number
   coincidence: the loss change measurably changes *how many* boundary-crossing decisions the
   network makes, in the predicted direction.
2. **BPSK: not confirmed.** Boundary-hinge made BPSK slightly *worse* at -10/-5dB (0.3329 vs
   0.3294 at -10dB) and gave negligible-to-no improvement elsewhere. BPSK's single, wide (180°)
   decision margin apparently leaves little of the specific failure mode (frequent near-boundary
   ambiguity) that a hinge penalty is designed to fix — consistent with this project's broader,
   repeated finding that BPSK and QPSK experience the same intervention very differently
   (report.md Sections 4.6, 4.7, and the higher-order-modulation findings).
3. **High SNR (10-20dB), both modulations: the effect vanishes completely** — MSE and Hinge
   produce *identical* error counts at every SNR level from 10-20dB (BPSK: 104/104, 103/103,
   103/103; QPSK: 197/197 at both 15 and 20dB). This points to a separate, loss-independent error
   floor (plausibly a window/overlap-add reconstruction artifact at window boundaries, or a small
   population of structurally hard-to-denoise samples) that neither loss function touches — not
   root-caused in this diagnostic, flagged as a specific open question rather than a re-derived
   guess.
4. **The predicted trade-off is confirmed directly**: Boundary-Hinge has systematically *lower*
   output SNR than MSE (e.g. QPSK 15dB: 18.04dB vs 21.28dB, a 3.2dB deficit) — it genuinely trades
   squared-error accuracy for boundary margin, exactly as designed. That cost only pays off in the
   regime (QPSK, low-moderate SNR) where boundary ambiguity, not some other structural error
   source, is the dominant cause of excess BER.

## Conclusion

**The "MSE is boundary-blind" diagnosis is confirmed as a real, causal, partial mechanism for
QPSK — and shown to not be the whole story for BPSK.** Case Study 1's headline finding (every
method loses to No-Processing) is not overturned — the gap is never fully closed at any SNR or
modulation tested — but it is now more precisely understood: for QPSK, a large fraction of the
excess BER is directly attributable to boundary-blind training and can be measurably (if not
fully) recovered by a boundary-aware loss; for BPSK, the mechanism must be something else (or the
margin is already wide enough that this particular failure mode barely occurs); and at high SNR,
a third, loss-independent error source dominates for both modulations. This turns the original
one-sentence explanation into three separately-testable sub-claims, two of which now have direct
evidence and one of which is a concrete, named open question.
