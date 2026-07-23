"""
LMS Adaptive Filter
=====================
Standard Least-Mean-Squares adaptive filter for denoising complex-valued
signals.

The filter treats the clean signal as the "desired" signal and the noisy
signal as the "input".  In a realistic scenario the desired signal is not
available; we use a *reference* approach where the noisy signal is both the
input and the filter estimates the clean component.  For this comparative
study we use the common supervised-style LMS formulation where the clean
signal serves as the reference during an initial *training* pass (simulating
a preamble). Afterwards, the filter switches to Decision-Directed mode where
it uses its own hard-decisions as the reference signal.

Stability notes (see issue write-up in project report)
--------------------------------------------------------
Two distinct failure modes were found and fixed here:

1. The preamble training itself was unstable at low input SNR. The classical
   mean-square stability bound mu < 2/(taps*input_power) only guarantees
   stability of the *expected* weight vector; a single realisation run for
   only ~1000 iterations with noise-dominated input still showed large
   weight excursions (error power spiking 1000x) well inside that bound.
   Clamping to a much smaller fraction of the bound (20% rather than 90%)
   keeps steady-state misadjustment low enough to avoid this.
2. Naive decision-directed adaptation after the preamble adapts toward the
   filter's own (possibly wrong) hard decisions. Gating on |decision - y| (as
   a "confidence" proxy) does not work: that residual is small whenever y is
   close to *any* constellation point, right or wrong, so it cannot actually
   detect a bad decision. A reliability gate (measuring the hard-decision
   error rate on the preamble tail, where the true symbols are known) and a
   local confidence gate (distance-to-decision-boundary) were tried as
   safeguards, but even a fairly strict reliability threshold (10% mismatch)
   still let decision-directed mode engage-and-diverge in ~30-40% of random
   trials at borderline SNR (0 dB, BPSK): a channel-static-in-time scenario
   like this one gets no benefit from continuing to adapt past the preamble
   (there is nothing to track), so any residual chance of feedback-driven
   divergence is pure downside. Monte Carlo testing across many random seeds
   also showed permanently freezing after the preamble performs equal-or-
   better than decision-directed continuation at *every* tested SNR level, so
   decision-directed mode defaults to off (`enable_decision_directed=False`).
   The gating machinery is kept and can be re-enabled for a future
   time-varying/ISI channel where tracking would actually have a job to do.
Additionally, the frozen "converged" weight vector is the *average* of the
last portion of the preamble trajectory (Polyak/Ruppert averaging) rather
than the single final iterate, which reduces the residual variance of the
point estimate and noticeably tightens the achievable output SNR at high
input SNR (where the channel is otherwise near-noiseless and the only
remaining error source is finite-sample training noise).

Implementation note:
    Because the signal is complex-valued, we operate on the real (I) and
    imaginary (Q) components independently with identical filter
    coefficients, which is equivalent to a widely-linear real-valued LMS on
    the interleaved I/Q stream.  This avoids phase-ambiguity issues with
    complex LMS for this particular denoising task.
"""

import numpy as np
from dataclasses import dataclass, field


def _decision_fn(modulation: str):
    if modulation == "BPSK":
        return lambda y: 1.0 + 0j if y.real > 0 else -1.0 + 0j
    return lambda y: (1.0 if y.real > 0 else -1.0) / np.sqrt(2) + 1j * (1.0 if y.imag > 0 else -1.0) / np.sqrt(2)


def _confidence_fn(modulation: str):
    """Distance from a filter output to the nearest decision boundary,
    normalised so that a perfectly-placed symbol has confidence 1.0."""
    if modulation == "BPSK":
        return lambda y: abs(y.real)
    return lambda y: min(abs(y.real), abs(y.imag)) * np.sqrt(2)


@dataclass
class LMSFilter:
    """LMS adaptive filter.

    Parameters
    ----------
    num_taps : int
        Filter order (number of coefficients).
    mu : float
        Step size.  Must satisfy mu < 2 / (num_taps * input_power) for
        stability.  Default 0.01 is conservative for unit-power signals.
    """
    num_taps: int = 16
    mu: float = 0.01
    _weights: np.ndarray = field(default=None, repr=False)
    _error_history: list = field(default_factory=list, repr=False)
    _dd_enabled: bool = field(default=True, repr=False)

    def denoise(self, clean: np.ndarray, noisy: np.ndarray, training_length: int = 1000,
                modulation: str = "BPSK", leakage: float = 1e-4,
                min_confidence: float = 0.3, reliability_threshold: float = 0.1,
                enable_decision_directed: bool = False):
        """Denoise a complex signal using LMS with a preamble + (optional)
        Decision-Directed continuation.

        Parameters
        ----------
        clean : np.ndarray, complex
            Clean reference signal (used for adaptation only during preamble).
        noisy : np.ndarray, complex
            Noisy signal to denoise.
        training_length : int
            Number of initial samples to use the clean reference (preamble).
        modulation : str
            'BPSK' or 'QPSK', used to set the decision boundaries in decision-directed mode.
        leakage : float
            Leakage factor for Leaky LMS (regularises against tap blow-up).
        min_confidence : float
            Minimum normalised distance-to-decision-boundary required to allow
            a decision-directed weight update on a given sample (local gate).
            Only used if enable_decision_directed=True.
        reliability_threshold : float
            Maximum tolerable hard-decision error rate measured on the tail of
            the preamble, required to allow decision-directed continuation at
            all (global gate). Only used if enable_decision_directed=True.
        enable_decision_directed : bool
            If False (default), the filter holds its preamble-converged
            weights fixed for the rest of the signal -- the safe choice for a
            time-invariant channel, see module docstring. If True, decision-
            directed adaptation is attempted after the preamble, subject to
            the reliability/confidence gates above (still not guaranteed safe
            at borderline SNR).

        Returns
        -------
        denoised : np.ndarray, complex
            Filtered output.
        error_history : np.ndarray, float
            Per-sample squared error magnitude (for convergence plot).
        """
        N = len(noisy)
        w = np.zeros(self.num_taps, dtype=np.complex128)
        output = np.zeros(N, dtype=np.complex128)
        errors = np.zeros(N, dtype=np.complex128)

        # Check stability bound. NOTE: 2/(taps*input_power) is only the bound
        # for *mean* stability of the weight vector; a single realisation run
        # for ~1000 iterations still shows large excursions/misadjustment
        # anywhere close to that bound (empirically, LMS diverged with error
        # power reaching 1e4+ at 90% of the bound at -10 dB input SNR, where
        # noise dominates input_power). Clamping to 20% of the bound keeps
        # steady-state misadjustment low enough for stable convergence even
        # when the input is noise-dominated.
        input_power = np.mean(np.abs(noisy) ** 2) + 1e-12
        mu_max = 2.0 / (self.num_taps * input_power)
        mu = min(self.mu, 0.2 * mu_max)

        decide = _decision_fn(modulation)
        confidence = _confidence_fn(modulation)

        training_length = min(training_length, N)

        # --- Phase 1: preamble (clean reference) ---
        avg_start = max(self.num_taps, int(training_length - 0.7 * (training_length - self.num_taps)))
        w_sum = np.zeros(self.num_taps, dtype=np.complex128)
        w_count = 0
        for n in range(self.num_taps, training_length):
            x = noisy[n : n - self.num_taps : -1]
            y = np.vdot(w, x)
            d = clean[n]
            e = d - y
            w = (1 - mu * leakage) * w + mu * np.conj(e) * x
            output[n] = y
            errors[n] = e
            if n >= avg_start:
                w_sum += w
                w_count += 1

        # Use the averaged tail of the preamble trajectory as the converged
        # filter -- reduces single-realisation variance vs. the raw final
        # iterate (Polyak/Ruppert averaging).
        w = w_sum / w_count if w_count > 0 else w

        # --- Reliability check: would our own decisions be trustworthy? ---
        # (only matters if decision-directed mode is requested at all)
        if enable_decision_directed:
            eval_start = max(self.num_taps, training_length - 500)
            if training_length > eval_start:
                tail_out = output[eval_start:training_length]
                tail_true = clean[eval_start:training_length]
                tail_decisions = np.array([decide(y) for y in tail_out])
                mismatch_rate = np.mean(np.abs(tail_decisions - tail_true) > 1e-9)
            else:
                mismatch_rate = 1.0  # no data to judge reliability -> be conservative
            dd_enabled = mismatch_rate <= reliability_threshold
        else:
            dd_enabled = False
        self._dd_enabled = bool(dd_enabled)

        # --- Phase 2: decision-directed (or frozen) ---
        for n in range(training_length, N):
            x = noisy[n : n - self.num_taps : -1]
            y = np.vdot(w, x)
            d = decide(y)
            e = d - y

            if dd_enabled and confidence(y) >= min_confidence:
                mu_eff = mu
            else:
                mu_eff = 0.0
            w = (1 - mu_eff * leakage) * w + mu_eff * np.conj(e) * x

            output[n] = y
            errors[n] = e

        # Fill initial samples with noisy input (filter hasn't converged yet)
        output[: self.num_taps] = noisy[: self.num_taps]

        self._weights = w
        self._error_history = np.abs(errors) ** 2
        return output, self._error_history
