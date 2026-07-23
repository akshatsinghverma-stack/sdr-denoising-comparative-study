"""
NLMS Adaptive Filter
=====================
Normalized Least-Mean-Squares adaptive filter.  Same interface as LMS but
normalises the step size by the instantaneous input power, giving faster
convergence and automatic stability regardless of input power level.

    w(n+1) = w(n) + mu / (eps + ||x(n)||^2) * e(n) * x(n)

See lms_filter.py module docstring for the decision-directed stability
safeguards (global preamble-reliability gate + local confidence gate, and
preamble tail-weight averaging) shared by both filters.
"""

import numpy as np
from dataclasses import dataclass, field

from src.lms_filter import _decision_fn, _confidence_fn


@dataclass
class NLMSFilter:
    """NLMS adaptive filter.

    Parameters
    ----------
    num_taps : int
        Filter order.
    mu : float
        Normalised step size (0 < mu < 2 for stability, typically 0.1-1.0).
    eps : float
        Regularisation constant to avoid division by zero.
    """
    num_taps: int = 16
    mu: float = 0.5
    eps: float = 1e-6
    _weights: np.ndarray = field(default=None, repr=False)
    _error_history: list = field(default_factory=list, repr=False)
    _dd_enabled: bool = field(default=True, repr=False)

    def denoise(self, clean: np.ndarray, noisy: np.ndarray, training_length: int = 1000,
                modulation: str = "BPSK", leakage: float = 1e-4,
                min_confidence: float = 0.3, reliability_threshold: float = 0.1,
                enable_decision_directed: bool = False):
        """Denoise a complex signal using NLMS with a preamble + (optional)
        Decision-Directed continuation.

        Same interface as LMSFilter.denoise().
        """
        N = len(noisy)
        w = np.zeros(self.num_taps, dtype=np.complex128)
        output = np.zeros(N, dtype=np.complex128)
        errors = np.zeros(N, dtype=np.complex128)

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
            norm_factor = self.mu / (self.eps + np.real(np.vdot(x, x)))
            w = (1 - self.mu * leakage) * w + norm_factor * np.conj(e) * x
            output[n] = y
            errors[n] = e
            if n >= avg_start:
                w_sum += w
                w_count += 1

        w = w_sum / w_count if w_count > 0 else w

        # --- Reliability check: would our own decisions be trustworthy? ---
        if enable_decision_directed:
            eval_start = max(self.num_taps, training_length - 500)
            if training_length > eval_start:
                tail_out = output[eval_start:training_length]
                tail_true = clean[eval_start:training_length]
                tail_decisions = np.array([decide(y) for y in tail_out])
                mismatch_rate = np.mean(np.abs(tail_decisions - tail_true) > 1e-9)
            else:
                mismatch_rate = 1.0
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
                mu_eff = self.mu
            else:
                mu_eff = 0.0
            norm_factor = mu_eff / (self.eps + np.real(np.vdot(x, x)))
            w = (1 - mu_eff * leakage) * w + norm_factor * np.conj(e) * x

            output[n] = y
            errors[n] = e

        output[: self.num_taps] = noisy[: self.num_taps]
        self._weights = w
        self._error_history = np.abs(errors) ** 2
        return output, self._error_history
