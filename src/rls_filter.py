"""
RLS Adaptive Filter
===================
Recursive Least Squares adaptive filter for complex-valued signals.
Provides faster convergence than LMS/NLMS, particularly for correlated inputs
(like ISI channels), at the expense of higher computational complexity.

Decision-directed default (see src/lms_filter.py module docstring for the
full history): this module's original implementation continued adapting
past the preamble by default, gated only by `np.abs(d - y) <= freeze_threshold`
with a loose default threshold (2.0) that almost always passes -- exactly the
"gating on |decision - output|" pattern documented in Section 3.2 as broken
(that residual is small whenever y is close to *any* constellation point,
right or wrong, so it cannot actually detect a bad decision). Since this
class was unused anywhere in the project when that hardening was done, the
same fix applies here now, before its first real use (Case Study 2's RLS
baseline): decision-directed continuation defaults OFF
(`enable_decision_directed=False`), matching LMSFilter/NLMSFilter's
established safe default -- on a static channel, freezing after the preamble
matches or beats continued adaptation (Section 3.2), so there is nothing to
lose by defaulting to the safe choice here too.

Tail-averaging (added later, see report/findings_rls_comparison.md): this
class originally froze on the raw final preamble iterate, unlike LMS/NLMS's
Polyak/Ruppert average of the last ~70% of the preamble trajectory. This was
flagged during a self-critique pass as an unexplained, untested asymmetry --
tested directly (not just fixed and assumed to help): tail-averaging is now
applied identically to LMS/NLMS's, and the RLS-vs-LMS/NLMS comparison was
re-run to measure the actual effect (see the findings file for the result).
"""

import numpy as np
from dataclasses import dataclass, field

@dataclass
class RLSFilter:
    """Complex RLS adaptive filter.

    Parameters
    ----------
    num_taps : int
        Filter order (number of coefficients).
    lam : float
        Forgetting factor (0 < lam <= 1). Typical values: 0.99 to 1.0.
    delta : float
        Initialization constant for the inverse correlation matrix P.
    """
    num_taps: int = 16
    lam: float = 0.99
    delta: float = 1.0
    _weights: np.ndarray = field(default=None, repr=False)
    _error_history: list = field(default_factory=list, repr=False)

    def denoise(self, clean: np.ndarray, noisy: np.ndarray, training_length: int = 1000,
                modulation: str = "BPSK", enable_decision_directed: bool = False,
                freeze_threshold: float = 2.0):
        """Denoise a complex signal using RLS with a preamble + (optional)
        Decision-Directed continuation.

        Parameters
        ----------
        enable_decision_directed : bool
            If False (default, the safe choice for a time-invariant channel --
            see module docstring), the filter holds its preamble-converged
            weights fixed for the rest of the signal, identical in spirit to
            LMSFilter/NLMSFilter's default. If True, RLS continues adapting
            using its own hard decisions, gated only by the (weak)
            `freeze_threshold` check -- not recommended without the same
            reliability/confidence gating LMS/NLMS use, which this class does
            not yet implement.
        """
        N = len(noisy)
        w = np.zeros(self.num_taps, dtype=np.complex128)
        P = np.eye(self.num_taps, dtype=np.complex128) / self.delta
        output = np.zeros(N, dtype=np.complex128)
        errors = np.zeros(N, dtype=np.complex128)

        if modulation == "BPSK":
            decide = lambda y: 1.0 + 0j if y.real > 0 else -1.0 + 0j
        else:
            decide = lambda y: (1.0 if y.real > 0 else -1.0) / np.sqrt(2) + 1j * (1.0 if y.imag > 0 else -1.0) / np.sqrt(2)

        training_length = min(training_length, N)
        # Polyak/Ruppert tail-averaging over the last ~70% of the preamble,
        # identical convention to LMSFilter/NLMSFilter -- reduces the
        # single-realization variance of the frozen estimate vs. the raw
        # final iterate.
        avg_start = max(self.num_taps, int(training_length - 0.7 * (training_length - self.num_taps)))
        w_sum = np.zeros(self.num_taps, dtype=np.complex128)
        w_count = 0

        for n in range(self.num_taps, N):
            x = noisy[n : n - self.num_taps : -1]
            y = np.vdot(w, x)  # w^H x

            if n < training_length:
                d = clean[n]
                update = True
            elif enable_decision_directed:
                d = decide(y)
                update = (freeze_threshold is None) or (np.abs(d - y) <= freeze_threshold)
            else:
                d = decide(y)  # for error-tracking only; weights stay frozen
                update = False

            e = d - y

            if update:
                # Calculate Kalman gain k
                Px = P @ x
                xH_P = np.conj(x).T @ P
                denom = self.lam + np.vdot(x, Px) # x^H P x
                k = Px / denom

                # Update weights
                w = w + k * np.conj(e)

                # Update inverse correlation matrix
                P = (P - np.outer(k, xH_P)) / self.lam

            if n < training_length and n >= avg_start:
                w_sum += w
                w_count += 1

            output[n] = y
            errors[n] = e

            # Once the preamble ends, switch to the tail-averaged estimate
            # for every subsequent sample's output (this only affects
            # reporting/downstream use of w -- the recursion above already
            # used the raw per-sample w while inside the preamble).
            if n == training_length - 1 and w_count > 0:
                w = w_sum / w_count

        output[: self.num_taps] = noisy[: self.num_taps]
        self._weights = w
        self._error_history = np.abs(errors) ** 2
        return output, self._error_history
