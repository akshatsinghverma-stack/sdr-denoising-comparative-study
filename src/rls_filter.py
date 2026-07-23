"""
RLS Adaptive Filter
===================
Recursive Least Squares adaptive filter for complex-valued signals.
Provides faster convergence than LMS/NLMS, particularly for correlated inputs
(like ISI channels), at the expense of higher computational complexity.
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

    def denoise(self, clean: np.ndarray, noisy: np.ndarray, training_length: int = 1000, modulation: str = "BPSK", freeze_threshold: float = 2.0):
        N = len(noisy)
        w = np.zeros(self.num_taps, dtype=np.complex128)
        P = np.eye(self.num_taps, dtype=np.complex128) / self.delta
        output = np.zeros(N, dtype=np.complex128)
        errors = np.zeros(N, dtype=np.complex128)

        if modulation == "BPSK":
            decide = lambda y: 1.0 + 0j if y.real > 0 else -1.0 + 0j
        else:
            decide = lambda y: (1.0 if y.real > 0 else -1.0) / np.sqrt(2) + 1j * (1.0 if y.imag > 0 else -1.0) / np.sqrt(2)

        for n in range(self.num_taps, N):
            x = noisy[n : n - self.num_taps : -1]
            y = np.vdot(w, x)  # w^H x
            
            if n < training_length:
                d = clean[n]
                update = True
            else:
                d = decide(y)
                update = (freeze_threshold is None) or (np.abs(d - y) <= freeze_threshold)

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
                
            output[n] = y
            errors[n] = e

        output[: self.num_taps] = noisy[: self.num_taps]
        self._weights = w
        self._error_history = np.abs(errors) ** 2
        return output, self._error_history
