"""
Genie-Aided Linear MMSE Equalizer
===================================
A closed-form linear MMSE equalizer computed directly from the *known*
channel (pulse shape + multipath taps) and the known noise level, used as an
upper-bound reference in Case Study 2 -- "how close do LMS/NLMS/CNN get to
the best a linear equalizer could possibly do for this exact channel", as
opposed to only comparing methods against each other.

"Genie-aided" means the channel and noise level are assumed perfectly known,
which no adaptive method in this study actually has access to -- this is a
reference ceiling, not a competing practical method.

Design
------
The symbol-spaced channel is measured empirically (send a unit impulse
through the same Tx-pulse-shape -> multipath -> Rx-matched-filter chain used
everywhere else in this project, rather than deriving it analytically --
consistent with how the SNR-calibration bug fix in run_case2_isi.py measures
gain empirically instead of assuming a formula). Given the resulting taps
g[k], the standard Wiener/MMSE linear equalizer is:

    R w = p,   R[i,j] = c[i-j] + sigma_v^2 * delta[i-j],   p[i] = g[D-i]

where c[m] = sum_k g[k] g*[k+m] is g's autocorrelation, sigma_v^2 is the
post-matched-filter noise variance (measured empirically from the actual
No-Processing residual, same as everywhere else in this codebase), D is a
decision delay aligned to the channel's peak tap, and unit symbol power is
assumed (P_s=1, matching this project's constellations).
"""

import numpy as np

from src.signal_gen import apply_pulse_shaping
from src.channel import add_multipath
from src.utils import receiver_frontend


def measure_channel_taps(h_rrc, multipath, sps, num_taps=21, span_before=10):
    """Measure the symbol-spaced channel impulse response g[k] by sending a
    unit impulse through the same Tx-shape -> multipath -> Rx-matched-filter
    chain used for the actual signals.

    Returns
    -------
    g : complex ndarray, length num_taps
    peak_idx : int
        Index within g corresponding to zero delay (the "D=0" tap).
    """
    n_symbols = 4 * span_before + num_taps + 20
    mid = n_symbols // 2
    symbols = np.zeros(n_symbols, dtype=np.complex128)
    symbols[mid] = 1.0
    tx, _ = apply_pulse_shaping(symbols, sps=sps)
    rx_clean = add_multipath(tx, multipath)
    g_full = receiver_frontend(rx_clean, h_rrc, sps=sps)

    start = mid - span_before
    end = start + num_taps
    g = g_full[start:end]
    peak_idx = mid - start
    return g, peak_idx


def design_mmse_equalizer(g, peak_idx, noise_var, num_taps=16, delay=None):
    """Solve for the linear MMSE equalizer taps given a measured channel.

    Parameters
    ----------
    g : complex ndarray
        Measured symbol-spaced channel taps (from measure_channel_taps).
    peak_idx : int
        Index of the zero-delay tap within g.
    noise_var : float
        Post-matched-filter noise variance (P_n), measured empirically.
    num_taps : int
        Equalizer length (matched to LMS/NLMS's 16 taps for fair comparison).
    delay : int, optional
        Decision delay; defaults to num_taps // 2 (centered), a standard
        choice that lets the equalizer use both pre- and post-cursor taps.

    Returns
    -------
    w : complex ndarray, length num_taps
        Equalizer taps such that s_hat[n] = sum_i conj(w[i]) * y[n+delay-i].
    delay : int
    """
    if delay is None:
        delay = num_taps // 2

    L = len(g)
    # Autocorrelation of g, c[m] for m = -(num_taps-1) .. (num_taps-1)
    c_full = np.correlate(g, g, mode="full")  # c_full[L-1+m] = c[m]

    def c(m):
        idx = (L - 1) + m
        if 0 <= idx < len(c_full):
            return c_full[idx]
        return 0.0 + 0.0j

    R = np.zeros((num_taps, num_taps), dtype=np.complex128)
    for i in range(num_taps):
        for j in range(num_taps):
            R[i, j] = c(i - j)
    R += noise_var * np.eye(num_taps)

    def g_at(k):
        # g is indexed so that g[peak_idx] corresponds to k=0 (zero delay).
        idx = peak_idx + k
        if 0 <= idx < L:
            return g[idx]
        return 0.0 + 0.0j

    p = np.array([g_at(delay - i) for i in range(num_taps)], dtype=np.complex128)

    w = np.linalg.solve(R, p)
    return w, delay


def apply_mmse_equalizer(y, w, delay):
    """Apply the designed equalizer to a symbol-rate received sequence y.

    s_hat[n] = sum_i conj(w[i]) * y[n + delay - i]

    Implemented as a correlation (matched to the same "newest first" tap
    convention used by LMSFilter/NLMSFilter's x = y[n : n-taps : -1]).
    """
    num_taps = len(w)
    N = len(y)
    out = np.zeros(N, dtype=np.complex128)
    w_conj = np.conj(w)
    for n in range(num_taps, N - delay):
        y_vec = y[n + delay : n + delay - num_taps : -1]
        out[n] = np.dot(w_conj, y_vec)
    # Edge samples (insufficient history/lookahead) pass through unmodified.
    out[:num_taps] = y[:num_taps]
    out[N - delay:] = y[N - delay:]
    return out
