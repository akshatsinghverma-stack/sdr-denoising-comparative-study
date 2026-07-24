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
    # Autocorrelation of g, c[m] for m = -(num_taps-1) .. (num_taps-1).
    # np.correlate(g, g)[L-1+m] computes sum_n g[n+m]*conj(g[n]), which is
    # conj(c[m]) as defined below (c[m] = sum_k g[k]*conj(g[k+m])), not c[m]
    # itself -- found via a code-correctness critique pass and verified
    # numerically against a from-scratch reimplementation of c(m). This was
    # a real, silent bug (this project's eighth): R[i,j]=c(i-j) was being
    # built from the conjugated value, so the equalizer solved
    # conj(R_correct)@w=p instead of R_correct@w=p. Negligible for BPSK
    # (real symbols largely mask a conjugate error) but severe for QPSK
    # (measured ~50x worse BER at 5dB on Case Study 2's channel before this fix).
    c_full = np.correlate(g, g, mode="full")  # c_full[L-1+m] = conj(c[m])

    def c(m):
        idx = (L - 1) + m
        if 0 <= idx < len(c_full):
            return np.conj(c_full[idx])
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


def design_zf_equalizer(g, peak_idx, num_taps=16, delay=None, reg=1e-10):
    """Zero-Forcing (ZF) linear equalizer -- added after a self-critique pass
    found published equalization-comparison literature (e.g. arXiv:2312.06084,
    a ZF/LMS/RLS comparative study) treats ZF as a standard baseline that this
    project was missing entirely (it had MMSE but not ZF).

    ZF inverts the channel completely, ignoring noise -- exactly the same
    linear system as design_mmse_equalizer with noise_var=0 (`R = autocorr(g)`
    instead of `autocorr(g) + noise_var*I`). A tiny `reg` (default 1e-10, pure
    numerical-stability regularization, NOT a noise-modeling choice) is added
    because R can be poorly conditioned when the equalizer has more taps than
    the channel's essential support (16 taps vs. this project's 3-tap
    channel), unlike MMSE where the actual noise variance already regularizes
    R. Unlike MMSE, ZF does not trade off noise amplification against
    residual ISI -- it always fully removes ISI, whatever the noise cost --
    so it is expected to underperform MMSE at low SNR and match it only in
    the noiseless limit; this is the textbook prediction, tested directly
    against this project's actual channel rather than assumed.

    Returns
    -------
    w : complex ndarray, length num_taps
    delay : int
    """
    return design_mmse_equalizer(g, peak_idx, noise_var=reg, num_taps=num_taps, delay=delay)


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


# apply_mmse_equalizer's actual logic is generic to any linear FIR equalizer
# taps -- this alias documents that design_zf_equalizer's output uses the
# same application function, not a separate implementation.
apply_zf_equalizer = apply_mmse_equalizer
