"""
AWGN Channel Module
====================
Adds Additive White Gaussian Noise at a specified SNR (dB) to a complex
baseband signal.
"""

import numpy as np


def add_multipath(signal: np.ndarray, h: list[complex] | np.ndarray = None) -> np.ndarray:
    """Apply a static multipath channel (ISI) via causal convolution:
    r[n] = sum_k h[k] * s[n-k], with h[0] the direct/zero-delay path.

    If h is None, defaults to no multipath (delta function).

    NOTE: this must be a *causal* convolution (mode='full', truncated to the
    input length), not mode='same'. 'same' mode centers the kernel assuming
    it's roughly symmetric (correct for a matched filter like h_rrc, wrong
    for a causal tap-delay channel model where h[0] means "no delay"). Using
    'same' mode here previously introduced a spurious ~1-sample shift for any
    non-centered kernel -- verified concretely: convolving with h=[1,0,0]
    (which should be an exact identity channel) instead returned the signal
    shifted by one sample. That shift silently misaligned the receiver's
    symbol-timing grid, and was present for every multipath profile actually
    used (not just edge cases), including the one used throughout the
    originally-reported Case Study 2 results.
    """
    if h is None or len(h) == 0 or (len(h) == 1 and h[0] == 1.0):
        return signal
    return np.convolve(signal, h, mode="full")[: len(signal)]


def add_awgn(signal: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add AWGN to a complex signal at the specified SNR.

    SNR is defined as:
        SNR_dB = 10 * log10(P_signal / P_noise)

    where P_signal is the *average* power of the input signal and P_noise is
    the variance of the additive noise.

    Parameters
    ----------
    signal : np.ndarray, complex
        Clean input signal.
    snr_db : float
        Desired signal-to-noise ratio in dB.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    noisy : np.ndarray, complex
        Signal with additive white Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    # Signal power (mean |s|^2)
    sig_power = np.mean(np.abs(signal) ** 2)

    # Noise power from SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    # Generate complex AWGN: each of I, Q has variance noise_power/2
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def add_time_varying_multipath(signal: np.ndarray, h_base: list[complex] | np.ndarray = None,
                                sps: int = 4, mode: str = "sinusoidal",
                                n_cycles: float = 1.5, drift_depth: float = 0.6,
                                coherence_symbols: float = 4000.0,
                                vary_direct_path: bool = False,
                                seed: int | None = None) -> np.ndarray:
    """Slowly time-varying multipath channel: a causal tap-delay system whose
    per-tap complex gains drift smoothly over the duration of `signal`,
    modelling a slow fade / Doppler-like drift (as opposed to `add_multipath`,
    whose taps are fixed for the entire signal).

    r[n] = sum_k h_k(n) * s[n-k],  with h_0 kept FIXED (the direct/LOS path
    acts as a stable amplitude/phase reference, as in a Rician-style model),
    and h_1, h_2, ... (the reflected/multipath taps) drifting around their
    base value from `h_base` (default: Case Study 2's profile
    `[1.0, 0.4+0.3j, -0.1+0.1j]`).

    Two drift models are provided:

    - mode="sinusoidal": each non-direct tap oscillates deterministically as
      `h_base[k] * (1 + drift_depth * sin(2*pi*n_cycles*n/N + phi_k))`, with
      an independent random phase offset `phi_k` per tap (so taps don't move
      in lockstep -- distinct reflectors fade somewhat independently).
      `n_cycles` full oscillations occur across the whole signal duration.
    - mode="random_walk": each non-direct tap follows a stochastic,
      band-limited (AR(1)-smoothed complex Gaussian) drift around its base
      value, with `coherence_symbols` controlling how many symbols it takes
      for the process to decorrelate (smaller = faster drift). This is the
      standard way to synthesize a slow-fading tap trajectory without an
      unbounded random walk.

    Rate-of-drift design note, CORRECTED after a later self-audit (see
    report/findings_preamble_drift_correction.md): this docstring originally
    claimed the default parameters make the channel "look static" across the
    1000-symbol preamble. That claim was never actually measured and is
    FALSE for the random_walk config used in Case Study 3's main flat-fading
    control (coherence_symbols=4000, drift_depth=0.4, sps=1): the single tap's
    magnitude was measured (via a constant-input probe through the real
    function) to swing by 60-91% of its base value *within the first 1000
    samples alone*, across 5 tested seeds. The sinusoidal mode's non-direct
    taps also drift by up to +-22% within a 4000-sample (SPS=4) preamble
    window, not "negligibly." Neither mode gives the near-static preamble
    this function's docstring and report.md Section 6.1 originally described
    -- see the findings file above for what this does and does not change
    about Case Study 3's conclusions.

    Parameters
    ----------
    signal : np.ndarray, complex
        Input (already pulse-shaped, oversampled) signal.
    h_base : list[complex] or np.ndarray, optional
        Starting/reference tap magnitudes; h_base[0] is the direct path.
        Defaults to Case Study 2's static profile.
    sps : int
        Samples per symbol (used only to convert `coherence_symbols` to a
        sample-domain time constant for the random-walk mode).
    mode : {"sinusoidal", "random_walk"}
    n_cycles : float
        Number of full drift oscillations across the signal (sinusoidal mode).
    drift_depth : float
        Fractional modulation depth of each non-direct tap's magnitude
        around its base value (both modes).
    coherence_symbols : float
        Approximate number of symbols over which the random-walk process
        decorrelates (random_walk mode only); larger = slower drift.
    vary_direct_path : bool
        If True, the direct/zero-delay tap (h_base[0]) drifts the same way
        as the other taps instead of being held fixed. Needed to model pure
        flat/Rayleigh-style fading with a single tap (h_base=[1.0]) -- the
        default (False) models frequency-selective fading of the *reflected*
        paths only, on top of a stable direct/LOS path, which is the
        physically-motivated default for a multi-tap channel.
    seed : int or None
        Random seed (phase offsets for sinusoidal mode; noise realization for
        random_walk mode).

    Returns
    -------
    r : np.ndarray, complex
        Channel output, same length as `signal`.
    """
    if h_base is None:
        h_base = [1.0]
    h_base = np.asarray(h_base, dtype=np.complex128)
    K = len(h_base)
    N = len(signal)
    t = np.arange(N)
    rng = np.random.default_rng(seed)

    h_t = np.zeros((K, N), dtype=np.complex128)
    k_start = 0 if vary_direct_path else 1
    if not vary_direct_path:
        h_t[0, :] = h_base[0]  # direct/LOS path: fixed, the stable phase/amplitude reference

    if mode == "sinusoidal":
        phase_offsets = rng.uniform(0, 2 * np.pi, size=K)
        for k in range(k_start, K):
            osc = np.sin(2 * np.pi * n_cycles * t / N + phase_offsets[k])
            h_t[k, :] = h_base[k] * (1.0 + drift_depth * osc)
    elif mode == "random_walk":
        from scipy.signal import lfilter
        # Single-pole low-pass filter applied to white complex Gaussian noise
        # produces a smooth, band-limited (bounded, mean-reverting-in-
        # distribution) random process -- the standard cheap way to
        # synthesize a slow-fading trajectory without an unbounded random
        # walk. `alpha` sets the pole location from the desired coherence
        # time (in samples).
        coherence_samples = max(coherence_symbols * sps, 2.0)
        alpha = np.exp(-1.0 / coherence_samples)
        for k in range(k_start, K):
            white = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            smoothed = lfilter([np.sqrt(1 - alpha ** 2)], [1.0, -alpha], white)
            # Normalise realised fluctuation std to drift_depth*|h_base[k]|
            fluct_std = np.std(np.abs(smoothed)) + 1e-12
            smoothed *= (drift_depth * np.abs(h_base[k]) / fluct_std)
            h_t[k, :] = h_base[k] + smoothed
    else:
        raise ValueError(f"Unknown mode: {mode!r}, expected 'sinusoidal' or 'random_walk'")

    # r[n] = sum_k h_k(n) * s[n-k]  -- vectorised per-tap (causal, matches
    # add_multipath's convention that h[0] is the zero-delay/direct path).
    out = np.zeros(N, dtype=np.complex128)
    for k in range(K):
        shifted = np.zeros(N, dtype=np.complex128)
        if k == 0:
            shifted = signal
        else:
            shifted[k:] = signal[: N - k]
        out += h_t[k, :] * shifted
    return out
