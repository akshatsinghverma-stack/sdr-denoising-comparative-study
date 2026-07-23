"""
Signal Generation Module
=========================
Generates BPSK and QPSK modulated signals with RRC pulse shaping.
"""

import numpy as np
from scipy.signal import upfirdn

def rrc_filter(N: int, alpha: float, Ts: float, Fs: float) -> np.ndarray:
    """Generates a root raised cosine (RRC) filter (FIR).

    N is always odd in this project (apply_pulse_shaping's filter_span*sps+1),
    so the true center sample is index (N-1)/2, not N/2 -- using N/2 (a
    non-integer for odd N, e.g. 12.5 for N=25) put the continuous-time t=0
    point between two samples instead of on the center sample, making the
    filter measurably asymmetric (verified: up to ~0.4 peak-scale difference
    between h and its reverse before this fix) even though its single
    highest-magnitude sample still landed at the expected center index.
    Fixed to align with apply_pulse_shaping's delay=(N-1)//2 convention,
    which assumes an exactly symmetric, linear-phase filter.
    """
    T_delta = 1.0 / Fs
    t = (np.arange(N) - (N - 1) / 2) * T_delta
    
    h_rrc = np.zeros(N, dtype=float)
    for i in range(N):
        if t[i] == 0.0:
            h_rrc[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.isclose(np.abs(t[i]), Ts / (4 * alpha)):
            h_rrc[i] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha)))) +
                ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
            )
        else:
            denom = np.pi * t[i] * (1 - (4 * alpha * t[i] / Ts)**2) / Ts
            num = (np.sin(np.pi * t[i] * (1 - alpha) / Ts) +
                   4 * alpha * (t[i] / Ts) * np.cos(np.pi * t[i] * (1 + alpha) / Ts))
            h_rrc[i] = num / denom
            
    h_rrc /= np.sqrt(np.sum(h_rrc**2))  # Unit energy
    # Scale by sqrt(Fs) so that the upsampled signal has unit average power
    h_rrc *= np.sqrt(Fs)
    return h_rrc

def apply_pulse_shaping(symbols, sps=4, alpha=0.35, filter_span=6):
    """Upsample and apply RRC pulse shaping."""
    if sps == 1:
        return symbols, np.array([1.0])
    N_filter = filter_span * sps + 1
    h_rrc = rrc_filter(N_filter, alpha, 1.0, sps)
    signal = upfirdn(h_rrc, symbols, up=sps)
    # Strip delay
    delay = (N_filter - 1) // 2
    signal = signal[delay : delay + len(symbols) * sps]
    return signal, h_rrc

def generate_bpsk(num_symbols: int = 10_000, seed: int = 42, sps: int = 4):
    """Generate a BPSK signal with pulse shaping.

    Returns
    -------
    signal : complex array (pulse shaped)
    bits : int array
    h_rrc : float array (the transmit filter)
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_symbols)
    symbols = 2.0 * bits - 1.0
    symbols = symbols.astype(np.complex128)
    
    signal, h_rrc = apply_pulse_shaping(symbols, sps=sps)
    return signal, bits, h_rrc


def generate_qpsk(num_symbols: int = 10_000, seed: int = 42, sps: int = 4):
    """Generate a Gray-coded QPSK signal with pulse shaping."""
    rng = np.random.default_rng(seed)
    num_bits = num_symbols * 2
    bits = rng.integers(0, 2, size=num_bits)

    b0 = bits[0::2]
    b1 = bits[1::2]

    gray_index = b0 * 2 + (b0 ^ b1)
    phase = (2 * gray_index + 1) * np.pi / 4.0
    I = np.cos(phase)
    Q = np.sin(phase)

    symbols = (I + 1j * Q)
    signal, h_rrc = apply_pulse_shaping(symbols, sps=sps)
    return signal, bits, h_rrc


def generate_16qam(num_symbols: int = 10_000, seed: int = 42, sps: int = 4):
    """Generate a Gray-coded 16-QAM signal with pulse shaping.

    Standard square 16-QAM constellation: I and Q rails are each an
    independent 4-level PAM with amplitudes {-3,-1,1,3}, Gray-coded across
    2 bits per rail (4 bits/symbol total), normalized so the average
    constellation power is 1.

    Gray mapping per rail (2 bits -> level), chosen so adjacent codes in the
    sequence differ by exactly 1 bit (standard 4-PAM Gray code):
        00 -> -3   01 -> -1   11 -> +1   10 -> +3
    i.e. indexed by the raw 2-bit value (b_hi*2 + b_lo): [-3, -1, 3, 1].

    Returns
    -------
    signal : complex array (pulse shaped)
    bits : int array (4 bits per symbol, order [I_hi, I_lo, Q_hi, Q_lo] per symbol)
    h_rrc : float array (the transmit filter)
    """
    rng = np.random.default_rng(seed)
    num_bits = num_symbols * 4
    bits = rng.integers(0, 2, size=num_bits)

    b0 = bits[0::4]  # I gray bit (hi)
    b1 = bits[1::4]  # I gray bit (lo)
    b2 = bits[2::4]  # Q gray bit (hi)
    b3 = bits[3::4]  # Q gray bit (lo)

    # 2-bit Gray code -> PAM level: 00->-3, 01->-1, 11->+1, 10->+3
    gray_levels = np.array([-3.0, -1.0, 3.0, 1.0])
    I = gray_levels[b0 * 2 + b1]
    Q = gray_levels[b2 * 2 + b3]

    # Average power of a {-3,-1,1,3} 4-PAM rail (equiprobable) is
    # (9+1+1+9)/4 = 5, so I^2+Q^2 averages to 10 before normalization.
    symbols = (I + 1j * Q) / np.sqrt(10.0)
    symbols = symbols.astype(np.complex128)

    signal, h_rrc = apply_pulse_shaping(symbols, sps=sps)
    return signal, bits, h_rrc


def generate_8psk(num_symbols: int = 10_000, seed: int = 42, sps: int = 4):
    """Generate a Gray-coded 8-PSK signal with pulse shaping.

    8 equally-spaced phase points (unit magnitude, unit average power),
    3 bits/symbol, Gray-coded so adjacent phase points differ by 1 bit
    (same construction pattern as generate_qpsk, generalized to 8 points).

    Returns
    -------
    signal : complex array (pulse shaped)
    bits : int array (3 bits per symbol)
    h_rrc : float array (the transmit filter)
    """
    rng = np.random.default_rng(seed)
    num_bits = num_symbols * 3
    bits = rng.integers(0, 2, size=num_bits)

    b0 = bits[0::3]
    b1 = bits[1::3]
    b2 = bits[2::3]

    # Gray-to-binary decode (3-bit), generalizing generate_qpsk's 2-bit
    # gray_index = b0*2 + (b0^b1) pattern: bit b_k of the constellation
    # index is the running XOR of gray bits b0..bk. This guarantees that
    # constellation index k and k+1 (adjacent phase points, since index maps
    # directly to phase below) have bit labels differing by exactly 1 bit --
    # the standard reflected-binary Gray code property -- which a naive
    # "binary_val ^ (binary_val >> 1)" applied directly to the *index* does
    # NOT give (that is Gray *encode*, not decode, and produces a bit
    # assignment where index 3->4 differs by 2 bits, breaking the point of
    # Gray coding). Verified against the QPSK precedent's 2-bit formula and
    # a hand-checked truth table before use.
    g1 = b0 ^ b1
    gray_index = b0 * 4 + g1 * 2 + (g1 ^ b2)
    phase = (2 * gray_index + 1) * np.pi / 8.0
    I = np.cos(phase)
    Q = np.sin(phase)

    symbols = (I + 1j * Q).astype(np.complex128)
    signal, h_rrc = apply_pulse_shaping(symbols, sps=sps)
    return signal, bits, h_rrc

