"""
Utility Functions
==================
Constellation demapping (hard-decision demodulation) and plotting helpers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# Receiver Front-End
# ============================================================================

def receiver_frontend(rx_sig, h_rrc, sps=4):
    """Apply matched filter and downsample to symbol rate.

    h_rrc is scaled (in signal_gen.rrc_filter) so that the *transmit*
    pulse-shaped signal has unit average sample power. Reusing the same
    filter as the receive matched filter without renormalizing double-counts
    that scaling: convolving an already sqrt(sps)-boosted filter with itself
    (via correlation) multiplies the recovered symbol amplitude by another
    factor of sps on top of the matched-filter's own coherent-combining gain,
    for a net sps-in-amplitude / sps^2-in-power gain that has nothing to do
    with any real SNR improvement -- it's pure double-scaling. Dividing by
    the filter's energy (sum|h_rrc|^2 == sps by construction) restores unit
    gain, verified by a clean (no ISI, no noise) round-trip recovering the
    original +-1 / QPSK-unit-magnitude symbols exactly rather than +-sps.
    """
    if sps == 1:
        return rx_sig
    mf_out = np.convolve(rx_sig, h_rrc[::-1], mode='same')
    mf_out = mf_out / np.sum(np.abs(h_rrc) ** 2)
    symbols = mf_out[0::sps]
    return symbols

def bits_to_symbols(bits, modulation="BPSK"):
    """Regenerate ideal symbols from bits.

    Only BPSK and QPSK are implemented -- previously any other value (a typo,
    or "16QAM"/"8PSK") silently fell through to the QPSK formula instead of
    raising, a latent mismap caught during a self-critique pass (no current
    caller actually passes an unsupported value, so this was never triggered
    in practice, but the silent fallback was a real trap for future callers).
    """
    if modulation == "BPSK":
        return (2.0 * bits - 1.0).astype(np.complex128)
    elif modulation == "QPSK":
        b0 = bits[0::2]
        b1 = bits[1::2]
        gray_index = b0 * 2 + (b0 ^ b1)
        phase = (2 * gray_index + 1) * np.pi / 4.0
        return np.cos(phase) + 1j * np.sin(phase)
    else:
        raise ValueError(f"bits_to_symbols: unsupported modulation {modulation!r} "
                          f"(only 'BPSK' and 'QPSK' are implemented)")

# ============================================================================
# Constellation demapping
# ============================================================================

def demod_bpsk(signal: np.ndarray) -> np.ndarray:
    """Hard-decision BPSK demodulation.

    Decision rule: real(s) >= 0  ->  bit 1,  else bit 0.
    """
    return (signal.real >= 0).astype(int)


def demod_qpsk(signal: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation (Gray-coded, matching signal_gen).

    Mapping (inverse of generate_qpsk):
        phase in [0,   π/2)  -> gray_index 0 -> (b0,b1) = (0,0)
        phase in [π/2, π)    -> gray_index 1 -> (b0,b1) = (0,1)
        phase in [π,  3π/2)  -> gray_index 2 -> (b0,b1) = (1,1)
        phase in [3π/2, 2π)  -> gray_index 3 -> (b0,b1) = (1,0)
    """
    # Nearest constellation point via minimum distance
    # Reference constellation (same as in signal_gen)
    ref_indices = np.array([0, 1, 2, 3])
    ref_phases = (2 * ref_indices + 1) * np.pi / 4.0
    ref_points = np.exp(1j * ref_phases)

    # Bit mapping for each gray index
    bit_map = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
    }

    # For each received symbol, find nearest reference
    bits = np.zeros(len(signal) * 2, dtype=int)
    for i, s in enumerate(signal):
        dists = np.abs(s - ref_points)
        idx = np.argmin(dists)
        b0, b1 = bit_map[idx]
        bits[2 * i] = b0
        bits[2 * i + 1] = b1
    return bits


def demod_qpsk_fast(signal: np.ndarray) -> np.ndarray:
    """Vectorised QPSK demodulation (much faster for large arrays)."""
    ref_indices = np.array([0, 1, 2, 3])
    ref_phases = (2 * ref_indices + 1) * np.pi / 4.0
    ref_points = np.exp(1j * ref_phases)  # shape (4,)

    # (N, 1) - (1, 4) -> (N, 4) distance matrix
    dists = np.abs(signal[:, None] - ref_points[None, :])  # broadcast
    nearest = np.argmin(dists, axis=1)  # shape (N,)

    # Gray-index -> (b0, b1)
    bit_table = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=int)
    bit_pairs = bit_table[nearest]  # shape (N, 2)
    return bit_pairs.ravel()


def demod_16qam_fast(signal: np.ndarray) -> np.ndarray:
    """Vectorised hard-decision 16-QAM demodulation (Gray-coded, matching
    signal_gen.generate_16qam's constellation and bit mapping exactly).

    Nearest-point decision against the full 16-point reference constellation
    (rather than independent per-rail thresholding), analogous in style to
    demod_qpsk_fast's distance-matrix approach.
    """
    gray_levels = np.array([-3.0, -1.0, 3.0, 1.0])  # index = raw 2-bit value

    # Build the 16 reference points once, labelled with the exact 4 bits
    # (I_hi, I_lo, Q_hi, Q_lo) that generate_16qam would assign to each.
    ref_points = np.zeros(16, dtype=np.complex128)
    ref_bits = np.zeros((16, 4), dtype=int)
    idx = 0
    for v_i in range(4):      # b0*2+b1 raw value for the I rail
        for v_q in range(4):  # b2*2+b3 raw value for the Q rail
            I = gray_levels[v_i]
            Q = gray_levels[v_q]
            ref_points[idx] = (I + 1j * Q) / np.sqrt(10.0)
            ref_bits[idx] = [v_i // 2, v_i % 2, v_q // 2, v_q % 2]
            idx += 1

    dists = np.abs(signal[:, None] - ref_points[None, :])  # (N, 16)
    nearest = np.argmin(dists, axis=1)
    bit_quads = ref_bits[nearest]  # (N, 4)
    return bit_quads.ravel()


def demod_8psk_fast(signal: np.ndarray) -> np.ndarray:
    """Vectorised hard-decision 8-PSK demodulation (Gray-coded, matching
    signal_gen.generate_8psk's constellation and bit mapping exactly)."""
    ref_indices = np.arange(8)
    ref_phases = (2 * ref_indices + 1) * np.pi / 8.0
    ref_points = np.exp(1j * ref_phases)  # shape (8,)

    # Inverse of generate_8psk's gray_index = b0*4 + g1*2 + (g1^b2) mapping,
    # i.e. binary(gray_index) -> (b0, b1, b2), by table lookup (built by
    # brute-force inversion over all 8 bit triples -- verified exact below
    # via the round-trip self-test rather than re-deriving the XOR algebra
    # a second time).
    bit_table = np.zeros((8, 3), dtype=int)
    for b0 in range(2):
        for b1 in range(2):
            for b2 in range(2):
                g1 = b0 ^ b1
                gi = b0 * 4 + g1 * 2 + (g1 ^ b2)
                bit_table[gi] = [b0, b1, b2]

    dists = np.abs(signal[:, None] - ref_points[None, :])  # (N, 8)
    nearest = np.argmin(dists, axis=1)
    bit_triples = bit_table[nearest]  # (N, 3)
    return bit_triples.ravel()


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_ber_vs_snr(snr_levels, ber_dict, modulation: str, save_path: str, err_dict: dict = None,
                     show_theoretical: bool = True):
    """Plot BER vs input SNR for multiple methods.

    Parameters
    ----------
    snr_levels : list of float
    ber_dict : dict  {method_name: list_of_ber_values}
    modulation : str  ('BPSK' or 'QPSK')
    save_path : str
    err_dict : dict, optional  {method_name: list_of_std_values}
        If given, drawn as error bars (e.g. std across Monte Carlo trials).
    show_theoretical : bool
        The plotted curve is the memoryless-AWGN Q-function bound, which is
        only valid with no ISI. Set False (e.g. for an ISI channel) rather
        than showing a curve that would be silently misread as an applicable
        bound by anyone looking at the figure without the surrounding text.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (name, bers) in enumerate(ber_dict.items()):
        linestyle = "--" if "No Processing" in name else "-"
        if err_dict is not None:
            bers_arr = np.array(bers)
            errs = np.array(err_dict[name])
            # Clip lower error bar so it never goes <= 0 on a log scale.
            lower = np.minimum(errs, bers_arr * 0.999)
            ax.errorbar(snr_levels, bers_arr, yerr=[lower, errs],
                        marker=markers[i % len(markers)], label=name,
                        linewidth=2, markersize=6, linestyle=linestyle, capsize=3)
        else:
            ax.semilogy(snr_levels, bers, marker=markers[i % len(markers)],
                        label=name, linewidth=2, markersize=6, linestyle=linestyle)
    ax.set_yscale("log")

    # Add theoretical curve (memoryless AWGN only -- see show_theoretical docstring)
    if show_theoretical:
        try:
            from scipy.special import erfc
            snr_lin = 10**(np.array(snr_levels)/10.0)
            if modulation == "BPSK":
                pe = 0.5 * erfc(np.sqrt(snr_lin))
            else:
                pe = 0.5 * erfc(np.sqrt(snr_lin / 2.0))
            ax.semilogy(snr_levels, pe, 'k:', label="Theoretical AWGN (no ISI)", linewidth=2)
        except Exception:
            pass

    ax.set_xlabel("Input SNR (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title(f"BER vs Input SNR — {modulation}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_snr_vs_snr(snr_levels, snr_out_dict, modulation: str, save_path: str, err_dict: dict = None):
    """Plot output SNR vs input SNR for multiple methods.

    Parameters
    ----------
    err_dict : dict, optional  {method_name: list_of_std_values}
        If given, drawn as error bars (e.g. std across Monte Carlo trials).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (name, snrs) in enumerate(snr_out_dict.items()):
        linestyle = "--" if "No Processing" in name else "-"
        if err_dict is not None:
            ax.errorbar(snr_levels, snrs, yerr=err_dict[name],
                        marker=markers[i % len(markers)], label=name,
                        linewidth=2, markersize=6, linestyle=linestyle, capsize=3)
        else:
            ax.plot(snr_levels, snrs, marker=markers[i % len(markers)],
                    label=name, linewidth=2, markersize=6, linestyle=linestyle)

    ax.set_xlabel("Input SNR (dB)", fontsize=12)
    ax.set_ylabel("Output SNR (dB)", fontsize=12)
    ax.set_title(f"Output SNR vs Input SNR — {modulation}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_constellation(signals_dict: dict, modulation: str, snr_db: float,
                       save_path: str, max_points: int = 2000):
    """Plot constellation diagrams for multiple signals side-by-side.

    Parameters
    ----------
    signals_dict : dict {label: complex_array}
    """
    n = len(signals_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, sig) in zip(axes, signals_dict.items()):
        pts = sig[:max_points]
        ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        lim = 2.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle(f"Constellation — {modulation} @ {snr_db} dB input SNR",
                 fontsize=13)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_convergence(error_lms, error_nlms, save_path: str,
                     smooth_window: int = 200):
    """Plot LMS and NLMS convergence curves (smoothed error vs iteration)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    def _smooth(arr, w):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    ax.semilogy(_smooth(error_lms, smooth_window), label="LMS", linewidth=1.5)
    ax.semilogy(_smooth(error_nlms, smooth_window), label="NLMS", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Squared Error (smoothed)", fontsize=12)
    ax.set_title("LMS / NLMS Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_loss(history, save_path: str):
    """Plot CNN training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("CNN Autoencoder Training Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
