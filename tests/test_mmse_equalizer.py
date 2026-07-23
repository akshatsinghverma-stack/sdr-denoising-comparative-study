"""
Regression tests for the genie-aided MMSE equalizer (Case Study 2 upper
bound reference). The genie has perfect channel + noise knowledge, so it
should never be worse than doing nothing at all.
"""
import numpy as np
import pytest

from src.signal_gen import generate_bpsk
from src.channel import add_multipath, add_awgn
from src.utils import receiver_frontend, demod_bpsk
from src.metrics import compute_snr_db, compute_ber_count
from src.mmse_equalizer import (measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer,
                                  design_zf_equalizer, apply_zf_equalizer)

SPS = 4
MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]
NUM_SYMBOLS = 20_000
TAPS = 16


def test_measure_channel_taps_peak_matches_dominant_multipath_tap():
    tx, bits, h_rrc = generate_bpsk(NUM_SYMBOLS, seed=1, sps=SPS)
    g, peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)
    # The dominant tap should be at (or immediately around) peak_idx and be
    # the largest-magnitude tap in the measured response.
    assert np.argmax(np.abs(g)) == peak_idx


@pytest.mark.parametrize("snr", [-10, 0, 10, 20])
def test_mmse_never_worse_than_no_processing(snr):
    tx, bits, h_rrc = generate_bpsk(NUM_SYMBOLS, seed=1, sps=SPS)
    rx_clean = add_multipath(tx, MULTIPATH)
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    g, peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)

    noisy = add_awgn(rx_clean, snr, seed=42 + snr)
    syms_noproc = receiver_frontend(noisy, h_rrc, sps=SPS)
    noise_var = np.mean(np.abs(isi_ref - syms_noproc) ** 2)

    w, delay = design_mmse_equalizer(g, peak_idx, noise_var, num_taps=TAPS)
    mmse_out = apply_mmse_equalizer(syms_noproc, w, delay)

    e_noproc, n = compute_ber_count(bits, demod_bpsk(syms_noproc))
    e_mmse, _ = compute_ber_count(bits, demod_bpsk(mmse_out))

    assert e_mmse <= e_noproc, (
        f"Genie MMSE ({e_mmse} errors) should never be worse than No-Processing "
        f"({e_noproc} errors) at SNR={snr}dB -- it has perfect channel+noise knowledge."
    )


# ============================================================================
# Zero-Forcing (ZF) equalizer -- added after a self-critique pass found
# published equalization-comparison literature treats ZF as a standard
# baseline this project was missing. ZF ignores noise entirely (unlike MMSE),
# so unconditional "never worse than No-Processing" is NOT asserted here --
# that is exactly the open question the comparison experiment tests.
# ============================================================================

def test_zf_recovers_symbols_with_negligible_noise():
    """With essentially no real noise, ZF (which fully inverts the channel,
    ignoring noise) must recover symbols almost exactly -- this is ZF's one
    unconditional guarantee."""
    tx, bits, h_rrc = generate_bpsk(NUM_SYMBOLS, seed=2, sps=SPS)
    rx_clean = add_multipath(tx, MULTIPATH)
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)
    g, peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)

    noisy = add_awgn(rx_clean, 40.0, seed=99)  # ~negligible noise
    syms = receiver_frontend(noisy, h_rrc, sps=SPS)

    w, delay = design_zf_equalizer(g, peak_idx, num_taps=TAPS)
    zf_out = apply_zf_equalizer(syms, w, delay)

    e_zf, n = compute_ber_count(bits, demod_bpsk(zf_out))
    assert e_zf == 0, f"ZF should recover symbols essentially perfectly at ~40dB SNR, got {e_zf}/{n} errors"


def test_zf_converges_to_mmse_as_noise_vanishes():
    """As the MMSE noise term shrinks to ZF's own numerical-stability
    regularizer, the two designs must converge to (nearly) the same taps --
    ZF is mathematically the noise_var->0 limit of MMSE."""
    tx, bits, h_rrc = generate_bpsk(NUM_SYMBOLS, seed=3, sps=SPS)
    g, peak_idx = measure_channel_taps(h_rrc, MULTIPATH, sps=SPS, num_taps=21)

    w_zf, delay_zf = design_zf_equalizer(g, peak_idx, num_taps=TAPS)
    w_mmse_tiny, delay_mmse = design_mmse_equalizer(g, peak_idx, noise_var=1e-10, num_taps=TAPS)

    assert delay_zf == delay_mmse
    assert np.allclose(w_zf, w_mmse_tiny, atol=1e-6)
