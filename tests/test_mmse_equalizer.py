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
from src.mmse_equalizer import measure_channel_taps, design_mmse_equalizer, apply_mmse_equalizer

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
