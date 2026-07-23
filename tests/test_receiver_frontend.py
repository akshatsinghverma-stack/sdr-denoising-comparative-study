"""
Regression tests for the receiver_frontend / add_awgn SNR-calibration bugs
found and fixed during the Case Study 2 (RRC + multipath ISI) review.

Both bugs were caught by a definitional identity: No-Processing's measured
output SNR (against the correct reference) must equal the nominal input SNR,
within measurement noise. These tests codify that identity so a future edit
to signal_gen.py / channel.py / utils.py can't silently reintroduce either bug.
"""
import numpy as np
import pytest

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_multipath, add_awgn
from src.utils import receiver_frontend
from src.metrics import compute_snr_db

NUM_SYMBOLS = 20_000
SPS = 4
MULTIPATH = [1.0, 0.4 + 0.3j, -0.1 + 0.1j]


@pytest.mark.parametrize("gen", [generate_bpsk, generate_qpsk])
def test_receiver_frontend_unit_gain_no_isi(gen):
    """A clean (no ISI, no noise) Tx-shape -> Rx-matched-filter round trip
    must recover unit-amplitude symbols, not sps-scaled ones. Regression test
    for the bug where reusing the sqrt(sps)-boosted RRC filter as the receive
    matched filter without renormalizing gave a sps-fold amplitude gain
    (verified: recovered +-4.0 instead of +-1.0 for BPSK at sps=4)."""
    tx, bits, h_rrc = gen(NUM_SYMBOLS, seed=1, sps=SPS)
    recovered = receiver_frontend(tx, h_rrc, sps=SPS)
    mean_power = np.mean(np.abs(recovered) ** 2)
    assert mean_power == pytest.approx(1.0, abs=0.02), (
        f"Expected unit average power after a clean matched-filter round trip, got {mean_power:.4f}. "
        "This indicates the matched filter is not properly normalized (sps-fold gain bug)."
    )


@pytest.mark.parametrize("gen", [generate_bpsk, generate_qpsk])
@pytest.mark.parametrize("multipath", [None, MULTIPATH])
def test_no_processing_snr_matches_nominal(gen, multipath):
    """No-Processing's measured SNR-out (vs the correct noise-free reference)
    must equal the nominal SNR passed to add_awgn, within measurement noise --
    an exact definitional identity regardless of oversampling or ISI. This is
    the check that caught the +5.77dB SNR-calibration bug (add_awgn defines
    SNR at the raw sample level, but matched filtering + ISI both change the
    effective gain the signal receives relative to noise)."""
    tx, bits, h_rrc = gen(NUM_SYMBOLS, seed=2, sps=SPS)
    rx_clean = add_multipath(tx, multipath)
    isi_ref = receiver_frontend(rx_clean, h_rrc, sps=SPS)

    # Empirical calibration -- same approach used in run_case2_isi.py.
    probe_db = 0.0
    noisy_probe = add_awgn(rx_clean, probe_db, seed=777)
    syms_probe = receiver_frontend(noisy_probe, h_rrc, sps=SPS)
    gain_db = compute_snr_db(isi_ref, syms_probe) - probe_db

    for nominal_snr in [-10, 0, 10, 20]:
        noisy = add_awgn(rx_clean, nominal_snr - gain_db, seed=1000 + nominal_snr)
        syms = receiver_frontend(noisy, h_rrc, sps=SPS)
        snr_out = compute_snr_db(isi_ref, syms)
        assert snr_out == pytest.approx(nominal_snr, abs=0.5), (
            f"No-Processing SNR_out={snr_out:.3f}dB does not match nominal {nominal_snr}dB "
            f"(multipath={'yes' if multipath else 'no'}). This is the calibration identity "
            "that caught the original +5.77dB bug -- do not relax this tolerance without investigating."
        )


def test_receiver_frontend_sps1_is_passthrough():
    """Case Study 1 relies on sps=1 bypassing the matched-filter path entirely
    (so it is provably unaffected by any receiver_frontend normalization
    change). This must stay true."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    out = receiver_frontend(signal, np.array([1.0]), sps=1)
    assert np.array_equal(out, signal)
