"""
Regression tests for the LMS/NLMS divergence bugs fixed for Case Study 1.

Two distinct failure modes were found and fixed:
  1. The preamble training itself diverging at low SNR (step-size safety
     margin was too aggressive -- 90% of the theoretical bound instead of
     20%).
  2. Naive decision-directed continuation engaging-and-diverging in ~30-40%
     of random trials at borderline SNR, even with a reliability gate.

These tests codify the resulting acceptance criterion across many random
trials (not just one lucky seed) so a future change to num_taps, mu, or the
freeze/decision-directed logic can't silently reintroduce either failure.
"""
import numpy as np
import pytest

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn
from src.lms_filter import LMSFilter
from src.nlms_filter import NLMSFilter
from src.utils import bits_to_symbols
from src.metrics import compute_snr_db

NUM_SYMBOLS = 20_000
TAPS = 16
LMS_MU = 0.01
NLMS_MU = 0.5
TRAINING_LENGTH = 1000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
N_TRIALS = 5  # fewer than the full study's 10, kept small so tests run fast


@pytest.mark.parametrize("gen,modulation", [(generate_bpsk, "BPSK"), (generate_qpsk, "QPSK")])
@pytest.mark.parametrize("snr", SNR_LEVELS)
def test_lms_nlms_never_worse_than_no_processing(gen, modulation, snr):
    """LMS and NLMS output SNR, *averaged* over multiple random trials, must
    never be meaningfully worse than the raw No-Processing SNR at any tested
    level. This is the acceptance criterion for the original catastrophic-
    divergence bug (BPSK -10dB LMS was measured at -15.07dB before the fix,
    i.e. 5dB *worse* than doing nothing) -- catching a ~30dB-scale failure
    doesn't require a tight per-trial bound. Comparing on the mean (rather
    than per-trial) matches how the actual study reports these results and
    avoids false failures from ordinary single-trial misadjustment variance
    (the known, documented, expected LMS noise floor at high SNR -- see
    report.md Section 3.2/4.5 -- is a few tenths of a dB on average but can
    swing close to 1dB on any individual trial at this test's reduced
    NUM_SYMBOLS / N_TRIALS, well short of a full 100k-symbol/10-trial run).
    """
    lms_snrs, nlms_snrs, noproc_snrs = [], [], []
    for trial in range(N_TRIALS):
        tx, bits, h_rrc = gen(NUM_SYMBOLS, seed=100 + trial, sps=1)
        ref = bits_to_symbols(bits, modulation)
        noisy = add_awgn(tx, snr, seed=200 + trial * 7 + snr)
        noproc_snrs.append(compute_snr_db(ref, noisy))

        lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
        lms_out, _ = lms.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
        lms_snrs.append(compute_snr_db(ref, lms_out))

        nlms = NLMSFilter(num_taps=TAPS, mu=NLMS_MU)
        nlms_out, _ = nlms.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
        nlms_snrs.append(compute_snr_db(ref, nlms_out))

    mean_noproc = np.mean(noproc_snrs)
    mean_lms = np.mean(lms_snrs)
    mean_nlms = np.mean(nlms_snrs)

    # Tolerance is 1.5dB, not the ~0.2-0.3dB shortfall reported in report.md
    # Section 3.2/4.5 for the full study: that number was measured at 100k
    # symbols/10 trials. This test uses a smaller NUM_SYMBOLS (20k) to stay
    # fast, and the fixed 1000-symbol preamble is consequently a larger
    # fraction of the signal the SNR metric averages over here (5% vs 1%),
    # so the still-converging early preamble drags the average down more.
    # 1.5dB comfortably covers that scaling effect while remaining nowhere
    # close to the 5-30dB drops the original divergence bug produced.
    assert mean_lms >= mean_noproc - 1.5, (
        f"LMS diverged: {modulation} SNR_in={snr}dB mean No-Proc={mean_noproc:.2f}dB "
        f"but mean LMS output={mean_lms:.2f}dB over {N_TRIALS} trials"
    )
    assert mean_nlms >= mean_noproc - 1.5, (
        f"NLMS diverged: {modulation} SNR_in={snr}dB mean No-Proc={mean_noproc:.2f}dB "
        f"but mean NLMS output={mean_nlms:.2f}dB over {N_TRIALS} trials"
    )


def test_decision_directed_off_by_default():
    """The default must stay off: Monte Carlo testing showed decision-directed
    continuation engages-and-diverges in ~30-40% of trials at borderline SNR
    even with the reliability/confidence gates, and freezing after the
    preamble matched-or-beat it at every tested SNR level on this static
    channel. If this default is ever flipped back to True, the low/mid-SNR
    stability tests above need to be re-run across many more trials first."""
    lms = LMSFilter(num_taps=TAPS, mu=LMS_MU)
    tx, bits, h_rrc = generate_bpsk(5000, seed=1, sps=1)
    noisy = add_awgn(tx, 0.0, seed=1)
    lms.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation="BPSK")
    # dd_enabled should be False whenever enable_decision_directed=False (the default)
    assert lms._dd_enabled is False
