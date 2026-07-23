"""
Regression tests for RLSFilter's decision-directed default.

src/rls_filter.py existed but was unused before the RLS-comparison follow-up
(report/report.md, "Named, scoped-out future work"). Its original
implementation continued adapting past the preamble by default, gated only
by a loose `|d - y| <= 2.0` threshold that almost always passes -- the same
"gating on |decision - output|" pattern report.md Section 3.2 documented as
broken for LMS/NLMS (that residual is small whenever the filter output is
close to *any* constellation point, right or wrong, so it cannot actually
detect a bad decision). Before RLS's first real use, this was fixed to
default to frozen-after-preamble (`enable_decision_directed=False`), matching
LMSFilter/NLMSFilter's established safe default. These tests codify that
fix the same way test_lms_stability.py codifies it for LMS/NLMS.
"""
import numpy as np
import pytest

from src.signal_gen import generate_bpsk, generate_qpsk
from src.channel import add_awgn
from src.rls_filter import RLSFilter
from src.utils import bits_to_symbols
from src.metrics import compute_snr_db

NUM_SYMBOLS = 20_000
TAPS = 16
RLS_LAMBDA = 0.99
TRAINING_LENGTH = 1000
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
N_TRIALS = 5


@pytest.mark.parametrize("gen,modulation", [(generate_bpsk, "BPSK"), (generate_qpsk, "QPSK")])
@pytest.mark.parametrize("snr", SNR_LEVELS)
def test_rls_never_worse_than_no_processing(gen, modulation, snr):
    """With the (now default) frozen-after-preamble behavior, RLS's output
    SNR, averaged over multiple trials, must never be meaningfully worse than
    raw No-Processing -- the same acceptance criterion test_lms_stability.py
    applies to LMS/NLMS."""
    rls_snrs, noproc_snrs = [], []
    for trial in range(N_TRIALS):
        tx, bits, h_rrc = gen(NUM_SYMBOLS, seed=100 + trial, sps=1)
        ref = bits_to_symbols(bits, modulation)
        noisy = add_awgn(tx, snr, seed=200 + trial * 7 + snr)
        noproc_snrs.append(compute_snr_db(ref, noisy))

        rls = RLSFilter(num_taps=TAPS, lam=RLS_LAMBDA)
        rls_out, _ = rls.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation=modulation)
        rls_snrs.append(compute_snr_db(ref, rls_out))

    mean_noproc = np.mean(noproc_snrs)
    mean_rls = np.mean(rls_snrs)

    # Same 1.5dB tolerance and rationale as test_lms_stability.py's analogous
    # test (reduced NUM_SYMBOLS/N_TRIALS vs. the full study inflates the
    # still-converging preamble's share of the averaged signal).
    assert mean_rls >= mean_noproc - 1.5, (
        f"RLS diverged: {modulation} SNR_in={snr}dB mean No-Proc={mean_noproc:.2f}dB "
        f"but mean RLS output={mean_rls:.2f}dB over {N_TRIALS} trials"
    )


def test_rls_decision_directed_off_by_default():
    """enable_decision_directed must default to False -- if this default is
    ever flipped, RLS needs the same reliability/confidence gating LMS/NLMS
    use (not yet implemented here) before it can be trusted with DD enabled."""
    import inspect
    sig = inspect.signature(RLSFilter.denoise)
    assert sig.parameters["enable_decision_directed"].default is False


def test_rls_frozen_matches_explicit_false():
    """Sanity check: relying on the default vs. explicitly passing
    enable_decision_directed=False must give identical output."""
    tx, bits, h_rrc = generate_bpsk(5000, seed=1, sps=1)
    noisy = add_awgn(tx, 5.0, seed=1)

    rls_default = RLSFilter(num_taps=TAPS, lam=RLS_LAMBDA)
    out_default, _ = rls_default.denoise(tx, noisy, training_length=TRAINING_LENGTH, modulation="BPSK")

    rls_explicit = RLSFilter(num_taps=TAPS, lam=RLS_LAMBDA)
    out_explicit, _ = rls_explicit.denoise(tx, noisy, training_length=TRAINING_LENGTH,
                                            modulation="BPSK", enable_decision_directed=False)

    assert np.allclose(out_default, out_explicit)
