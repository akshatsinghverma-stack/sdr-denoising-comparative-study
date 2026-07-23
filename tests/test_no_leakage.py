"""
Regression test for the train/test leakage question raised during Case
Study 2 review. Verifies the general property the experiment scripts rely
on: different seeds passed to the signal generators produce statistically
independent (in practice, exactly distinct) bit sequences, so a training
seed and a set of Monte Carlo test-trial seeds can never accidentally
collide on the same underlying data.
"""
import numpy as np

from src.signal_gen import generate_bpsk, generate_qpsk


def _distinct_bits_across_seeds(gen):
    seeds = [42] + [42 + 5000 + t * 97 for t in range(10)]
    bit_sequences = [gen(5000, seed=s, sps=1)[1] for s in seeds]
    for i in range(len(bit_sequences)):
        for j in range(i + 1, len(bit_sequences)):
            assert not np.array_equal(bit_sequences[i], bit_sequences[j]), (
                f"Seeds {seeds[i]} and {seeds[j]} produced identical bit sequences -- "
                "training and test data would leak into each other."
            )


def test_bpsk_seeds_produce_distinct_bit_sequences():
    _distinct_bits_across_seeds(generate_bpsk)


def test_qpsk_seeds_produce_distinct_bit_sequences():
    _distinct_bits_across_seeds(generate_qpsk)
