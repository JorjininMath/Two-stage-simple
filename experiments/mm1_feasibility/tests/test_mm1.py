import numpy as np

from experiments.mm1_feasibility.experiment.mm1 import average_sojourn, compute_sojourn_times, simulate_mm1_output, simulate_mm1_outputs


def test_queue_recursion_hand_example():
    inter_arrivals = np.array([1.0, 1.0])
    services = np.array([2.0, 1.0])
    np.testing.assert_allclose(compute_sojourn_times(inter_arrivals, services), np.array([2.0, 2.0]))
    assert average_sojourn(inter_arrivals, services) == 2.0


def test_mm1_output_reproducible_and_positive():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    y1 = simulate_mm1_output(0.7, 1.0, rng1)
    y2 = simulate_mm1_output(0.7, 1.0, rng2)
    assert y1 == y2
    assert y1 > 0


def test_vectorized_mm1_outputs_are_finite_positive():
    rng = np.random.default_rng(321)
    values = simulate_mm1_outputs(0.7, 1.0, rng, n_outputs=5)
    assert values.shape == (5,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
