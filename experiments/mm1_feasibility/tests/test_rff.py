import numpy as np

from experiments.mm1_feasibility.experiment.rff import RFFTransformer1D


def test_rff_transformer_dimension_and_finite_values():
    rng = np.random.default_rng(11)
    observations = rng.normal(size=20)
    transformer = RFFTransformer1D(n_features=7).fit(observations, rng)
    features = transformer.transform(observations[:5])
    assert features.shape == (5, 7)
    assert np.all(np.isfinite(features))


def test_rff_same_seed_gives_same_features():
    observations = np.linspace(-1.0, 1.0, 20)
    rng1 = np.random.default_rng(12)
    rng2 = np.random.default_rng(12)
    first = RFFTransformer1D(n_features=8).fit(observations, rng1).transform(observations)
    second = RFFTransformer1D(n_features=8).fit(observations, rng2).transform(observations)
    np.testing.assert_allclose(first, second)
