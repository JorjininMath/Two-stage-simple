import numpy as np

from ckme_dcp_mm1.experiment.dgp import generate_training_scenarios
from ckme_dcp_mm1.experiment.features import KMEFeatureBuilder, Standardizer, bootstrap_query_features, input_uncertainty


def test_kme_feature_dimensions():
    rng = np.random.default_rng(1)
    scenarios = generate_training_scenarios(4, rng)
    builder = KMEFeatureBuilder(rff_dim=6).fit(scenarios, rng)
    features = builder.transform(scenarios)
    assert features.shape == (4, 2 * 6 + 3)
    assert input_uncertainty(scenarios).shape == (4,)


def test_bootstrap_query_features_uses_same_dimension():
    rng = np.random.default_rng(2)
    scenarios = generate_training_scenarios(5, rng)
    builder = KMEFeatureBuilder(rff_dim=5).fit(scenarios, rng)
    standardizer = Standardizer().fit(builder.transform(scenarios))
    z_boot, u_boot = bootstrap_query_features(scenarios[0], builder, standardizer, rng, n_bootstrap=7)
    assert z_boot.shape == (7, 2 * 5 + 3)
    assert u_boot.shape == (7,)


def test_standardizer_uses_fit_statistics_for_other_splits():
    fit = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    val = np.array([[3.0, 6.0]])
    standardizer = Standardizer().fit(fit)
    transformed_fit = standardizer.transform(fit)
    transformed_val = standardizer.transform(val)
    np.testing.assert_allclose(np.mean(transformed_fit, axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(transformed_val, np.zeros((1, 2)), atol=1e-12)
