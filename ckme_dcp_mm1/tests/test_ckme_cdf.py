import numpy as np

from ckme_dcp_mm1.experiment.ckme_cdf import (
    CKMECDFEstimator,
    KRRConditionalCDFEstimator,
    make_cdf_grid,
    pointwise_band,
    rbf_kernel,
    tune_krr_cdf,
)


def test_ckme_cdf_is_bounded_and_monotone():
    rng = np.random.default_rng(3)
    z_fit = rng.normal(size=(30, 4))
    y_fit = rng.exponential(size=30)
    u_fit = np.ones(30)
    estimator = CKMECDFEstimator(k_neighbors=5).fit(z_fit, y_fit, u_fit)
    t_grid = np.linspace(0, 5, 100)
    curve = estimator.predict_cdf(z_fit[:2], t_grid, u_query=np.ones(2))
    assert curve.shape == (2, 100)
    assert np.all(curve >= 0)
    assert np.all(curve <= 1)
    assert np.all(np.diff(curve, axis=1) >= -1e-12)


def test_pointwise_band_shapes_and_order():
    curves = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.1, 0.3, 1.0],
            [0.2, 0.4, 1.0],
        ]
    )
    low, med, high = pointwise_band(curves, alpha=0.2)
    assert low.shape == med.shape == high.shape == (3,)
    assert np.all(low <= med)
    assert np.all(med <= high)


def test_krr_cdf_gram_shape_prediction_bounds_and_validation():
    rng = np.random.default_rng(4)
    z_fit = rng.normal(size=(25, 3))
    y_fit = rng.exponential(size=(25, 2))
    z_val = rng.normal(size=(8, 3))
    y_val = rng.exponential(size=8)
    t_grid = make_cdf_grid(y_fit, grid_size=12)

    gram = rbf_kernel(z_fit, z_fit, tau=1.5)
    assert gram.shape == (25, 25)

    estimator = KRRConditionalCDFEstimator(tau=1.5, ridge=1e-2, t_grid=t_grid).fit(z_fit, y_fit)
    pred = estimator.predict_cdf(z_val)
    assert pred.shape == (8, 12)
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)
    assert np.all(np.diff(pred, axis=1) >= -1e-12)
    assert np.isfinite(estimator.validation_mse(z_val, y_val))


def test_krr_cdf_tuning_returns_finite_objective():
    rng = np.random.default_rng(5)
    z_fit = rng.normal(size=(20, 2))
    y_fit = rng.exponential(size=20)
    z_val = rng.normal(size=(6, 2))
    y_val = rng.exponential(size=6)
    t_grid = make_cdf_grid(y_fit, grid_size=10)
    estimator, info = tune_krr_cdf(
        z_fit,
        y_fit,
        z_val,
        y_val,
        t_grid,
        tau_candidates=[0.5, 1.0],
        ridge_candidates=(1e-3, 1e-1),
    )
    assert estimator.predict_cdf(z_val).shape == (6, 10)
    assert np.isfinite(info["val_mse"])
