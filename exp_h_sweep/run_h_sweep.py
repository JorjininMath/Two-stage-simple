"""
run_h_sweep.py

Sweep the CKME indicator bandwidth h across a wide range and record:
  - F̂(t|x) at representative x values (CDF shape)
  - f̂(t|x) via finite difference (density shape)
  - Quantile errors |q̂_τ − q_oracle| for several τ
  - Pointwise CRPS(x)

Other hyperparameters are fixed: ell_x=0.5, lam=1e-3.
Simulator: nongauss_B2L, n_0=200, r_0=5.

Usage:
  python exp_h_sweep/run_h_sweep.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from CKME.ckme import CKMEModel
from CKME.loss_functions.pinball import _invert_cdf
from CKME.parameters import Params
from Two_stage.data_collection import collect_stage1_data
from Two_stage.sim_functions import get_experiment_config

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SIM = "nongauss_B2L"
N_0 = 200
R_0 = 5
N_MACRO = 5
BASE_SEED = 42
ALPHA = 0.1
T_GRID_SIZE = 500

H_LIST = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ELL_X_FIXED = 0.5
LAM_FIXED = 1e-3

REP_X = [1.0, float(np.pi), 5.0]
TAU_LIST = [0.05, 0.25, 0.5, 0.75, 0.95]

# Dense grid for per-x quantile/CRPS evaluation
N_TEST = 200


# ---------------------------------------------------------------------------
# Oracle (dispatches on simulator name)
# ---------------------------------------------------------------------------
def oracle_quantile(x: np.ndarray, tau: float, sim: str) -> np.ndarray:
    from Two_stage.sim_functions.exp2 import exp2_true_function
    f = exp2_true_function(x)
    if sim.startswith("nongauss_B2"):
        from scipy.stats import gamma as gamma_dist
        from math import pi as _PI, sqrt
        k = 2.0 if sim.endswith("L") else 9.0
        sigma_tar = 0.1 + 0.1 * (x - _PI) ** 2
        theta = sigma_tar / sqrt(k)
        return f + gamma_dist.ppf(tau, k, scale=theta) - k * theta
    else:  # exp2: Gaussian
        from scipy.stats import norm
        from Two_stage.sim_functions.exp2 import exp2_noise_variance_function
        sigma = np.sqrt(exp2_noise_variance_function(x))
        return f + sigma * norm.ppf(tau)


def oracle_cdf(x_scalar: float, t_grid: np.ndarray, sim: str) -> np.ndarray:
    from Two_stage.sim_functions.exp2 import exp2_true_function
    f = exp2_true_function(np.array([x_scalar]))[0]
    if sim.startswith("nongauss_B2"):
        from scipy.stats import gamma as gamma_dist
        from math import pi as _PI, sqrt
        k = 2.0 if sim.endswith("L") else 9.0
        sigma_tar = 0.1 + 0.1 * (x_scalar - _PI) ** 2
        theta = sigma_tar / sqrt(k)
        return gamma_dist.cdf(t_grid - f + k * theta, k, scale=theta)
    else:
        from scipy.stats import norm
        from Two_stage.sim_functions.exp2 import exp2_noise_variance_function
        sigma = np.sqrt(exp2_noise_variance_function(np.array([x_scalar]))[0])
        return norm.cdf(t_grid, loc=f, scale=sigma)


def oracle_pdf(x_scalar: float, t_grid: np.ndarray, sim: str) -> np.ndarray:
    from Two_stage.sim_functions.exp2 import exp2_true_function
    f = exp2_true_function(np.array([x_scalar]))[0]
    if sim.startswith("nongauss_B2"):
        from scipy.stats import gamma as gamma_dist
        from math import pi as _PI, sqrt
        k = 2.0 if sim.endswith("L") else 9.0
        sigma_tar = 0.1 + 0.1 * (x_scalar - _PI) ** 2
        theta = sigma_tar / sqrt(k)
        return gamma_dist.pdf(t_grid - f + k * theta, k, scale=theta)
    else:
        from scipy.stats import norm
        from Two_stage.sim_functions.exp2 import exp2_noise_variance_function
        sigma = np.sqrt(exp2_noise_variance_function(np.array([x_scalar]))[0])
        return norm.pdf(t_grid, loc=f, scale=sigma)


# ---------------------------------------------------------------------------
# Pointwise CRPS
# ---------------------------------------------------------------------------
def pointwise_crps(F_pred: np.ndarray, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Return per-sample CRPS, shape (n,)."""
    ind = (Y[:, None] <= t_grid[None, :]).astype(float)
    dt = np.mean(np.diff(t_grid))
    return np.sum((F_pred - ind) ** 2, axis=1) * dt


# ---------------------------------------------------------------------------
# One macrorep
# ---------------------------------------------------------------------------
def run_one_macrorep(macrorep_id: int, sim: str):
    seed = BASE_SEED + macrorep_id * 10000
    exp_config = get_experiment_config(sim)
    X_bounds = exp_config["bounds"]
    d = exp_config["d"]

    X_all, Y_all = collect_stage1_data(
        n_0=N_0, d=d, r_0=R_0, simulator_func=sim,
        X_bounds=X_bounds, design_method="lhs", random_state=seed,
    )

    Y_lo = np.percentile(Y_all, 0.5)
    Y_hi = np.percentile(Y_all, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, T_GRID_SIZE)

    X_test = np.linspace(X_bounds[0][0], X_bounds[1][0], N_TEST).reshape(-1, 1)
    sim_func = exp_config["simulator"]
    Y_test = np.asarray(sim_func(X_test, random_state=seed + 5000)).ravel()

    # Oracle quantiles at test x (for each tau)
    x_test_arr = X_test.ravel()
    q_oracle = {tau: oracle_quantile(x_test_arr, tau, sim) for tau in TAU_LIST}

    rows = []
    cdf_curves = {}  # {h: {x_rep: F_hat}}
    pdf_curves = {}  # {h: {x_rep: f_hat}}

    for h in H_LIST:
        t0 = time.time()
        params = Params(ell_x=ELL_X_FIXED, lam=LAM_FIXED, h=h)
        model = CKMEModel(indicator_type="logistic")
        model.fit(X=X_all, Y=Y_all, params=params, r=R_0)

        F_test = model.predict_cdf(X_test, t_grid, clip=True)  # (n_test, M)

        # Quantile errors per tau
        tau_errors = {}
        for tau in TAU_LIST:
            q_hat = _invert_cdf(F_test, t_grid, tau)
            tau_errors[tau] = np.abs(q_hat - q_oracle[tau])

        # Pointwise CRPS
        crps_vals = pointwise_crps(F_test, Y_test, t_grid)

        # Store per-test-point rows
        for i in range(N_TEST):
            row = {
                "macrorep": macrorep_id,
                "h": h,
                "x": float(x_test_arr[i]),
                "crps": float(crps_vals[i]),
            }
            for tau in TAU_LIST:
                row[f"err_tau{tau}"] = float(tau_errors[tau][i])
            rows.append(row)

        # CDF/PDF curves at representative x (macrorep 0 only for plotting)
        if macrorep_id == 0:
            cdf_curves[h] = {}
            pdf_curves[h] = {}
            for x_rep in REP_X:
                idx = np.argmin(np.abs(x_test_arr - x_rep))
                F_row = F_test[idx]
                cdf_curves[h][x_rep] = F_row.copy()
                # Finite difference density
                f_row = np.gradient(F_row, t_grid)
                pdf_curves[h][x_rep] = f_row

        elapsed = time.time() - t0
        print(f"    h={h:.3f}: ({elapsed:.1f}s)")

    return pd.DataFrame(rows), cdf_curves, pdf_curves, t_grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default=DEFAULT_SIM,
                        help="Simulator name: nongauss_B2L or exp2")
    args = parser.parse_args()
    sim = args.sim

    out_dir = _HERE / f"output_{sim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"h sweep: sim={sim}, n_0={N_0}, r_0={R_0}, n_macro={N_MACRO}")
    print(f"  h_list: {H_LIST}")
    print(f"  Fixed: ell_x={ELL_X_FIXED}, lam={LAM_FIXED}")
    print(f"  Rep x: {REP_X}")

    frames = []
    saved_curves = None
    saved_t_grid = None

    for k in range(N_MACRO):
        print(f"\n--- Macrorep {k}/{N_MACRO} ---")
        df, cdf_curves, pdf_curves, t_grid = run_one_macrorep(k, sim)
        frames.append(df)
        if k == 0:
            saved_curves = (cdf_curves, pdf_curves)
            saved_t_grid = t_grid

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(out_dir / "h_sweep_perpoint.csv", index=False)
    print(f"\nSaved: {out_dir / 'h_sweep_perpoint.csv'}")

    # Save CDF/PDF curves (macrorep 0)
    if saved_curves is not None:
        cdf_curves, pdf_curves = saved_curves
        arrs = {"t_grid": saved_t_grid,
                "h_list": np.array(H_LIST),
                "rep_x": np.array(REP_X)}
        for h in H_LIST:
            for x_rep in REP_X:
                arrs[f"cdf_h{h}_x{x_rep:.2f}"] = cdf_curves[h][x_rep]
                arrs[f"pdf_h{h}_x{x_rep:.2f}"] = pdf_curves[h][x_rep]
        np.savez(out_dir / "h_sweep_curves.npz", **arrs)
        print(f"Saved curves: {out_dir / 'h_sweep_curves.npz'}")

    # Quick summary
    print(f"\n=== Mean CRPS(x) and mean quantile errors, by h ===")
    for h in H_LIST:
        sub = df_all[df_all["h"] == h]
        crps_mean = sub["crps"].mean()
        errs = {tau: sub[f"err_tau{tau}"].mean() for tau in TAU_LIST}
        print(f"  h={h:.3f}: CRPS={crps_mean:.4f}  "
              f"err_0.05={errs[0.05]:.3f}  err_0.5={errs[0.5]:.3f}  "
              f"err_0.95={errs[0.95]:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
