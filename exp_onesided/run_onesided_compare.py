"""
run_onesided_compare.py

One-sided quantile comparison: CKME vs linear QR (plug-in, no calibration).

Layer 1 (estimation) comparison only:
  - QR:   directly fits conditional quantile q_tau(x) via linear quantile regression
  - CKME: estimates conditional CDF F(y|x), then inverts to get q_tau(x)

Simulators: exp1 (Gaussian), exp2 (heteroscedastic Gaussian), nongauss_B2L (Gamma strong skew)
Tau levels:  0.05 (lower bound) and 0.95 (upper bound)

Additional diagnostics:
  - True quantile approximation q_true_tau(x) via Monte Carlo
  - sup_x |q_hat_tau(x) - q_true_tau(x)| boxplot-ready metrics
  - Tail probability check at true quantile:
      delta_tau(x) = [1 - F_hat(q_true_tau(x) | x)] - (1 - tau)
  - Fixed-threshold tail probability curves:
      P(Y > t | X = x) for selected t values

Usage:
    python exp_onesided/run_onesided_compare.py
    python exp_onesided/run_onesided_compare.py --n_macro 5 --base_seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CKME.ckme import CKMEModel
from CKME.coefficients import compute_ckme_coeffs
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config

_PRETRAINED_PATH = Path(__file__).resolve().parent / "pretrained_params.json"
_DEFAULT_PARAMS = Params(ell_x=0.5, lam=0.001, h=0.1)


def _load_pretrained() -> dict:
    """Load per-simulator pretrained params from JSON. Returns empty dict if not found."""
    if _PRETRAINED_PATH.exists():
        with open(_PRETRAINED_PATH) as f:
            raw = json.load(f)
        return {sim: Params(**raw[sim]) for sim in raw}
    print(
        f"Warning: {_PRETRAINED_PATH} not found; using default params for all simulators.\n"
        "Run 'python exp_onesided/pretrain_params.py' first to improve accuracy.",
        file=sys.stderr,
    )
    return {}

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not found. QR baseline will be skipped.")

SIMULATORS = ["exp1", "exp2", "nongauss_B2L"]


def _tau_tag(tau: float) -> str:
    """Stable tau string for filenames."""
    return f"{tau:.2f}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    residual = y - q
    return float(np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual)))


def coverage_lower(y: np.ndarray, q: np.ndarray) -> float:
    """Empirical P(Y >= q)."""
    return float(np.mean(y >= q))


def coverage_upper(y: np.ndarray, q: np.ndarray) -> float:
    """Empirical P(Y <= q)."""
    return float(np.mean(y <= q))


# ---------------------------------------------------------------------------
# QR helpers
# ---------------------------------------------------------------------------

def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    # degree=1 matches R dcp.qr which uses cbind(1, X0) — linear features only
    x = x.ravel()
    return np.column_stack([x ** d for d in range(degree + 1)])


def fit_qr_predict(X_train: np.ndarray, Y_train: np.ndarray,
                   tau: float, degree: int):
    """Fit linear QR with polynomial features, return predict function."""
    X_feat = poly_features(X_train, degree)
    model = sm.QuantReg(Y_train, X_feat)
    result = model.fit(q=tau, max_iter=5000)

    def predict(X_test: np.ndarray) -> np.ndarray:
        return result.predict(poly_features(X_test, degree))

    return predict


def _estimate_true_quantiles(
    simulator,
    X_test: np.ndarray,
    taus: list[float],
    n_mc: int,
    rng: np.random.Generator,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """
    Approximate true quantiles q_tau(x) by Monte Carlo.

    Returns
    -------
    q_true : dict[tau -> ndarray shape (n_test,)]
    y_mc : ndarray shape (n_mc, n_test)
        Monte Carlo samples Y^(b)(x_i) for each x_i on test grid.
    """
    y_mc = np.empty((n_mc, X_test.shape[0]), dtype=float)
    x_flat = X_test.ravel()
    for b in range(n_mc):
        y_mc[b, :] = simulator(x_flat, random_state=int(rng.integers(0, 2**31)))
    q_true = {tau: np.quantile(y_mc, tau, axis=0) for tau in taus}
    return q_true, y_mc


def _predict_ckme_cdf_at_thresholds(
    ckme: CKMEModel, X_query: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """
    Compute F_hat(thresholds[i] | X_query[i]) for i=1..q.
    """
    C = compute_ckme_coeffs(ckme.L, ckme.kx, ckme.X, X_query)  # (n_train, q)
    G_var = np.column_stack(
        [ckme.indicator.g_vector(ckme.Y, float(ti)) for ti in thresholds]
    )  # (n_train, q)
    F = np.sum(C * G_var, axis=0)  # (q,)
    return np.clip(F, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-simulator run
# ---------------------------------------------------------------------------

def run_one_simulator(sim_name: str, config: dict, rng: np.random.Generator,
                      pretrained: dict, monotone: bool = False) -> tuple:
    """
    Run CKME vs QR comparison for one simulator.

    Returns
    -------
    rows : list of dicts  — summary metrics per (tau, method)
    sup_rows : list of dicts  — quantile error summaries per (tau, method)
    tail_rows : list of dicts — tail-prob summaries per tau (CKME)
    tail_curve_rows : list of dicts — fixed-threshold tail-prob curves
    per_point : dict      — {tau: DataFrame with x, y, q_ckme, q_qr, q_true, ...}
    """
    sim_cfg = get_experiment_config(sim_name)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    n_train = config["n_train"]
    r_train = config["r_train"]
    n_test = config["n_test"]
    t_grid_size = config["t_grid_size"]
    degree = config["poly_degree"]
    taus = config["taus"]
    n_true_mc = config["n_true_mc"]
    tail_t_values = config["tail_t_values"]
    tail_t_quantiles = config["tail_t_quantiles"]
    params = pretrained.get(sim_name, config["ckme_params"])

    # Training data: n_train sites x r_train reps each
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_reps = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(r_train)
    ]
    X_train = np.tile(X_sites, (r_train, 1))   # (n_train * r_train, 1)
    Y_train = np.concatenate(Y_reps)            # (n_train * r_train,)

    # Test data: uniform grid for clean visualization
    X_test = np.linspace(x_lo, x_hi, n_test).reshape(-1, 1)
    Y_test = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))

    # True quantiles and tail probabilities approximated via Monte Carlo
    q_true_dict, y_mc = _estimate_true_quantiles(simulator, X_test, taus, n_true_mc, rng)

    # t_grid: cover training Y range with 10% margin (percentile-based to resist outliers)
    Y_lo = np.percentile(Y_train, 0.5)
    Y_hi = np.percentile(Y_train, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    # Fit CKME once
    ckme = CKMEModel(indicator_type="logistic")
    ckme.fit(X_train, Y_train, params=params)

    rows = []
    sup_rows = []
    tail_rows = []
    tail_curve_rows = []
    per_point = {}

    # Fixed-threshold t values for P(Y > t | X=x)
    if tail_t_values is None:
        fixed_t_values = np.quantile(Y_train, tail_t_quantiles).astype(float)
    else:
        fixed_t_values = np.asarray(tail_t_values, dtype=float).ravel()

    for tau in taus:
        side = "lower" if tau < 0.5 else "upper"
        cov_fn = coverage_lower if side == "lower" else coverage_upper

        # CKME quantile
        q_ckme = ckme.predict_quantile(X_test, tau, t_grid, monotone=monotone)
        q_true = q_true_dict[tau]
        rows.append({
            "simulator": sim_name, "tau": tau, "side": side, "method": "CKME",
            "coverage": cov_fn(Y_test, q_ckme),
            "pinball_loss": pinball_loss(Y_test, q_ckme, tau),
        })
        sup_rows.append({
            "simulator": sim_name, "tau": tau, "side": side, "method": "CKME",
            "sup_abs_quantile_error": float(np.max(np.abs(q_ckme - q_true))),
            "mean_abs_quantile_error": float(np.mean(np.abs(q_ckme - q_true))),
        })

        # QR quantile
        q_qr = np.full(n_test, np.nan)
        if HAS_STATSMODELS:
            try:
                predict_qr = fit_qr_predict(X_train, Y_train, tau, degree)
                q_qr = predict_qr(X_test)
                rows.append({
                    "simulator": sim_name, "tau": tau, "side": side, "method": "QR",
                    "coverage": cov_fn(Y_test, q_qr),
                    "pinball_loss": pinball_loss(Y_test, q_qr, tau),
                })
                sup_rows.append({
                    "simulator": sim_name, "tau": tau, "side": side, "method": "QR",
                    "sup_abs_quantile_error": float(np.max(np.abs(q_qr - q_true))),
                    "mean_abs_quantile_error": float(np.mean(np.abs(q_qr - q_true))),
                })
            except Exception as e:
                print(f"    QR failed for {sim_name} tau={tau}: {e}")

        # Tail probability diagnostic at q_true_tau(x)
        # target: 1 - F_true(q_true_tau(x) | x) ~= 1 - tau
        F_ckme_at_qtrue = _predict_ckme_cdf_at_thresholds(ckme, X_test, q_true)
        tail_prob_ckme = 1.0 - F_ckme_at_qtrue
        tail_prob_true = np.mean(y_mc > q_true[np.newaxis, :], axis=0)
        delta = tail_prob_ckme - (1.0 - tau)
        tail_rows.append({
            "simulator": sim_name,
            "tau": tau,
            "side": side,
            "method": "CKME",
            "target_tail_prob": float(1.0 - tau),
            "mean_tail_prob_true_at_qtrue": float(np.mean(tail_prob_true)),
            "mean_tail_prob_ckme_at_qtrue": float(np.mean(tail_prob_ckme)),
            "sup_abs_delta_tailprob": float(np.max(np.abs(delta))),
            "mean_abs_delta_tailprob": float(np.mean(np.abs(delta))),
        })

        per_point[tau] = pd.DataFrame({
            "x": X_test.ravel(),
            "y": Y_test,
            "q_ckme": q_ckme,
            "q_qr": q_qr,
            "q_true": q_true,
            "tail_prob_true_at_qtrue": tail_prob_true,
            "tail_prob_ckme_at_qtrue": tail_prob_ckme,
            "delta_tailprob_ckme": delta,
        })

    # Fixed-threshold tail probability curve diagnostics
    # For each t: compare true MC curve vs CKME 1-F_hat(t|x)
    x_flat = X_test.ravel()
    for t_val in fixed_t_values:
        tail_true_t = np.mean(y_mc > float(t_val), axis=0)
        F_ckme_t = ckme.predict_cdf(X_test, t=float(t_val))
        tail_ckme_t = 1.0 - np.asarray(F_ckme_t, dtype=float).ravel()
        abs_err_t = np.abs(tail_ckme_t - tail_true_t)
        for x_i, p_true, p_ckme, ae in zip(x_flat, tail_true_t, tail_ckme_t, abs_err_t):
            tail_curve_rows.append({
                "simulator": sim_name,
                "x": float(x_i),
                "t_value": float(t_val),
                "tail_prob_true": float(p_true),
                "tail_prob_ckme": float(p_ckme),
                "abs_error": float(ae),
            })

    return rows, sup_rows, tail_rows, tail_curve_rows, per_point


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macro_k: int,
    seed: int,
    config: dict,
    pretrained: dict,
    output_dir: str,
    monotone: bool = False,
) -> tuple[list, list, list, list]:
    """
    Run all simulators for one macrorep. Writes per-point CSVs directly to disk.

    Returns (rows, sup_rows, tail_rows, tail_curve_rows) — no large arrays.
    """
    rng = np.random.default_rng(seed)
    macro_dir = os.path.join(output_dir, f"macrorep_{macro_k}")
    os.makedirs(macro_dir, exist_ok=True)

    rows, sup_rows, tail_rows, tail_curve_rows = [], [], [], []

    for sim_name in SIMULATORS:
        r, s, t, tc, per_point = run_one_simulator(sim_name, config, rng, pretrained,
                                                    monotone=monotone)

        for row in r:
            print(f"  [rep{macro_k}] {sim_name}  {row['method']:4s} "
                  f"tau={row['tau']:.2f}: "
                  f"cov={row['coverage']:.3f}  pinball={row['pinball_loss']:.4f}")

        rows.extend([{**row, "macrorep": macro_k} for row in r])
        sup_rows.extend([{**row, "macrorep": macro_k} for row in s])
        tail_rows.extend([{**row, "macrorep": macro_k} for row in t])
        tail_curve_rows.extend([{**row, "macrorep": macro_k} for row in tc])

        for tau, df in per_point.items():
            fname = f"{sim_name}_tau{_tau_tag(tau)}_perpoint.csv"
            df.to_csv(os.path.join(macro_dir, fname), index=False)

    return rows, sup_rows, tail_rows, tail_curve_rows


def parse_args():
    parser = argparse.ArgumentParser(description="One-sided quantile comparison")
    parser.add_argument("--n_macro", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel workers for macroreps (default: 1 = sequential).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    parser.add_argument(
        "--monotone",
        action="store_true",
        default=False,
        help="Apply isotonic regression to enforce CDF monotonicity before quantile inversion.",
    )
    parser.add_argument(
        "--tail_t_values",
        type=str,
        default=None,
        help=(
            "Comma-separated fixed t values for tail curves, e.g. --tail_t_values 0.0,1.0,2.0. "
            "If omitted, t values are chosen by training-Y quantiles."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "n_train": 1000,
        "r_train": 5,
        "n_test": 1000,
        "n_true_mc": 10000,
        "t_grid_size": 500,
        "poly_degree": 1,  # linear QR to match R dcp.qr: rq.fit.br(cbind(1, X0), Y0, tau)
        "taus": [0.05, 0.95],
        "tail_t_values": (
            [float(v) for v in args.tail_t_values.split(",")] if args.tail_t_values else None
        ),
        "tail_t_quantiles": [0.1, 0.5, 0.9],
        "ckme_params": Params(ell_x=0.5, lam=0.001, h=0.1),
    }

    pretrained = _load_pretrained()
    if pretrained:
        print("Loaded pretrained params:")
        for sim in SIMULATORS:
            p = pretrained.get(sim)
            if p is None:
                print(f"  {sim}: (missing) -> use default params")
            else:
                print(f"  {sim}: ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

    seeds = [args.base_seed + k for k in range(args.n_macro)]
    worker = partial(run_one_macrorep, config=config, pretrained=pretrained,
                     output_dir=args.output_dir, monotone=args.monotone)

    all_rows, all_sup_rows, all_tail_rows, all_tail_curve_rows = [], [], [], []

    n_jobs = min(args.n_jobs, args.n_macro)
    if n_jobs <= 1:
        # Sequential
        for macro_k, seed in enumerate(seeds):
            print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
            r, s, t, tc = worker(macro_k, seed)
            all_rows.extend(r)
            all_sup_rows.extend(s)
            all_tail_rows.extend(t)
            all_tail_curve_rows.extend(tc)
    else:
        print(f"Running {args.n_macro} macroreps with {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(worker, macro_k, seed): macro_k
                for macro_k, seed in enumerate(seeds)
            }
            for fut in as_completed(futures):
                macro_k = futures[fut]
                r, s, t, tc = fut.result()
                print(f"  Macrorep {macro_k} done.")
                all_rows.extend(r)
                all_sup_rows.extend(s)
                all_tail_rows.extend(t)
                all_tail_curve_rows.extend(tc)

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.output_dir, "onesided_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nDone. Summary → {summary_path}")

    sup_df = pd.DataFrame(all_sup_rows)
    sup_path = os.path.join(args.output_dir, "onesided_quantile_sup_error.csv")
    sup_df.to_csv(sup_path, index=False)
    print(f"Quantile sup-error summary → {sup_path}")

    tail_df = pd.DataFrame(all_tail_rows)
    tail_path = os.path.join(args.output_dir, "onesided_tailprob_summary.csv")
    tail_df.to_csv(tail_path, index=False)
    print(f"Tail-prob summary → {tail_path}")

    tail_curve_df = pd.DataFrame(all_tail_curve_rows)
    tail_curve_path = os.path.join(args.output_dir, "onesided_tailprob_curve.csv")
    tail_curve_df.to_csv(tail_curve_path, index=False)
    print(f"Tail-prob curve data → {tail_curve_path}")

    # Print aggregate table
    if args.n_macro > 1:
        agg = summary.groupby(["simulator", "tau", "method"])[
            ["coverage", "pinball_loss"]
        ].agg(["mean", "std"]).round(4)
        print("\n--- Aggregate (mean ± std) ---")
        print(agg.to_string())


if __name__ == "__main__":
    main()
