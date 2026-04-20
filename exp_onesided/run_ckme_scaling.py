"""
run_ckme_scaling.py

CKME scaling study: quantile estimation error vs training data size.

Simulators : nongauss_A1S (Student-t ν=10) and nongauss_A1L (Student-t ν=3)
Metric     : mean and sup of |q̂_τ(x) - q_true_τ(x)| over test grid
Taus       : 0.05 (lower tail), 0.95 (upper tail)
Varies     : n_train in N_TRAIN_LIST, r_train=10 (10 reps per site)

Fixed hyperparameters are loaded from pretrained_params.json (same file as
run_onesided_compare.py). Run pretrain_params.py --sims nongauss_A1S,nongauss_A1L
first if not already done.

Outputs (in --output_dir):
  scaling_raw.csv     — one row per (macrorep, n_train, simulator, tau)
  scaling_summary.csv — mean ± std across macroreps per (n_train, simulator, tau)

Usage:
    python exp_onesided/run_ckme_scaling.py
    python exp_onesided/run_ckme_scaling.py --n_macro 20 --n_jobs 4
    python exp_onesided/run_ckme_scaling.py --monotone --n_macro 10
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

from CKME.coefficients import build_cholesky_factor, solve_ckme_system
from CKME.indicators import make_indicator
from CKME.kernels import make_x_rbf_kernel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not found. QR baseline will be skipped.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIMULATORS = ["exp1", "exp2", "nongauss_A1L"]
N_TRAIN_LIST = [500, 1000, 2000, 5000]
R_TRAIN = 10         # replications per training site
TAUS = [0.05, 0.95]
QR_POLY_DEGREE = 1  # linear QR to match R dcp.qr: rq.fit.br(cbind(1, X0), Y0, tau)

_PRETRAINED_PATH = Path(__file__).resolve().parent / "pretrained_params.json"
_DEFAULT_PARAMS = Params(ell_x=0.5, lam=0.001, h=0.1)


def _load_pretrained() -> dict:
    if _PRETRAINED_PATH.exists():
        with open(_PRETRAINED_PATH) as f:
            raw = json.load(f)
        return {sim: Params(**raw[sim]) for sim in raw}
    print(
        f"Warning: {_PRETRAINED_PATH} not found; using default params.\n"
        "Run 'python exp_onesided/pretrain_params.py --sims nongauss_A1S,nongauss_A1L' first.",
        file=sys.stderr,
    )
    return {}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# QR helpers
# ---------------------------------------------------------------------------

def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    # degree=1 → [x], then sm.add_constant prepends 1 → [1, x],
    # matching R dcp.qr: rq.fit.br(cbind(1, X0), Y0, tau)
    x = x.ravel()
    return np.column_stack([x**k for k in range(1, degree + 1)])


def _fit_qr_predict(X_train: np.ndarray, Y_train: np.ndarray, tau: float, degree: int):
    """Fit polynomial QR and return a predict function."""
    X_feat = sm.add_constant(_poly_features(X_train.ravel(), degree))
    result = sm.QuantReg(Y_train, X_feat).fit(q=tau, max_iter=5000, disp=False)
    def predict(X_query: np.ndarray) -> np.ndarray:
        Xq = sm.add_constant(
            _poly_features(X_query.ravel(), degree), has_constant="add"
        )
        return result.predict(Xq)
    return predict


# ---------------------------------------------------------------------------
# True quantile helper
# ---------------------------------------------------------------------------

def _true_quantiles(
    simulator,
    X_test: np.ndarray,
    taus: list[float],
    n_mc: int,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Monte Carlo approximation of true conditional quantiles."""
    y_mc = np.empty((n_mc, X_test.shape[0]), dtype=float)
    x_flat = X_test.ravel()
    for b in range(n_mc):
        y_mc[b] = simulator(x_flat, random_state=int(rng.integers(0, 2**31)))
    return {tau: np.quantile(y_mc, tau, axis=0) for tau in taus}


def run_one_case(
    sim_name: str,
    n_train: int,
    config: dict,
    rng: np.random.Generator,
    params: Params,
    monotone: bool,
    save_perpoint: bool = False,
    methods: tuple = ("CKME", "QR"),
) -> tuple[list[dict], list[dict]]:
    """
    One (simulator, n_train) case.

    Returns
    -------
    summary_rows : list of dicts, one per tau — aggregate metrics
    perpoint_rows : list of dicts, one per (tau, x) — per-point data.
        Empty if save_perpoint=False.
    """
    sim_cfg = get_experiment_config(sim_name)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    n_test      = config["n_test"]
    t_grid_size = config["t_grid_size"]
    n_true_mc   = config["n_true_mc"]
    r_train     = config.get("r_train", 1)

    # Training data: n_train sites, r_train reps each
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_reps = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(r_train)
    ]
    # Test grid (uniform, deterministic given x_lo/x_hi)
    X_test = np.linspace(x_lo, x_hi, n_test).reshape(-1, 1)

    # True quantiles via Monte Carlo
    q_true_dict = _true_quantiles(simulator, X_test, TAUS, n_true_mc, rng)

    # t_grid: cover training Y range with 10% margin
    Y_all = np.concatenate(Y_reps)
    Y_lo = np.percentile(Y_all, 0.5)
    Y_hi = np.percentile(Y_all, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)
    t_grid_min = float(t_grid[0])
    t_grid_max = float(t_grid[-1])

    # Collapsed CKME (skipped if "CKME" not in methods)
    F_all = None
    if "CKME" in methods:
        kx        = make_x_rbf_kernel(params.ell_x)
        indicator = make_indicator("logistic", params.h)
        K_sites = kx(X_sites, X_sites)
        L_sites = build_cholesky_factor(K_sites, n_train, params.lam)
        K_sq  = kx(X_sites, X_test)
        D_bar = solve_ckme_system(L_sites, K_sq)
        G_bar = np.zeros((n_train, t_grid_size))
        for rep_y in Y_reps:
            G_bar += indicator.g_matrix(rep_y, t_grid)
        G_bar /= r_train
        F_all = np.clip(D_bar.T @ G_bar, 0.0, 1.0)
        if monotone:
            from sklearn.isotonic import IsotonicRegression
            _ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            t_idx = np.arange(t_grid_size, dtype=float)
            for i in range(F_all.shape[0]):
                F_all[i] = _ir.fit_transform(t_idx, F_all[i])

    # QR training data: stack all reps
    X_train_qr = np.repeat(X_sites, r_train, axis=0)  # (n*r, 1)
    Y_train_qr = np.concatenate(Y_reps)               # (n*r,)

    summary_rows, perpoint_rows = [], []
    x_flat = X_test.ravel()

    for tau in TAUS:
        q_true = q_true_dict[tau]

        def _row(method, ae):
            return {
                "simulator":      sim_name,
                "n_train":        n_train,
                "tau":            tau,
                "method":         method,
                "mean_abs_err":   float(np.mean(ae)),
                "sup_abs_err":    float(np.max(ae)),
                "median_abs_err": float(np.median(ae)),
                "q95_abs_err":    float(np.quantile(ae, 0.95)),
                "t_grid_min":     t_grid_min,
                "t_grid_max":     t_grid_max,
            }

        if "CKME" in methods and F_all is not None:
            mask      = F_all >= tau
            has_valid = mask.any(axis=1)
            idx       = mask.argmax(axis=1)
            q_hat     = np.where(has_valid, t_grid[idx], t_grid[-1])
            abs_err   = np.abs(q_hat - q_true)
            summary_rows.append(_row("CKME", abs_err))

            if save_perpoint:
                # Distance of q_hat from grid boundary (negative = extrapolated)
                dist_to_max = t_grid_max - q_hat   # small → near right boundary
                dist_to_min = q_hat - t_grid_min   # small → near left boundary
                for i in range(len(x_flat)):
                    perpoint_rows.append({
                        "simulator":      sim_name,
                        "n_train":        n_train,
                        "tau":            tau,
                        "x":              float(x_flat[i]),
                        "q_hat":          float(q_hat[i]),
                        "q_true":         float(q_true[i]),
                        "abs_err":        float(abs_err[i]),
                        "t_grid_min":     t_grid_min,
                        "t_grid_max":     t_grid_max,
                        "dist_to_max":    float(dist_to_max[i]),
                        "dist_to_min":    float(dist_to_min[i]),
                        # 1 if q_hat is at grid boundary (forced extrapolation)
                        "at_right_bnd":   int(q_hat[i] >= t_grid_max - 1e-10),
                        "at_left_bnd":    int(q_hat[i] <= t_grid_min + 1e-10),
                    })

        if "QR" in methods and HAS_STATSMODELS:
            try:
                predict_qr = _fit_qr_predict(X_train_qr, Y_train_qr, tau, QR_POLY_DEGREE)
                q_qr = predict_qr(X_test)
                summary_rows.append(_row("QR", np.abs(q_qr - q_true)))
            except Exception as e:
                print(f"    QR failed ({sim_name} n={n_train} tau={tau}): {e}")

    return summary_rows, perpoint_rows


# ---------------------------------------------------------------------------
# Macrorep runner
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macro_k: int,
    seed: int,
    config: dict,
    pretrained: dict,
    output_dir: str,
    monotone: bool = False,
    save_perpoint: bool = False,
    methods: tuple = ("CKME", "QR"),
) -> list[dict]:
    """
    Run all (simulator, n_train) combinations for one macrorep.

    Returns list of summary result dicts.
    Per-point CSVs are written to output_dir/perpoint/ when save_perpoint=True.
    """
    rng = np.random.default_rng(seed)
    all_summary = []
    all_perpoint = []

    for sim_name in SIMULATORS:
        params = pretrained.get(sim_name, _DEFAULT_PARAMS)
        for n_train in N_TRAIN_LIST:
            summary, perpoint = run_one_case(
                sim_name, n_train, config, rng, params, monotone, save_perpoint, methods
            )
            for row in summary:
                print(
                    f"  [rep{macro_k}] {sim_name}  n={n_train:5d}"
                    f"  tau={row['tau']:.2f}"
                    f"  mean_err={row['mean_abs_err']:.4f}"
                    f"  sup_err={row['sup_abs_err']:.4f}"
                    f"  q95_err={row['q95_abs_err']:.4f}"
                )
            all_summary.extend([{**r, "macrorep": macro_k} for r in summary])
            all_perpoint.extend([{**r, "macrorep": macro_k} for r in perpoint])

    # Write per-point CSV for this macrorep
    if save_perpoint and all_perpoint:
        pp_dir = os.path.join(output_dir, "perpoint")
        os.makedirs(pp_dir, exist_ok=True)
        pp_path = os.path.join(pp_dir, f"perpoint_rep{macro_k}.csv")
        pd.DataFrame(all_perpoint).to_csv(pp_path, index=False)

    return all_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CKME scaling study (A1 simulators)")
    parser.add_argument("--n_macro",   type=int, default=10,
                        help="Number of macroreps (default: 10)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base random seed; rep k uses base_seed+k (default: 42)")
    parser.add_argument("--n_jobs",    type=int, default=1,
                        help="Parallel workers for macroreps (default: 1 = sequential)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "output_scaling"))
    parser.add_argument("--monotone", action="store_true", default=False,
                        help="Apply isotonic regression to enforce CDF monotonicity")
    parser.add_argument("--save_perpoint", action="store_true", default=False,
                        help="Save per-point (x, q_hat, q_true, boundary flags) CSVs "
                             "to output_dir/perpoint/ for diagnostic analysis")
    parser.add_argument("--methods", type=str, default="CKME,QR",
                        help="Comma-separated list of methods to run: CKME, QR, or CKME,QR (default: CKME,QR)")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Append new results to existing scaling_raw.csv instead of overwriting")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "n_test":      500,
        "n_true_mc":   10000,
        "t_grid_size": 500,
        "r_train":     R_TRAIN,
    }

    pretrained = _load_pretrained()
    print("Params in use:")
    for sim in SIMULATORS:
        p = pretrained.get(sim, _DEFAULT_PARAMS)
        src = "pretrained" if sim in pretrained else "default"
        print(f"  {sim}: ell_x={p.ell_x}, lam={p.lam}, h={p.h}  [{src}]")

    methods = tuple(m.strip() for m in args.methods.split(","))
    print(f"\nN_TRAIN_LIST : {N_TRAIN_LIST}")
    print(f"n_macro      : {args.n_macro}")
    print(f"methods      : {methods}")
    print(f"monotone     : {args.monotone}\n")

    seeds  = [args.base_seed + k for k in range(args.n_macro)]
    worker = partial(
        run_one_macrorep,
        config=config,
        pretrained=pretrained,
        output_dir=args.output_dir,
        monotone=args.monotone,
        save_perpoint=args.save_perpoint,
        methods=methods,
    )

    all_rows: list[dict] = []
    n_jobs = min(args.n_jobs, args.n_macro)

    if n_jobs <= 1:
        for macro_k, seed in enumerate(seeds):
            print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
            all_rows.extend(worker(macro_k, seed))
    else:
        print(f"Running {args.n_macro} macroreps with {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(worker, macro_k, seed): macro_k
                for macro_k, seed in enumerate(seeds)
            }
            for fut in as_completed(futures):
                macro_k = futures[fut]
                rows = fut.result()
                all_rows.extend(rows)
                print(f"  Macrorep {macro_k} done.")

    # Save raw results (optionally append to existing)
    raw_df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.output_dir, "scaling_raw.csv")
    if args.append and os.path.exists(raw_path):
        existing = pd.read_csv(raw_path)
        raw_df = pd.concat([existing, raw_df], ignore_index=True)
        print(f"Appending to existing {raw_path}")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw results  → {raw_path}")

    # Aggregate: mean ± std across macroreps
    agg = (
        raw_df
        .groupby(["simulator", "n_train", "tau", "method"])[
            ["mean_abs_err", "sup_abs_err", "median_abs_err", "q95_abs_err"]
        ]
        .agg(["mean", "std"])
        .round(6)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    summary_path = os.path.join(args.output_dir, "scaling_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"Summary      → {summary_path}")

    # Quick print
    print("\n--- Mean absolute quantile error (mean across macroreps) ---")
    pivot = agg.pivot_table(
        index=["simulator", "tau", "method"],
        columns="n_train",
        values="mean_abs_err_mean",
    ).round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
