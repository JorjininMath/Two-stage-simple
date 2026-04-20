"""
exp2_sup_vs_tau.py

For each tau in a user-specified list, plot sup |q_hat - q_true| vs n_train
as boxplots (across macroreps).  CKME only.

Layout: one subplot per tau value.

Output:
  output_sup_tau/sup_vs_tau_raw_{indicator}.csv
  output_sup_tau/sup_vs_tau_{indicator}.png

Usage:
    python exp_onesided/exp2_sup_vs_tau.py
    python exp_onesided/exp2_sup_vs_tau.py --taus 0.5 0.7 0.9 0.95
    python exp_onesided/exp2_sup_vs_tau.py --n_macro 20 --n_jobs 4
    python exp_onesided/exp2_sup_vs_tau.py --no_run --plot --save my_fig.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.coefficients import build_cholesky_factor, solve_ckme_system
from CKME.indicators import make_indicator
from CKME.kernels import make_x_rbf_kernel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp2 import exp2_true_function, exp2_noise_variance_function

# ── fixed config ──────────────────────────────────────────────────────────────
SIM_NAME     = "exp2"
N_TRAIN_LIST = [500, 1000, 2000, 5000]
R_TRAIN      = 10
N_TEST       = 500
N_TRUE_MC    = 10000
T_GRID_SIZE  = 500

_PARAM_PATH = Path(__file__).resolve().parent / "pretrained_params.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_oracle_t_grid() -> np.ndarray:
    from scipy import stats
    x_dense = np.linspace(0, 2 * np.pi, 10_000)
    mu      = exp2_true_function(x_dense)
    sigma   = np.sqrt(exp2_noise_variance_function(x_dense))
    Y_lo    = float(np.min(mu + sigma * stats.norm.ppf(0.005)))
    Y_hi    = float(np.max(mu + sigma * stats.norm.ppf(0.995)))
    margin  = 0.10 * (Y_hi - Y_lo)
    t_grid  = np.linspace(Y_lo - margin, Y_hi + margin, T_GRID_SIZE)
    print(f"Oracle t_grid: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]  (size={T_GRID_SIZE})")
    return t_grid


def _load_params() -> Params:
    if not _PARAM_PATH.exists():
        raise FileNotFoundError(
            f"pretrained_params.json not found at {_PARAM_PATH}.\n"
            "Run pretrain_params.py --sims exp2 first."
        )
    with open(_PARAM_PATH) as f:
        raw = json.load(f)
    if SIM_NAME not in raw:
        raise KeyError(f"Key '{SIM_NAME}' not found in pretrained_params.json.")
    p = raw[SIM_NAME]
    params = Params(ell_x=p["ell_x"], lam=p["lam"], h=p["h"])
    print(f"Loaded params: ell_x={params.ell_x:.4f}, lam={params.lam:.2e}, h={params.h:.4f}")
    return params


# ── core: one macrorep ────────────────────────────────────────────────────────

def run_one_macrorep(
    macro_k: int,
    seed: int,
    params: Params,
    t_grid: np.ndarray,
    taus: list[float],
    indicator_type: str = "step",
) -> list[dict]:
    rng = np.random.default_rng(seed)

    sim_cfg   = get_experiment_config(SIM_NAME)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    X_test = np.linspace(x_lo, x_hi, N_TEST).reshape(-1, 1)

    # True quantiles via Monte Carlo — computed once, shared across n_train
    y_mc = np.empty((N_TRUE_MC, N_TEST))
    for b in range(N_TRUE_MC):
        y_mc[b] = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))
    q_true_dict = {tau: np.quantile(y_mc, tau, axis=0) for tau in taus}

    rows = []
    for n_train in N_TRAIN_LIST:
        # Training data
        X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
        Y_reps  = [
            simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
            for _ in range(R_TRAIN)
        ]

        # CKME: fit once per n_train
        kx        = make_x_rbf_kernel(params.ell_x)
        indicator = make_indicator(indicator_type, params.h)
        K_sites   = kx(X_sites, X_sites)
        L_sites   = build_cholesky_factor(K_sites, n_train, params.lam)
        K_sq      = kx(X_sites, X_test)
        D_bar     = solve_ckme_system(L_sites, K_sq)
        G_bar     = np.zeros((n_train, T_GRID_SIZE))
        for rep_y in Y_reps:
            G_bar += indicator.g_matrix(rep_y, t_grid)
        G_bar /= R_TRAIN
        F_hat = np.clip(D_bar.T @ G_bar, 0.0, 1.0)  # (N_TEST, T_GRID_SIZE)

        # Invert CDF for each tau
        for tau in taus:
            q_true = q_true_dict[tau]
            mask   = F_hat >= tau
            q_ckme = np.where(mask.any(axis=1), t_grid[mask.argmax(axis=1)], t_grid[-1])
            sup_err = float(np.max(np.abs(q_ckme - q_true)))
            rows.append({
                "macrorep":   macro_k,
                "n_train":    n_train,
                "tau":        tau,
                "sup_abs_err": sup_err,
            })
            print(
                f"  [rep{macro_k}]  n={n_train:5d}  tau={tau:.2f}"
                f"  sup={sup_err:.4f}"
            )

    return rows


# ── plot ──────────────────────────────────────────────────────────────────────

def make_figure(
    df: pd.DataFrame,
    taus: list[float],
    indicator_type: str = "",
) -> plt.Figure:
    n_taus = len(taus)
    ncols  = min(n_taus, 3)
    nrows  = (n_taus + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharey=False, squeeze=False)

    n_vals = sorted(df["n_train"].unique())
    positions = np.arange(len(n_vals))

    for idx, tau in enumerate(taus):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df[df["tau"] == tau]

        data = [sub[sub["n_train"] == n]["sup_abs_err"].to_numpy() for n in n_vals]
        ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor="#2166ac", alpha=0.75),
            whiskerprops=dict(color="#2166ac"),
            capprops=dict(color="#2166ac"),
            flierprops=dict(marker=".", markersize=4, color="#2166ac", alpha=0.5),
        )

        ax.set_title(f"τ = {tau:.2f}", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
        ax.set_xlabel("n_train")
        ax.set_ylabel("Sup |q̂ − q_true|")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(n_taus, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    ind_label = f"  [indicator: {indicator_type}]" if indicator_type else ""
    fig.suptitle(
        f"CKME  —  Sup quantile error vs n_train  (exp2){ind_label}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    default_dir = os.path.join(os.path.dirname(__file__), "output_sup_tau")
    parser = argparse.ArgumentParser(
        description="Sup quantile error vs n_train, one subplot per tau (CKME only)"
    )
    parser.add_argument(
        "--taus", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        help="List of tau values (default: 0.5 0.6 0.7 0.8 0.9 0.95)",
    )
    parser.add_argument("--n_macro",    type=int, default=10)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_jobs",     type=int, default=1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--output_dir", type=str, default=default_dir)
    parser.add_argument("--no_run",     action="store_true",
                        help="Skip computation; load existing CSV and plot")
    parser.add_argument("--plot",       action="store_true",
                        help="Plot after running (or with --no_run)")
    parser.add_argument("--indicator",  type=str, default="step",
                        choices=["logistic", "step", "gaussian_cdf"])
    parser.add_argument("--save",       type=str, default=None,
                        help="Save figure to this path")
    return parser.parse_args()


def main():
    args = parse_args()
    taus = sorted(args.taus)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"sup_vs_tau_raw_{args.indicator}.csv")

    # ── run ───────────────────────────────────────────────────────────────────
    if not args.no_run:
        params = _load_params()
        t_grid = _build_oracle_t_grid()

        print(f"\ntaus        : {taus}")
        print(f"N_TRAIN_LIST: {N_TRAIN_LIST}")
        print(f"n_macro     : {args.n_macro}")
        print(f"indicator   : {args.indicator}\n")

        seeds  = [args.base_seed + k for k in range(args.n_macro)]
        worker = partial(
            run_one_macrorep,
            params=params, t_grid=t_grid, taus=taus,
            indicator_type=args.indicator,
        )
        all_rows: list[dict] = []

        n_jobs = args.n_jobs if args.n_jobs != -1 else os.cpu_count()
        if min(n_jobs, args.n_macro) <= 1:
            for macro_k, seed in enumerate(seeds):
                print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
                all_rows.extend(worker(macro_k, seed))
        else:
            n_jobs = min(n_jobs, args.n_macro)
            print(f"Running {args.n_macro} macroreps with {n_jobs} workers...")
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(worker, macro_k, seed): macro_k
                    for macro_k, seed in enumerate(seeds)
                }
                for fut in as_completed(futures):
                    macro_k = futures[fut]
                    all_rows.extend(fut.result())
                    print(f"  Macrorep {macro_k} done.")

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved -> {csv_path}  ({len(df)} rows)")

        agg = (
            df.groupby(["tau", "n_train"])["sup_abs_err"]
            .agg(["mean", "std"]).round(4)
        )
        print("\n--- Sup abs error (mean ± std across macroreps) ---")
        print(agg.to_string())

    # ── plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.no_run:
        if not os.path.exists(csv_path):
            print(f"No CSV found at {csv_path}. Run without --no_run first.")
            return
        df = pd.read_csv(csv_path)
        taus_plot = sorted(df["tau"].unique().tolist())
        fig = make_figure(df, taus_plot, indicator_type=args.indicator)

        save_path = args.save or os.path.join(args.output_dir, f"sup_vs_tau_{args.indicator}.png")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
