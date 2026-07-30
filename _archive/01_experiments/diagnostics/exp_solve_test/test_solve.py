"""
test_solve.py

Compare four quantile-extraction methods on the exp2 simulator:

    logistic_grid  — logistic indicator, predict_quantile(monotone=True)
    logistic_solve — logistic indicator, predict_quantile_solve (brentq)
    step_grid      — step indicator,     predict_quantile(monotone=True)
    step_solve     — step indicator,     predict_quantile_solve (sort+scan)

Metrics : mean and sup |q̂_τ(x) − q_true_τ(x)| over a uniform test grid
Taus    : 0.05 and 0.95
n_train : 500, 1000, 2000, 5000  (r_train=10 reps per site)
n_macro : 20 macroreps (default)

Outputs (in --output_dir, default exp_solve_test/output/):
    solve_raw.csv          — one row per (macrorep, indicator, method, n_train, tau)
    exp2_box_tau_95.png    — boxplots of mean/sup error at tau=0.95 (all 4 methods)
    exp2_box_tau_05.png    — same for tau=0.05

Usage:
    python exp_solve_test/test_solve.py
    python exp_solve_test/test_solve.py --n_macro 20 --n_jobs 4
    python exp_solve_test/test_solve.py --plot_only   # skip run, only plot from CSV
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIMULATOR     = "exp2"
N_TRAIN_LIST  = [500, 1000, 2000, 5000]
R_TRAIN       = 10
TAUS          = [0.05, 0.95]

# exp2 CV-tuned params (from exp_onesided/pretrained_params.json)
_EXP2_PARAMS  = Params(ell_x=0.23853323044733007, lam=0.0002782559402207126, h=0.079)

_RUN_CONFIG = {
    "n_test":      500,
    "n_true_mc":   10_000,
    "t_grid_size": 500,
    "r_train":     R_TRAIN,
}

# plot aesthetics — pairs: (fill color, label)
_METHOD_STYLE = {
    "logistic_grid":  ("#2196F3", "logistic · grid+isotonic"),
    "logistic_solve": ("#64B5F6", "logistic · solve (brentq)"),
    "step_grid":      ("#F44336", "step · grid+isotonic"),
    "step_solve":     ("#EF9A9A", "step · solve (sort+scan)"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_quantiles(
    simulator,
    X_test: np.ndarray,
    taus: list[float],
    n_mc: int,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Monte Carlo approximation of true conditional quantiles."""
    y_mc = np.empty((n_mc, X_test.shape[0]))
    x_flat = X_test.ravel()
    for b in range(n_mc):
        y_mc[b] = simulator(x_flat, random_state=int(rng.integers(0, 2**31)))
    return {tau: np.quantile(y_mc, tau, axis=0) for tau in taus}


# ---------------------------------------------------------------------------
# Core: one (n_train) case
# ---------------------------------------------------------------------------

def run_one_case(
    n_train: int,
    config: dict,
    rng: np.random.Generator,
    params: Params,
) -> list[dict]:
    """
    Fit and evaluate all four methods for one n_train value.

    Training data layout for CKMEModel (r > 1 distinct-sites mode):
      X_flat[i*r : (i+1)*r] are all replicates of site i
      Y_flat[i*r : (i+1)*r] are the corresponding Y values
    """
    sim_cfg    = get_experiment_config(SIMULATOR)
    simulator  = sim_cfg["simulator"]
    x_lo       = float(sim_cfg["bounds"][0].item() if hasattr(sim_cfg["bounds"][0], "item") else sim_cfg["bounds"][0])
    x_hi       = float(sim_cfg["bounds"][1].item() if hasattr(sim_cfg["bounds"][1], "item") else sim_cfg["bounds"][1])

    r_train     = config["r_train"]
    n_test      = config["n_test"]
    n_true_mc   = config["n_true_mc"]
    t_grid_size = config["t_grid_size"]

    # --- training data ---
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_reps  = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(r_train)
    ]
    # consecutive layout: [y_{0,0}, y_{0,1}, ..., y_{0,r-1}, y_{1,0}, ...]
    X_flat = np.repeat(X_sites, r_train, axis=0)        # (n*r, 1)
    Y_flat = np.stack(Y_reps, axis=1).ravel()           # (n*r,)

    # --- t_grid: percentile-based ---
    Y_all    = np.concatenate(Y_reps)
    Y_lo     = np.percentile(Y_all, 0.5)
    Y_hi     = np.percentile(Y_all, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid   = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    # --- test grid and true quantiles ---
    X_test       = np.linspace(x_lo, x_hi, n_test).reshape(-1, 1)
    q_true_dict  = _true_quantiles(simulator, X_test, TAUS, n_true_mc, rng)

    rows = []
    for ind_type in ["logistic", "step"]:
        model = CKMEModel(indicator_type=ind_type)
        model.fit(X_flat, Y_flat, params=params, r=r_train)

        for tau in TAUS:
            q_true = q_true_dict[tau]

            # --- grid + isotonic (monotone=True) ---
            q_grid  = model.predict_quantile(X_test, tau, t_grid, monotone=True)
            ae_grid = np.abs(q_grid - q_true)
            rows.append({
                "indicator":    ind_type,
                "method":       "grid",
                "n_train":      n_train,
                "tau":          tau,
                "mean_abs_err": float(np.mean(ae_grid)),
                "sup_abs_err":  float(np.max(ae_grid)),
            })

            # --- direct solve (brentq / sort+scan) ---
            q_solve  = model.predict_quantile_solve(X_test, tau)
            ae_solve = np.abs(q_solve - q_true)
            rows.append({
                "indicator":    ind_type,
                "method":       "solve",
                "n_train":      n_train,
                "tau":          tau,
                "mean_abs_err": float(np.mean(ae_solve)),
                "sup_abs_err":  float(np.max(ae_solve)),
            })

    return rows


# ---------------------------------------------------------------------------
# Macrorep runner
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macro_k: int,
    seed: int,
    config: dict,
    params: Params,
) -> list[dict]:
    rng      = np.random.default_rng(seed)
    all_rows = []
    for n_train in N_TRAIN_LIST:
        case_rows = run_one_case(n_train, config, rng, params)
        for row in case_rows:
            ind, meth, tau = row["indicator"], row["method"], row["tau"]
            key = f"{ind}_{meth}"
            print(
                f"  [rep{macro_k:2d}] n={n_train:5d}  tau={tau:.2f}"
                f"  {key:<20s}"
                f"  mean={row['mean_abs_err']:.4f}"
                f"  sup={row['sup_abs_err']:.4f}"
            )
        all_rows.extend([{**r, "macrorep": macro_k} for r in case_rows])
    return all_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_boxplot_group(ax, raw: pd.DataFrame, col: str, tau: float, n_vals: list):
    """
    Draw 4 boxplots per n_train (one per indicator+method combination),
    with slight horizontal offsets so they don't overlap.
    """
    method_keys = list(_METHOD_STYLE.keys())
    n_methods   = len(method_keys)
    # offsets within each n_train group: spread from -0.4 to +0.4
    offsets = np.linspace(-0.38, 0.38, n_methods)

    for m_idx, key in enumerate(method_keys):
        ind_type, meth = key.rsplit("_", 1)
        color, label   = _METHOD_STYLE[key]

        sub = raw[
            (raw["tau"]       == tau)
            & (raw["indicator"] == ind_type)
            & (raw["method"]    == meth)
        ]
        if sub.empty:
            continue

        data      = [sub[sub["n_train"] == n][col].values for n in n_vals]
        positions = [i + offsets[m_idx] for i in range(len(n_vals))]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", lw=1.5),
            boxprops=dict(facecolor=color, alpha=0.5),
            whiskerprops=dict(color=color, lw=1.2),
            capprops=dict(color=color, lw=1.2),
            flierprops=dict(marker="x", color=color, markersize=3, alpha=0.6),
        )
        # invisible line for legend
        ax.plot([], [], color=color, lw=2.5, label=label)

    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
    ax.set_xlabel("n_train", fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper right")


def plot_tau(raw: pd.DataFrame, tau: float, save_path: str | None = None):
    """
    2-panel figure: mean abs error (left) and sup abs error (right)
    for the given tau, boxplots over macroreps.
    """
    n_vals = sorted(raw["n_train"].unique())
    tau_str = f"{int(tau * 100):02d}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"exp2 — quantile estimation error  (τ = {tau:.2f})  |  logistic vs step  ·  grid+isotonic vs solve",
        fontsize=11, fontweight="bold",
    )

    _draw_boxplot_group(axes[0], raw, "mean_abs_err", tau, n_vals)
    axes[0].set_ylabel("mean |q̂ − q_true|", fontsize=9)
    axes[0].set_title(f"Mean absolute error  (τ = {tau:.2f})", fontsize=10)

    _draw_boxplot_group(axes[1], raw, "sup_abs_err", tau, n_vals)
    axes[1].set_ylabel("sup |q̂ − q_true|", fontsize=9)
    axes[1].set_title(f"Sup absolute error  (τ = {tau:.2f})", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare quantile methods: grid+isotonic vs solve, logistic vs step"
    )
    parser.add_argument("--n_macro",    type=int, default=20,
                        help="Number of macroreps (default: 20)")
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_jobs",     type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "output"))
    parser.add_argument("--plot_only",  action="store_true", default=False,
                        help="Skip run, load existing solve_raw.csv and plot only")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "solve_raw.csv")

    # ---- run ----
    if not args.plot_only:
        seeds  = [args.base_seed + k for k in range(args.n_macro)]
        worker = partial(
            run_one_macrorep,
            config=_RUN_CONFIG,
            params=_EXP2_PARAMS,
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

        raw_df = pd.DataFrame(all_rows)
        raw_df.to_csv(raw_path, index=False)
        print(f"\nRaw results → {raw_path}")
    else:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"--plot_only: {raw_path} not found. Run without --plot_only first.")
        raw_df = pd.read_csv(raw_path)
        print(f"Loaded {raw_path}  ({len(raw_df)} rows)")

    # ---- plot ----
    for tau in TAUS:
        tau_str  = f"{int(tau * 100):02d}"
        out_path = os.path.join(args.output_dir, f"exp2_box_tau_{tau_str}.png")
        plot_tau(raw_df, tau, save_path=out_path)

    # ---- quick summary table ----
    print("\n--- Sup absolute error: mean across macroreps ---")
    summary = (
        raw_df
        .assign(method_key=lambda df: df["indicator"] + "_" + df["method"])
        .groupby(["method_key", "tau", "n_train"])["sup_abs_err"]
        .mean()
        .unstack("n_train")
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
