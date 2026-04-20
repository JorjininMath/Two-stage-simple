"""
exp2_twostage_error.py

Compare one-stage vs two-stage CKME quantile estimation on exp2.

Stage 1 : n_train uniform sites × r_train reps  →  train CKME
Stage 2 : n_1=500 additional sites selected ∝ S⁰ score (sampling)
          × r_1=r_train reps  →  retrain CKME on combined data

Metrics : mean / sup |q̂_τ(x) - q_true_τ(x)| at τ ∈ {0.05, 0.95}

No CP calibration — pure quantile estimation comparison.

Usage
-----
    python exp_onesided/exp2_twostage_error.py --n_macro 20 --n_jobs 4
    python exp_onesided/exp2_twostage_error.py --n_macro 1 --n_train 500
    python exp_onesided/exp2_twostage_error.py --no_run --plot
    python exp_onesided/exp2_twostage_error.py --no_run --plot --save exp_onesided/output_twostage/fig.png
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

# ── config ─────────────────────────────────────────────────────────────────────
SIM_NAME     = "exp2"
N_TRAIN_LIST = [500, 1000, 2000, 5000]   # Stage 1 site counts
N_1          = 500                        # Stage 2 fixed budget (sites)
R_TRAIN      = 10                         # reps per site (Stage 1 and Stage 2)
N_CAND       = 1000                       # candidate pool for S⁰ selection
TAUS         = [0.05, 0.95]
ALPHA        = 0.1                        # for S⁰ = q_{0.95} - q_{0.05}
N_TEST       = 500
N_TRUE_MC    = 10000
T_GRID_SIZE  = 500
INDICATOR    = "step"                     # consistent with exp2_quantile_error.py default

_PARAM_PATH = Path(__file__).resolve().parent / "pretrained_params.json"

# ── style ──────────────────────────────────────────────────────────────────────
COLORS  = {"stage1_only": "#d6604d", "stage2_sampling": "#2166ac"}
LABELS  = {"stage1_only": "Stage 1 only", "stage2_sampling": "Stage 1 + Stage 2 (adaptive)"}
LS      = {"stage1_only": "--",            "stage2_sampling": "-"}
MARKERS = {"stage1_only": "s",             "stage2_sampling": "o"}


# ── helpers ────────────────────────────────────────────────────────────────────

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
    print(f"Loaded {SIM_NAME} params: ell_x={params.ell_x:.4f}, "
          f"lam={params.lam:.2e}, h={params.h:.4f}")
    return params


def _true_quantiles(simulator, X_test: np.ndarray, rng: np.random.Generator) -> dict:
    """Monte Carlo approximation of true conditional quantiles."""
    y_mc = np.empty((N_TRUE_MC, X_test.shape[0]))
    for b in range(N_TRUE_MC):
        y_mc[b] = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))
    return {tau: np.quantile(y_mc, tau, axis=0) for tau in TAUS}


def _build_ckme(X_sites: np.ndarray, Y_reps: list[np.ndarray],
                kx, indicator, params: Params, t_grid: np.ndarray) -> np.ndarray:
    """
    Build CKME CDF matrix for X_sites with given reps.

    Returns
    -------
    G_bar : ndarray, shape (n_sites, T_GRID_SIZE)
        Per-site averaged indicator matrix (needed to reuse when combining).
    L     : Cholesky factor, shape (n_sites, n_sites)
    """
    n_sites = X_sites.shape[0]
    K = kx(X_sites, X_sites)
    L = build_cholesky_factor(K, n_sites, params.lam)
    G_bar = np.zeros((n_sites, len(t_grid)))
    for rep_y in Y_reps:
        G_bar += indicator.g_matrix(rep_y, t_grid)
    G_bar /= len(Y_reps)
    return G_bar, L


def _predict_cdf(L: np.ndarray, kx, X_sites: np.ndarray, X_query: np.ndarray,
                 G_bar: np.ndarray) -> np.ndarray:
    """Predict CDF at X_query; returns (n_query, M)."""
    K_sq = kx(X_sites, X_query)
    D    = solve_ckme_system(L, K_sq)         # (n_sites, n_query)
    return np.clip(D.T @ G_bar, 0.0, 1.0)    # (n_query, M)


def _s0_scores(F_cand: np.ndarray, t_grid: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    S⁰(x) = q_{1-α/2}(x) - q_{α/2}(x)  from CDF matrix F_cand (n_cand, M).

    Simple argmax-based inversion (step CDF).
    """
    tau_lo = alpha / 2
    tau_hi = 1.0 - alpha / 2

    def _quantile_inv(F: np.ndarray, tau: float) -> np.ndarray:
        mask = F >= tau
        has = mask.any(axis=1)
        idx = mask.argmax(axis=1)
        return np.where(has, t_grid[idx], t_grid[-1])

    return _quantile_inv(F_cand, tau_hi) - _quantile_inv(F_cand, tau_lo)


def _select_stage2_sites(X_cand: np.ndarray, scores: np.ndarray,
                         n_1: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_1 sites from X_cand proportional to S⁰ scores (without replacement)."""
    weights = np.maximum(scores, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(weights))
    probs = weights / weights.sum()
    idx = rng.choice(len(X_cand), size=n_1, replace=False, p=probs)
    return X_cand[idx]


def _invert_quantile(F: np.ndarray, t_grid: np.ndarray, tau: float) -> np.ndarray:
    """CDF inversion: first t where F >= tau (per row). Clips to t_grid[-1]."""
    mask = F >= tau
    has  = mask.any(axis=1)
    idx  = mask.argmax(axis=1)
    return np.where(has, t_grid[idx], t_grid[-1])


# ── core: one (n_train, macrorep) ─────────────────────────────────────────────

def run_one_case(n_train: int, rng: np.random.Generator,
                 params: Params) -> list[dict]:
    """
    Run one (n_train, macrorep) case.

    Returns list of dicts with keys:
      n_train, n_1, tau, method, mean_abs_err, sup_abs_err,
      median_abs_err, q95_abs_err
    """
    sim_cfg   = get_experiment_config(SIM_NAME)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    kx        = make_x_rbf_kernel(params.ell_x)
    indicator = make_indicator(INDICATOR, params.h)

    # ── Stage 1: uniform sites ────────────────────────────────────────────────
    X_s1 = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_s1 = [
        simulator(X_s1.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(R_TRAIN)
    ]

    # Test grid (deterministic)
    X_test = np.linspace(x_lo, x_hi, N_TEST).reshape(-1, 1)

    # True quantiles
    q_true_dict = _true_quantiles(simulator, X_test, rng)

    # t_grid from Stage 1 data
    Y_all_s1 = np.concatenate(Y_s1)
    Y_lo = np.percentile(Y_all_s1, 0.5)
    Y_hi = np.percentile(Y_all_s1, 99.5)
    margin = 0.10 * (Y_hi - Y_lo)
    t_grid_s1 = np.linspace(Y_lo - margin, Y_hi + margin, T_GRID_SIZE)

    # Build Stage 1 CKME
    G_bar_s1, L_s1 = _build_ckme(X_s1, Y_s1, kx, indicator, params, t_grid_s1)
    F_s1 = _predict_cdf(L_s1, kx, X_s1, X_test, G_bar_s1)   # (N_TEST, M)

    # ── S⁰ scores on candidate pool ───────────────────────────────────────────
    X_cand = rng.uniform(x_lo, x_hi, size=(N_CAND, 1))
    F_cand = _predict_cdf(L_s1, kx, X_s1, X_cand, G_bar_s1)  # (N_CAND, M)
    scores = _s0_scores(F_cand, t_grid_s1)                    # (N_CAND,)

    # ── Stage 2: adaptive site selection & data collection ────────────────────
    X_s2 = _select_stage2_sites(X_cand, scores, N_1, rng)     # (N_1, 1)
    Y_s2 = [
        simulator(X_s2.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(R_TRAIN)
    ]

    # t_grid from combined data (Stage 1 + Stage 2)
    Y_all_s2   = np.concatenate(Y_s2)
    Y_all_comb = np.concatenate([Y_all_s1, Y_all_s2])
    Y_lo_c = np.percentile(Y_all_comb, 0.5)
    Y_hi_c = np.percentile(Y_all_comb, 99.5)
    margin_c = 0.10 * (Y_hi_c - Y_lo_c)
    t_grid_c = np.linspace(Y_lo_c - margin_c, Y_hi_c + margin_c, T_GRID_SIZE)

    # Recompute G_bar for Stage 1 sites on the new (wider) t_grid
    G_bar_s1_c = np.zeros((n_train, T_GRID_SIZE))
    for rep_y in Y_s1:
        G_bar_s1_c += indicator.g_matrix(rep_y, t_grid_c)
    G_bar_s1_c /= R_TRAIN

    # Build G_bar for Stage 2 sites
    G_bar_s2, _ = _build_ckme(X_s2, Y_s2, kx, indicator, params, t_grid_c)

    # Combined CKME: all sites together
    X_comb  = np.vstack([X_s1, X_s2])                         # (n_train+N_1, 1)
    G_comb  = np.vstack([G_bar_s1_c, G_bar_s2])               # (n_train+N_1, M)
    n_comb  = X_comb.shape[0]
    K_comb  = kx(X_comb, X_comb)
    L_comb  = build_cholesky_factor(K_comb, n_comb, params.lam)
    F_comb  = _predict_cdf(L_comb, kx, X_comb, X_test, G_comb)  # (N_TEST, M)

    # ── compute errors ────────────────────────────────────────────────────────
    rows = []

    def _row(method, ae, n1_val):
        return {
            "n_train":        n_train,
            "n_1":            n1_val,
            "tau":            tau,
            "method":         method,
            "mean_abs_err":   float(np.mean(ae)),
            "sup_abs_err":    float(np.max(ae)),
            "median_abs_err": float(np.median(ae)),
            "q95_abs_err":    float(np.quantile(ae, 0.95)),
        }

    for tau in TAUS:
        q_true = q_true_dict[tau]

        # Stage 1 only
        q_s1  = _invert_quantile(F_s1, t_grid_s1, tau)
        rows.append(_row("stage1_only",      np.abs(q_s1  - q_true), 0))

        # Stage 1 + Stage 2 combined
        q_c   = _invert_quantile(F_comb, t_grid_c, tau)
        rows.append(_row("stage2_sampling",  np.abs(q_c   - q_true), N_1))

    return rows


# ── macrorep runner ────────────────────────────────────────────────────────────

def run_one_macrorep(macro_k: int, seed: int, params: Params,
                     n_train_list: list[int]) -> list[dict]:
    rng = np.random.default_rng(seed)
    all_rows = []
    for n_train in n_train_list:
        rows = run_one_case(n_train, rng, params)
        for r in rows:
            print(
                f"  [rep{macro_k}] n_train={n_train:5d}  n_1={r['n_1']:3d}"
                f"  tau={r['tau']:.2f}  {r['method']:18s}"
                f"  mean={r['mean_abs_err']:.4f}  sup={r['sup_abs_err']:.4f}"
            )
        all_rows.extend([{**r, "macrorep": macro_k} for r in rows])
    return all_rows


# ── plotting ───────────────────────────────────────────────────────────────────

PANEL_SPECS = [
    ("mean_abs_err", "Mean |q̂ − q_true|", "Mean absolute error  (τ=0.05)", 0.05),
    ("mean_abs_err", "Mean |q̂ − q_true|", "Mean absolute error  (τ=0.95)", 0.95),
    ("sup_abs_err",  "Sup |q̂ − q_true|",  "Sup absolute error   (τ=0.05)", 0.05),
    ("sup_abs_err",  "Sup |q̂ − q_true|",  "Sup absolute error   (τ=0.95)", 0.95),
]


def _draw_line(ax, df, col, tau, method, n_vals):
    sub = df[(df["tau"] == tau) & (df["method"] == method)]
    means, stds = [], []
    for n in n_vals:
        vals = sub[sub["n_train"] == n][col].values
        means.append(np.mean(vals) if len(vals) else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    means, stds = np.array(means), np.array(stds)
    c = COLORS.get(method, "gray")
    ax.plot(n_vals, means, color=c, lw=2.0, ls=LS.get(method, "-"),
            marker=MARKERS.get(method, "o"), markersize=5,
            label=LABELS.get(method, method), zorder=3)
    ax.fill_between(n_vals, means - stds, means + stds, color=c, alpha=0.15)


def _draw_boxplot(ax, df, col, tau, method, n_vals, offset):
    sub  = df[(df["tau"] == tau) & (df["method"] == method)]
    data = [sub[sub["n_train"] == n][col].values for n in n_vals]
    c = COLORS.get(method, "gray")
    positions = [i + offset for i in range(len(n_vals))]
    ax.boxplot(
        data,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor=c, alpha=0.35),
        whiskerprops=dict(color=c, lw=1.2),
        capprops=dict(color=c, lw=1.2),
        flierprops=dict(marker="x", color=c, markersize=4, alpha=0.6),
    )
    ax.plot([], [], color=c, lw=2.5, label=LABELS.get(method, method))


def make_figure(raw_path: str, plot_type: str = "line") -> plt.Figure:
    df = pd.read_csv(raw_path)
    methods  = ["stage1_only", "stage2_sampling"]
    n_vals   = sorted(df["n_train"].unique())
    n_macro  = df["macrorep"].nunique() if "macrorep" in df.columns else "?"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"exp2 — Stage1 vs Stage1+Stage2 (adaptive, n_1={N_1})\n"
        f"CKME quantile estimation error  "
        f"({n_macro} macroreps, R_train={R_TRAIN})",
        fontsize=12, fontweight="bold",
    )

    offsets = {"stage1_only": -0.18, "stage2_sampling": 0.18}

    for ax, (col, ylabel, title, tau) in zip(axes.ravel(), PANEL_SPECS):
        for method in methods:
            sub = df[(df["tau"] == tau) & (df["method"] == method)]
            if sub.empty:
                continue
            if plot_type == "boxplot":
                _draw_boxplot(ax, df, col, tau, method, n_vals, offsets[method])
                ax.set_xticks(range(len(n_vals)))
                ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
            else:
                _draw_line(ax, df, col, tau, method, n_vals)
                ax.set_xscale("log")
                ax.set_xticks(n_vals)
                ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)

        ax.set_xlabel("n_train  (Stage 1 sites)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    default_dir = os.path.join(os.path.dirname(__file__), "output_twostage")
    parser = argparse.ArgumentParser(
        description="exp2 two-stage quantile error: Stage1 vs Stage1+Stage2"
    )
    parser.add_argument("--n_macro",    type=int, default=10)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_jobs",     type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=default_dir)
    parser.add_argument("--n_train",    type=int, default=None,
                        help="Single n_train value to run (overrides N_TRAIN_LIST)")
    parser.add_argument("--no_run",     action="store_true",
                        help="Skip running; load existing raw CSV to plot")
    parser.add_argument("--plot",       action="store_true",
                        help="Plot after running (or with --no_run)")
    parser.add_argument("--plot_type",  type=str, default="line",
                        choices=["line", "boxplot"])
    parser.add_argument("--save",       type=str, default=None,
                        help="Save figure to this path (PNG)")
    parser.add_argument("--append",     action="store_true",
                        help="Append to existing CSV instead of overwriting")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "twostage_raw.csv")

    n_train_list = [args.n_train] if args.n_train is not None else N_TRAIN_LIST

    # ── run ───────────────────────────────────────────────────────────────────
    if not args.no_run:
        params = _load_params()
        print(f"\nN_TRAIN_LIST : {n_train_list}")
        print(f"N_1 (Stage 2 budget) : {N_1}  sites × {R_TRAIN} reps")
        print(f"N_CAND       : {N_CAND}")
        print(f"n_macro      : {args.n_macro}")
        print(f"INDICATOR    : {INDICATOR}\n")

        seeds  = [args.base_seed + k for k in range(args.n_macro)]
        worker = partial(run_one_macrorep, params=params, n_train_list=n_train_list)
        all_rows: list[dict] = []

        if min(args.n_jobs, args.n_macro) <= 1:
            for macro_k, seed in enumerate(seeds):
                print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
                all_rows.extend(worker(macro_k, seed))
        else:
            n_jobs = min(args.n_jobs, args.n_macro)
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

        raw_df = pd.DataFrame(all_rows)
        if args.append and os.path.exists(raw_path):
            existing = pd.read_csv(raw_path)
            raw_df = pd.concat([existing, raw_df], ignore_index=True)
        raw_df.to_csv(raw_path, index=False)
        print(f"\nSaved -> {raw_path}  ({len(raw_df)} rows)")

        # Quick summary table
        agg = (
            raw_df.groupby(["n_train", "n_1", "tau", "method"])[
                ["mean_abs_err", "sup_abs_err"]
            ]
            .agg(["mean", "std"])
            .round(4)
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        print("\n--- Quantile error summary (mean ± std across macroreps) ---")
        print(agg.to_string())

    # ── plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.no_run:
        if not os.path.exists(raw_path):
            print(f"No raw CSV found at {raw_path}. Run without --no_run first.")
            return
        fig = make_figure(raw_path, plot_type=args.plot_type)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Figure saved -> {args.save}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
