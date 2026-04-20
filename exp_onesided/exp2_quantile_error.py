"""
exp2_quantile_error.py

Run quantile estimation error experiment on exp2 (CKME vs QR),
then optionally plot boxplots across macroreps.

Metrics: mean |q_hat - q_true| and sup |q_hat - q_true| over test grid,
         at tau = 0.05 and 0.95.

Params loaded from pretrained_params.json (key "exp2") -- no CV tuning needed.

Outputs (in --output_dir):
  exp2_raw.csv      -- one row per (macrorep, n_train, tau, method)

Usage:
    # Run experiment only
    python exp_onesided/exp2_quantile_error.py --n_macro 10

    # Run then immediately plot boxplot
    python exp_onesided/exp2_quantile_error.py --n_macro 10 --plot

    # Plot from existing CSV (no re-run)
    python exp_onesided/exp2_quantile_error.py --no_run --plot

    # Save figure
    python exp_onesided/exp2_quantile_error.py --no_run --plot --save exp_onesided/output_exp2/exp2_box.png

    # Parallel macroreps
    python exp_onesided/exp2_quantile_error.py --n_macro 20 --n_jobs 4 --plot
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

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("Warning: statsmodels not found; QR will be skipped.", file=sys.stderr)

# ── config ────────────────────────────────────────────────────────────────────
SIM_NAME    = "exp2"
N_TRAIN_LIST = [500, 1000, 2000, 5000]
R_TRAIN     = 10
TAUS        = [0.05, 0.95]   # default; overridden by --tau_hi at runtime
N_TEST      = 500
N_TRUE_MC   = 10000
T_GRID_SIZE = 500
QR_DEGREE   = 1      # linear QR, matching R dcp.qr: rq.fit.br(cbind(1,X),Y,tau)

_PARAM_PATH = Path(__file__).resolve().parent / "pretrained_params.json"

# ── style ─────────────────────────────────────────────────────────────────────
COLORS  = {"CKME": "#2166ac", "QR": "#d6604d"}
LS      = {"CKME": "-",       "QR": "--"}
MARKERS = {"CKME": "o",       "QR": "s"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_oracle_t_grid() -> np.ndarray:
    """Fixed t_grid from oracle Gaussian quantiles, shared across all macroreps."""
    from scipy import stats
    x_dense = np.linspace(0, 2 * np.pi, 10_000)
    mu    = exp2_true_function(x_dense)
    sigma = np.sqrt(exp2_noise_variance_function(x_dense))
    Y_lo  = float(np.min(mu + sigma * stats.norm.ppf(0.005)))
    Y_hi  = float(np.max(mu + sigma * stats.norm.ppf(0.995)))
    margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - margin, Y_hi + margin, T_GRID_SIZE)
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
    print(f"Loaded {SIM_NAME} params: ell_x={params.ell_x:.4f}, lam={params.lam:.2e}, h={params.h:.4f}")
    return params


def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = x.ravel()
    return np.column_stack([x**k for k in range(1, degree + 1)])


def _fit_qr(X_tr: np.ndarray, Y_tr: np.ndarray, tau: float):
    X_feat = sm.add_constant(_poly_features(X_tr, QR_DEGREE))
    res = sm.QuantReg(Y_tr, X_feat).fit(q=tau, max_iter=5000, disp=False)
    def predict(X_q):
        Xq = sm.add_constant(_poly_features(X_q, QR_DEGREE), has_constant="add")
        return res.predict(Xq)
    return predict


def _true_quantiles(simulator, X_test: np.ndarray, rng: np.random.Generator) -> dict:
    y_mc = np.empty((N_TRUE_MC, X_test.shape[0]))
    for b in range(N_TRUE_MC):
        y_mc[b] = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))
    return {tau: np.quantile(y_mc, tau, axis=0) for tau in TAUS}


# ── core: one (n_train, macrorep) ─────────────────────────────────────────────

def run_one_case(n_train: int, rng: np.random.Generator, params: Params,
                 t_grid: np.ndarray,
                 indicator_type: str = "step") -> list[dict]:
    sim_cfg   = get_experiment_config(SIM_NAME)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    # Training data
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_reps = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(R_TRAIN)
    ]

    # Test grid
    X_test = np.linspace(x_lo, x_hi, N_TEST).reshape(-1, 1)

    # True quantiles (Monte Carlo)
    q_true_dict = _true_quantiles(simulator, X_test, rng)

    Y_all = np.concatenate(Y_reps)

    # ── CKME ─────────────────────────────────────────────────────────────────
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
    F_hat = np.clip(D_bar.T @ G_bar, 0.0, 1.0)   # (N_TEST, T_GRID_SIZE)

    # ── QR ───────────────────────────────────────────────────────────────────
    X_tr_qr = np.repeat(X_sites, R_TRAIN, axis=0)
    Y_tr_qr = Y_all

    rows = []
    for tau in TAUS:
        q_true = q_true_dict[tau]

        # CKME quantile: first t where F_hat >= tau
        mask      = F_hat >= tau
        has_valid = mask.any(axis=1)
        q_ckme    = np.where(has_valid, t_grid[mask.argmax(axis=1)], t_grid[-1])
        ae_ckme   = np.abs(q_ckme - q_true)
        rows.append({
            "n_train":        n_train,
            "tau":            tau,
            "method":         "CKME",
            "mean_abs_err":   float(np.mean(ae_ckme)),
            "sup_abs_err":    float(np.max(ae_ckme)),
            "median_abs_err": float(np.median(ae_ckme)),
            "q95_abs_err":    float(np.quantile(ae_ckme, 0.95)),
        })

        # QR
        if HAS_SM:
            try:
                pred_qr = _fit_qr(X_tr_qr, Y_tr_qr, tau)
                q_qr    = pred_qr(X_test)
                ae_qr   = np.abs(q_qr - q_true)
                rows.append({
                    "n_train":        n_train,
                    "tau":            tau,
                    "method":         "QR",
                    "mean_abs_err":   float(np.mean(ae_qr)),
                    "sup_abs_err":    float(np.max(ae_qr)),
                    "median_abs_err": float(np.median(ae_qr)),
                    "q95_abs_err":    float(np.quantile(ae_qr, 0.95)),
                })
            except Exception as e:
                print(f"    QR failed (n={n_train}, tau={tau}): {e}")

    return rows


# ── macrorep runner ───────────────────────────────────────────────────────────

def run_one_macrorep(macro_k: int, seed: int, params: Params,
                     t_grid: np.ndarray,
                     indicator_type: str = "step") -> list[dict]:
    rng = np.random.default_rng(seed)
    all_rows = []
    for n_train in N_TRAIN_LIST:
        rows = run_one_case(n_train, rng, params, t_grid, indicator_type)
        for r in rows:
            print(
                f"  [rep{macro_k}] n={n_train:5d}  tau={r['tau']:.2f}"
                f"  {r['method']:4s}  mean={r['mean_abs_err']:.4f}  sup={r['sup_abs_err']:.4f}"
            )
        all_rows.extend([{**r, "macrorep": macro_k} for r in rows])
    return all_rows


# ── plotting ──────────────────────────────────────────────────────────────────

def _make_panel_specs(tau_hi: float) -> list:
    return [
        ("mean_abs_err", "Mean |q̂ − q_true|", f"Mean absolute error  (τ=0.05)", 0.05),
        ("mean_abs_err", "Mean |q̂ − q_true|", f"Mean absolute error  (τ={tau_hi:.2f})", tau_hi),
        ("sup_abs_err",  "Sup |q̂ − q_true|",  f"Sup absolute error   (τ=0.05)", 0.05),
        ("sup_abs_err",  "Sup |q̂ − q_true|",  f"Sup absolute error   (τ={tau_hi:.2f})", tau_hi),
    ]

PANEL_SPECS = _make_panel_specs(0.95)  # default; updated in main()


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
            marker=MARKERS.get(method, "o"), markersize=5, label=method, zorder=3)
    ax.fill_between(n_vals, means - stds, means + stds, color=c, alpha=0.15)


def _draw_boxplot(ax, df, col, tau, method, n_vals, positions):
    sub  = df[(df["tau"] == tau) & (df["method"] == method)]
    data = [sub[sub["n_train"] == n][col].values for n in n_vals]
    c = COLORS.get(method, "gray")
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
    ax.plot([], [], color=c, lw=2.5, label=method)  # legend proxy


def make_figure(raw_path: str, plot_type: str = "boxplot",
                method_filter: str | None = None,
                indicator_type: str = "") -> plt.Figure:
    df = pd.read_csv(raw_path)
    df = df[df["simulator"] == SIM_NAME].copy() if "simulator" in df.columns else df

    methods = ["CKME"]

    n_vals   = sorted(df["n_train"].unique())
    n_macro  = df["macrorep"].nunique() if "macrorep" in df.columns else "?"
    offsets  = {m: 0.0 for m in methods}

    ind_label = f"  [indicator: {indicator_type}]" if indicator_type else ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"exp2  —  CKME  |  tau-quantile estimation error{ind_label}\n"
        f"({n_macro} macroreps, R_train={R_TRAIN})",
        fontsize=12, fontweight="bold",
    )

    for ax, (col, ylabel, title, tau) in zip(axes.ravel(), PANEL_SPECS):
        for method in methods:
            sub = df[(df["tau"] == tau) & (df["method"] == method)]
            if sub.empty:
                continue
            if plot_type == "boxplot":
                pos = [i + offsets[method] for i in range(len(n_vals))]
                _draw_boxplot(ax, df, col, tau, method, n_vals, pos)
                ax.set_xticks(range(len(n_vals)))
                ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
            else:
                _draw_line(ax, df, col, tau, method, n_vals)
                ax.set_xscale("log")
                ax.set_xticks(n_vals)
                ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)

        ax.set_xlabel("n_train", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    default_dir = os.path.join(os.path.dirname(__file__), "output_exp2")
    parser = argparse.ArgumentParser(description="exp2 quantile error: run + plot")
    parser.add_argument("--n_macro",    type=int, default=10)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_jobs",     type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=default_dir)
    parser.add_argument("--no_run",     action="store_true",
                        help="Skip running; load existing exp2_raw.csv to plot")
    parser.add_argument("--plot",       action="store_true",
                        help="Plot after running (or with --no_run)")
    parser.add_argument("--plot_type",  type=str, default="boxplot",
                        choices=["line", "boxplot"])
    parser.add_argument("--method",     type=str, default=None,
                        choices=["CKME", "QR"])
    parser.add_argument("--save",       type=str, default=None,
                        help="Save figure to this path (PNG)")
    parser.add_argument("--indicator",   type=str, default="step",
                        choices=["logistic", "step", "gaussian_cdf"],
                        help="Smooth indicator type for CKME (default: step)")
    parser.add_argument("--append",     action="store_true",
                        help="Append to existing CSV instead of overwriting")
    parser.add_argument("--tau_hi",    type=float, default=0.95,
                        help="Upper quantile level (default: 0.95)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Override module-level TAUS and PANEL_SPECS with CLI value
    global TAUS, PANEL_SPECS
    TAUS = [0.05, args.tau_hi]
    PANEL_SPECS = _make_panel_specs(args.tau_hi)

    raw_path = os.path.join(args.output_dir, f"exp2_raw_{args.indicator}.csv")

    # ── run ───────────────────────────────────────────────────────────────────
    if not args.no_run:
        params = _load_params()
        print(f"\nN_TRAIN_LIST : {N_TRAIN_LIST}")
        print(f"n_macro      : {args.n_macro}")
        print(f"R_TRAIN      : {R_TRAIN}")
        print(f"indicator    : {args.indicator}\n")

        t_grid = _build_oracle_t_grid()
        seeds  = [args.base_seed + k for k in range(args.n_macro)]
        worker = partial(run_one_macrorep, params=params, t_grid=t_grid,
                         indicator_type=args.indicator)
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
        print(f"\nSaved raw results -> {raw_path}  ({len(raw_df)} rows)")

        # Quick summary table
        agg = (
            raw_df.groupby(["n_train", "tau", "method"])[
                ["mean_abs_err", "sup_abs_err"]
            ]
            .agg(["mean", "std"])
            .round(4)
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        print("\n--- Mean absolute quantile error (mean ± std across macroreps) ---")
        print(agg.to_string())

    # ── plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.no_run:
        if not os.path.exists(raw_path):
            print(f"No raw CSV found at {raw_path}. Run without --no_run first.")
            return
        fig = make_figure(raw_path, plot_type=args.plot_type,
                          method_filter=args.method, indicator_type=args.indicator)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Figure saved -> {args.save}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
