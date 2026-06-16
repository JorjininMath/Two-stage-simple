"""
plot_exp4b.py

Two figures for Exp4b on nongauss_A1L (Student-t_3, where plug-in sigma_hat
estimates std = scale * sqrt(3) ≠ scale; the prediction is that CP calibration
absorbs the misspecification).

Figure 1 (exp4b_coverage_curves.png) — single panel at the largest budget:
    Conditional coverage cov(x) for the three arms (fixed / plug-in / oracle).
    P1: plug-in and oracle curves overlap; both flatter than fixed.

Figure 2 (exp4b_qhat_ratio_vs_budget.png) — diagnostic ratios vs budget:
    Left  panel: q_plug / q_oracle. Empirical signature is that this stabilizes
                 (not necessarily at 1) — direction depends on how the smoother
                 plug-in indicator reshapes the |F_hat - 0.5| score distribution.
                 Goal is just to show convergence as B grows, not >1.
    Right panel: h_plug / h_oracle = sigma_hat / s. For Student-t_nu with
                 nu=3, the asymptotic ratio is sqrt(nu/(nu-2)) = sqrt(3) ≈ 1.732
                 (since plug-in estimates std, oracle uses scale).
                 Convergence to that line is the cleanest empirical check that
                 plug-in is doing what the theory says.

Reads:
    exp_adaptive_h/output_exp4/exp4_paired_deltas.csv
    exp_adaptive_h/output_exp4/macrorep_*/budget_*/case_nongauss_A1L_*/per_point.csv

Writes:
    exp_adaptive_h/output_exp4/exp4b_coverage_curves.png
    exp_adaptive_h/output_exp4/exp4b_qhat_ratio_vs_budget.png

Usage (from project root):
    python exp_adaptive_h/plot_exp4b.py
    python exp_adaptive_h/plot_exp4b.py --output_dir exp_adaptive_h/output_exp4 --c_scale 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Two_stage.sim_functions import get_experiment_config

ARMS = ["fixed", "plugin", "oracle"]
ARM_COLOR = {"fixed": "tab:gray", "plugin": "tab:blue", "oracle": "tab:red"}
ARM_LS    = {"fixed": "--",       "plugin": "-",        "oracle": "-"}
ARM_LABEL = {"fixed": "fixed h (CV)", "plugin": "plug-in $h(x)$", "oracle": "oracle $h(x)$"}


def _bin_coverage(x: np.ndarray, cov: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = cov[mask].mean()
    return out


def _load_arm_curves(out_dir: Path, simulator: str, budget: int, arm: str, bin_edges: np.ndarray):
    paths = sorted(out_dir.glob(f"macrorep_*/budget_{budget}/case_{simulator}_{arm}/per_point.csv"))
    if not paths:
        return None
    cov_per = np.empty((len(paths), len(bin_edges) - 1))
    marg = np.empty(len(paths))
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        cov_per[i] = _bin_coverage(
            df["x0"].to_numpy(), df["covered_score"].to_numpy(), bin_edges
        )
        marg[i] = df["covered_score"].mean()
    return cov_per, marg


def _figure1_coverage(out_dir: Path, simulator: str, paired: pd.DataFrame, n_bins: int, alpha: float):
    df_sim = paired[paired["simulator"] == simulator]
    if df_sim.empty:
        print(f"WARN: no rows for {simulator} in paired csv; skipping Fig 1", file=sys.stderr)
        return
    budgets = sorted(df_sim["budget"].unique())
    B_max = int(budgets[-1])

    cfg = get_experiment_config(simulator)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    target = 1.0 - alpha

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for arm in ARMS:
        loaded = _load_arm_curves(out_dir, simulator, B_max, arm, bin_edges)
        if loaded is None:
            continue
        cov_per, marg = loaded
        med = np.nanmedian(cov_per, axis=0)
        marg_med = float(np.median(marg))
        ax.plot(
            bin_centers, med,
            color=ARM_COLOR[arm], ls=ARM_LS[arm], lw=2.0, marker="o", ms=4,
            label=f"{ARM_LABEL[arm]} (marg={marg_med:.3f})",
        )

    ax.axhline(target, ls=":", color="black", alpha=0.6,
               label=f"target $1-\\alpha={target:.2f}$")
    ax.set_xlabel("x")
    ax.set_ylabel("conditional coverage (median over macroreps)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Exp4b ({simulator}, $B={B_max}$): conditional coverage — "
        f"plug-in vs oracle vs fixed"
    )
    fig.tight_layout()
    out_path = out_dir / f"exp4b_coverage_curves_{simulator}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _figure2_qhat_ratio(out_dir: Path, simulator: str, paired: pd.DataFrame, c_scale: float):
    df_sim = paired[paired["simulator"] == simulator].copy()
    if df_sim.empty:
        print(f"WARN: no rows for {simulator} in paired csv; skipping Fig 2", file=sys.stderr)
        return

    g = df_sim.groupby("budget", sort=True)
    rows = []
    for B, df in g:
        rs = df["q_ratio_plugin_over_oracle"].to_numpy()
        rs = rs[~np.isnan(rs)]
        hr = df["h_ratio_plugin_over_oracle"].to_numpy()
        hr = hr[~np.isnan(hr)]
        rows.append({
            "budget": int(B),
            "med_q":  float(np.median(rs)) if rs.size else np.nan,
            "q1_q":   float(np.quantile(rs, 0.25)) if rs.size else np.nan,
            "q3_q":   float(np.quantile(rs, 0.75)) if rs.size else np.nan,
            "med_h":  float(np.median(hr)) if hr.size else np.nan,
            "q1_h":   float(np.quantile(hr, 0.25)) if hr.size else np.nan,
            "q3_h":   float(np.quantile(hr, 0.75)) if hr.size else np.nan,
        })
    agg = pd.DataFrame(rows)
    x = agg["budget"].to_numpy(dtype=float)

    # Population limit of (IQR/1.349) / scale: 1 for Gaussian DGPs;
    # kappa_nu = (q_{0.75}(t_nu) - q_{0.25}(t_nu)) / 1.349 for Student-t_nu.
    if simulator == "nongauss_A1L":
        from scipy.stats import t as _t
        nu = 3.0
        h_target = float((_t.ppf(0.75, df=nu) - _t.ppf(0.25, df=nu)) / 1.349)
        h_target_label = rf"$\kappa_{{\nu={int(nu)}}}\approx{h_target:.3f}$ (asymp.)"
    else:
        h_target = 1.0
        h_target_label = "asymp. ratio = 1 (Gaussian)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax = axes[0]
    ax.fill_between(x, agg["q1_q"], agg["q3_q"], alpha=0.25, color="tab:blue",
                    label="25-75% paired")
    ax.plot(x, agg["med_q"], "o-", color="tab:blue", lw=2, ms=7, label="median")
    ax.axhline(1.0, ls=":", color="black", alpha=0.6, label="ratio = 1")
    ax.set_xscale("log")
    ax.set_xlabel("Stage 1 budget $B = n_0 \\cdot r_0$")
    ax.set_ylabel(r"$\hat{q}_{\mathrm{plug}}/\hat{q}_{\mathrm{oracle}}$")
    ax.set_title("P2: $\\hat{q}$ ratio stabilizes as $B$ grows")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in x])
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    ax.fill_between(x, agg["q1_h"], agg["q3_h"], alpha=0.25, color="tab:green",
                    label="25-75% paired")
    ax.plot(x, agg["med_h"], "o-", color="tab:green", lw=2, ms=7, label="median")
    ax.axhline(1.0, ls=":", color="black", alpha=0.6)
    if not np.isclose(h_target, 1.0):
        ax.axhline(h_target, ls="--", color="tab:orange", label=h_target_label)
    ax.set_xscale("log")
    ax.set_xlabel("Stage 1 budget $B = n_0 \\cdot r_0$")
    ax.set_ylabel(r"$\bar{h}_{\mathrm{plug}}/\bar{h}_{\mathrm{oracle}}$ (mean over $X_\mathrm{test}$)")
    ax.set_title(r"Diagnostic: bandwidth ratio $\hat{\sigma}/s$")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in x])
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Exp4b ({simulator}, $c={c_scale:g}$): plug-in vs oracle scaling",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / f"exp4b_qhat_ratio_vs_budget_{simulator}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _figure_h_ratio_grid(out_dir: Path, paired: pd.DataFrame, c_scale: float,
                          sims: list, finite_sample_factor: float = 0.87):
    """2x2 grid: h_plug/h_oracle vs B, one panel per DGP. Population limits
    differ across DGPs (1 for Gaussian, kappa_nu for Student-t_nu); a dashed
    finite-r_0 prediction multiplies the population limit by ~0.87 at r_0=10.
    """
    from scipy.stats import t as _t

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, sim in zip(axes.ravel(), sims):
        df_sim = paired[paired["simulator"] == sim].copy()
        if df_sim.empty:
            ax.set_visible(False)
            continue
        rows = []
        for B, df in df_sim.groupby("budget", sort=True):
            hr = df["h_ratio_plugin_over_oracle"].to_numpy()
            hr = hr[~np.isnan(hr)]
            rows.append({
                "budget": int(B),
                "med":  float(np.median(hr)) if hr.size else np.nan,
                "q1":   float(np.quantile(hr, 0.25)) if hr.size else np.nan,
                "q3":   float(np.quantile(hr, 0.75)) if hr.size else np.nan,
            })
        agg = pd.DataFrame(rows)
        x = agg["budget"].to_numpy(dtype=float)

        if sim == "nongauss_A1L":
            nu = 3.0
            pop_limit = float((_t.ppf(0.75, df=nu) - _t.ppf(0.25, df=nu)) / 1.349)
            pop_label = rf"pop. limit $\kappa_3\approx{pop_limit:.3f}$"
        else:
            pop_limit = 1.0
            pop_label = "pop. limit $= 1$ (Gauss)"

        ax.fill_between(x, agg["q1"], agg["q3"], alpha=0.25, color="tab:green",
                        label="25-75%")
        ax.plot(x, agg["med"], "o-", color="tab:green", lw=2, ms=6, label="median")
        ax.axhline(pop_limit, ls="--", color="tab:orange", label=pop_label)
        ax.axhline(pop_limit * finite_sample_factor, ls=":", color="tab:red",
                   label=rf"finite-$r_0$ pred. $\approx {pop_limit * finite_sample_factor:.2f}$")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(b)}" for b in x])
        ax.set_xlabel(r"Stage 1 budget $B = n_0 \cdot r_0$")
        ax.set_ylabel(r"$\bar h_{\mathrm{plug}}/\bar h_{\mathrm{oracle}}$")
        ax.set_title(sim)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        rf"Exp4b: bandwidth ratio across DGPs ($c={c_scale:g}$, $r_0=10$)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_path = out_dir / "exp4b_h_ratio_grid_4dgp.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _figure_cov_grid(out_dir: Path, paired: pd.DataFrame, n_bins: int, alpha: float,
                      sims: list):
    """2x2 grid: bin-wise coverage curves at the largest budget per DGP."""
    target = 1.0 - alpha
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, sim in zip(axes.ravel(), sims):
        df_sim = paired[paired["simulator"] == sim]
        if df_sim.empty:
            ax.set_visible(False)
            continue
        B_max = int(sorted(df_sim["budget"].unique())[-1])
        cfg = get_experiment_config(sim)
        x_lo = float(cfg["bounds"][0][0])
        x_hi = float(cfg["bounds"][1][0])
        bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for arm in ARMS:
            loaded = _load_arm_curves(out_dir, sim, B_max, arm, bin_edges)
            if loaded is None:
                continue
            cov_per, marg = loaded
            med = np.nanmedian(cov_per, axis=0)
            marg_med = float(np.median(marg))
            ax.plot(
                bin_centers, med,
                color=ARM_COLOR[arm], ls=ARM_LS[arm], lw=1.7, marker="o", ms=3,
                label=f"{ARM_LABEL[arm]} (m={marg_med:.3f})",
            )
        ax.axhline(target, ls=":", color="black", alpha=0.6,
                   label=rf"target $1-\alpha={target:.2f}$")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("conditional coverage")
        ax.set_title(f"{sim} ($B={B_max}$)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        "Exp4b: conditional coverage across DGPs (median over macroreps)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_path = out_dir / "exp4b_coverage_curves_grid_4dgp.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Exp4b: plug-in vs oracle, per simulator")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp4")
    parser.add_argument("--simulator",  type=str, default="nongauss_A1L",
                        help="DGP name (one of exp1, gibbs_s1, wsc_gauss, nongauss_A1L) "
                             "or 'all' to loop over all four.")
    parser.add_argument("--n_bins",     type=int,   default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--c_scale",    type=float, default=1.0,
                        help="c used in run_exp4_plugin (for plateau ref line).")
    args = parser.parse_args()

    out_dir = (
        (_root / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    paired_path = out_dir / "exp4_paired_deltas.csv"
    if not paired_path.exists():
        print(f"ERROR: {paired_path} not found. Run summarize_exp4.py first.",
              file=sys.stderr)
        sys.exit(1)
    paired = pd.read_csv(paired_path)

    if args.simulator == "all":
        sims = ["exp1", "gibbs_s1", "wsc_gauss", "nongauss_A1L"]
    else:
        sims = [args.simulator]
    for sim in sims:
        _figure1_coverage(out_dir, sim, paired, n_bins=args.n_bins, alpha=args.alpha)
        _figure2_qhat_ratio(out_dir, sim, paired, c_scale=args.c_scale)
    if args.simulator == "all":
        _figure_h_ratio_grid(out_dir, paired, c_scale=args.c_scale, sims=sims)
        _figure_cov_grid(out_dir, paired, n_bins=args.n_bins, alpha=args.alpha, sims=sims)


if __name__ == "__main__":
    main()
