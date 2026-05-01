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

SIMULATOR = "nongauss_A1L"
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


def _load_arm_curves(out_dir: Path, budget: int, arm: str, bin_edges: np.ndarray):
    paths = sorted(out_dir.glob(f"macrorep_*/budget_{budget}/case_{SIMULATOR}_{arm}/per_point.csv"))
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


def _figure1_coverage(out_dir: Path, paired: pd.DataFrame, n_bins: int, alpha: float):
    df_sim = paired[paired["simulator"] == SIMULATOR]
    if df_sim.empty:
        print(f"WARN: no rows for {SIMULATOR} in paired csv; skipping Fig 1", file=sys.stderr)
        return
    budgets = sorted(df_sim["budget"].unique())
    B_max = int(budgets[-1])

    cfg = get_experiment_config(SIMULATOR)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    target = 1.0 - alpha

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for arm in ARMS:
        loaded = _load_arm_curves(out_dir, B_max, arm, bin_edges)
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
        f"Exp4b ({SIMULATOR}, $B={B_max}$): conditional coverage — "
        f"plug-in vs oracle vs fixed"
    )
    fig.tight_layout()
    out_path = out_dir / "exp4b_coverage_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _figure2_qhat_ratio(out_dir: Path, paired: pd.DataFrame, c_scale: float):
    df_sim = paired[paired["simulator"] == SIMULATOR].copy()
    if df_sim.empty:
        print(f"WARN: no rows for {SIMULATOR} in paired csv; skipping Fig 2", file=sys.stderr)
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

    nu = 3.0
    h_target = float(np.sqrt(nu / (nu - 2)))  # = sqrt(3) for Student-t_3

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
    ax.axhline(h_target, ls="--", color="tab:orange",
               label=f"$\\sqrt{{\\nu/(\\nu-2)}}={h_target:.3f}$ (asymp.)")
    ax.set_xscale("log")
    ax.set_xlabel("Stage 1 budget $B = n_0 \\cdot r_0$")
    ax.set_ylabel(r"$\bar{h}_{\mathrm{plug}}/\bar{h}_{\mathrm{oracle}}$ (mean over $X_\mathrm{test}$)")
    ax.set_title(r"Diagnostic: bandwidth ratio $\hat{\sigma}/s$ converges to $\sqrt{\nu/(\nu-2)}$")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in x])
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Exp4b ({SIMULATOR}, $c={c_scale:g}$): plug-in vs oracle scaling",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "exp4b_qhat_ratio_vs_budget.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Exp4b: nongauss_A1L plug-in vs oracle")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp4")
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

    _figure1_coverage(out_dir, paired, n_bins=args.n_bins, alpha=args.alpha)
    _figure2_qhat_ratio(out_dir, paired, c_scale=args.c_scale)


if __name__ == "__main__":
    main()
