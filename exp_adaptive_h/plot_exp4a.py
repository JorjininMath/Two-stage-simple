"""
plot_exp4a.py

Exp4a figure: Gap Theorem decay.
For each Gaussian DGP (wsc_gauss, gibbs_s1, exp1), plot |cov_plug - cov_oracle|
vs Stage1 budget B = n_0 * r_0 (paired within macrorep, then median + IQR).

We expect the gap to decay roughly as O(B^{-alpha}) — this is the testable
prediction of the plug-in Gap Theorem.

Reads:
    exp_adaptive_h/output_exp4/exp4_paired_deltas.csv

Writes:
    exp_adaptive_h/output_exp4/exp4a_cov_gap_vs_budget.png

Usage (from project root):
    python exp_adaptive_h/plot_exp4a.py
    python exp_adaptive_h/plot_exp4a.py --output_dir exp_adaptive_h/output_exp4
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

GAUSSIAN_SIMS = ["wsc_gauss", "gibbs_s1", "exp1"]


def _agg_per_budget(df_sim: pd.DataFrame) -> pd.DataFrame:
    g = df_sim.groupby("budget", sort=True)
    rows = []
    for B, df in g:
        gap = df["abs_cov_gap_plugin_oracle"].to_numpy()
        gap = gap[~np.isnan(gap)]
        rows.append({
            "budget":   int(B),
            "median":   float(np.median(gap)) if gap.size else np.nan,
            "q1":       float(np.quantile(gap, 0.25)) if gap.size else np.nan,
            "q3":       float(np.quantile(gap, 0.75)) if gap.size else np.nan,
            "max":      float(np.max(gap)) if gap.size else np.nan,
            "n":        int(gap.size),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot Exp4a Gap Theorem decay")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp4")
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
    sims_present = [s for s in GAUSSIAN_SIMS if s in paired["simulator"].unique()]
    if not sims_present:
        print(f"ERROR: none of {GAUSSIAN_SIMS} found in {paired_path}", file=sys.stderr)
        sys.exit(1)

    n_sims = len(sims_present)
    fig, axes = plt.subplots(1, n_sims, figsize=(4.5 * n_sims, 4.5), sharey=True)
    if n_sims == 1:
        axes = [axes]

    for ax, sim in zip(axes, sims_present):
        df_sim = paired[paired["simulator"] == sim].copy()
        agg = _agg_per_budget(df_sim)
        x = agg["budget"].to_numpy(dtype=float)
        med = agg["median"].to_numpy()
        lo, hi = agg["q1"].to_numpy(), agg["q3"].to_numpy()

        ax.fill_between(x, lo, hi, alpha=0.25, color="tab:blue", label="25-75% paired")
        ax.plot(x, med, "o-", color="tab:blue", lw=2, ms=7, label="median")

        if (med > 0).all() and len(x) >= 2:
            slope, intercept = np.polyfit(np.log(x), np.log(med), 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ys = np.exp(intercept) * xs ** slope
            ax.plot(xs, ys, "--", color="black", lw=1,
                    label=f"$\\propto B^{{{slope:.2f}}}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Stage 1 budget $B = n_0 \\cdot r_0$")
        ax.set_title(sim)
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(b)}" for b in x])
        ax.legend(fontsize=9, loc="best")

    axes[0].set_ylabel(r"$|\mathrm{cov}_\mathrm{plug-in} - \mathrm{cov}_\mathrm{oracle}|$ "
                       r"(paired per macrorep)")
    fig.suptitle(
        "Exp4a: plug-in vs oracle adaptive $h$ — coverage gap decay (Gap Theorem).",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "exp4a_cov_gap_vs_budget.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
