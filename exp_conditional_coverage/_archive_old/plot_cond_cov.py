"""
plot_cond_cov.py

Generate three figures from asymptotic conditional coverage experiment results.

Figure 1: Conditional coverage curves x -> Cov_n(x) for multiple n values
          (one subplot per simulator, lines colored by n)
Figure 2: MAE-Cov(n) vs n with macro-rep error bands (mean ± 1 SD)
Figure 3: Oracle endpoint error vs n (exp1 only)

Usage (from project root):
  python exp_conditional_coverage/plot_cond_cov.py
  python exp_conditional_coverage/plot_cond_cov.py --save --output_dir exp_conditional_coverage/output
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
import matplotlib.cm as cm

_HERE = Path(__file__).parent

SIM_LABELS = {
    "exp1":         "exp1 (MG1 queue, Gaussian, 1D)",
    "exp2":         "exp2 (Gaussian, 1D)",
    "nongauss_B2L": "nongauss_B2L (Gamma skew, 1D)",
}

# n-value colors: sequential colormap
_CMAP = cm.get_cmap("viridis_r")


def _n_colors(n_vals):
    idxs = np.linspace(0.1, 0.9, len(n_vals))
    return {n: _CMAP(i) for n, i in zip(n_vals, idxs)}


# ---------------------------------------------------------------------------
# Figure 1: Coverage curves
# ---------------------------------------------------------------------------

def plot_fig1(results: dict[str, pd.DataFrame], alpha: float, save_path: Path | None):
    """One subplot per simulator. Multiple n-value curves (averaged over macroreops)."""
    sims = list(results.keys())
    fig, axes = plt.subplots(1, len(sims), figsize=(6 * len(sims), 4.5), squeeze=False)

    for col, sim in enumerate(sims):
        ax = axes[0, col]
        df = results[sim]
        n_vals = sorted(df["n_0"].unique())
        colors = _n_colors(n_vals)

        for n in n_vals:
            g = df[df["n_0"] == n].groupby("x_eval")["cov_mc"].mean().reset_index()
            ax.plot(g["x_eval"], g["cov_mc"], color=colors[n], label=f"n={n}", lw=1.5)

        ax.axhline(1 - alpha, color="k", ls="--", lw=1.2, label=f"target {1-alpha:.0%}")
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("Conditional coverage", fontsize=11)
        ax.set_title(SIM_LABELS.get(sim, sim), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 1: Conditional Coverage Curves by Sample Size", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved Figure 1 -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: MAE-Cov vs n
# ---------------------------------------------------------------------------

def plot_fig2(summaries: dict[str, pd.DataFrame], alpha: float, save_path: Path | None):
    """MAE-Cov mean ± 1 SD for each simulator, plotted on the same axes."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = ["o", "s", "^", "D"]
    colors_sim = ["tab:blue", "tab:orange"]

    for i, (sim, df) in enumerate(summaries.items()):
        df = df.sort_values("n_0")
        n = df["n_0"].values
        m = df["mae_cov_mean"].values
        s = df["mae_cov_sd"].values
        label = SIM_LABELS.get(sim, sim)
        c = colors_sim[i % len(colors_sim)]
        ax.plot(n, m, marker=markers[i], color=c, lw=1.8, ms=6, label=label)
        ax.fill_between(n, m - s, m + s, alpha=0.18, color=c)

    ax.set_xlabel("n  (Stage1 = Stage2 design points)", fontsize=11)
    ax.set_ylabel("MAE-Cov  (mean |Cov_n(x) − (1−α)|)", fontsize=11)
    ax.set_title("Figure 2: Mean Conditional Coverage Error vs Sample Size", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved Figure 2 -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Oracle endpoint error vs n (exp1 only)
# ---------------------------------------------------------------------------

def plot_fig3(summaries: dict[str, pd.DataFrame], save_path: Path | None):
    """Endpoint error (L and U) vs n for exp1."""
    sim_key = "exp1" if "exp1" in summaries else ("exp2" if "exp2" in summaries else None)
    if sim_key is None:
        print("  Figure 3 skipped: no exp1/exp2 results found.")
        return None

    df = summaries[sim_key].sort_values("n_0")
    if "endpoint_err_L_mean" not in df.columns:
        print(f"  Figure 3 skipped: endpoint_err columns missing in summary_{sim_key}.csv.")
        return None
    n = df["n_0"].values
    errL = df["endpoint_err_L_mean"].values
    sdL  = df["endpoint_err_L_sd"].values
    errU = df["endpoint_err_U_mean"].values
    sdU  = df["endpoint_err_U_sd"].values

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n, errL, marker="o", color="tab:blue",   lw=1.8, ms=6, label="Lower endpoint error")
    ax.fill_between(n, errL - sdL, errL + sdL, alpha=0.18, color="tab:blue")
    ax.plot(n, errU, marker="s", color="tab:orange", lw=1.8, ms=6, label="Upper endpoint error")
    ax.fill_between(n, errU - sdU, errU + sdU, alpha=0.18, color="tab:orange")

    ax.set_xlabel("n  (Stage1 = Stage2 design points)", fontsize=11)
    ax.set_ylabel("Mean |endpoint − oracle quantile|", fontsize=11)
    ax.set_title(f"Figure 3: Oracle Endpoint Error vs n  ({sim_key}, Gaussian)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved Figure 3 -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot asymptotic conditional coverage results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save",       action="store_true", help="Save figures to disk")
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (_HERE / "output")

    # Load results
    results: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    for sim in ["exp1", "nongauss_B2L"]:
        rp = out_dir / f"results_{sim}.csv"
        sp = out_dir / f"summary_{sim}.csv"
        if rp.exists():
            results[sim] = pd.read_csv(rp)
        if sp.exists():
            summaries[sim] = pd.read_csv(sp)

    if not results:
        print("No result files found. Run run_cond_cov.py first.")
        return

    alpha = args.alpha

    save1 = (out_dir / "fig1_coverage_curves.png") if args.save else None
    save2 = (out_dir / "fig2_mae_cov_vs_n.png")    if args.save else None
    save3 = (out_dir / "fig3_endpoint_err_vs_n.png") if args.save else None

    print("Generating Figure 1...")
    plot_fig1(results, alpha, save1)

    if summaries:
        print("Generating Figure 2...")
        plot_fig2(summaries, alpha, save2)
    else:
        print("No summary files found; skipping Figure 2.")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
