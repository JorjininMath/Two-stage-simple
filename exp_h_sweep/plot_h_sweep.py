"""
plot_h_sweep.py

Figures:
  1. CDF curves F̂(t|x) vs h at representative x
  2. PDF curves f̂(t|x) vs h at representative x
  3. Quantile error vs h (U-shape), one line per tau
  4. CRPS(x, h) heatmap (x-axis h, y-axis x, color=CRPS)

Usage:
  python exp_h_sweep/plot_h_sweep.py [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize

from exp_h_sweep.run_h_sweep import (
    H_LIST, REP_X, TAU_LIST, oracle_cdf, oracle_pdf,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="nongauss_B2L")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    sim = args.sim

    here = Path(__file__).parent
    inp = here / f"output_{sim}"
    df = pd.read_csv(inp / "h_sweep_perpoint.csv")
    curves = np.load(inp / "h_sweep_curves.npz", allow_pickle=True)
    t_grid = curves["t_grid"]

    # Color map for h (log scale): cool → warm
    h_arr = np.array(H_LIST)
    norm = LogNorm(vmin=h_arr.min(), vmax=h_arr.max())
    cmap = cm.viridis
    h_colors = [cmap(norm(h)) for h in h_arr]

    # -----------------------------------------------------------------------
    # Fig 1: CDF curves vs h at representative x
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, len(REP_X), figsize=(5 * len(REP_X), 5),
                                sharey=True)

    for ax, x_val in zip(axes1, REP_X):
        # Oracle reference
        F_oracle = oracle_cdf(float(x_val), t_grid, sim)
        ax.plot(t_grid, F_oracle, "k--", lw=2.5, label="Oracle", zorder=10)

        for h, c in zip(H_LIST, h_colors):
            key = f"cdf_h{h}_x{x_val:.2f}"
            F = curves[key]
            ax.plot(t_grid, F, color=c, lw=1.3, alpha=0.9)

        ax.set_title(f"F̂(t | x={x_val:.2f})")
        ax.set_xlabel("t")
        if ax is axes1[0]:
            ax.set_ylabel("F(t|x)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    # Colorbar for h
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=axes1, fraction=0.02, pad=0.02)
    cbar.set_label("h (log scale)")
    axes1[0].legend(loc="upper left", fontsize=9)

    fig1.savefig(inp / "fig1_cdf_vs_h.png", dpi=150, bbox_inches="tight")
    print("Saved fig1")

    # -----------------------------------------------------------------------
    # Fig 2: PDF curves vs h at representative x
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, len(REP_X), figsize=(5 * len(REP_X), 5),
                                sharey=False)

    for ax, x_val in zip(axes2, REP_X):
        f_oracle = oracle_pdf(float(x_val), t_grid, sim)
        ax.plot(t_grid, f_oracle, "k--", lw=2.5, label="Oracle", zorder=10)

        for h, c in zip(H_LIST, h_colors):
            key = f"pdf_h{h}_x{x_val:.2f}"
            f = curves[key]
            ax.plot(t_grid, f, color=c, lw=1.3, alpha=0.9)

        ax.set_title(f"f̂(t | x={x_val:.2f})")
        ax.set_xlabel("t")
        if ax is axes2[0]:
            ax.set_ylabel("f(t|x)")
        ax.grid(alpha=0.3)

    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=axes2, fraction=0.02, pad=0.02)
    cbar2.set_label("h (log scale)")
    axes2[0].legend(loc="upper right", fontsize=9)

    fig2.savefig(inp / "fig2_pdf_vs_h.png", dpi=150, bbox_inches="tight")
    print("Saved fig2")

    # -----------------------------------------------------------------------
    # Fig 3: Mean quantile error vs h (U-shape), one line per tau
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5.5))

    tau_cmap = cm.plasma
    tau_colors = [tau_cmap(i / (len(TAU_LIST) - 1)) for i in range(len(TAU_LIST))]

    for tau, tc in zip(TAU_LIST, tau_colors):
        err_col = f"err_tau{tau}"
        means = df.groupby("h")[err_col].mean()
        # Aggregate to per-macrorep mean first, then SE across macroreps
        per_macro = df.groupby(["h", "macrorep"])[err_col].mean().reset_index()
        stats = per_macro.groupby("h")[err_col].agg(["mean", "std", "count"])
        se = stats["std"] / np.sqrt(stats["count"])

        ax3.errorbar(stats.index, stats["mean"], yerr=se,
                     marker="o", label=f"τ={tau}", color=tc,
                     capsize=3, lw=1.8)

    ax3.set_xscale("log")
    ax3.set_xlabel("h (log scale)")
    ax3.set_ylabel("Mean |q̂_τ − q_oracle|")
    ax3.set_title(f"Quantile error vs bandwidth h  ({sim}, n_0=200)")
    ax3.legend(title="Quantile level")
    ax3.grid(alpha=0.3, which="both")

    fig3.tight_layout()
    fig3.savefig(inp / "fig3_qerror_vs_h.png", dpi=150)
    print("Saved fig3")

    # -----------------------------------------------------------------------
    # Fig 4: CRPS(x, h) heatmap
    # -----------------------------------------------------------------------
    # Aggregate over macroreps: mean CRPS per (x, h)
    pivot = df.groupby(["x", "h"])["crps"].mean().unstack("h")
    # Ensure h column order
    pivot = pivot[H_LIST]

    fig4, ax4 = plt.subplots(figsize=(9, 6))
    im = ax4.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[0, len(H_LIST), pivot.index.min(), pivot.index.max()],
        cmap="viridis",
    )
    ax4.set_xticks(np.arange(len(H_LIST)) + 0.5)
    ax4.set_xticklabels([f"{h:g}" for h in H_LIST])
    ax4.set_xlabel("h")
    ax4.set_ylabel("x")
    ax4.set_title("Pointwise CRPS(x, h)  — darker = lower CRPS")
    cbar4 = fig4.colorbar(im, ax=ax4)
    cbar4.set_label("CRPS")

    # Overlay: for each x-row, mark the argmin h
    argmin_idx = np.argmin(pivot.values, axis=1)
    x_vals = pivot.index.values
    ax4.plot(argmin_idx + 0.5, x_vals, "rx", markersize=6, label="argmin_h")
    ax4.legend(loc="upper right")

    fig4.tight_layout()
    fig4.savefig(inp / "fig4_crps_heatmap.png", dpi=150)
    print("Saved fig4")

    if args.show:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
