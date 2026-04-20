"""
diag_ckme_scaling.py

Diagnostic analysis of CKME scaling results.
Requires per-point CSVs produced by run_ckme_scaling.py --save_perpoint.

先查 1  (--mode worst_x):
    For each macrorep, find x* = argmax_x |q̂_tau(x) - q_true_tau(x)|.
    Plot distribution of x* across macroreps to see if worst-case errors
    concentrate at boundaries, high-noise regions, or heavy-tail regions.

先查 3  (--mode boundary):
    Check what fraction of (macrorep, x) has q̂_0.95(x) at or near t_grid_max.
    Plots: (a) fraction of grid-boundary hits vs n_train,
           (b) distance of q̂ and q_true from t_grid_max vs x.

先查 4  (--mode q95err):
    Plot Q_0.95,x(|q̂_tau - q_true|) vs n_train, alongside mean and sup.
    If Q_0.95 is much lower than sup, the problem is confined to a few x points.

Usage:
    python exp_onesided/diag_ckme_scaling.py --mode worst_x
    python exp_onesided/diag_ckme_scaling.py --mode boundary
    python exp_onesided/diag_ckme_scaling.py --mode q95err
    python exp_onesided/diag_ckme_scaling.py --mode all --save exp_onesided/output_scaling/diag.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

SIMULATORS = ["exp1", "exp2", "nongauss_A1L"]
SIM_LABELS = {
    "exp1":         "exp1 (Gaussian)",
    "exp2":         "exp2 (Gaussian)",
    "nongauss_A1L": "A1L (nu=3)",
}
TAUS = [0.05, 0.95]
TAU_LABELS = {0.05: "τ=0.05", 0.95: "τ=0.95"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_perpoint(output_dir: str) -> pd.DataFrame:
    pp_dir = os.path.join(output_dir, "perpoint")
    if not os.path.exists(pp_dir):
        raise FileNotFoundError(
            f"Per-point directory not found: {pp_dir}\n"
            "Run: python exp_onesided/run_ckme_scaling.py --save_perpoint"
        )
    files = sorted(Path(pp_dir).glob("perpoint_rep*.csv"))
    if not files:
        raise FileNotFoundError(f"No perpoint_rep*.csv files in {pp_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} macrorep files, {len(df)} rows total.")
    return df


def load_summary(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "scaling_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nRun run_ckme_scaling.py first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 先查 1: worst-case x* distribution
# ---------------------------------------------------------------------------

def plot_worst_x(pp: pd.DataFrame, save_path: str | None = None):
    """
    For each (macrorep, n_train, sim, tau), find x* = argmax_x |error|.
    Plot histogram of x* to see if worst-case errors cluster structurally.
    """
    n_train_vals = sorted(pp["n_train"].unique())
    fig, axes = plt.subplots(
        len(SIMULATORS), len(TAUS),
        figsize=(10, 6),
        sharex=True,
    )
    fig.suptitle("Diag 1: Worst-case x* distribution\n"
                 "x* = argmax_x |q_hat_tau(x) - q_true_tau(x)|  per macrorep",
                 fontsize=11)

    cmap = plt.cm.viridis
    colors = {n: cmap(i / max(1, len(n_train_vals) - 1))
              for i, n in enumerate(n_train_vals)}

    for row, sim in enumerate(SIMULATORS):
        for col, tau in enumerate(TAUS):
            ax = axes[row, col]
            sub = pp[(pp["simulator"] == sim) & (pp["tau"] == tau)]

            for n_train in n_train_vals:
                grp = sub[sub["n_train"] == n_train]
                # For each macrorep, find x* (worst-case x)
                worst_x = (
                    grp.groupby("macrorep")
                    .apply(lambda df: df.loc[df["abs_err"].idxmax(), "x"],
                           include_groups=False)
                    .values
                )
                ax.hist(worst_x, bins=20, alpha=0.5, color=colors[n_train],
                        label=f"n={n_train}", density=True)

            if row == 0:
                ax.set_title(TAU_LABELS[tau], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{SIM_LABELS[sim]}\ndensity", fontsize=8)
            ax.set_xlabel("x*", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 先查 3: right-boundary problem
# ---------------------------------------------------------------------------

def plot_boundary(pp: pd.DataFrame, save_path: str | None = None):
    """
    Two panels:
    (a) Fraction of test points with q̂ at right boundary vs n_train
    (b) For the largest n_train, scatter of (x, dist_to_max) for q̂ and q_true
    """
    n_train_vals = sorted(pp["n_train"].unique())
    n_max = max(n_train_vals)

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("Diag 3: Right-boundary problem diagnosis", fontsize=11)
    gs = gridspec.GridSpec(2, len(SIMULATORS) * len(TAUS), figure=fig)

    # Panel row 0: boundary hit fraction vs n_train
    panel_axes_top = []
    for i, (sim, tau) in enumerate(
        [(s, t) for s in SIMULATORS for t in TAUS]
    ):
        ax = fig.add_subplot(gs[0, i])
        panel_axes_top.append((ax, sim, tau))

    for ax, sim, tau in panel_axes_top:
        sub = pp[(pp["simulator"] == sim) & (pp["tau"] == tau)]
        fracs = []
        for n in n_train_vals:
            grp = sub[sub["n_train"] == n]
            frac = grp["at_right_bnd"].mean() if tau > 0.5 else grp["at_left_bnd"].mean()
            fracs.append(frac)
        bnd_label = "at t_grid_max (τ=0.95)" if tau > 0.5 else "at t_grid_min (τ=0.05)"
        ax.plot(n_train_vals, fracs, marker="o", lw=1.8, color="tomato")
        ax.set_xscale("log")
        ax.set_xticks(n_train_vals)
        ax.set_xticklabels([str(n) for n in n_train_vals], fontsize=7, rotation=45)
        ax.set_title(f"{SIM_LABELS[sim]} | {TAU_LABELS[tau]}\n{bnd_label}", fontsize=8)
        ax.set_ylabel("fraction hit boundary", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)

    # Panel row 1: distance to boundary vs x (largest n_train, tau=0.95 only)
    for j, sim in enumerate(SIMULATORS):
        ax = fig.add_subplot(gs[1, j * len(TAUS): j * len(TAUS) + len(TAUS)])
        sub = pp[
            (pp["simulator"] == sim) &
            (pp["tau"] == 0.95) &
            (pp["n_train"] == n_max)
        ]
        if sub.empty:
            ax.set_visible(False)
            continue
        # One macrorep for clarity
        rep0 = sub[sub["macrorep"] == sub["macrorep"].min()]
        ax.scatter(rep0["x"], rep0["dist_to_max"], s=6, alpha=0.5,
                   color="steelblue", label="t_max − q̂_0.95(x)")
        q_true_dist = rep0["t_grid_max"] - rep0["q_true"]
        ax.scatter(rep0["x"], q_true_dist, s=6, alpha=0.5,
                   color="orange", label="t_max − q_true_0.95(x)")
        ax.axhline(0, color="red", ls="--", lw=1, label="boundary")
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("distance to t_grid_max", fontsize=8)
        ax.set_title(f"{SIM_LABELS[sim]} | τ=0.95 | n={n_max}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 先查 4: Q_0.95 vs mean vs sup error
# ---------------------------------------------------------------------------

def plot_q95err(raw: pd.DataFrame, save_path: str | None = None):
    """
    Plot three error metrics vs n_train on same axes:
      - mean_abs_err  (solid blue)
      - q95_abs_err   (solid green)  ← 95th percentile over x
      - sup_abs_err   (dashed red)

    If sup >> q95 >> mean, problem is confined to very few x points (pathological).
    """
    n_train_vals = sorted(raw["n_train"].unique())

    fig, axes = plt.subplots(
        len(SIMULATORS), len(TAUS),
        figsize=(10, 7),
        sharex=False,
    )
    fig.suptitle(
        "Diag 4: Error percentiles vs n_train\n"
        "If sup >> Q95 >> mean -> few pathological x points",
        fontsize=11,
    )

    for row, sim in enumerate(SIMULATORS):
        for col, tau in enumerate(TAUS):
            ax = axes[row, col]
            sub = raw[(raw["simulator"] == sim) & (raw["tau"] == tau)]

            def _plot_metric(col_name, color, label, ls="-"):
                means, stds = [], []
                for n in n_train_vals:
                    vals = sub[sub["n_train"] == n][col_name].values
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                means, stds = np.array(means), np.array(stds)
                ax.plot(n_train_vals, means, color=color, ls=ls, lw=1.8,
                        marker="o", markersize=4, label=label)
                ax.fill_between(n_train_vals, means - stds, means + stds,
                                alpha=0.12, color=color)

            _plot_metric("mean_abs_err", "steelblue",  "mean |err|",    ls="-")
            _plot_metric("q95_abs_err",  "seagreen",   "Q95 |err|",     ls="-")
            _plot_metric("sup_abs_err",  "tomato",     "sup |err|",     ls="--")

            ax.set_xscale("log")
            ax.set_xticks(n_train_vals)
            ax.set_xticklabels([str(n) for n in n_train_vals], fontsize=7, rotation=30)
            ax.set_xlabel("n_train", fontsize=9)
            ax.set_ylabel("|q̂ − q_true|", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", ls=":", alpha=0.4)

            if row == 0:
                ax.set_title(TAU_LABELS[tau], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{SIM_LABELS[sim]}\n|q̂ − q_true|", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: str | None):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CKME scaling diagnostics")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "output_scaling"),
    )
    parser.add_argument(
        "--mode", type=str, default="q95err",
        choices=["worst_x", "boundary", "q95err", "all"],
        help=(
            "worst_x: x* distribution (先查 1)  | "
            "boundary: right-boundary hits (先查 3)  | "
            "q95err: error percentiles (先查 4)  | "
            "all: run all three"
        ),
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Base save path. For --mode all, suffixes _worst_x/_boundary/_q95err are appended.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    modes = ["worst_x", "boundary", "q95err"] if args.mode == "all" else [args.mode]

    needs_perpoint = set(modes) & {"worst_x", "boundary"}
    needs_raw      = set(modes) & {"q95err"}

    pp  = load_perpoint(args.output_dir) if needs_perpoint else None
    raw = load_summary(args.output_dir)  if needs_raw      else None

    for mode in modes:
        save = None
        if args.save:
            base, ext = os.path.splitext(args.save)
            save = f"{base}_{mode}{ext}" if args.mode == "all" else args.save

        if mode == "worst_x":
            plot_worst_x(pp, save)
        elif mode == "boundary":
            plot_boundary(pp, save)
        elif mode == "q95err":
            plot_q95err(raw, save)


if __name__ == "__main__":
    main()
