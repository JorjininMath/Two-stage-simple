"""
plot_exp1.py

Plot Exp1 (fixed-h baseline) conditional coverage curves on a 2x2 grid.

For each of the 4 DGPs:
  - bin X_test into n_bins equal-width bins covering the DGP's x-domain
  - per macrorep, compute per-bin coverage rate (using covered_score column)
  - across macroreps, plot median + 5-95% IQR shaded band
  - horizontal dashed line at 1 - alpha (target marginal coverage)

Reads:
    exp_adaptive_h/output_exp1/macrorep_{k}/case_{sim}/per_point.csv

Writes:
    exp_adaptive_h/output_exp1/exp1_coverage_curves.png

Usage (from project root):
    python exp_adaptive_h/plot_exp1.py
    python exp_adaptive_h/plot_exp1.py --output_dir exp_adaptive_h/output_exp1 --n_bins 25
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

SIMULATORS = [
    "wsc_gauss",
    "gibbs_s1",
    "exp1",
    "nongauss_A1L",
]

PRETTY_NAME = {
    "wsc_gauss":    "wsc_gauss  (Gaussian, smooth U)",
    "gibbs_s1":     "gibbs_s1  (Gaussian, interior zero)",
    "exp1":         "exp1  (Gaussian MG1, boundary explosion)",
    "nongauss_A1L": "nongauss_A1L  (Student-t $\\nu=3$, smooth U)",
}

PANEL_COLOR = {
    "wsc_gauss":    "tab:blue",
    "gibbs_s1":     "tab:orange",
    "exp1":         "tab:green",
    "nongauss_A1L": "tab:red",
}


def _load_macrorep_dirs(out_dir: Path, sim: str) -> list[Path]:
    """Return list of case_dir paths across all macroreps for one simulator."""
    macrorep_dirs = sorted(out_dir.glob("macrorep_*"))
    case_dirs = []
    for mdir in macrorep_dirs:
        case_dir = mdir / f"case_{sim}"
        if (case_dir / "per_point.csv").exists():
            case_dirs.append(case_dir)
    return case_dirs


def _bin_coverage(df: pd.DataFrame, bin_edges: np.ndarray) -> np.ndarray:
    """Return per-bin mean coverage. NaN where bin is empty."""
    x = df["x0"].to_numpy()
    cov = df["covered_score"].to_numpy()
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = cov[mask].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot Exp1 conditional coverage curves")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp1")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--out_png",    type=str, default=None)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, sim in zip(axes, SIMULATORS):
        cfg = get_experiment_config(sim)
        x_lo = float(cfg["bounds"][0][0])
        x_hi = float(cfg["bounds"][1][0])
        bin_edges = np.linspace(x_lo, x_hi, args.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        case_dirs = _load_macrorep_dirs(out_dir, sim)
        if not case_dirs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(PRETTY_NAME[sim])
            continue

        cov_per_macrorep = np.empty((len(case_dirs), args.n_bins))
        marginal_per_macrorep = np.empty(len(case_dirs))
        for i, cdir in enumerate(case_dirs):
            df = pd.read_csv(cdir / "per_point.csv")
            cov_per_macrorep[i] = _bin_coverage(df, bin_edges)
            marginal_per_macrorep[i] = df["covered_score"].mean()

        med = np.nanmedian(cov_per_macrorep, axis=0)
        lo  = np.nanpercentile(cov_per_macrorep, 5,  axis=0)
        hi  = np.nanpercentile(cov_per_macrorep, 95, axis=0)

        color = PANEL_COLOR[sim]
        ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.20,
                        label="5-95% across macroreps")
        ax.plot(bin_centers, med, color=color, lw=2, marker="o", ms=4,
                label="median coverage(x)")
        ax.axhline(target, ls="--", color="gray", alpha=0.7,
                   label=f"target $1-\\alpha={target:.2f}$")

        marg_med = float(np.median(marginal_per_macrorep))
        ax.set_title(
            f"{PRETTY_NAME[sim]}\n"
            f"marginal coverage (median over {len(case_dirs)} macroreps): {marg_med:.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("conditional coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(x_lo, x_hi)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Exp1: fixed-h baseline (CV-tuned scalar h) — conditional coverage curves",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    out_png = Path(args.out_png) if args.out_png else out_dir / "exp1_coverage_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
