"""
plot_exp2.py

Plot Exp2 (fixed h vs oracle adaptive h(x) = c*s(x)) conditional coverage curves.

For each of the 4 DGPs (2x2 grid), overlay TWO arms:
  - fixed  (Exp1 baseline, scalar h from CV)
  - oracle (adaptive h(x) = c * s(x))
Each arm shows median coverage(x) and 5-95% IQR shaded band across macroreps.
Horizontal dashed line at 1 - alpha (target marginal coverage).

Reads:
    exp_adaptive_h/output_exp2/macrorep_{k}/case_{sim}_{fixed|oracle}/per_point.csv

Writes:
    exp_adaptive_h/output_exp2/exp2_coverage_curves.png

Usage (from project root):
    python exp_adaptive_h/plot_exp2.py
    python exp_adaptive_h/plot_exp2.py --output_dir exp_adaptive_h/output_exp2 --n_bins 25
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

ARM_COLOR = {
    "fixed":  "tab:blue",
    "oracle": "tab:red",
}
ARM_LABEL = {
    "fixed":  "fixed h (CV)",
    "oracle": "oracle h(x) = c·s(x)",
}


def _load_case_paths(out_dir: Path, sim: str, arm: str) -> list[Path]:
    case_paths = sorted(out_dir.glob(f"macrorep_*/case_{sim}_{arm}/per_point.csv"))
    return case_paths


def _bin_coverage(df: pd.DataFrame, bin_edges: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Plot Exp2 fixed vs oracle conditional coverage")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp2")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--out_png",    type=str, default=None)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for ax, sim in zip(axes, SIMULATORS):
        cfg = get_experiment_config(sim)
        x_lo = float(cfg["bounds"][0][0])
        x_hi = float(cfg["bounds"][1][0])
        bin_edges = np.linspace(x_lo, x_hi, args.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        marg_lines = []
        any_data = False
        for arm in ["fixed", "oracle"]:
            case_paths = _load_case_paths(out_dir, sim, arm)
            if not case_paths:
                continue
            any_data = True

            cov_per = np.empty((len(case_paths), args.n_bins))
            marg_per = np.empty(len(case_paths))
            for i, p in enumerate(case_paths):
                df = pd.read_csv(p)
                cov_per[i] = _bin_coverage(df, bin_edges)
                marg_per[i] = df["covered_score"].mean()

            med = np.nanmedian(cov_per, axis=0)
            lo  = np.nanpercentile(cov_per, 5,  axis=0)
            hi  = np.nanpercentile(cov_per, 95, axis=0)

            color = ARM_COLOR[arm]
            ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.15)
            ax.plot(
                bin_centers, med, color=color, lw=2,
                marker="o" if arm == "fixed" else "s", ms=4,
                label=f"{ARM_LABEL[arm]} (n={len(case_paths)})",
            )
            marg_lines.append(f"{arm}: {float(np.median(marg_per)):.3f}")

        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(PRETTY_NAME[sim])
            continue

        ax.axhline(target, ls="--", color="gray", alpha=0.7,
                   label=f"target $1-\\alpha={target:.2f}$")

        ax.set_title(
            f"{PRETTY_NAME[sim]}\n"
            f"marginal cov (median):  " + " | ".join(marg_lines)
        )
        ax.set_xlabel("x")
        ax.set_ylabel("conditional coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(x_lo, x_hi)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Exp2: fixed h vs oracle adaptive h(x) = c·s(x) — conditional coverage curves",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    out_png = Path(args.out_png) if args.out_png else out_dir / "exp2_coverage_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
