"""
plot_nongauss_A1L_compare.py

Single-panel conditional coverage plot for the nongauss_A1L DGP, comparing CKME,
DCP-DR and hetGP across binned x. Mirrors the standalone style used by
exp_adaptive_h/plot_exp2.py: 6.5x4.5 figure, no title, no suptitle, median
curve plus 5-95% inter-macrorep band per method, nominal-coverage dashed line.

Reads:
    exp_nongauss/output/macrorep_{k}/case_{sim}_{method}/per_point.csv

Writes:
    exp_nongauss/output/nongauss_compare_coverage_curves_{sim}.png

Usage (from project root):
    python exp_nongauss/plot_nongauss_A1L_compare.py
    python exp_nongauss/plot_nongauss_A1L_compare.py --simulator nongauss_A1L --n_bins 20
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

METHOD_COL = {
    "CKME":   "covered_interval",
    "DCP-DR": "covered_interval_dr",
    "hetGP":  "covered_interval_hetgp",
}
METHOD_COLOR = {
    "CKME":   "tab:blue",
    "DCP-DR": "tab:green",
    "hetGP":  "tab:orange",
}
METHOD_MARKER = {
    "CKME":   "o",
    "DCP-DR": "s",
    "hetGP":  "^",
}


def _load_case_paths(out_dir: Path, sim: str, method: str) -> list[Path]:
    pattern = f"macrorep_*/case_{sim}_{method}/per_point.csv"
    return sorted(out_dir.glob(pattern))


def _bin_coverage(df: pd.DataFrame, col: str, bin_edges: np.ndarray) -> np.ndarray:
    x = df["x0"].to_numpy()
    cov = df[col].to_numpy()
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = cov[mask].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="nongauss coverage(x) comparison plot")
    parser.add_argument("--output_dir", type=str, default="exp_nongauss/output")
    parser.add_argument("--simulator",  type=str, default="nongauss_A1L")
    parser.add_argument("--method",     type=str, default="lhs")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--out_png",    type=str, default=None)
    args = parser.parse_args()

    out_dir = (
        (_root / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    cfg = get_experiment_config(args.simulator)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, args.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    target = 1.0 - args.alpha

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    case_paths = _load_case_paths(out_dir, args.simulator, args.method)
    if not case_paths:
        print(
            f"ERROR: no per_point.csv files matched "
            f"case_{args.simulator}_{args.method}",
            file=sys.stderr,
        )
        sys.exit(1)

    for method_name, col in METHOD_COL.items():
        cov_per = np.empty((len(case_paths), args.n_bins))
        for i, p in enumerate(case_paths):
            df = pd.read_csv(p)
            if col not in df.columns:
                print(f"WARN: column {col} missing in {p}", file=sys.stderr)
                cov_per[i, :] = np.nan
                continue
            cov_per[i] = _bin_coverage(df, col, bin_edges)

        med = np.nanmedian(cov_per, axis=0)
        lo = np.nanpercentile(cov_per, 5, axis=0)
        hi = np.nanpercentile(cov_per, 95, axis=0)

        color = METHOD_COLOR[method_name]
        ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.15)
        ax.plot(
            bin_centers, med, color=color, lw=2,
            marker=METHOD_MARKER[method_name], ms=4,
            label=method_name,
        )

    ax.axhline(
        target, ls="--", color="gray", alpha=0.7,
        label=f"target $1-\\alpha={target:.2f}$",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out_png = (
        Path(args.out_png) if args.out_png
        else out_dir / f"nongauss_compare_coverage_curves_{args.simulator}.png"
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
