"""
Plot coverage and width vs x for one macrorep in exp2_test.

Produces two figures:
  1) Coverage vs x (CKME, DCP-DR, hetGP)
  2) Width vs x (CKME, DCP-DR, hetGP)

It re-runs a single macrorep using the same pipeline as run_exp2_compare.py,
so it uses the current exp2_test noise and settings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from exp2_test.config_utils import load_config_from_file, get_config
from exp2_test.run_exp2_compare import (
    METHOD_COV,
    METHOD_WIDTH,
    run_one_macrorep,
)


def _pick_x_col(df: pd.DataFrame, x_col: str) -> str:
    if x_col in df.columns:
        return x_col
    for c in df.columns:
        if c.startswith("x") and c[1:].isdigit():
            return c
    raise ValueError(f"No x column '{x_col}' and no x0,x1,... in {list(df.columns)}")


def run_plot(
    macrorep_id: int,
    config_path: Path,
    out_dir: Path,
    method: str,
    x_col: str = "x0",
    smooth_window: int | None = 31,
) -> tuple[Path, Path]:
    config = load_config_from_file(config_path)
    config = get_config(config, quick=False)
    n_grid = config.get("t_grid_size", 500)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, df = run_one_macrorep(
        macrorep_id=macrorep_id,
        base_seed=42,
        config=config,
        out_dir=out_dir,
        method=method,
        n_grid=n_grid,
        return_df=True,
    )

    x_name = _pick_x_col(df, x_col)
    x_vals = df[x_name].values
    order = np.argsort(x_vals)
    x_sorted = x_vals[order]

    methods = list(METHOD_COV.keys())
    prefix = f"exp2_macrorep_{macrorep_id}_{method}"

    # Figure 1: Coverage vs x
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for m in methods:
        col = METHOD_COV[m]
        if col not in df.columns:
            continue
        y = df[col].values[order]
        if smooth_window and len(y) >= smooth_window:
            ys = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean()
            ax1.plot(x_sorted, ys, lw=2, label=m)
        else:
            ax1.plot(x_sorted, y, lw=1, alpha=0.7, label=m)
    ax1.set_xlabel(x_name)
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"Coverage vs {x_name} — macrorep_{macrorep_id}, method {method}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path_cov = out_dir / f"{prefix}_cov.png"
    fig1.savefig(path_cov, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Width vs x
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for m in methods:
        col = METHOD_WIDTH[m]
        if col not in df.columns:
            continue
        y = df[col].values[order]
        if smooth_window and len(y) >= smooth_window:
            ys = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean()
            ax2.plot(x_sorted, ys, lw=2, label=m)
        else:
            ax2.plot(x_sorted, y, lw=1, alpha=0.7, label=m)
    ax2.set_xlabel(x_name)
    ax2.set_ylabel("Width")
    ax2.set_title(f"Width vs {x_name} — macrorep_{macrorep_id}, method {method}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path_width = out_dir / f"{prefix}_width.png"
    fig2.savefig(path_width, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return path_cov, path_width


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage and width vs x for one macrorep in exp2_test (CKME, DCP-DR, hetGP)."
    )
    parser.add_argument("--macrorep_id", type=int, default=0, help="Which macrorep (default: 0)")
    parser.add_argument("--config", type=str, default="exp2_test/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="lhs", choices=("lhs", "sampling", "mixed"))
    parser.add_argument("--x_col", type=str, default="x0", help="Column for x-axis (default: x0)")
    parser.add_argument(
        "--smooth", type=int, default=31, help="Rolling window for smoothed line; 0 to disable (default: 31)"
    )
    args = parser.parse_args()

    config_path = _root / args.config
    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp2_test" / "output"

    smooth_window = args.smooth if args.smooth > 0 else None
    path_cov, path_width = run_plot(
        macrorep_id=args.macrorep_id,
        config_path=config_path,
        out_dir=out_dir,
        method=args.method,
        x_col=args.x_col,
        smooth_window=smooth_window,
    )
    print(f"Saved {path_cov}")
    print(f"Saved {path_width}")


if __name__ == "__main__":
    main()

