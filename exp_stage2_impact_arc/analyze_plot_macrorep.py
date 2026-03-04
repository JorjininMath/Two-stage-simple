"""
Plot coverage and width vs x for one macrorep and one case.
Produces two figures: (1) coverage vs x, (2) width vs x; each with 3 methods (CKME, DCP-DR, hetGP).
Reads macrorep_<id>.xlsx from output/, writes to analysis_output/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COV_COL = {"CKME": "covered_interval", "DCP-DR": "covered_interval_dr", "hetGP": "covered_interval_hetgp"}
WIDTH_COL = {"CKME": "width", "DCP-DR": "width_dr", "hetGP": "width_hetgp"}


def _pick_x_col(df: pd.DataFrame, x_col: str) -> str:
    if x_col in df.columns:
        return x_col
    # Fallback: first column like x0, x1, ...
    for c in df.columns:
        if c.startswith("x") and c[1:].isdigit():
            return c
    raise ValueError(f"No x column '{x_col}' and no x0,x1,... in {list(df.columns)}")


def run(
    macrorep_id: int,
    case: str,
    input_dir: Path,
    output_dir: Path,
    x_col: str = "x0",
    smooth_window: int | None = 31,
) -> tuple[Path, Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = input_dir / f"macrorep_{macrorep_id}.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Not found: {xlsx_path}")

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    if case not in sheets:
        raise KeyError(f"Case '{case}' not in workbook. Available: {sorted(sheets.keys())}")

    df = sheets[case]
    x_name = _pick_x_col(df, x_col)
    x_vals = df[x_name].values
    order = np.argsort(x_vals)
    x_sorted = x_vals[order]

    methods = list(COV_COL.keys())
    prefix = f"stage2_macrorep_{macrorep_id}_case_{case}"

    # ---- Figure 1: Coverage vs x (3 methods) ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for method in methods:
        col = COV_COL[method]
        if col not in df.columns:
            continue
        y = df[col].values[order]
        if smooth_window and len(y) >= smooth_window:
            ys = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean()
            ax1.plot(x_sorted, ys, lw=2, label=method)
        else:
            ax1.plot(x_sorted, y, lw=1, alpha=0.7, label=method)
    ax1.set_xlabel(x_name)
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"Coverage vs {x_name} — macrorep_{macrorep_id}, case {case}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path_cov = output_dir / f"{prefix}_cov.png"
    fig1.savefig(path_cov, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ---- Figure 2: Width vs x (3 methods) ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for method in methods:
        col = WIDTH_COL[method]
        if col not in df.columns:
            continue
        y = df[col].values[order]
        if smooth_window and len(y) >= smooth_window:
            ys = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean()
            ax2.plot(x_sorted, ys, lw=2, label=method)
        else:
            ax2.plot(x_sorted, y, lw=1, alpha=0.7, label=method)
    ax2.set_xlabel(x_name)
    ax2.set_ylabel("Width")
    ax2.set_title(f"Width vs {x_name} — macrorep_{macrorep_id}, case {case}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path_width = output_dir / f"{prefix}_width.png"
    fig2.savefig(path_width, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return path_cov, path_width


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage and width vs x for one macrorep and one case (two figs, 3 methods each)"
    )
    parser.add_argument("--macrorep_id", type=int, default=0, help="Which macrorep (default: 0)")
    parser.add_argument("--case", type=str, default="n1000_r5_lhs",
                        help="Case sheet name, e.g. n1000_r5_lhs, n250_r20_mixed (default: n1000_r5_lhs)")
    parser.add_argument("--x_col", type=str, default="x0",
                        help="Column for x-axis, e.g. x0, x1, y (default: x0)")
    parser.add_argument("--smooth", type=int, default=31,
                        help="Rolling window for smoothed line; 0 to disable (default: 31)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory with macrorep_<id>.xlsx (default: exp_stage2_impact_arc/output)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write figures (default: exp_stage2_impact_arc/analysis_output)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir) if args.input_dir else base / "output"
    output_dir = Path(args.output_dir) if args.output_dir else base / "analysis_output"

    path_cov, path_width = run(
        args.macrorep_id,
        args.case,
        input_dir,
        output_dir,
        x_col=args.x_col,
        smooth_window=args.smooth if args.smooth > 0 else None,
    )
    print(f"Wrote {path_cov}")
    print(f"Wrote {path_width}")


if __name__ == "__main__":
    main()
