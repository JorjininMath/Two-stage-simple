"""
evaluate.py

Evaluation utilities for the C-MAPSS experiment.
Produces formatted Table 1 (overall) and Table 2 (high-risk subset) from saved predictions.

Usage (standalone)
------------------
python exp_cmapss/evaluate.py --pred exp_cmapss/output/tables/cmapss_predictions.csv \
                               --alpha 0.1 --late_rul 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from CP.evaluation import compute_interval_score


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def compute_metrics(
    Y_true: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    alpha: float,
    Y_pred: np.ndarray | None = None,
) -> dict:
    """
    Compute standard metrics for one method on one subset.

    Returns
    -------
    dict with: coverage, width, interval_score, rmse (nan if Y_pred not given)
    """
    Y_true = np.asarray(Y_true, dtype=float)
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)

    covered = (Y_true >= L) & (Y_true <= U)
    coverage = float(covered.mean())
    width = float((U - L).mean())
    _, is_mean = compute_interval_score(Y_true, L, U, alpha)

    if Y_pred is not None:
        Y_pred = np.asarray(Y_pred, dtype=float)
        rmse = float(np.sqrt(np.mean((Y_pred - Y_true) ** 2)))
        mae  = float(np.mean(np.abs(Y_pred - Y_true)))
    else:
        rmse = float("nan")
        mae  = float("nan")

    return dict(
        n=int(len(Y_true)),
        coverage=coverage,
        width=width,
        interval_score=is_mean,
        rmse=rmse,
        mae=mae,
    )


# ---------------------------------------------------------------------------
# Build summary tables from predictions CSV
# ---------------------------------------------------------------------------

def build_tables(
    pred_df: pd.DataFrame,
    alpha: float = 0.1,
    late_rul: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Table 1 (overall) and Table 2 (late-stage) from predictions DataFrame.

    Expected columns in pred_df:
        rul_true, rul_pred_ckme, L_stage1, U_stage1, L_stage2, U_stage2
        optionally: rul_pred_ridge, L_ridge, U_ridge

    Returns
    -------
    table_all : pd.DataFrame  (overall metrics)
    table_late : pd.DataFrame (late-stage metrics, RUL <= late_rul)
    """
    Y_true = pred_df["rul_true"].values

    methods = []

    # Ridge baselines (optional)
    if "L_ridge" in pred_df.columns:
        methods.append((
            "Ridge (point only)",
            pred_df["rul_pred_ridge"].values,  # L = U = point pred for zero-width
            pred_df["rul_pred_ridge"].values,
            pred_df["rul_pred_ridge"].values,
        ))
        methods.append((
            "Ridge + split CP",
            pred_df["rul_pred_ridge"].values,
            pred_df["L_ridge"].values,
            pred_df["U_ridge"].values,
        ))

    # CKME methods
    methods.append((
        "CKME Stage 1 CP",
        pred_df["rul_pred_ckme"].values,
        pred_df["L_stage1"].values,
        pred_df["U_stage1"].values,
    ))
    methods.append((
        "CKME Stage 2 CP",
        pred_df["rul_pred_ckme"].values,
        pred_df["L_stage2"].values,
        pred_df["U_stage2"].values,
    ))

    mask_late = Y_true <= late_rul
    rows_all, rows_late = [], []

    for method_name, Y_pred, L, U in methods:
        m_all = compute_metrics(Y_true, L, U, alpha, Y_pred)
        m_all["method"] = method_name
        rows_all.append(m_all)

        if mask_late.any():
            m_late = compute_metrics(Y_true[mask_late], L[mask_late], U[mask_late],
                                     alpha, Y_pred[mask_late])
            m_late["method"] = method_name
            rows_late.append(m_late)

    col_order = ["method", "n", "coverage", "width", "interval_score", "rmse", "mae"]

    table_all = pd.DataFrame(rows_all)[col_order].round(4)
    table_late = pd.DataFrame(rows_late)[col_order].round(4) if rows_late else pd.DataFrame()

    return table_all, table_late


# ---------------------------------------------------------------------------
# RUL-bin analysis: coverage and width per RUL bucket
# ---------------------------------------------------------------------------

def rul_bin_analysis(
    pred_df: pd.DataFrame,
    alpha: float = 0.1,
    bins: list[int] | None = None,
) -> pd.DataFrame:
    """
    Coverage and width of Stage 1 and Stage 2 broken down by RUL bins.
    Useful for showing Stage 2 improves in the high-risk region.
    """
    if bins is None:
        bins = [0, 10, 20, 30, 50, 75, 100, 125, 9999]

    Y_true = pred_df["rul_true"].values
    rows = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (Y_true > lo) & (Y_true <= hi)
        if mask.sum() == 0:
            continue
        label = f"({lo}, {hi}]" if hi < 9999 else f">{lo}"
        for method_name, L_col, U_col in [
            ("Stage 1", "L_stage1", "U_stage1"),
            ("Stage 2", "L_stage2", "U_stage2"),
        ]:
            L, U = pred_df[L_col].values[mask], pred_df[U_col].values[mask]
            m = compute_metrics(Y_true[mask], L, U, alpha)
            rows.append(dict(
                rul_bin=label,
                method=method_name,
                n=m["n"],
                coverage=round(m["coverage"], 4),
                width=round(m["width"], 4),
                interval_score=round(m["interval_score"], 4),
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="exp_cmapss/output/tables/cmapss_predictions.csv")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--late_rul", type=int, default=30)
    parser.add_argument("--output_dir", default="exp_cmapss/output/tables")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_all, table_late = build_tables(pred_df, alpha=args.alpha, late_rul=args.late_rul)
    bin_df = rul_bin_analysis(pred_df, alpha=args.alpha)

    print("\n=== Table 1: Overall Performance ===")
    print(table_all.to_string(index=False))

    print(f"\n=== Table 2: High-Risk Subset (RUL <= {args.late_rul}) ===")
    print(table_late.to_string(index=False))

    print("\n=== RUL-Bin Analysis ===")
    print(bin_df.to_string(index=False))

    table_all.to_csv(output_dir / "table1_overall.csv", index=False)
    table_late.to_csv(output_dir / "table2_highrisk.csv", index=False)
    bin_df.to_csv(output_dir / "rul_bin_analysis.csv", index=False)
    print(f"\nSaved tables to {output_dir}/")


if __name__ == "__main__":
    main()
