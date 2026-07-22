"""
make_tables.py

Read wsc_summary.csv and print formatted Tables 2-3.

Table 2: Exp 1 (wsc_gauss,    Gaussian noise)
Table 3: Exp 2 (nongauss_A1L, Student-t nu=3)

Each table has rows = Stage 2 budget (n_1 x r_1) and
columns = method x metric (Coverage / Width / IS).

Usage (from project root):
    python exp_wsc/make_tables.py
    python exp_wsc/make_tables.py --output_dir exp_wsc/output
    python exp_wsc/make_tables.py --format latex   # LaTeX tabular
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np

METHOD_LABELS = {"lhs": "LHS", "sampling": "Adaptive", "mixed": "Mixture"}
DGP_TO_TABLE  = {
    "Exp1_Gaussian":    "Table 2 (Exp 1 — Gaussian noise)",
    "Exp2_Student_t3":  "Table 3 (Exp 2 — Student-t, nu=3)",
}
METRICS = [
    ("CKME_coverage_mean",       "CKME_coverage_std",       "Cov",   "CKME"),
    ("CKME_width_mean",          "CKME_width_std",          "Width", "CKME"),
    ("CKME_interval_score_mean", "CKME_interval_score_std", "IS",    "CKME"),
    ("DCP-DR_coverage_mean",     "DCP-DR_coverage_std",     "Cov",   "DCP-DR"),
    ("DCP-DR_width_mean",        "DCP-DR_width_std",        "Width", "DCP-DR"),
    ("DCP-DR_interval_score_mean","DCP-DR_interval_score_std","IS",  "DCP-DR"),
    ("hetGP_coverage_mean",      "hetGP_coverage_std",      "Cov",   "hetGP"),
    ("hetGP_width_mean",         "hetGP_width_std",         "Width", "hetGP"),
    ("hetGP_interval_score_mean","hetGP_interval_score_std","IS",    "hetGP"),
]


def fmt(mean: float, std: float, metric: str) -> str:
    if np.isnan(mean):
        return "—"
    if metric == "Cov":
        return f"{mean:.3f} ({std:.3f})"
    return f"{mean:.2f} ({std:.2f})"


def print_table(df: pd.DataFrame, title: str, output_format: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    budget_labels = {(100, 50): "n=100, r=50", (200, 25): "n=200, r=25", (500, 10): "n=500, r=10"}

    for method_key, method_label in METHOD_LABELS.items():
        print(f"\n  Method: {method_label}")
        print(f"  {'Budget':<16} {'':>4} {'CKME':^30} {'DCP-DR':^30} {'hetGP':^30}")
        print(f"  {'':16} {'Cov':>10} {'Width':>10} {'IS':>10}"
              f" {'Cov':>10} {'Width':>10} {'IS':>10}"
              f" {'Cov':>10} {'Width':>10} {'IS':>10}")
        print(f"  {'-'*106}")

        sub = df[df["method"] == method_key]
        for (n_1, r_1), blabel in budget_labels.items():
            row = sub[(sub["n_1"] == n_1) & (sub["r_1"] == r_1)]
            if row.empty:
                print(f"  {blabel:<16}  (no data)")
                continue
            row = row.iloc[0]

            cells = []
            for mean_col, std_col, metric, _ in METRICS:
                mean = row.get(mean_col, np.nan)
                std  = row.get(std_col,  np.nan)
                cells.append(fmt(float(mean), float(std), metric))

            print(
                f"  {blabel:<16}"
                f" {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}"
                f" {cells[3]:>10} {cells[4]:>10} {cells[5]:>10}"
                f" {cells[6]:>10} {cells[7]:>10} {cells[8]:>10}"
            )


def main():
    parser = argparse.ArgumentParser(description="Format WSC Tables 2-3")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--format",     type=str, default="text", choices=("text", "latex"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_wsc" / "output"
    csv_path = out_dir / "wsc_summary.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run run_wsc_compare.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    for dgp_label, table_title in DGP_TO_TABLE.items():
        sub = df[df["dgp"] == dgp_label]
        if sub.empty:
            print(f"No data for {dgp_label}.")
            continue
        print_table(sub, table_title, args.format)

    print()


if __name__ == "__main__":
    main()
