"""
Summary table of Stage 2 results: mean coverage, width, interval score by case and method,
aggregated across 50 macroreps. Reads macrorep_*.xlsx from output/, writes CSV to analysis_output/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_COLS = {
    "CKME": ("covered_interval", "width", "interval_score"),
    "DCP-DR": ("covered_interval_dr", "width_dr", "interval_score_dr"),
    "hetGP": ("covered_interval_hetgp", "width_hetgp", "interval_score_hetgp"),
}
# Score coverage (conformal score-based): CKME and DCP-DR have it; hetGP does not.
COV_SCORE_COL = {"CKME": "covered_score", "DCP-DR": "covered_score_dr"}


def load_all_macroreps(input_dir: Path):
    """Load all macrorep_*.xlsx; return list of (macrorep_id, dict[sheet_name -> DataFrame])."""
    input_dir = Path(input_dir)
    results = []
    for p in sorted(input_dir.glob("macrorep_*.xlsx")):
        stem = p.stem
        if not stem.startswith("macrorep_"):
            continue
        try:
            mid = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
        results.append((mid, sheets))
    return results


def mean_per_sheet(df: pd.DataFrame):
    """Per-method mean interval coverage, score coverage (if any), mean width, mean interval_score."""
    out = {}
    for method, (cov_col, width_col, score_col) in METHOD_COLS.items():
        if cov_col in df.columns:
            out[f"{method}_coverage"] = df[cov_col].mean()
        if method in COV_SCORE_COL and COV_SCORE_COL[method] in df.columns:
            out[f"{method}_coverage_score"] = df[COV_SCORE_COL[method]].mean()
        if width_col in df.columns:
            out[f"{method}_width"] = df[width_col].mean()
        if score_col in df.columns:
            out[f"{method}_interval_score"] = df[score_col].mean()
    return out


def run(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    macroreps = load_all_macroreps(input_dir)
    if not macroreps:
        raise FileNotFoundError(f"No macrorep_*.xlsx found in {input_dir}")

    # (case, method) -> list of values over macroreps
    case_method_coverage = {}
    case_method_coverage_score = {}
    case_method_width = {}
    case_method_score = {}

    for _macrorep_id, sheets in macroreps:
        for sheet_name, df in sheets.items():
            s = mean_per_sheet(df)
            for k, v in s.items():
                if k.endswith("_coverage_score"):
                    method = k.replace("_coverage_score", "")
                    case_method_coverage_score.setdefault((sheet_name, method), []).append(v)
                elif k.endswith("_coverage") and not k.endswith("_coverage_score"):
                    method = k.replace("_coverage", "")
                    case_method_coverage.setdefault((sheet_name, method), []).append(v)
                elif k.endswith("_width"):
                    method = k.replace("_width", "")
                    case_method_width.setdefault((sheet_name, method), []).append(v)
                elif k.endswith("_interval_score"):
                    method = k.replace("_interval_score", "")
                    case_method_score.setdefault((sheet_name, method), []).append(v)

    cases = sorted({c for c, _ in case_method_coverage})
    methods = sorted({m for _, m in case_method_coverage})

    rows = []
    for case in cases:
        for method in methods:
            cov_list = case_method_coverage.get((case, method), [])
            cov_score_list = case_method_coverage_score.get((case, method), [])
            w_list = case_method_width.get((case, method), [])
            sc_list = case_method_score.get((case, method), [])
            rows.append({
                "case": case,
                "method": method,
                "mean_coverage_interval": np.mean(cov_list) if cov_list else np.nan,
                "sd_coverage_interval": np.std(cov_list, ddof=1) if len(cov_list) > 1 else np.nan,
                "mean_coverage_score": np.mean(cov_score_list) if cov_score_list else np.nan,
                "sd_coverage_score": np.std(cov_score_list, ddof=1) if len(cov_score_list) > 1 else np.nan,
                "mean_width": np.mean(w_list) if w_list else np.nan,
                "sd_width": np.std(w_list, ddof=1) if len(w_list) > 1 else np.nan,
                "mean_interval_score": np.mean(sc_list) if sc_list else np.nan,
                "sd_interval_score": np.std(sc_list, ddof=1) if len(sc_list) > 1 else np.nan,
                "n_macroreps": len(cov_list),
            })
    summary = pd.DataFrame(rows)
    out_path = output_dir / "stage2_summary_table.csv"
    summary.to_csv(out_path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summary table: mean cov, width, score across macroreps")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory with macrorep_*.xlsx (default: exp_stage2_impact_arc/output)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write stage2_summary_table.csv (default: exp_stage2_impact_arc/analysis_output)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir) if args.input_dir else base / "output"
    output_dir = Path(args.output_dir) if args.output_dir else base / "analysis_output"

    summary = run(input_dir, output_dir)
    out_path = output_dir / "stage2_summary_table.csv"
    print(f"Wrote {out_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
