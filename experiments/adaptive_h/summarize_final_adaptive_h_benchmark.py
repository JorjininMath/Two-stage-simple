"""Summarize completed final adaptive-h benchmark outputs.

This script is intentionally read-only with respect to per-job results. It
rebuilds the aggregate tables from checkpointed job files, writes a main
budget table, and creates compact advisor-facing Markdown and LaTeX tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (_ROOT / "src", _ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import pandas as pd

from experiments.adaptive_h.run_final_adaptive_h_benchmark import (
    _atomic_csv,
    _paired_deltas,
    _summary,
)

DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")
ARM_LABELS = {
    "fixed": "Fixed",
    "plugin_sd_nw": "Sample-SD + NW",
    "oracle": "Oracle",
}
DGP_LABELS = {
    "mm1_sojourn": "M/M/1 sojourn time",
    "raised_floor_gauss": "Raised-floor Gaussian",
    "raised_floor_t3": "Raised-floor Student-t3",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def collect_per_arm(output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "jobs").glob("*/budget_*/macrorep_*/per_arm.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No completed per-arm jobs under {output_dir}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    return frame.sort_values(
        ["simulator", "budget", "macrorep", "arm"]
    ).reset_index(drop=True)


def main_table(summary: pd.DataFrame, budget: int) -> pd.DataFrame:
    table = summary.loc[summary["budget"] == budget].copy()
    if table.empty:
        raise ValueError(f"No summary rows found for main budget B={budget}")
    table["DGP"] = table["simulator"].map(DGP_LABELS).fillna(table["simulator"])
    table["Method"] = table["arm"].map(ARM_LABELS).fillna(table["arm"])
    columns = {
        "mean_coverage": "Coverage",
        "mcse_coverage": "Coverage MCSE",
        "mean_width": "Width",
        "mcse_width": "Width MCSE",
        "mean_interval_score": "Interval score",
        "mcse_interval_score": "Interval score MCSE",
        "mean_mean_group_coverage_gap": "Mean group gap",
        "mean_worst_group_coverage_gap": "Worst group gap",
    }
    result = table[["DGP", "Method", *columns]].rename(columns=columns)
    return result.sort_values(["DGP", "Method"]).reset_index(drop=True)


def _format_cell(mean: float, mcse: float) -> str:
    return f"{mean:.3f} ({mcse:.3f})"


def advisor_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table[["DGP", "Method"]].copy()
    result["Coverage (MCSE)"] = [
        _format_cell(mean, mcse)
        for mean, mcse in zip(table["Coverage"], table["Coverage MCSE"])
    ]
    result["Width (MCSE)"] = [
        _format_cell(mean, mcse)
        for mean, mcse in zip(table["Width"], table["Width MCSE"])
    ]
    result["Interval score (MCSE)"] = [
        _format_cell(mean, mcse)
        for mean, mcse in zip(
            table["Interval score"], table["Interval score MCSE"]
        )
    ]
    result["Mean group gap"] = table["Mean group gap"].map(lambda x: f"{x:.3f}")
    result["Worst group gap"] = table["Worst group gap"].map(
        lambda x: f"{x:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--main-budget", "--main_budget", type=int, default=1000)
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    per_arm = collect_per_arm(output_dir)
    summary = _summary(per_arm)
    paired = _paired_deltas(per_arm)
    _atomic_csv(per_arm, output_dir / "per_arm.csv")
    _atomic_csv(summary, output_dir / "summary.csv")
    _atomic_csv(paired, output_dir / "paired_deltas.csv")

    table = main_table(summary, args.main_budget)
    advisor = advisor_table(table)
    _atomic_csv(table, output_dir / f"main_table_B{args.main_budget}.csv")
    (output_dir / f"main_table_B{args.main_budget}.md").write_text(
        advisor.to_markdown(index=False) + "\n"
    )
    latex = advisor.to_latex(
        index=False,
        escape=True,
        caption=(
            "Final adaptive-bandwidth benchmark at "
            f"$B={args.main_budget}$. Parentheses report Monte Carlo "
            "standard errors."
        ),
        label="tab:final-adaptive-h",
    )
    (output_dir / f"main_table_B{args.main_budget}.tex").write_text(latex)
    print(
        f"Summarized {per_arm['macrorep'].nunique()} macroreps; "
        f"wrote {output_dir / f'main_table_B{args.main_budget}.md'}"
    )


if __name__ == "__main__":
    main()
