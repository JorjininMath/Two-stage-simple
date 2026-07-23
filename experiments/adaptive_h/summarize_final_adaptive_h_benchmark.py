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

import numpy as np
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
        "mean_coverage": "Raw-score coverage",
        "mcse_coverage": "Raw-score coverage MCSE",
        "mean_coverage_interval": "Projected-interval coverage",
        "mcse_coverage_interval": "Projected-interval coverage MCSE",
        "mean_score_interval_disagreement": "Score-set/interval disagreement",
        "mean_width": "Width",
        "mcse_width": "Width MCSE",
        "mean_interval_score": "Interval score",
        "mcse_interval_score": "Interval score MCSE",
        "mean_worst_group_coverage_gap": "Worst group gap",
    }
    result = table[["DGP", "Method", *columns]].rename(columns=columns)
    return result.sort_values(["DGP", "Method"]).reset_index(drop=True)


def _format_cell(mean: float, mcse: float) -> str:
    return f"{mean:.3f} ({mcse:.3f})"


def advisor_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table[["DGP", "Method"]].copy()
    result["Raw-score cov. (MCSE)"] = [
        _format_cell(mean, mcse)
        for mean, mcse in zip(
            table["Raw-score coverage"],
            table["Raw-score coverage MCSE"],
        )
    ]
    result["Interval cov. (MCSE)"] = [
        _format_cell(mean, mcse)
        for mean, mcse in zip(
            table["Projected-interval coverage"],
            table["Projected-interval coverage MCSE"],
        )
    ]
    result["Disagreement"] = table["Score-set/interval disagreement"].map(
        lambda x: f"{x:.3f}"
    )
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
    result["Worst group gap"] = table["Worst group gap"].map(
        lambda x: f"{x:.3f}"
    )
    return result


def scale_diagnostics(per_arm: pd.DataFrame) -> pd.DataFrame:
    plugin = per_arm.loc[per_arm["arm"] == "plugin_sd_nw"].copy()
    rows = []
    for key, group in plugin.groupby(["simulator", "budget"], sort=True):
        row = {"simulator": key[0], "budget": int(key[1])}
        row["n_macroreps"] = int(group["macrorep"].nunique())
        for metric in (
            "mean_abs_scale_relative_error",
            "mean_h_over_s",
            "sd_h_over_s",
        ):
            values = group[metric].astype(float)
            sd = float(values.std(ddof=1))
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = sd
            row[f"mcse_{metric}"] = sd / np.sqrt(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


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
    _atomic_csv(
        scale_diagnostics(per_arm),
        output_dir / "scale_diagnostics_summary.csv",
    )

    table = main_table(summary, args.main_budget)
    advisor = advisor_table(table)
    _atomic_csv(table, output_dir / f"main_table_B{args.main_budget}.csv")
    (output_dir / f"main_table_B{args.main_budget}.md").write_text(
        advisor.to_markdown(index=False) + "\n"
    )
    latex_advisor = advisor.copy()
    latex_advisor["DGP"] = latex_advisor["DGP"].replace(
        {
            "M/M/1 sojourn time": "M/M/1",
            "Raised-floor Gaussian": "Gaussian",
            "Raised-floor Student-t3": r"Student-$t_3$",
        }
    )
    latex_advisor["Method"] = latex_advisor["Method"].replace(
        {"Sample-SD + NW": "Plug-in"}
    )
    latex_advisor = latex_advisor.rename(
        columns={
            "Raw-score cov. (MCSE)": "Raw cov. (MCSE)",
            "Interval cov. (MCSE)": "Interval cov. (MCSE)",
            "Disagreement": "Disagree.",
            "Worst group gap": "Worst gap",
        }
    )
    coverage_columns = [
        "DGP",
        "Method",
        "Raw cov. (MCSE)",
        "Interval cov. (MCSE)",
        "Disagree.",
        "Worst gap",
    ]
    interval_columns = [
        "DGP",
        "Method",
        "Width (MCSE)",
        "Interval score (MCSE)",
    ]
    coverage_tabular = latex_advisor[coverage_columns].to_latex(
        index=False,
        escape=False,
        column_format="@{}llcccc@{}",
    )
    interval_tabular = latex_advisor[interval_columns].to_latex(
        index=False,
        escape=False,
        column_format="@{}llcc@{}",
    )
    latex = (
        "\\begin{table}[p]\n"
        "\\centering\n"
        "\\caption{Final adaptive-bandwidth benchmark at "
        f"$B={args.main_budget}$. Parentheses report Monte Carlo "
        "standard errors.}\n"
        "\\label{tab:final-adaptive-h}\n"
        "\\textbf{(a) Coverage and score-set/interval agreement}"
        "\\par\\smallskip\n"
        "\\small\n"
        f"{coverage_tabular}"
        "\\par\\medskip\n"
        "\\textbf{(b) Projected-interval efficiency}"
        "\\par\\smallskip\n"
        f"{interval_tabular}"
        "\\par\\medskip\n"
        "\\begin{minipage}{0.98\\linewidth}\n"
        "\\footnotesize\\textit{Notes.} Raw-score coverage is "
        "guarantee-bearing. Interval coverage, width, and interval score "
        "use the monotone-projected generalized-inverse interval. "
        "Disagreement is the fraction of test points for which raw "
        "score-set membership and projected-interval membership differ. "
        "Worst group gaps use raw-score coverage in ten equal-count input "
        "bins. Gaussian and Student-$t_3$ denote the two raised-floor "
        "DGPs; plug-in denotes sample-SD plus Nadaraya--Watson smoothing.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )
    (output_dir / f"main_table_B{args.main_budget}.tex").write_text(latex)
    print(
        f"Summarized {per_arm['macrorep'].nunique()} macroreps; "
        f"wrote {output_dir / f'main_table_B{args.main_budget}.md'}"
    )


if __name__ == "__main__":
    main()
