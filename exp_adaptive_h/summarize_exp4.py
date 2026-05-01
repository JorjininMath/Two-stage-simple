"""
summarize_exp4.py

Aggregate Exp4 (plug-in vs fixed vs oracle adaptive h) into:
  1. Per (simulator, budget, arm) summary table.
  2. Paired-deltas table (within each macrorep, plug-in vs oracle):
       - |cov_plug - cov_oracle|              (Gap Theorem target)
       - q_plug / q_oracle                    (Exp4b prediction P2a/P2b)
       - mean_h_plug / mean_h_oracle
       - width ratios
  3. Conditional-coverage bin table per (simulator, budget, arm) for plot_exp4b.

Reads:
    exp_adaptive_h/output_exp4/exp4_per_arm.csv          (one row per macrorep,sim,budget,arm)
    exp_adaptive_h/output_exp4/macrorep_*/budget_*/case_*/per_point.csv
                                                          (per-test-point detail)

Writes:
    exp_adaptive_h/output_exp4/exp4_summary_arm.csv      (mean/sd of metrics per cell)
    exp_adaptive_h/output_exp4/exp4_paired_deltas.csv    (per macrorep paired comparisons)
    exp_adaptive_h/output_exp4/exp4_paired_summary.csv   (median/IQR of paired deltas)
    exp_adaptive_h/output_exp4/exp4_table.tex            (LaTeX summary)

Usage (from project root):
    python exp_adaptive_h/summarize_exp4.py
    python exp_adaptive_h/summarize_exp4.py --output_dir exp_adaptive_h/output_exp4 --alpha 0.1
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

ARMS = ["fixed", "plugin", "oracle"]


def _summarize_arm_table(per_arm: pd.DataFrame) -> pd.DataFrame:
    g = per_arm.groupby(["simulator", "budget", "arm"], sort=False)
    out = g.agg(
        n_macroreps=("coverage", "count"),
        mean_coverage=("coverage", "mean"),
        sd_coverage=("coverage", "std"),
        median_coverage=("coverage", "median"),
        mean_width=("width", "mean"),
        sd_width=("width", "std"),
        mean_interval_score=("interval_score", "mean"),
        sd_interval_score=("interval_score", "std"),
        mean_q_hat=("q_hat", "mean"),
        sd_q_hat=("q_hat", "std"),
        mean_h=("mean_h_query", "mean"),
    ).reset_index()
    return out


def _paired_deltas(per_arm: pd.DataFrame) -> pd.DataFrame:
    """For each (macrorep, simulator, budget): join plug-in & oracle and compute paired deltas."""
    pivot_cov = per_arm.pivot_table(
        index=["macrorep", "simulator", "budget"],
        columns="arm", values="coverage",
    )
    pivot_q = per_arm.pivot_table(
        index=["macrorep", "simulator", "budget"],
        columns="arm", values="q_hat",
    )
    pivot_h = per_arm.pivot_table(
        index=["macrorep", "simulator", "budget"],
        columns="arm", values="mean_h_query",
    )
    pivot_w = per_arm.pivot_table(
        index=["macrorep", "simulator", "budget"],
        columns="arm", values="width",
    )
    pivot_is = per_arm.pivot_table(
        index=["macrorep", "simulator", "budget"],
        columns="arm", values="interval_score",
    )

    out = pd.DataFrame(index=pivot_cov.index)
    out["cov_fixed"]  = pivot_cov.get("fixed")
    out["cov_plugin"] = pivot_cov.get("plugin")
    out["cov_oracle"] = pivot_cov.get("oracle")

    out["abs_cov_gap_plugin_oracle"] = (out["cov_plugin"] - out["cov_oracle"]).abs()
    out["abs_cov_gap_plugin_fixed"]  = (out["cov_plugin"] - out["cov_fixed"]).abs()

    out["q_plugin"] = pivot_q.get("plugin")
    out["q_oracle"] = pivot_q.get("oracle")
    out["q_ratio_plugin_over_oracle"] = out["q_plugin"] / out["q_oracle"]

    out["h_plugin_mean"] = pivot_h.get("plugin")
    out["h_oracle_mean"] = pivot_h.get("oracle")
    out["h_ratio_plugin_over_oracle"] = out["h_plugin_mean"] / out["h_oracle_mean"]

    out["width_plugin"] = pivot_w.get("plugin")
    out["width_oracle"] = pivot_w.get("oracle")
    out["width_ratio_plugin_over_oracle"] = out["width_plugin"] / out["width_oracle"]

    out["is_plugin"] = pivot_is.get("plugin")
    out["is_oracle"] = pivot_is.get("oracle")

    return out.reset_index()


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    g = paired.groupby(["simulator", "budget"], sort=False)
    rows = []
    for (sim, B), df in g:
        rows.append({
            "simulator":               sim,
            "budget":                  B,
            "n_macroreps":             int(df.shape[0]),
            "median_abs_cov_gap":      float(df["abs_cov_gap_plugin_oracle"].median()),
            "q1_abs_cov_gap":          float(df["abs_cov_gap_plugin_oracle"].quantile(0.25)),
            "q3_abs_cov_gap":          float(df["abs_cov_gap_plugin_oracle"].quantile(0.75)),
            "max_abs_cov_gap":         float(df["abs_cov_gap_plugin_oracle"].max()),
            "median_q_ratio":          float(df["q_ratio_plugin_over_oracle"].median()),
            "q1_q_ratio":              float(df["q_ratio_plugin_over_oracle"].quantile(0.25)),
            "q3_q_ratio":              float(df["q_ratio_plugin_over_oracle"].quantile(0.75)),
            "median_h_ratio":          float(df["h_ratio_plugin_over_oracle"].median()),
            "median_width_ratio":      float(df["width_ratio_plugin_over_oracle"].median()),
            "mean_cov_plugin":         float(df["cov_plugin"].mean()),
            "mean_cov_oracle":         float(df["cov_oracle"].mean()),
            "mean_cov_fixed":          float(df["cov_fixed"].mean()),
        })
    return pd.DataFrame(rows)


def _to_latex(arm_summary: pd.DataFrame, target: float) -> str:
    header = (
        r"\begin{tabular}{llrrrrrr}" "\n"
        r"\toprule" "\n"
        r"sim & arm & budget & cov & width & IS & q\_hat & h \\" "\n"
        r"\midrule" "\n"
    )
    body = []
    for _, r in arm_summary.iterrows():
        body.append(
            f"{r['simulator']} & {r['arm']} & {int(r['budget'])} "
            f"& {r['mean_coverage']:.3f} ({r['sd_coverage']:.3f}) "
            f"& {r['mean_width']:.3f} ({r['sd_width']:.3f}) "
            f"& {r['mean_interval_score']:.3f} ({r['sd_interval_score']:.3f}) "
            f"& {r['mean_q_hat']:.3f} "
            f"& {r['mean_h']:.3f} \\\\"
        )
    footer = r"\bottomrule" "\n" r"\end{tabular}" "\n"
    caption = f"% Exp4 per-arm summary. Target marginal coverage = {target:.2f}.\n"
    return caption + header + "\n".join(body) + "\n" + footer


def main():
    parser = argparse.ArgumentParser(description="Summarize Exp4 (plug-in adaptive h)")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp4")
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = (
        (_root / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    per_arm_path = out_dir / "exp4_per_arm.csv"
    if not per_arm_path.exists():
        print(f"ERROR: {per_arm_path} not found. Run run_exp4_plugin.py first.",
              file=sys.stderr)
        sys.exit(1)

    per_arm = pd.read_csv(per_arm_path)
    print(f"Loaded {len(per_arm)} arm rows from {per_arm_path}")

    arm_summary = _summarize_arm_table(per_arm)
    arm_summary_path = out_dir / "exp4_summary_arm.csv"
    arm_summary.to_csv(arm_summary_path, index=False)
    print(f"Saved {arm_summary_path}")

    paired = _paired_deltas(per_arm)
    paired_path = out_dir / "exp4_paired_deltas.csv"
    paired.to_csv(paired_path, index=False)
    print(f"Saved {paired_path}")

    paired_sum = _paired_summary(paired)
    paired_sum_path = out_dir / "exp4_paired_summary.csv"
    paired_sum.to_csv(paired_sum_path, index=False)
    print(f"Saved {paired_sum_path}")

    tex = _to_latex(arm_summary, target=1.0 - args.alpha)
    tex_path = out_dir / "exp4_table.tex"
    tex_path.write_text(tex)
    print(f"Saved {tex_path}")

    print("\n=== Per-arm summary ===")
    show_cols = [
        "simulator", "budget", "arm", "n_macroreps",
        "mean_coverage", "mean_width", "mean_interval_score",
        "mean_q_hat", "mean_h",
    ]
    print(arm_summary[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Paired (plug-in vs oracle) summary ===")
    print(paired_sum.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
