"""
summarize_exp2.py

Aggregate Exp2 (fixed h vs oracle adaptive h(x)) into a paired comparison table.

For each DGP, compute across macroreps for BOTH arms (fixed / oracle):
  - marginal_coverage    : median over macroreps
  - worst_bin_dev        : max_b |cov_b - (1-alpha)|, median over macroreps
  - mean_bin_dev         : mean_b |cov_b - (1-alpha)|, median over macroreps
  - coverage_range       : max_b cov_b - min_b cov_b, median
  - mean_width           : E[U-L], mean over macroreps
  - mean_interval_score  : E[IS], mean over macroreps

Decision rule (Exp_plan.md):
  Oracle should beat fixed on >=3 of 4 DGPs in worst-bin dev (or interval score).
  Otherwise the Score Homogeneity story is in trouble.

Reads:
    exp_adaptive_h/output_exp2/macrorep_{k}/case_{sim}_{fixed|oracle}/per_point.csv

Writes:
    exp_adaptive_h/output_exp2/exp2_table.csv
    exp_adaptive_h/output_exp2/exp2_table.tex   (booktabs LaTeX)

Usage (from project root):
    python exp_adaptive_h/summarize_exp2.py
    python exp_adaptive_h/summarize_exp2.py --n_bins 20 --alpha 0.1
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

from Two_stage.sim_functions import get_experiment_config

SIMULATORS = [
    "wsc_gauss",
    "gibbs_s1",
    "exp1",
    "nongauss_A1L",
]

ARMS = ["fixed", "oracle"]


def _bin_coverage(x: np.ndarray, cov: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = cov[mask].mean()
    return out


def _macrorep_metrics(per_point: pd.DataFrame, bin_edges: np.ndarray, target: float) -> dict:
    x = per_point["x0"].to_numpy()
    cov = per_point["covered_score"].to_numpy()
    width = per_point["width"].to_numpy()
    is_ = per_point["interval_score"].to_numpy()

    cov_bins = _bin_coverage(x, cov, bin_edges)
    valid = ~np.isnan(cov_bins)
    devs = np.abs(cov_bins[valid] - target)
    metrics = {
        "marginal":     cov.mean(),
        "worst_bin":    devs.max() if devs.size else np.nan,
        "mean_bin":     devs.mean() if devs.size else np.nan,
        "cov_range":    (cov_bins[valid].max() - cov_bins[valid].min()) if valid.any() else np.nan,
        "mean_width":   width.mean(),
        "mean_is":      is_.mean(),
    }
    if "y_in_grid" in per_point.columns:
        metrics["frac_y_outside"] = float(1.0 - per_point["y_in_grid"].mean())
        metrics["frac_L_clipped"] = float(per_point["L_at_grid_lo"].mean())
        metrics["frac_U_clipped"] = float(per_point["U_at_grid_hi"].mean())
    else:
        metrics["frac_y_outside"] = np.nan
        metrics["frac_L_clipped"] = np.nan
        metrics["frac_U_clipped"] = np.nan
    return metrics


def _summarize_arm(out_dir: Path, sim: str, arm: str, n_bins: int, target: float) -> dict:
    cfg = get_experiment_config(sim)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    case_paths = sorted(out_dir.glob(f"macrorep_*/case_{sim}_{arm}/per_point.csv"))
    if not case_paths:
        return {"simulator": sim, "arm": arm, "n_macroreps": 0}

    per_rep = [
        _macrorep_metrics(pd.read_csv(p), bin_edges, target)
        for p in case_paths
    ]
    df = pd.DataFrame(per_rep)
    return {
        "simulator":              sim,
        "arm":                    arm,
        "n_macroreps":            len(df),
        "marginal_coverage":      float(df["marginal"].median()),
        "marginal_coverage_sd":   float(df["marginal"].std(ddof=1)),
        "worst_bin_dev":          float(df["worst_bin"].median()),
        "worst_bin_dev_sd":       float(df["worst_bin"].std(ddof=1)),
        "mean_bin_dev":           float(df["mean_bin"].median()),
        "mean_bin_dev_sd":        float(df["mean_bin"].std(ddof=1)),
        "coverage_range":         float(df["cov_range"].median()),
        "coverage_range_sd":      float(df["cov_range"].std(ddof=1)),
        "mean_width":             float(df["mean_width"].mean()),
        "mean_width_sd":          float(df["mean_width"].std(ddof=1)),
        "mean_interval_score":    float(df["mean_is"].mean()),
        "mean_interval_score_sd": float(df["mean_is"].std(ddof=1)),
        "frac_y_outside":         float(df["frac_y_outside"].mean()),
        "frac_L_clipped":         float(df["frac_L_clipped"].mean()),
        "frac_U_clipped":         float(df["frac_U_clipped"].mean()),
    }


def _paired_macrorep_deltas(
    out_dir: Path, sim: str, n_bins: int, target: float
) -> pd.DataFrame:
    """Per-macrorep paired deltas (oracle - fixed) for matching macrorep IDs."""
    cfg = get_experiment_config(sim)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    fixed_paths = {
        int(p.parent.parent.name.split("_")[1]): p
        for p in out_dir.glob(f"macrorep_*/case_{sim}_fixed/per_point.csv")
    }
    oracle_paths = {
        int(p.parent.parent.name.split("_")[1]): p
        for p in out_dir.glob(f"macrorep_*/case_{sim}_oracle/per_point.csv")
    }
    common_ks = sorted(set(fixed_paths) & set(oracle_paths))
    rows = []
    for k in common_ks:
        m_fix = _macrorep_metrics(pd.read_csv(fixed_paths[k]), bin_edges, target)
        m_or  = _macrorep_metrics(pd.read_csv(oracle_paths[k]), bin_edges, target)
        rows.append({
            "macrorep":      k,
            "d_marginal":    m_or["marginal"] - m_fix["marginal"],
            "d_worst_bin":   m_or["worst_bin"] - m_fix["worst_bin"],
            "d_mean_bin":    m_or["mean_bin"] - m_fix["mean_bin"],
            "d_width":       m_or["mean_width"] - m_fix["mean_width"],
            "d_is":          m_or["mean_is"] - m_fix["mean_is"],
        })
    return pd.DataFrame(rows)


def _to_latex(rows_long: pd.DataFrame, target: float) -> str:
    """Wide LaTeX table: one row per DGP, columns grouped by metric (fixed vs oracle)."""
    fixed = rows_long[rows_long["arm"] == "fixed"].set_index("simulator")
    oracle = rows_long[rows_long["arm"] == "oracle"].set_index("simulator")

    header = (
        r"\begin{tabular}{l ccc ccc cc}" "\n"
        r"\toprule" "\n"
        r" & \multicolumn{3}{c}{Worst-bin $|c-(1-\alpha)|$} "
        r"& \multicolumn{3}{c}{Mean width} "
        r"& \multicolumn{2}{c}{Marginal cov} \\" "\n"
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}" "\n"
        r"DGP & fixed & oracle & $\Delta$ & fixed & oracle & ratio & fixed & oracle \\" "\n"
        r"\midrule" "\n"
    )
    body_lines = []
    for sim in SIMULATORS:
        if sim not in fixed.index or sim not in oracle.index:
            body_lines.append(f"{sim.replace('_', r'_')} & --- & --- & --- & --- & --- & --- & --- & --- \\\\")
            continue
        f_row, o_row = fixed.loc[sim], oracle.loc[sim]
        d_worst = o_row["worst_bin_dev"] - f_row["worst_bin_dev"]
        ratio_w = (
            o_row["mean_width"] / f_row["mean_width"] if f_row["mean_width"] > 0 else np.nan
        )
        sim_tex = sim.replace("_", r"\_")
        body_lines.append(
            f"{sim_tex} "
            f"& {f_row['worst_bin_dev']:.3f} "
            f"& {o_row['worst_bin_dev']:.3f} "
            f"& {d_worst:+.3f} "
            f"& {f_row['mean_width']:.3f} "
            f"& {o_row['mean_width']:.3f} "
            f"& {ratio_w:.2f} "
            f"& {f_row['marginal_coverage']:.3f} "
            f"& {o_row['marginal_coverage']:.3f} \\\\"
        )
    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    caption = (
        f"% Exp2 paired comparison. Target marginal coverage = {target:.2f}.\n"
        "% Worst-bin / marginal cov: median across macroreps. "
        "Width: mean across macroreps. Delta = oracle - fixed (negative is better).\n"
    )
    return caption + header + "\n".join(body_lines) + "\n" + footer


def main():
    parser = argparse.ArgumentParser(description="Summarize Exp2 fixed vs oracle adaptive h")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp2")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha

    # Long-format table: row per (sim, arm)
    rows = []
    for sim in SIMULATORS:
        for arm in ARMS:
            rows.append(_summarize_arm(out_dir, sim, arm, args.n_bins, target))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp2_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Paired per-macrorep deltas (more informative for inference)
    delta_rows = []
    for sim in SIMULATORS:
        d_df = _paired_macrorep_deltas(out_dir, sim, args.n_bins, target)
        if d_df.empty:
            continue
        delta_rows.append({
            "simulator":          sim,
            "n_paired":           len(d_df),
            "median_d_worst":     float(d_df["d_worst_bin"].median()),
            "median_d_mean_bin":  float(d_df["d_mean_bin"].median()),
            "median_d_marginal":  float(d_df["d_marginal"].median()),
            "median_d_width":     float(d_df["d_width"].median()),
            "median_d_is":        float(d_df["d_is"].median()),
            "frac_oracle_better_worst": float((d_df["d_worst_bin"] < 0).mean()),
            "frac_oracle_better_is":    float((d_df["d_is"] < 0).mean()),
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_csv = out_dir / "exp2_paired_deltas.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"Saved: {delta_csv}")

    tex_path = out_dir / "exp2_table.tex"
    tex_path.write_text(_to_latex(df, target))
    print(f"Saved: {tex_path}")

    show_cols = [
        "simulator", "arm", "n_macroreps",
        "marginal_coverage", "worst_bin_dev", "mean_bin_dev",
        "coverage_range", "mean_width", "mean_interval_score",
    ]
    print("\nPer-arm summary:")
    print(df[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not delta_df.empty:
        print("\nPaired deltas (oracle - fixed; negative = oracle better):")
        print(delta_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        n_total = len(SIMULATORS)
        n_better_worst = int((delta_df["median_d_worst"] < 0).sum())
        n_better_is    = int((delta_df["median_d_is"] < 0).sum())
        print(
            f"\nDecision rule (Exp_plan.md): oracle should beat fixed on >=3 of {n_total} DGPs.\n"
            f"  worst-bin dev: oracle better on {n_better_worst}/{n_total}\n"
            f"  interval score: oracle better on {n_better_is}/{n_total}"
        )

    diag_cols = ["simulator", "arm", "frac_y_outside", "frac_L_clipped", "frac_U_clipped"]
    print("\nt_grid truncation diagnostics (mean across macroreps):")
    print(df[diag_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
