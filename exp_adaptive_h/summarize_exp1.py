"""
summarize_exp1.py

Aggregate Exp1 (fixed-h baseline) results into a 4-row report table.

For each DGP, compute across macroreps:
  - marginal_coverage           : mean over all test points (median across macroreps)
  - worst_bin_dev               : max_b |cov_b - (1-alpha)|, median across macroreps
  - mean_bin_dev                : mean_b |cov_b - (1-alpha)|, median across macroreps
  - coverage_range              : max_b cov_b - min_b cov_b, median across macroreps
  - mean_width                  : E[U-L], mean across macroreps
  - mean_interval_score         : E[IS], mean across macroreps

Reads:
    exp_adaptive_h/output_exp1/macrorep_{k}/case_{sim}/per_point.csv

Writes:
    exp_adaptive_h/output_exp1/exp1_table.csv
    exp_adaptive_h/output_exp1/exp1_table.tex   (booktabs LaTeX)

Usage (from project root):
    python exp_adaptive_h/summarize_exp1.py
    python exp_adaptive_h/summarize_exp1.py --n_bins 20 --alpha 0.1
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
    # t_grid truncation diagnostics (Method 1)
    if "y_in_grid" in per_point.columns:
        metrics["frac_y_outside"]  = float(1.0 - per_point["y_in_grid"].mean())
        metrics["frac_L_clipped"]  = float(per_point["L_at_grid_lo"].mean())
        metrics["frac_U_clipped"]  = float(per_point["U_at_grid_hi"].mean())
    else:
        metrics["frac_y_outside"]  = np.nan
        metrics["frac_L_clipped"]  = np.nan
        metrics["frac_U_clipped"]  = np.nan
    return metrics


def _summarize_simulator(out_dir: Path, sim: str, n_bins: int, target: float) -> dict:
    cfg = get_experiment_config(sim)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    case_paths = sorted((out_dir).glob(f"macrorep_*/case_{sim}/per_point.csv"))
    if not case_paths:
        return {"simulator": sim, "n_macroreps": 0}

    per_rep = [
        _macrorep_metrics(pd.read_csv(p), bin_edges, target)
        for p in case_paths
    ]
    df = pd.DataFrame(per_rep)
    return {
        "simulator":               sim,
        "n_macroreps":             len(df),
        "marginal_coverage":       float(df["marginal"].median()),
        "marginal_coverage_sd":    float(df["marginal"].std(ddof=1)),
        "worst_bin_dev":           float(df["worst_bin"].median()),
        "worst_bin_dev_sd":        float(df["worst_bin"].std(ddof=1)),
        "mean_bin_dev":            float(df["mean_bin"].median()),
        "mean_bin_dev_sd":         float(df["mean_bin"].std(ddof=1)),
        "coverage_range":          float(df["cov_range"].median()),
        "coverage_range_sd":       float(df["cov_range"].std(ddof=1)),
        "mean_width":              float(df["mean_width"].mean()),
        "mean_width_sd":           float(df["mean_width"].std(ddof=1)),
        "mean_interval_score":     float(df["mean_is"].mean()),
        "mean_interval_score_sd":  float(df["mean_is"].std(ddof=1)),
        "frac_y_outside":          float(df["frac_y_outside"].mean()),
        "frac_L_clipped":          float(df["frac_L_clipped"].mean()),
        "frac_U_clipped":          float(df["frac_U_clipped"].mean()),
    }


def _to_latex(rows: list[dict], target: float) -> str:
    header = (
        r"\begin{tabular}{lrrrrrrr}" "\n"
        r"\toprule" "\n"
        r"DGP & $K$ & marginal cov & worst-bin $|c-(1-\alpha)|$ & mean-bin $|c-(1-\alpha)|$ "
        r"& cov range & mean width & mean IS \\" "\n"
        r"\midrule" "\n"
    )
    body_lines = []
    for r in rows:
        if r.get("n_macroreps", 0) == 0:
            body_lines.append(f"{r['simulator']} & 0 & --- & --- & --- & --- & --- & --- \\\\")
            continue
        sim_tex = r["simulator"].replace("_", r"\_")
        body_lines.append(
            f"{sim_tex} "
            f"& {r['n_macroreps']} "
            f"& {r['marginal_coverage']:.3f} ({r['marginal_coverage_sd']:.3f}) "
            f"& {r['worst_bin_dev']:.3f} ({r['worst_bin_dev_sd']:.3f}) "
            f"& {r['mean_bin_dev']:.3f} ({r['mean_bin_dev_sd']:.3f}) "
            f"& {r['coverage_range']:.3f} ({r['coverage_range_sd']:.3f}) "
            f"& {r['mean_width']:.3f} ({r['mean_width_sd']:.3f}) "
            f"& {r['mean_interval_score']:.3f} ({r['mean_interval_score_sd']:.3f}) \\\\"
        )
    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    caption = (
        f"% Exp1 fixed-h baseline. Target marginal coverage = {target:.2f}.\n"
        "% Cell format: median (sd across macroreps) for binwise stats; "
        "mean (sd) for width and interval score.\n"
    )
    return caption + header + "\n".join(body_lines) + "\n" + footer


def main():
    parser = argparse.ArgumentParser(description="Summarize Exp1 fixed-h baseline")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp1")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha
    rows = [_summarize_simulator(out_dir, sim, args.n_bins, target) for sim in SIMULATORS]

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp1_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    tex_path = out_dir / "exp1_table.tex"
    tex_path.write_text(_to_latex(rows, target))
    print(f"Saved: {tex_path}")

    show_cols = [
        "simulator", "n_macroreps",
        "marginal_coverage", "worst_bin_dev", "mean_bin_dev",
        "coverage_range", "mean_width", "mean_interval_score",
    ]
    print("\n" + df[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if "frac_y_outside" in df.columns:
        diag_cols = ["simulator", "frac_y_outside", "frac_L_clipped", "frac_U_clipped"]
        print("\nt_grid truncation diagnostics (mean across macroreps):")
        print(df[diag_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
