"""
summarize_exp3.py

Aggregate Exp3 (c-sensitivity sweep on nongauss_A1L) into a per-arm table.

Goal (Exp_plan.md): show plateau, not knife-edge optimum, in conditional
coverage deviation as c varies in {0.3, 0.5, 1.0, 2.0}.

For each arm (fixed + each c), compute across macroreps:
  - marginal_coverage    : median over macroreps
  - worst_bin_dev        : max_b |cov_b - (1-alpha)|, median
  - mean_bin_dev         : mean_b |cov_b - (1-alpha)|, median
  - coverage_range       : max_b cov_b - min_b cov_b, median
  - mean_width           : E[U-L], mean
  - mean_interval_score  : E[IS], mean
  - mean_h_query         : average bandwidth used (sanity check oracle scaling)

Reads:
    exp_adaptive_h/output_exp3/macrorep_{k}/case_nongauss_A1L_{arm}/per_point.csv

Writes:
    exp_adaptive_h/output_exp3/exp3_table.csv
    exp_adaptive_h/output_exp3/exp3_table.tex

Usage (from project root):
    python exp_adaptive_h/summarize_exp3.py
    python exp_adaptive_h/summarize_exp3.py --n_bins 20 --alpha 0.1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from Two_stage.sim_functions import get_experiment_config

SIMULATOR = "nongauss_A1L"


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
        "marginal":   cov.mean(),
        "worst_bin":  devs.max() if devs.size else np.nan,
        "mean_bin":   devs.mean() if devs.size else np.nan,
        "cov_range":  (cov_bins[valid].max() - cov_bins[valid].min()) if valid.any() else np.nan,
        "mean_width": width.mean(),
        "mean_is":    is_.mean(),
    }
    metrics["mean_h_query"] = (
        float(per_point["h_query"].mean()) if "h_query" in per_point.columns else np.nan
    )
    return metrics


def _discover_arms(out_dir: Path) -> list[str]:
    """Return arm labels in display order: 'fixed' first, then ascending c."""
    arms = set()
    pat = re.compile(rf"^case_{re.escape(SIMULATOR)}_(.+)$")
    for case_dir in out_dir.glob("macrorep_*/case_*"):
        m = pat.match(case_dir.name)
        if m:
            arms.add(m.group(1))
    arms_list = sorted(arms, key=lambda s: (s != "fixed", float(s[1:]) if s.startswith("c") else 0.0))
    return arms_list


def _summarize_arm(out_dir: Path, arm: str, n_bins: int, target: float) -> dict:
    cfg = get_experiment_config(SIMULATOR)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    case_paths = sorted(out_dir.glob(f"macrorep_*/case_{SIMULATOR}_{arm}/per_point.csv"))
    if not case_paths:
        return {"arm": arm, "n_macroreps": 0}

    per_rep = [
        _macrorep_metrics(pd.read_csv(p), bin_edges, target)
        for p in case_paths
    ]
    df = pd.DataFrame(per_rep)
    c_val = float(arm[1:]) if arm.startswith("c") else float("nan")
    return {
        "arm":                    arm,
        "c":                      c_val,
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
        "mean_h_query":           float(df["mean_h_query"].mean()),
    }


def _to_latex(rows: list[dict], target: float) -> str:
    header = (
        r"\begin{tabular}{lrrrrrr}" "\n"
        r"\toprule" "\n"
        r"arm & $c$ & marginal cov & worst-bin $|c-(1-\alpha)|$ "
        r"& mean-bin $|c-(1-\alpha)|$ & mean width & mean IS \\" "\n"
        r"\midrule" "\n"
    )
    body_lines = []
    for r in rows:
        if r.get("n_macroreps", 0) == 0:
            body_lines.append(f"{r['arm']} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        c_str = "---" if not np.isfinite(r.get("c", np.nan)) else f"{r['c']:g}"
        body_lines.append(
            f"{r['arm']} "
            f"& {c_str} "
            f"& {r['marginal_coverage']:.3f} ({r['marginal_coverage_sd']:.3f}) "
            f"& {r['worst_bin_dev']:.3f} ({r['worst_bin_dev_sd']:.3f}) "
            f"& {r['mean_bin_dev']:.3f} ({r['mean_bin_dev_sd']:.3f}) "
            f"& {r['mean_width']:.3f} ({r['mean_width_sd']:.3f}) "
            f"& {r['mean_interval_score']:.3f} ({r['mean_interval_score_sd']:.3f}) \\\\"
        )
    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    caption = (
        f"% Exp3 c-sweep on nongauss_A1L. Target marginal coverage = {target:.2f}.\n"
        "% Cell format: median (sd) for binwise stats; mean (sd) for width and IS.\n"
    )
    return caption + header + "\n".join(body_lines) + "\n" + footer


def main():
    parser = argparse.ArgumentParser(description="Summarize Exp3 c-sweep")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp3")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha
    arms = _discover_arms(out_dir)
    if not arms:
        print(f"ERROR: no case_{SIMULATOR}_* directories under {out_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered arms: {arms}")

    rows = [_summarize_arm(out_dir, arm, args.n_bins, target) for arm in arms]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp3_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    tex_path = out_dir / "exp3_table.tex"
    tex_path.write_text(_to_latex(rows, target))
    print(f"Saved: {tex_path}")

    show_cols = [
        "arm", "c", "n_macroreps",
        "marginal_coverage", "worst_bin_dev", "mean_bin_dev",
        "coverage_range", "mean_width", "mean_interval_score", "mean_h_query",
    ]
    print("\n" + df[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
