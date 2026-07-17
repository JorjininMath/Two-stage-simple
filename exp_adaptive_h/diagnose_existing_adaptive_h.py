"""
Post-hoc diagnostics for existing adaptive-h experiment outputs.

This script does not rerun CKME-DCP and does not mutate existing experiment
outputs. It reads the current Exp2, Exp3, and Exp4-IQR per-point CSV files,
adds diagnostics based on the known oracle scale functions, and writes a
separate diagnostic output directory.

Important limitation:
    Existing per_point.csv files contain the thresholded coverage indicator
    `covered_score`, but they do not contain the raw conformity score
    R(X,Y)=|F_hat(Y|X)-0.5|. Exact score-homogeneity diagnostics are therefore
    reported as unavailable unless future outputs include a raw score column.

Usage:
    python exp_adaptive_h/diagnose_existing_adaptive_h.py
    python exp_adaptive_h/diagnose_existing_adaptive_h.py --max-files-per-source 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ckme_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ckme_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_adaptive_h.adaptive_h_utils import ORACLE_SCALE, get_oracle_h
from Two_stage.sim_functions import get_experiment_config


SIMULATORS = [
    "exp2_gauss_low",
    "wsc_gauss",
    "gibbs_s1",
    "exp1",
    "nongauss_A1L",
]
ARM_ORDER = ["fixed", "plugin", "oracle"]
ARM_LABEL = {
    "fixed": "fixed h",
    "plugin": "plug-in h(x)",
    "oracle": "oracle h(x)",
}
ARM_COLOR = {"fixed": "tab:gray", "plugin": "tab:blue", "oracle": "tab:red"}
ARM_LS = {"fixed": "--", "plugin": "-", "oracle": "-"}
SCORE_COL_CANDIDATES = (
    "score",
    "raw_score",
    "conformity_score",
    "nonconformity_score",
    "pre_threshold_score",
    "R",
)
SCALE_FLOOR = 1e-3


@dataclass(frozen=True)
class CaseInfo:
    source: str
    path: Path
    macrorep: int
    simulator: str
    arm: str
    budget: float
    c_value: float
    case_name: str


def _finite_or_nan(values: np.ndarray, fn) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(fn(arr))


def _sample_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    values = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _load_fixed_h(root: Path) -> dict[str, float]:
    path = root / "exp_adaptive_h" / "pretrained_params.json"
    raw = json.loads(path.read_text())
    return {sim: float(params["h"]) for sim, params in raw.items()}


def _parse_case_name(case_dir: str, default_c: float) -> tuple[str, str, float]:
    if not case_dir.startswith("case_"):
        raise ValueError(f"Unexpected case directory name: {case_dir}")
    name = case_dir[len("case_") :]
    if name.endswith("_fixed"):
        return name[: -len("_fixed")], "fixed", float("nan")
    if name.endswith("_plugin"):
        return name[: -len("_plugin")], "plugin", default_c
    if name.endswith("_oracle"):
        return name[: -len("_oracle")], "oracle", default_c
    match = re.match(r"(.+)_c([0-9.]+)$", name)
    if match:
        return match.group(1), "oracle", float(match.group(2))
    raise ValueError(f"Could not parse case directory name: {case_dir}")


def _iter_cases(source: str, root: Path, default_c: float) -> Iterable[CaseInfo]:
    if source in {"exp2", "exp3"}:
        for path in sorted(root.glob("macrorep_*/case_*/per_point.csv")):
            macrorep = int(path.parts[-3].split("_")[1])
            case_name = path.parts[-2]
            sim, arm, c_value = _parse_case_name(case_name, default_c)
            yield CaseInfo(
                source=source,
                path=path,
                macrorep=macrorep,
                simulator=sim,
                arm=arm,
                budget=float("nan"),
                c_value=c_value,
                case_name=case_name,
            )
    elif source == "exp4_iqr":
        for path in sorted(root.glob("macrorep_*/budget_*/case_*/per_point.csv")):
            macrorep = int(path.parts[-4].split("_")[1])
            budget = float(path.parts[-3].split("_")[1])
            case_name = path.parts[-2]
            sim, arm, c_value = _parse_case_name(case_name, default_c)
            yield CaseInfo(
                source=source,
                path=path,
                macrorep=macrorep,
                simulator=sim,
                arm=arm,
                budget=budget,
                c_value=c_value,
                case_name=case_name,
            )
    else:
        raise ValueError(f"Unknown source: {source}")


def _bin_edges_for_sim(simulator: str, n_bins: int) -> np.ndarray:
    cfg = get_experiment_config(simulator)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    return np.linspace(x_lo, x_hi, n_bins + 1)


def _detect_score_column(df: pd.DataFrame) -> str | None:
    for col in SCORE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _h_query_for_case(
    df: pd.DataFrame,
    case: CaseInfo,
    fixed_h: dict[str, float],
    x: np.ndarray,
) -> np.ndarray:
    if "h_query" in df.columns:
        return df["h_query"].to_numpy(dtype=float)
    if case.arm == "fixed":
        return np.full(x.shape[0], fixed_h[case.simulator], dtype=float)
    if case.arm == "oracle":
        c = case.c_value if math.isfinite(case.c_value) else 1.0
        return get_oracle_h(case.simulator, x, c)
    return np.full(x.shape[0], np.nan, dtype=float)


def _case_target_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    cov_col = "covered_score" if "covered_score" in df.columns else "covered_interval"
    interval_cov_col = "covered_interval" if "covered_interval" in df.columns else None
    return cov_col, interval_cov_col


def _process_case(
    case: CaseInfo,
    fixed_h: dict[str, float],
    n_bins: int,
    target: float,
) -> tuple[dict, list[dict], dict, dict]:
    df = pd.read_csv(case.path)
    if case.simulator not in ORACLE_SCALE:
        raise ValueError(f"No oracle scale function for {case.simulator}")

    x = df["x0"].to_numpy(dtype=float)
    s_true = ORACLE_SCALE[case.simulator](x)
    s_denom = np.maximum(s_true, SCALE_FLOOR)
    h_query = _h_query_for_case(df, case, fixed_h, x)
    eff_ratio = h_query / s_denom
    scale_floor_active = s_true <= SCALE_FLOOR

    cov_col, interval_cov_col = _case_target_columns(df)
    coverage = df[cov_col].to_numpy(dtype=float)
    interval_cov = (
        df[interval_cov_col].to_numpy(dtype=float)
        if interval_cov_col is not None
        else np.full(len(df), np.nan)
    )
    width = df["width"].to_numpy(dtype=float)
    interval_score = df["interval_score"].to_numpy(dtype=float)
    width_to_scale = width / s_denom

    score_col = _detect_score_column(df)
    score = df[score_col].to_numpy(dtype=float) if score_col else None

    bin_edges = _bin_edges_for_sim(case.simulator, n_bins)
    bin_idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)

    bin_rows: list[dict] = []
    cov_bins: list[float] = []
    score_q90_bins: list[float] = []
    pooled_score = score[np.isfinite(score)] if score is not None else None
    ks_bins: list[float] = []

    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            bin_cov = float("nan")
            bin_score_q90 = float("nan")
            bin_ks = float("nan")
        else:
            bin_cov = float(np.mean(coverage[mask]))
            if score is not None and pooled_score is not None and pooled_score.size:
                bin_scores = score[mask]
                bin_scores = bin_scores[np.isfinite(bin_scores)]
                bin_score_q90 = (
                    float(np.quantile(bin_scores, 0.90))
                    if bin_scores.size
                    else float("nan")
                )
                bin_ks = _ks_distance(bin_scores, pooled_score)
            else:
                bin_score_q90 = float("nan")
                bin_ks = float("nan")
        if math.isfinite(bin_cov):
            cov_bins.append(bin_cov)
        if math.isfinite(bin_score_q90):
            score_q90_bins.append(bin_score_q90)
        if math.isfinite(bin_ks):
            ks_bins.append(bin_ks)

        row = {
            "source": case.source,
            "macrorep": case.macrorep,
            "simulator": case.simulator,
            "budget": case.budget,
            "arm": case.arm,
            "c": case.c_value,
            "bin": b,
            "x_left": float(bin_edges[b]),
            "x_right": float(bin_edges[b + 1]),
            "x_mid": float(0.5 * (bin_edges[b] + bin_edges[b + 1])),
            "n": int(mask.sum()),
            "coverage": bin_cov,
            "coverage_interval": _finite_or_nan(interval_cov[mask], np.mean),
            "coverage_dev": abs(bin_cov - target) if math.isfinite(bin_cov) else float("nan"),
            "mean_width": _finite_or_nan(width[mask], np.mean),
            "mean_interval_score": _finite_or_nan(interval_score[mask], np.mean),
            "mean_s_true": _finite_or_nan(s_true[mask], np.mean),
            "mean_h": _finite_or_nan(h_query[mask], np.mean),
            "mean_effective_ratio": _finite_or_nan(eff_ratio[mask], np.mean),
            "median_effective_ratio": _finite_or_nan(eff_ratio[mask], np.median),
            "mean_width_to_scale": _finite_or_nan(width_to_scale[mask], np.mean),
            "score_available": score is not None,
            "score_q90": bin_score_q90,
            "score_ks_to_pooled": bin_ks,
        }
        bin_rows.append(row)

    cov_bins_arr = np.asarray(cov_bins, dtype=float)
    coverage_dev = np.abs(cov_bins_arr - target) if cov_bins_arr.size else np.asarray([])

    plugin_scale_valid = (
        (case.arm == "plugin")
        & np.isfinite(h_query)
        & (s_true > SCALE_FLOOR)
        & np.isfinite(s_true)
    )
    if np.any(plugin_scale_valid):
        c = case.c_value if math.isfinite(case.c_value) else 1.0
        s_hat = h_query[plugin_scale_valid] / c
        scale_log_error = np.log(s_hat / s_true[plugin_scale_valid])
        scale_rmse = float(np.sqrt(np.mean(scale_log_error * scale_log_error)))
        scale_sup = float(np.max(np.abs(scale_log_error)))
        scale_mean = float(np.mean(scale_log_error))
    else:
        scale_rmse = scale_sup = scale_mean = float("nan")

    boundary_cols = {
        "frac_y_outside": 1.0 - df["y_in_grid"].mean() if "y_in_grid" in df else float("nan"),
        "frac_L_clipped": df["L_at_grid_lo"].mean() if "L_at_grid_lo" in df else float("nan"),
        "frac_U_clipped": df["U_at_grid_hi"].mean() if "U_at_grid_hi" in df else float("nan"),
    }

    per_arm = {
        "source": case.source,
        "macrorep": case.macrorep,
        "simulator": case.simulator,
        "budget": case.budget,
        "arm": case.arm,
        "c": case.c_value,
        "case_name": case.case_name,
        "n_test": int(len(df)),
        "score_available": score is not None,
        "score_column": score_col or "",
        "marginal_coverage": float(np.mean(coverage)),
        "marginal_coverage_interval": _finite_or_nan(interval_cov, np.mean),
        "worst_bin_dev": float(np.max(coverage_dev)) if coverage_dev.size else float("nan"),
        "mean_bin_dev": float(np.mean(coverage_dev)) if coverage_dev.size else float("nan"),
        "coverage_range": (
            float(np.max(cov_bins_arr) - np.min(cov_bins_arr))
            if cov_bins_arr.size
            else float("nan")
        ),
        "mean_width": float(np.mean(width)),
        "sd_width": _sample_std(width),
        "mean_interval_score": float(np.mean(interval_score)),
        "sd_interval_score": _sample_std(interval_score),
        "mean_s_true": float(np.mean(s_true)),
        "sd_s_true": _sample_std(s_true),
        "mean_h": _finite_or_nan(h_query, np.mean),
        "sd_h": _sample_std(h_query),
        "mean_effective_ratio": _finite_or_nan(eff_ratio, np.mean),
        "sd_effective_ratio": _sample_std(eff_ratio),
        "min_effective_ratio": _finite_or_nan(eff_ratio, np.min),
        "max_effective_ratio": _finite_or_nan(eff_ratio, np.max),
        "effective_ratio_range": _finite_or_nan(eff_ratio, np.max)
        - _finite_or_nan(eff_ratio, np.min),
        "mean_width_to_scale": _finite_or_nan(width_to_scale, np.mean),
        "sd_width_to_scale": _sample_std(width_to_scale),
        "corr_width_scale": _safe_corr(width, s_true),
        "scale_floor_frac": float(np.mean(scale_floor_active)),
        "plugin_log_scale_error_mean": scale_mean,
        "plugin_log_scale_error_rmse": scale_rmse,
        "plugin_log_scale_error_sup": scale_sup,
        **boundary_cols,
    }

    score_diag = {
        "source": case.source,
        "macrorep": case.macrorep,
        "simulator": case.simulator,
        "budget": case.budget,
        "arm": case.arm,
        "c": case.c_value,
        "available": score is not None,
        "score_column": score_col or "",
        "mean_bin_KS_to_pooled": float(np.mean(ks_bins)) if ks_bins else float("nan"),
        "max_bin_KS_to_pooled": float(np.max(ks_bins)) if ks_bins else float("nan"),
        "std_bin_score_q90": _sample_std(np.asarray(score_q90_bins, dtype=float)),
        "reason": "" if score is not None else "raw conformity score not saved in per_point.csv",
    }

    scale_diag = {
        "source": case.source,
        "macrorep": case.macrorep,
        "simulator": case.simulator,
        "budget": case.budget,
        "arm": case.arm,
        "c": case.c_value,
        "available": case.arm == "plugin",
        "log_scale_error_mean": scale_mean,
        "log_scale_error_rmse": scale_rmse,
        "log_scale_error_sup": scale_sup,
        "scale_floor_frac": float(np.mean(scale_floor_active)),
        "reason": "" if case.arm == "plugin" else "not a plug-in arm",
    }

    return per_arm, bin_rows, score_diag, scale_diag


def _summarize(per_arm: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "marginal_coverage",
        "marginal_coverage_interval",
        "worst_bin_dev",
        "mean_bin_dev",
        "coverage_range",
        "mean_width",
        "mean_interval_score",
        "mean_effective_ratio",
        "sd_effective_ratio",
        "effective_ratio_range",
        "mean_width_to_scale",
        "corr_width_scale",
        "scale_floor_frac",
        "frac_y_outside",
        "frac_L_clipped",
        "frac_U_clipped",
        "plugin_log_scale_error_rmse",
        "plugin_log_scale_error_sup",
    ]
    rows = []
    group_cols = ["source", "simulator", "budget", "arm", "c"]
    for keys, df in per_arm.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_macroreps"] = int(df["macrorep"].nunique())
        row["score_available"] = bool(df["score_available"].any())
        for metric in metrics:
            vals = pd.to_numeric(df[metric], errors="coerce")
            row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else float("nan")
            row[f"se_{metric}"] = (
                float(vals.std(ddof=1) / np.sqrt(vals.notna().sum()))
                if vals.notna().sum() > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_red_flags(
    summary: pd.DataFrame,
    target: float,
    marginal_tol: float,
    boundary_tol: float,
    oracle_ratio_sd_tol: float,
    scale_rmse_tol: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(row, flag, severity, metric, value, threshold, message):
        rows.append(
            {
                "source": row["source"],
                "simulator": row["simulator"],
                "budget": row["budget"],
                "arm": row["arm"],
                "c": row["c"],
                "flag": flag,
                "severity": severity,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    for _, row in summary.iterrows():
        cov = row["mean_marginal_coverage"]
        if math.isfinite(cov) and abs(cov - target) > marginal_tol:
            add(
                row,
                "marginal_coverage_far_from_target",
                "warn",
                "mean_marginal_coverage",
                cov,
                f"{target:.3f} +/- {marginal_tol:.3f}",
                "Check split, calibration quantile, interval inversion, or test distribution.",
            )

        boundary = max(
            row.get("mean_frac_y_outside", float("nan")),
            row.get("mean_frac_L_clipped", float("nan")),
            row.get("mean_frac_U_clipped", float("nan")),
        )
        if math.isfinite(boundary) and boundary > boundary_tol:
            add(
                row,
                "grid_boundary_hits",
                "warn",
                "max_grid_boundary_fraction",
                boundary,
                boundary_tol,
                "Response-grid clipping can distort coverage, width, and interval score.",
            )

        if row["arm"] == "oracle":
            ratio_sd = row["mean_sd_effective_ratio"]
            floor_frac = row["mean_scale_floor_frac"]
            if math.isfinite(ratio_sd) and ratio_sd > oracle_ratio_sd_tol:
                severity = "note" if floor_frac > 0.01 else "warn"
                add(
                    row,
                    "oracle_effective_ratio_not_flat",
                    severity,
                    "mean_sd_effective_ratio",
                    ratio_sd,
                    oracle_ratio_sd_tol,
                    "For oracle h, h/s should be flat except where the scale floor is active.",
                )

        if row["arm"] == "plugin":
            rmse = row["mean_plugin_log_scale_error_rmse"]
            if math.isfinite(rmse) and rmse > scale_rmse_tol:
                add(
                    row,
                    "plugin_scale_error_large",
                    "note",
                    "mean_plugin_log_scale_error_rmse",
                    rmse,
                    scale_rmse_tol,
                    "Inspect plug-in scale estimation before blaming adaptive-h mechanics.",
                )

    compare_cols = ["source", "simulator", "budget"]
    for _, df in summary.groupby(compare_cols, dropna=False, sort=False):
        fixed = df[df["arm"] == "fixed"]
        oracle = df[df["arm"] == "oracle"]
        plugin = df[df["arm"] == "plugin"]
        if not fixed.empty and not oracle.empty:
            f = fixed.iloc[0]
            o = oracle.iloc[0]
            f_sd = f["mean_sd_effective_ratio"]
            o_sd = o["mean_sd_effective_ratio"]
            if math.isfinite(f_sd) and math.isfinite(o_sd) and o_sd >= f_sd:
                add(
                    o,
                    "oracle_not_less_variable_than_fixed",
                    "warn",
                    "mean_sd_effective_ratio",
                    o_sd,
                    f"fixed={f_sd:.4g}",
                    "Oracle adaptive h should reduce effective-ratio variation relative to fixed h.",
                )
        if not plugin.empty and not oracle.empty:
            p = plugin.iloc[0]
            o = oracle.iloc[0]
            p_worst = p["mean_worst_bin_dev"]
            o_worst = o["mean_worst_bin_dev"]
            if math.isfinite(p_worst) and math.isfinite(o_worst) and p_worst > o_worst + 0.03:
                add(
                    p,
                    "plugin_much_worse_than_oracle",
                    "note",
                    "mean_worst_bin_dev",
                    p_worst,
                    f"oracle+0.03={o_worst + 0.03:.4g}",
                    "This points to plug-in scale estimation rather than the oracle mechanism.",
                )

    return pd.DataFrame(rows)


def _aggregate_bin_for_plot(bin_df: pd.DataFrame, source: str, budget: float | None) -> pd.DataFrame:
    sub = bin_df[bin_df["source"] == source].copy()
    if budget is not None:
        sub = sub[np.isclose(sub["budget"], budget, equal_nan=False)]
    group_cols = ["simulator", "arm", "bin"]
    agg = (
        sub.groupby(group_cols, dropna=False, sort=False)
        .agg(
            x_mid=("x_mid", "mean"),
            coverage=("coverage", "median"),
            mean_effective_ratio=("mean_effective_ratio", "median"),
            mean_width_to_scale=("mean_width_to_scale", "median"),
            score_q90=("score_q90", "median"),
        )
        .reset_index()
    )
    return agg


def _plot_grid_metric(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    target_line: float | None = None,
) -> Path | None:
    if agg.empty or metric not in agg:
        return None
    sims = [s for s in SIMULATORS if s in set(agg["simulator"])]
    if not sims:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    for ax, sim in zip(axes.ravel(), sims):
        sub_sim = agg[agg["simulator"] == sim]
        for arm in ARM_ORDER:
            sub = sub_sim[sub_sim["arm"] == arm].sort_values("x_mid")
            vals = pd.to_numeric(sub[metric], errors="coerce")
            if sub.empty or vals.notna().sum() == 0:
                continue
            ax.plot(
                sub["x_mid"],
                vals,
                color=ARM_COLOR.get(arm),
                ls=ARM_LS.get(arm, "-"),
                marker="o",
                ms=3,
                lw=1.8,
                label=ARM_LABEL.get(arm, arm),
            )
        if target_line is not None:
            ax.axhline(target_line, ls=":", color="black", alpha=0.6)
        ax.set_title(sim)
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(sims) :]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_csweep(summary: pd.DataFrame, out_path: Path, target: float) -> Path | None:
    sub = summary[(summary["source"] == "exp3") & (summary["simulator"] == "nongauss_A1L")]
    if sub.empty:
        return None
    oracle = sub[(sub["arm"] == "oracle") & sub["c"].notna()].sort_values("c")
    fixed = sub[sub["arm"] == "fixed"]
    if oracle.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("mean_marginal_coverage", "marginal coverage", target),
        ("mean_worst_bin_dev", "worst-bin deviation", None),
        ("mean_mean_interval_score", "interval score", None),
    ]
    x = oracle["c"].to_numpy(dtype=float)
    for ax, (metric, label, ref) in zip(axes, metrics):
        y = oracle[metric].to_numpy(dtype=float)
        ax.plot(x, y, "o-", color="tab:red", lw=2, label="oracle c-sweep")
        if not fixed.empty:
            fixed_val = float(fixed.iloc[0][metric])
            if math.isfinite(fixed_val):
                ax.axhline(fixed_val, ls="--", color="tab:gray", label="fixed h")
        if ref is not None:
            ax.axhline(ref, ls=":", color="black", alpha=0.7, label="target")
        ax.set_xscale("log")
        ax.set_xlabel("c in h(x)=c*s(x)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Exp3 c-sweep diagnostics on nongauss_A1L", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _report_table(df: pd.DataFrame, cols: list[str], max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.loc[:, cols].head(max_rows).copy()
    return shown.to_markdown(index=False, floatfmt=".4f")


def _write_report(
    out_dir: Path,
    per_arm: pd.DataFrame,
    summary: pd.DataFrame,
    score_diag: pd.DataFrame,
    red_flags: pd.DataFrame,
    plot_paths: list[Path],
    target: float,
) -> Path:
    lines: list[str] = []
    lines.append("# Existing-DGP Adaptive-h Diagnostic Report")
    lines.append("")
    lines.append("This report is generated post-hoc from existing Exp2, Exp3, and Exp4-IQR outputs.")
    lines.append("The checks are diagnostic guidance, not hard pass/fail criteria.")
    lines.append("")

    lines.append("## Data Sources")
    src_counts = (
        per_arm.groupby("source", dropna=False)
        .agg(n_cases=("case_name", "count"), n_macroreps=("macrorep", "nunique"))
        .reset_index()
    )
    lines.append(_report_table(src_counts, ["source", "n_cases", "n_macroreps"]))
    lines.append("")

    score_available = bool(score_diag["available"].any()) if not score_diag.empty else False
    lines.append("## Score Homogeneity Availability")
    if score_available:
        lines.append("Raw conformity scores were found, so score-KS and score-q90 diagnostics were computed.")
    else:
        lines.append(
            "Raw conformity scores were not found in the existing per-point CSV files. "
            "The files contain `covered_score`, which is already thresholded, so exact "
            "score-KS and score-q90 diagnostics are marked unavailable. To compute them, "
            "rerun the experiments after saving R(X,Y)=|F_hat(Y|X)-0.5| per test point."
        )
    lines.append("")

    lines.append("## Exp4-IQR Largest-Budget Summary")
    exp4 = summary[summary["source"] == "exp4_iqr"].copy()
    if not exp4.empty:
        max_budget = float(exp4["budget"].dropna().max())
        exp4_max = exp4[np.isclose(exp4["budget"], max_budget)].copy()
        order = {arm: i for i, arm in enumerate(ARM_ORDER)}
        exp4_max["arm_order"] = exp4_max["arm"].map(order).fillna(99)
        exp4_max = exp4_max.sort_values(["simulator", "arm_order"])
        lines.append(f"Largest Stage-1 budget in Exp4-IQR: B={max_budget:g}.")
        lines.append(
            _report_table(
                exp4_max,
                [
                    "simulator",
                    "arm",
                    "mean_marginal_coverage",
                    "mean_worst_bin_dev",
                    "mean_sd_effective_ratio",
                    "mean_mean_width",
                    "mean_mean_interval_score",
                    "mean_plugin_log_scale_error_rmse",
                ],
                max_rows=30,
            )
        )
    else:
        lines.append("_No Exp4-IQR rows found._")
    lines.append("")

    lines.append("## Interpretation Notes By DGP")
    lines.append("- `wsc_gauss` and `nongauss_A1L`: fixed-h distortion is strongest near high-scale edges and weakest near the center.")
    lines.append("- `gibbs_s1`: the near-zero scale region activates the scale floor, so oracle effective-ratio flatness should be interpreted with that floor in mind.")
    lines.append("- `exp1`: instability should be read mostly near the high-scale boundary.")
    lines.append("- These existing DGPs do not establish the exact `R_s=1,2,4,8` trend; that would require a controlled `R_s` simulator.")
    lines.append("")

    lines.append("## Red Flags")
    if red_flags.empty:
        lines.append("No configured red flags were triggered.")
    else:
        lines.append(
            _report_table(
                red_flags.sort_values(["severity", "source", "simulator", "arm"]),
                [
                    "severity",
                    "flag",
                    "source",
                    "simulator",
                    "budget",
                    "arm",
                    "value",
                    "threshold",
                    "message",
                ],
                max_rows=30,
            )
        )
    lines.append("")

    lines.append("## Generated Plots")
    if plot_paths:
        for path in plot_paths:
            lines.append(f"- `{path.name}`")
    else:
        lines.append("_No plots were generated._")
    lines.append("")

    lines.append("## Files Written")
    for name in [
        "existing_diag_per_arm.csv",
        "existing_diag_summary.csv",
        "existing_diag_bin.csv",
        "existing_diag_score_homogeneity.csv",
        "existing_diag_scale_estimation.csv",
        "existing_diag_red_flags.csv",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")

    path = out_dir / "existing_diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose existing adaptive-h outputs")
    parser.add_argument("--exp2-dir", type=str, default="exp_adaptive_h/output_exp2")
    parser.add_argument("--exp3-dir", type=str, default="exp_adaptive_h/output_exp3")
    parser.add_argument("--exp4-dir", type=str, default="exp_adaptive_h/output_exp4_iqr")
    parser.add_argument("--output-dir", type=str, default="exp_adaptive_h/output_existing_diagnostics")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--marginal-tol", type=float, default=0.03)
    parser.add_argument("--boundary-tol", type=float, default=0.01)
    parser.add_argument("--oracle-ratio-sd-tol", type=float, default=0.02)
    parser.add_argument("--scale-rmse-tol", type=float, default=0.35)
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        default=None,
        help="Optional smoke-test limit per source.",
    )
    args = parser.parse_args()

    target = 1.0 - args.alpha
    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "exp2": (_root / args.exp2_dir).resolve(),
        "exp3": (_root / args.exp3_dir).resolve(),
        "exp4_iqr": (_root / args.exp4_dir).resolve(),
    }
    fixed_h = _load_fixed_h(_root)

    per_arm_rows: list[dict] = []
    bin_rows: list[dict] = []
    score_rows: list[dict] = []
    scale_rows: list[dict] = []

    for source, path in sources.items():
        if not path.exists():
            print(f"WARN: source directory not found, skipping: {path}", file=sys.stderr)
            continue
        cases = list(_iter_cases(source, path, args.c_scale))
        if args.max_files_per_source is not None:
            cases = cases[: args.max_files_per_source]
        print(f"{source}: processing {len(cases)} per-point files from {path}")
        for idx, case in enumerate(cases, start=1):
            per_arm, bins, score_diag, scale_diag = _process_case(
                case=case,
                fixed_h=fixed_h,
                n_bins=args.n_bins,
                target=target,
            )
            per_arm_rows.append(per_arm)
            bin_rows.extend(bins)
            score_rows.append(score_diag)
            scale_rows.append(scale_diag)
            if idx % 250 == 0:
                print(f"  processed {idx}/{len(cases)} files")

    per_arm = pd.DataFrame(per_arm_rows)
    bin_df = pd.DataFrame(bin_rows)
    score_diag = pd.DataFrame(score_rows)
    scale_diag = pd.DataFrame(scale_rows)
    summary = _summarize(per_arm) if not per_arm.empty else pd.DataFrame()
    red_flags = (
        _make_red_flags(
            summary=summary,
            target=target,
            marginal_tol=args.marginal_tol,
            boundary_tol=args.boundary_tol,
            oracle_ratio_sd_tol=args.oracle_ratio_sd_tol,
            scale_rmse_tol=args.scale_rmse_tol,
        )
        if not summary.empty
        else pd.DataFrame()
    )

    per_arm.to_csv(out_dir / "existing_diag_per_arm.csv", index=False)
    summary.to_csv(out_dir / "existing_diag_summary.csv", index=False)
    bin_df.to_csv(out_dir / "existing_diag_bin.csv", index=False)
    score_diag.to_csv(out_dir / "existing_diag_score_homogeneity.csv", index=False)
    scale_diag.to_csv(out_dir / "existing_diag_scale_estimation.csv", index=False)
    red_flags.to_csv(out_dir / "existing_diag_red_flags.csv", index=False)

    plot_paths: list[Path] = []
    exp4_rows = summary[summary["source"] == "exp4_iqr"] if not summary.empty else pd.DataFrame()
    if not exp4_rows.empty:
        max_budget = float(exp4_rows["budget"].dropna().max())
        agg = _aggregate_bin_for_plot(bin_df, source="exp4_iqr", budget=max_budget)
        for path in [
            _plot_grid_metric(
                agg,
                metric="mean_effective_ratio",
                ylabel="median bin h(x)/s(x)",
                title=f"Exp4-IQR B={max_budget:g}: effective bandwidth ratio",
                out_path=out_dir / "diag_effective_ratio_exp4_iqr_budget_max.png",
            ),
            _plot_grid_metric(
                agg,
                metric="coverage",
                ylabel="median bin coverage",
                title=f"Exp4-IQR B={max_budget:g}: bin coverage",
                out_path=out_dir / "diag_bin_coverage_exp4_iqr_budget_max.png",
                target_line=target,
            ),
            _plot_grid_metric(
                agg,
                metric="mean_width_to_scale",
                ylabel="median bin width / s(x)",
                title=f"Exp4-IQR B={max_budget:g}: interval length relative to local scale",
                out_path=out_dir / "diag_width_to_scale_exp4_iqr_budget_max.png",
            ),
        ]:
            if path is not None:
                plot_paths.append(path)
        if bool(score_diag["available"].any()) and "score_q90" in agg:
            path = _plot_grid_metric(
                agg,
                metric="score_q90",
                ylabel="median bin raw-score q90",
                title=f"Exp4-IQR B={max_budget:g}: raw score q90 by bin",
                out_path=out_dir / "diag_score_q90_exp4_iqr_budget_max.png",
            )
            if path is not None:
                plot_paths.append(path)

    csweep_path = _plot_csweep(summary, out_dir / "diag_csweep_exp3.png", target=target)
    if csweep_path is not None:
        plot_paths.append(csweep_path)

    report_path = _write_report(
        out_dir=out_dir,
        per_arm=per_arm,
        summary=summary,
        score_diag=score_diag,
        red_flags=red_flags,
        plot_paths=plot_paths,
        target=target,
    )

    print(f"\nSaved diagnostics to: {out_dir}")
    print(f"Report: {report_path}")
    print("Core files:")
    for name in [
        "existing_diag_per_arm.csv",
        "existing_diag_summary.csv",
        "existing_diag_bin.csv",
        "existing_diag_score_homogeneity.csv",
        "existing_diag_scale_estimation.csv",
        "existing_diag_red_flags.csv",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
