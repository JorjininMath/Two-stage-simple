"""Analyze raw-score homogeneity in final adaptive-h per-point outputs.

The diagnostic compares raw DCP score distributions across the ten equal-count
input bins. It reports the maximum pairwise two-sample KS distance, the mean
pairwise KS distance, and the range of binwise mean scores. These are
descriptive diagnostics; they are not finite-sample conditional-coverage
guarantees.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (_ROOT / "src", _ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from experiments.adaptive_h.run_final_adaptive_h_benchmark import _atomic_csv

DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")
METRICS = (
    "max_pairwise_score_ks",
    "mean_pairwise_score_ks",
    "bin_mean_score_range",
)
ARMS = {"fixed", "plugin_sd_nw", "oracle"}
COMPARISONS = (
    ("plugin_sd_nw_minus_oracle", "plugin_sd_nw", "oracle"),
    ("plugin_sd_nw_minus_fixed", "plugin_sd_nw", "fixed"),
    ("oracle_minus_fixed", "oracle", "fixed"),
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def analyze_file(path: Path, *, expected_groups: int) -> dict:
    frame = pd.read_csv(
        path,
        usecols=[
            "macrorep",
            "simulator",
            "budget",
            "arm",
            "group_bin",
            "raw_score",
        ],
    )
    grouped = list(frame.groupby("group_bin", sort=True))
    actual_bins = [int(bin_id) for bin_id, _ in grouped]
    expected_bins = list(range(expected_groups))
    if actual_bins != expected_bins:
        raise ValueError(
            f"Expected nonempty group bins {expected_bins} in {path}; "
            f"found {actual_bins}"
        )
    groups = [
        group["raw_score"].to_numpy(dtype=float) for _, group in grouped
    ]
    ks_values = np.array(
        [
            ks_2samp(left, right, method="auto").statistic
            for left, right in combinations(groups, 2)
        ],
        dtype=float,
    )
    group_means = np.array([values.mean() for values in groups])
    first = frame.iloc[0]
    return {
        "macrorep": int(first["macrorep"]),
        "simulator": str(first["simulator"]),
        "budget": int(first["budget"]),
        "arm": str(first["arm"]),
        "n_groups": len(groups),
        "max_pairwise_score_ks": float(ks_values.max()),
        "mean_pairwise_score_ks": float(ks_values.mean()),
        "bin_mean_score_range": float(group_means.max() - group_means.min()),
    }


def summarize(per_arm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in per_arm.groupby(
        ["simulator", "budget", "arm"], sort=True
    ):
        row = dict(zip(("simulator", "budget", "arm"), key))
        count = int(group["macrorep"].nunique())
        row["n_macroreps"] = count
        for metric in METRICS:
            values = group[metric].astype(float)
            sd = float(values.std(ddof=1)) if count > 1 else float("nan")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = sd
            row[f"mcse_{metric}"] = (
                sd / np.sqrt(count) if count > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(per_arm: pd.DataFrame) -> pd.DataFrame:
    keys = ["macrorep", "simulator", "budget"]
    if per_arm.duplicated(keys + ["arm"]).any():
        raise ValueError("Duplicate arm rows in score-homogeneity output")
    indexed = per_arm.set_index(keys + ["arm"])
    rows = []
    for key, group in per_arm.groupby(keys, sort=True):
        available = set(group["arm"])
        if available != ARMS:
            raise ValueError(
                f"Incomplete score arms for {key}: "
                f"expected {sorted(ARMS)}, found {sorted(available)}"
            )
        for label, left, right in COMPARISONS:
            left_row = indexed.loc[(*key, left)]
            right_row = indexed.loc[(*key, right)]
            row = dict(zip(keys, key))
            row["comparison"] = label
            for metric in METRICS:
                row[f"delta_{metric}"] = float(
                    left_row[metric] - right_row[metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_groups = int(
        manifest.get("resolved_config", {}).get("group_bins", 0)
    )
    if expected_groups < 2:
        raise ValueError(
            "Manifest must define resolved_config.group_bins >= 2"
        )
    paths = sorted(
        (output_dir / "jobs").glob(
            "*/budget_*/macrorep_*/per_point_*.csv"
        )
    )
    if not paths:
        raise FileNotFoundError(f"No per-point files under {output_dir}")
    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(analyze_file(path, expected_groups=expected_groups))
        if index % 100 == 0:
            print(f"Analyzed {index}/{len(paths)} per-point files")
    per_arm = pd.DataFrame(rows).sort_values(
        ["simulator", "budget", "macrorep", "arm"]
    )
    _atomic_csv(per_arm, output_dir / "score_homogeneity_per_arm.csv")
    _atomic_csv(
        summarize(per_arm),
        output_dir / "score_homogeneity_summary.csv",
    )
    _atomic_csv(
        paired_deltas(per_arm),
        output_dir / "score_homogeneity_paired_deltas.csv",
    )
    print(f"Wrote raw-score homogeneity diagnostics under {output_dir}")


if __name__ == "__main__":
    main()
