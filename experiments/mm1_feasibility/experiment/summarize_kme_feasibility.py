"""Summarize KME feasibility metrics over seeds."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUP_KEYS = ("dgp", "n_fit", "n_val", "n_test", "r_train", "r_oracle", "rff_dim", "grid_size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Metrics CSV path(s) or glob(s).")
    parser.add_argument("--out", default="results/kme_feas_summary.csv")
    return parser


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted({p.resolve() for p in paths if p.exists()})


def maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, str]] = []
    for path in expand_inputs(args.input):
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no metrics rows found")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in GROUP_KEYS)].append(row)

    out_rows: list[dict[str, object]] = []
    for key_values, group_rows in grouped.items():
        out: dict[str, object] = {key: value for key, value in zip(GROUP_KEYS, key_values)}
        out["n_seeds"] = len(group_rows)
        numeric_keys = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if key not in GROUP_KEYS and maybe_float(value) is not None
            }
        )
        for numeric_key in numeric_keys:
            values = [maybe_float(row.get(numeric_key, "")) for row in group_rows]
            clean = [v for v in values if v is not None and np.isfinite(v)]
            if clean:
                out[f"{numeric_key}_mean_over_seeds"] = float(np.mean(clean))
                out[f"{numeric_key}_se_over_seeds"] = standard_error(clean)
        out_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in out_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
