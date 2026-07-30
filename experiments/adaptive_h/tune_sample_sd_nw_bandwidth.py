"""Scale-only tuning for the sample-SD/NW adaptive-h estimator.

This script does not train CKME and does not run split conformal calibration.
It only checks whether the Stage 1 replicated data can recover the oracle
scale shape s(x) well enough for h(x) = c * sigma_hat(x).

Recommended use:
    python experiments/adaptive_h/tune_sample_sd_nw_bandwidth.py --simulators exp2_gauss_low
    python experiments/adaptive_h/tune_sample_sd_nw_bandwidth.py --simulators exp2_gauss_low,wsc_gauss,nongauss_A1L
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ckme_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ckme_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

_root = Path(__file__).resolve().parents[2]
# Keep direct script entrypoints working before an editable install.
for _import_path in (_root / "src", _root):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.adaptive_h.adaptive_bandwidth import ORACLE_SCALE
from experiments.adaptive_h.sample_sd_nw_scale import SampleSdNwScaleEstimator
from Two_stage.data_collection import collect_stage1_data
from Two_stage.sim_functions import get_experiment_config

DEFAULT_BW_FACTORS = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 1.0]
SCALE_FLOOR = 1e-3


def _parse_str_list(value: str) -> list[str]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one simulator must be provided")
    return vals


def _parse_float_list(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one bandwidth factor must be provided")
    if any(v <= 0 for v in vals):
        raise ValueError("All bandwidth factors must be positive")
    return vals


def _evaluate_one(
    simulator: str,
    macrorep: int,
    seed: int,
    n_0: int,
    r_0: int,
    n_eval: int,
    design_method: str,
    bw_factors: list[float],
) -> list[dict]:
    if simulator not in ORACLE_SCALE:
        raise ValueError(
            f"No oracle scale defined for {simulator}; valid: {sorted(ORACLE_SCALE)}"
        )

    cfg = get_experiment_config(simulator)
    if cfg["d"] != 1:
        raise ValueError(
            "tune_sample_sd_nw_bandwidth currently supports 1D simulators only"
        )
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])

    X_all, Y_all = collect_stage1_data(
        n_0=n_0,
        d=cfg["d"],
        r_0=r_0,
        simulator_func=simulator,
        X_bounds=cfg["bounds"],
        design_method=design_method,
        random_state=seed,
    )

    x_eval = np.linspace(x_lo, x_hi, n_eval)
    X_eval = x_eval.reshape(-1, 1)
    s_true = np.maximum(ORACLE_SCALE[simulator](x_eval), SCALE_FLOOR)

    rows: list[dict] = []
    for bw_factor in bw_factors:
        est = SampleSdNwScaleEstimator.fit(X_all, Y_all, n_0=n_0, r_0=r_0, bw_factor=bw_factor)
        s_hat = np.maximum(est.predict(X_eval), SCALE_FLOOR)
        ratio = s_hat / s_true
        log_err = np.log(ratio)

        rows.append(
            {
                "simulator": simulator,
                "macrorep": macrorep,
                "n_0": n_0,
                "r_0": r_0,
                "design_method": design_method,
                "bw_factor": float(bw_factor),
                "bw_mean": float(np.mean(est.bw)),
                "scale_rmse": float(np.sqrt(np.mean((s_hat - s_true) ** 2))),
                "log_scale_error_mean": float(np.mean(log_err)),
                "log_scale_error_rmse": float(np.sqrt(np.mean(log_err ** 2))),
                "log_scale_error_sup": float(np.max(np.abs(log_err))),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_sd": float(np.std(ratio, ddof=1)),
                "ratio_min": float(np.min(ratio)),
                "ratio_max": float(np.max(ratio)),
                "floor_frac": float(np.mean(s_hat <= SCALE_FLOOR * (1.0 + 1e-12))),
            }
        )
    return rows


def _summarize(per_macro: pd.DataFrame) -> pd.DataFrame:
    df = per_macro.copy()
    df["ratio_range"] = df["ratio_max"] - df["ratio_min"]
    return (
        df.groupby(["simulator", "bw_factor"], as_index=False)
        .agg(
            mean_scale_rmse=("scale_rmse", "mean"),
            sd_scale_rmse=("scale_rmse", "std"),
            mean_log_rmse=("log_scale_error_rmse", "mean"),
            sd_log_rmse=("log_scale_error_rmse", "std"),
            mean_log_sup=("log_scale_error_sup", "mean"),
            mean_ratio_sd=("ratio_sd", "mean"),
            mean_ratio_range=("ratio_range", "mean"),
            mean_floor_frac=("floor_frac", "mean"),
            n_macro=("macrorep", "count"),
        )
        .sort_values(["simulator", "bw_factor"])
        .reset_index(drop=True)
    )


def _plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    sims = list(summary["simulator"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)

    for sim in sims:
        sub = summary[summary["simulator"] == sim].sort_values("bw_factor")
        axes[0].plot(sub["bw_factor"], sub["mean_log_rmse"], "o-", label=sim)
        axes[1].plot(sub["bw_factor"], sub["mean_ratio_sd"], "s-", label=sim)
        axes[2].plot(sub["bw_factor"], sub["mean_log_sup"], "^-", label=sim)

    axes[0].set_ylabel("mean log-scale RMSE")
    axes[1].set_ylabel("mean sd(sigma_hat / sigma)")
    axes[2].set_ylabel("mean sup |log error|")
    for ax in axes:
        ax.set_xlabel("SampleSdNwScaleEstimator bw_factor")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Sample-SD/NW scale tuning: smoothing bandwidth", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_sd_nw_bandwidth_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune SampleSdNwScaleEstimator kernel bandwidth against oracle scale functions."
    )
    parser.add_argument("--simulators", type=str, default="exp2_gauss_low")
    parser.add_argument(
        "--bw-factors",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BW_FACTORS),
    )
    parser.add_argument("--n-macro", type=int, default=20)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--r0", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=400)
    parser.add_argument("--design-method", type=str, default="grid", choices=["grid", "lhs"])
    parser.add_argument("--base-seed", type=int, default=20260624)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/adaptive_h/output_sample_sd_nw_tuning",
    )
    args = parser.parse_args()

    simulators = _parse_str_list(args.simulators)
    bw_factors = _parse_float_list(args.bw_factors)
    # Relative output paths are interpreted from the repository root.
    out_dir = (_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sim_idx, sim in enumerate(simulators):
        print(f"\n=== {sim} ===")
        for k in range(args.n_macro):
            seed = args.base_seed + 10000 * sim_idx + k
            one = _evaluate_one(
                simulator=sim,
                macrorep=k,
                seed=seed,
                n_0=args.n0,
                r_0=args.r0,
                n_eval=args.n_eval,
                design_method=args.design_method,
                bw_factors=bw_factors,
            )
            rows.extend(one)
            best = min(one, key=lambda r: r["log_scale_error_rmse"])
            print(
                f"  k={k:3d} best_bw_factor={best['bw_factor']:.3g} "
                f"log_rmse={best['log_scale_error_rmse']:.3f} "
                f"ratio_sd={best['ratio_sd']:.3f}"
            )

    per_macro = pd.DataFrame(rows)
    summary = _summarize(per_macro)

    per_path = out_dir / "sample_sd_nw_tuning_per_macro.csv"
    sum_path = out_dir / "sample_sd_nw_tuning_summary.csv"
    per_macro.to_csv(per_path, index=False)
    summary.to_csv(sum_path, index=False)
    _plot_summary(summary, out_dir)

    print(f"\nWrote {per_path}")
    print(f"Wrote {sum_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
