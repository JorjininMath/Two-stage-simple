"""Run the protocol-aligned final adaptive-bandwidth benchmark.

The benchmark isolates output-side bandwidth adaptation. Stage 1 uses a
replicated grid, while calibration and test pairs are independent iid draws
from the target law q_X. All three arms share Stage-1, calibration, and test
data within a macrorep:

``fixed``
    Scalar response bandwidth selected before calibration.
``plugin_sd_nw``
    Query bandwidth ``h(x) = c * s_hat(x)`` from Stage-1 sample SDs smoothed
    by Nadaraya-Watson regression.
``oracle``
    Query bandwidth ``h(x) = c * s(x)`` using the known DGP scale.

Coverage is computed from raw point-evaluated scores. Reported intervals use
the shared monotone-projected CDF inversion in ``CP.interval``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (_ROOT / "src", _ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import numpy as np
import pandas as pd
import scipy

from CKME.parameters import Params
from CP.evaluation import compute_interval_score
from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from Two_stage.test_data import generate_test_data
from experiments.adaptive_h.adaptive_bandwidth import (
    ORACLE_SCALE,
    adaptive_point_scores,
    adaptive_predict_interval,
    adaptive_recalibrate_q,
    get_oracle_h,
)
from experiments.adaptive_h.sample_sd_nw_scale import (
    SampleSdNwScaleEstimator,
)

DEFAULT_CONFIG = Path(__file__).with_name("final_benchmark_config.json")
DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")
ARMS = ("fixed", "plugin_sd_nw", "oracle")
METRICS = (
    "coverage",
    "coverage_interval",
    "width",
    "interval_score",
    "mean_group_coverage_gap",
    "worst_group_coverage_gap",
)


def _parse_csv(value: str, cast=str) -> list:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one comma-separated value is required")
    return values


def _resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(_ROOT), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _seed_bundle(
    base_seed: int,
    macrorep: int,
    simulator_index: int,
    budget_index: int,
) -> dict[str, int]:
    root = (
        int(base_seed)
        + int(macrorep) * 100_000
        + int(simulator_index) * 10_000
        + int(budget_index) * 1_000
    )
    return {
        "stage1_design": root + 11,
        "stage1_output": root + 12,
        "calibration_x": root + 21,
        "calibration_y": root + 22,
        "test_x": root + 31,
        "test_y": root + 32,
    }


def _equal_count_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        return np.empty(0, dtype=int)
    bins = min(int(n_bins), values.size)
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(values.size, dtype=int)
    labels[order] = np.floor(
        np.arange(values.size, dtype=float) * bins / values.size
    ).astype(int)
    return labels


def _arm_frame(
    *,
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    t_grid: np.ndarray,
    alpha: float,
    q_hat: float,
    h_query: np.ndarray,
    s_oracle: np.ndarray,
    s_hat: np.ndarray,
    group_bin: np.ndarray,
    arm: str,
    macrorep: int,
    simulator: str,
    budget: int,
) -> pd.DataFrame:
    raw_score = adaptive_point_scores(model, X_test, Y_test, h_query)
    covered_score = (raw_score <= q_hat).astype(int)
    L, U = adaptive_predict_interval(
        model, X_test, h_query, t_grid, q_hat
    )
    covered_interval = ((Y_test >= L) & (Y_test <= U)).astype(int)
    interval_score, _ = compute_interval_score(Y_test, L, U, alpha)
    t_lo, t_hi = float(t_grid[0]), float(t_grid[-1])

    frame = pd.DataFrame(
        {
            "macrorep": macrorep,
            "simulator": simulator,
            "budget": budget,
            "arm": arm,
            "test_index": np.arange(len(Y_test), dtype=int),
            "x0": np.asarray(X_test)[:, 0],
            "y": Y_test,
            "L": L,
            "U": U,
            "covered_interval": covered_interval,
            "covered_score": covered_score,
            "raw_score": raw_score,
            "width": U - L,
            "interval_score": interval_score,
            "status": np.where(
                Y_test < L, "below", np.where(Y_test > U, "above", "in")
            ),
            "h_query": h_query,
            "s_oracle": s_oracle,
            "s_hat": s_hat,
            "h_over_s": h_query / np.maximum(s_oracle, 1e-12),
            "group_bin": group_bin,
            "y_in_grid": ((Y_test >= t_lo) & (Y_test <= t_hi)).astype(int),
            "L_at_grid_lo": (L <= t_lo + 1e-9).astype(int),
            "U_at_grid_hi": (U >= t_hi - 1e-9).astype(int),
            "score_interval_disagree": (
                covered_score != covered_interval
            ).astype(int),
            "q_hat": q_hat,
            "t_grid_lo": t_lo,
            "t_grid_hi": t_hi,
        }
    )
    X_2d = np.atleast_2d(X_test)
    for column in range(1, X_2d.shape[1]):
        frame[f"x{column}"] = X_2d[:, column]
    return frame


def _aggregate_arm(
    frame: pd.DataFrame,
    *,
    n_0: int,
    r_0: int,
    n_cal: int,
    n_test: int,
    params: Params,
    seeds: dict[str, int],
    runtime_seconds: float,
) -> dict[str, Any]:
    target = 1.0 - float(frame.attrs["alpha"])
    group_cov = frame.groupby("group_bin")["covered_score"].mean()
    group_gap = (group_cov - target).abs()
    return {
        "macrorep": int(frame["macrorep"].iloc[0]),
        "simulator": str(frame["simulator"].iloc[0]),
        "budget": int(frame["budget"].iloc[0]),
        "n_0": n_0,
        "r_0": r_0,
        "n_cal": n_cal,
        "n_test": n_test,
        "arm": str(frame["arm"].iloc[0]),
        "coverage": float(frame["covered_score"].mean()),
        "coverage_interval": float(frame["covered_interval"].mean()),
        "score_interval_disagreement": float(
            frame["score_interval_disagree"].mean()
        ),
        "width": float(frame["width"].mean()),
        "interval_score": float(frame["interval_score"].mean()),
        "mean_group_coverage_gap": float(group_gap.mean()),
        "worst_group_coverage_gap": float(group_gap.max()),
        "q_hat": float(frame["q_hat"].iloc[0]),
        "mean_h": float(frame["h_query"].mean()),
        "mean_h_over_s": float(frame["h_over_s"].mean()),
        "sd_h_over_s": float(frame["h_over_s"].std(ddof=1)),
        "mean_abs_scale_relative_error": float(
            np.mean(
                np.abs(frame["s_hat"] / np.maximum(frame["s_oracle"], 1e-12) - 1)
            )
        ),
        "y_outside_grid_rate": float(1.0 - frame["y_in_grid"].mean()),
        "lower_grid_clip_rate": float(frame["L_at_grid_lo"].mean()),
        "upper_grid_clip_rate": float(frame["U_at_grid_hi"].mean()),
        "ell_x": float(params.ell_x),
        "lam": float(params.lam),
        "h_fixed": float(params.h),
        "runtime_seconds": float(runtime_seconds),
        **{f"seed_{key}": value for key, value in seeds.items()},
    }


def _job_dir(
    output_dir: Path, simulator: str, budget: int, macrorep: int
) -> Path:
    return (
        output_dir
        / "jobs"
        / simulator
        / f"budget_{budget}"
        / f"macrorep_{macrorep:03d}"
    )


def _job_complete(path: Path) -> bool:
    status_path = path / "job_manifest.json"
    if not status_path.exists() or not (path / "per_arm.csv").exists():
        return False
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "complete" and all(
        (path / f"per_point_{arm}.csv").exists() for arm in ARMS
    )


def run_one_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    started = time.perf_counter()
    output_dir = Path(job["output_dir"])
    simulator = str(job["simulator"])
    budget = int(job["budget"])
    macrorep = int(job["macrorep"])
    job_dir = _job_dir(output_dir, simulator, budget, macrorep)
    if _job_complete(job_dir) and not bool(job["overwrite"]):
        return pd.read_csv(job_dir / "per_arm.csv").to_dict("records")

    r_0 = int(job["r0"])
    if budget % r_0:
        raise ValueError(f"budget={budget} is not divisible by r0={r_0}")
    n_0 = budget // r_0
    seeds = dict(job["seeds"])
    params = Params(**job["params"])
    alpha = float(job["alpha"])
    c_scale = float(job["c_scale"])

    stage1 = run_stage1_train(
        n_0=n_0,
        r_0=r_0,
        simulator_func=simulator,
        params=params,
        design_method=str(job["stage1_design"]),
        t_grid_size=int(job["t_grid_size"]),
        t_grid_margin=float(job["t_grid_margin"]),
        random_state=seeds["stage1_design"],
        verbose=False,
    )
    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=None,
        n_1=int(job["n_cal"]),
        r_1=1,
        simulator_func=simulator,
        method="iid",
        alpha=alpha,
        random_state=seeds["calibration_x"],
        sim_random_state=seeds["calibration_y"],
        verbose=False,
    )
    X_test, Y_test = generate_test_data(
        stage2_result=stage2,
        n_test=int(job["n_test"]),
        r_test=1,
        X_cand=None,
        simulator_func=simulator,
        random_state=seeds["test_x"],
        sim_random_state=seeds["test_y"],
    )
    Y_test = np.asarray(Y_test, dtype=float).ravel()

    scale_estimator = SampleSdNwScaleEstimator.fit(
        stage1.X_all,
        stage1.Y_all,
        n_0,
        r_0,
        bw_factor=float(job["scale_bw_factor"]),
    )
    s_hat_cal = scale_estimator.predict(stage2.X_stage2)
    s_hat_test = scale_estimator.predict(X_test)
    s_oracle_cal = np.asarray(
        ORACLE_SCALE[simulator](stage2.X_stage2), dtype=float
    ).ravel()
    s_oracle_test = np.asarray(
        ORACLE_SCALE[simulator](X_test), dtype=float
    ).ravel()
    group_bin = _equal_count_bins(X_test[:, 0], int(job["group_bins"]))

    h_by_arm = {
        "fixed": np.full(len(Y_test), float(params.h)),
        "plugin_sd_nw": np.maximum(c_scale * s_hat_test, 1e-3),
        "oracle": get_oracle_h(simulator, X_test, c_scale),
    }
    h_cal_by_arm = {
        "fixed": np.full(len(stage2.Y_stage2), float(params.h)),
        "plugin_sd_nw": np.maximum(c_scale * s_hat_cal, 1e-3),
        "oracle": np.maximum(c_scale * s_oracle_cal, 1e-3),
    }
    fixed_q_check = adaptive_recalibrate_q(
        stage2.model,
        stage2.X_stage2,
        stage2.Y_stage2,
        h_cal_by_arm["fixed"],
        alpha,
        t_grid=None,
    )
    if not np.isclose(
        fixed_q_check, float(stage2.cp.q_hat), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "Fixed-arm raw-score calibration disagrees with the canonical "
            f"CP implementation: {fixed_q_check} versus {stage2.cp.q_hat}"
        )
    q_by_arm = {
        "fixed": float(stage2.cp.q_hat),
        "plugin_sd_nw": adaptive_recalibrate_q(
            stage2.model,
            stage2.X_stage2,
            stage2.Y_stage2,
            h_cal_by_arm["plugin_sd_nw"],
            alpha,
            t_grid=None,
        ),
        "oracle": adaptive_recalibrate_q(
            stage2.model,
            stage2.X_stage2,
            stage2.Y_stage2,
            h_cal_by_arm["oracle"],
            alpha,
            t_grid=None,
        ),
    }

    arm_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        frame = _arm_frame(
            model=stage2.model,
            X_test=X_test,
            Y_test=Y_test,
            t_grid=stage2.t_grid,
            alpha=alpha,
            q_hat=q_by_arm[arm],
            h_query=h_by_arm[arm],
            s_oracle=s_oracle_test,
            s_hat=s_hat_test,
            group_bin=group_bin,
            arm=arm,
            macrorep=macrorep,
            simulator=simulator,
            budget=budget,
        )
        frame.attrs["alpha"] = alpha
        frames[arm] = frame
        arm_rows.append(
            _aggregate_arm(
                frame,
                n_0=n_0,
                r_0=r_0,
                n_cal=int(job["n_cal"]),
                n_test=int(job["n_test"]),
                params=params,
                seeds=seeds,
                runtime_seconds=time.perf_counter() - arm_started,
            )
        )

    job_dir.mkdir(parents=True, exist_ok=True)
    for arm, frame in frames.items():
        _atomic_csv(frame, job_dir / f"per_point_{arm}.csv")
    arm_frame = pd.DataFrame(arm_rows)
    _atomic_csv(arm_frame, job_dir / "per_arm.csv")
    _atomic_json(
        job_dir / "job_manifest.json",
        {
            "status": "complete",
            "schema_version": job["schema_version"],
            "simulator": simulator,
            "budget": budget,
            "macrorep": macrorep,
            "seeds": seeds,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return arm_rows


def _paired_deltas(per_arm: pd.DataFrame) -> pd.DataFrame:
    keys = ["macrorep", "simulator", "budget"]
    indexed = per_arm.set_index(keys + ["arm"])
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("plugin_sd_nw_minus_oracle", "plugin_sd_nw", "oracle"),
        ("plugin_sd_nw_minus_fixed", "plugin_sd_nw", "fixed"),
        ("oracle_minus_fixed", "oracle", "fixed"),
    )
    for key, group in per_arm.groupby(keys, sort=True):
        available = set(group["arm"])
        for label, left, right in comparisons:
            if left not in available or right not in available:
                continue
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


def _summary(per_arm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
            row[f"mcse_{metric}"] = sd / np.sqrt(count) if count > 1 else float("nan")
        row["mean_q_hat"] = float(group["q_hat"].mean())
        row["mean_h"] = float(group["mean_h"].mean())
        row["mean_score_interval_disagreement"] = float(
            group["score_interval_disagreement"].mean()
        )
        row["mean_y_outside_grid_rate"] = float(
            group["y_outside_grid_rate"].mean()
        )
        row["mean_lower_grid_clip_rate"] = float(
            group["lower_grid_clip_rate"].mean()
        )
        row["mean_upper_grid_clip_rate"] = float(
            group["upper_grid_clip_rate"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_completed(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "jobs").glob("*/budget_*/macrorep_*/per_arm.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _write_aggregate_outputs(output_dir: Path) -> pd.DataFrame:
    per_arm = _collect_completed(output_dir)
    if per_arm.empty:
        return per_arm
    per_arm = per_arm.sort_values(
        ["simulator", "budget", "macrorep", "arm"]
    ).reset_index(drop=True)
    _atomic_csv(per_arm, output_dir / "per_arm.csv")
    _atomic_csv(_paired_deltas(per_arm), output_dir / "paired_deltas.csv")
    _atomic_csv(_summary(per_arm), output_dir / "summary.csv")
    return per_arm


def _resolved_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = _resolve_from_root(args.config)
    config = json.loads(config_path.read_text())
    if args.simulators:
        config["simulators"] = _parse_csv(args.simulators)
    if args.budgets:
        config["budgets"] = _parse_csv(args.budgets, int)
    for name in ("n_macro", "n_cal", "n_test", "t_grid_size", "base_seed"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.pretrained_path:
        config["pretrained_params_path"] = args.pretrained_path
    return config, config_path


def _validate_config(
    config: dict[str, Any], pretrained: dict[str, dict[str, float]]
) -> None:
    if config.get("calibration_method") != "iid" or int(config["r_cal"]) != 1:
        raise ValueError("Final calibration must be iid with r_cal=1")
    if config.get("test_method") != "iid" or int(config["r_test"]) != 1:
        raise ValueError("Final test data must be iid with r_test=1")
    if config.get("stage1_design") != "grid":
        raise ValueError("The one-dimensional final benchmark uses a Stage-1 grid")
    if float(config["c_scale"]) <= 0 or float(config["scale_bw_factor"]) <= 0:
        raise ValueError("c_scale and scale_bw_factor must be positive")
    if int(config["r0"]) < 2:
        raise ValueError("r0 must be at least 2 for the sample-SD plug-in")
    missing_params = set(config["simulators"]) - set(pretrained)
    if missing_params:
        raise ValueError(
            f"Missing pretrained parameters for {sorted(missing_params)}"
        )
    missing_oracle = set(config["simulators"]) - set(ORACLE_SCALE)
    if missing_oracle:
        raise ValueError(f"Missing oracle scale for {sorted(missing_oracle)}")
    for simulator in config["simulators"]:
        dgp = get_experiment_config(simulator)
        if int(dgp["d"]) != 1:
            raise ValueError(
                f"Final benchmark expects one-dimensional DGPs; {simulator} "
                f"has d={dgp['d']}"
            )
    for budget in config["budgets"]:
        if int(budget) % int(config["r0"]):
            raise ValueError(
                f"budget={budget} is not divisible by r0={config['r0']}"
            )


def _manifest(
    config: dict[str, Any],
    config_path: Path,
    pretrained_path: Path,
    pretrained: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    for simulator in config["simulators"]:
        lower, upper = get_experiment_config(simulator)["bounds"]
        bounds[simulator] = {
            "distribution": "uniform",
            "lower": np.asarray(lower, dtype=float).ravel().tolist(),
            "upper": np.asarray(upper, dtype=float).ravel().tolist(),
        }
    scientific_hash = _canonical_hash(
        {"config": config, "pretrained_params": pretrained}
    )
    return {
        "schema_version": config["schema_version"],
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_config_hash": scientific_hash,
        "config_path": str(config_path.relative_to(_ROOT)),
        "config_sha256": _file_hash(config_path),
        "resolved_config": config,
        "pretrained_params_path": str(pretrained_path.relative_to(_ROOT)),
        "pretrained_params_sha256": _file_hash(pretrained_path),
        "pretrained_params": pretrained,
        "target_laws": bounds,
        "seed_scheme": {
            "formula": (
                "base + macrorep*100000 + simulator_index*10000 "
                "+ budget_index*1000 + named_offset"
            ),
            "offsets": {
                "stage1_design": 11,
                "stage1_output": 12,
                "calibration_x": 21,
                "calibration_y": 22,
                "test_x": 31,
                "test_y": 32,
            },
        },
        "uses_s0": False,
        "calibration_guarantee_layer": "raw_point_score",
        "reporting_interval_layer": "monotone_projected_cdf_generalized_inverse",
        "git": _git_state(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "command": sys.argv,
        "n_workers": int(args.n_workers),
        "executor": args.executor,
    }


def build_jobs(
    config: dict[str, Any],
    pretrained: dict[str, dict[str, float]],
    output_dir: Path,
    overwrite: bool,
) -> Iterable[dict[str, Any]]:
    for simulator_index, simulator in enumerate(config["simulators"]):
        for budget_index, budget in enumerate(config["budgets"]):
            for macrorep in range(int(config["n_macro"])):
                yield {
                    "schema_version": config["schema_version"],
                    "output_dir": str(output_dir),
                    "overwrite": overwrite,
                    "simulator": simulator,
                    "budget": int(budget),
                    "macrorep": macrorep,
                    "r0": int(config["r0"]),
                    "stage1_design": config["stage1_design"],
                    "n_cal": int(config["n_cal"]),
                    "n_test": int(config["n_test"]),
                    "alpha": float(config["alpha"]),
                    "c_scale": float(config["c_scale"]),
                    "scale_bw_factor": float(config["scale_bw_factor"]),
                    "group_bins": int(config["group_bins"]),
                    "t_grid_size": int(config["t_grid_size"]),
                    "t_grid_margin": float(
                        config["t_grid_margin"][simulator]
                    ),
                    "params": pretrained[simulator],
                    "seeds": _seed_bundle(
                        int(config["base_seed"]),
                        macrorep,
                        simulator_index,
                        budget_index,
                    ),
                }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final iid-calibrated adaptive-h benchmark"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pretrained-path", "--pretrained_path")
    parser.add_argument("--simulators")
    parser.add_argument("--budgets")
    parser.add_argument("--n-macro", "--n_macro", dest="n_macro", type=int)
    parser.add_argument("--n-cal", "--n_cal", dest="n_cal", type=int)
    parser.add_argument("--n-test", "--n_test", dest="n_test", type=int)
    parser.add_argument(
        "--t-grid-size", "--t_grid_size", dest="t_grid_size", type=int
    )
    parser.add_argument("--base-seed", "--base_seed", dest="base_seed", type=int)
    parser.add_argument("--n-workers", "--n_workers", type=int, default=1)
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Parallel backend when n_workers > 1. Threads avoid semaphore "
            "restrictions and work well when BLAS thread counts are set to 1."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, config_path = _resolved_config(args)
    output_dir = _resolve_from_root(args.output_dir)
    pretrained_path = _resolve_from_root(config["pretrained_params_path"])
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained parameters not found: {pretrained_path}. Run "
            "experiments/adaptive_h/pretrain_params.py for the final DGPs first."
        )
    pretrained_all = json.loads(pretrained_path.read_text())
    pretrained = {
        simulator: pretrained_all[simulator]
        for simulator in config["simulators"]
        if simulator in pretrained_all
    }
    _validate_config(config, pretrained)
    manifest = _manifest(
        config, config_path, pretrained_path, pretrained, args
    )

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("scientific_config_hash")
            != manifest["scientific_config_hash"]
        ):
            raise RuntimeError(
                f"{output_dir} contains a different scientific configuration. "
                "Use a new output directory or pass --overwrite explicitly."
            )
        manifest["created_at_utc"] = existing.get(
            "created_at_utc", manifest["created_at_utc"]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)

    jobs = list(build_jobs(config, pretrained, output_dir, args.overwrite))
    pending = [
        job
        for job in jobs
        if args.overwrite
        or not _job_complete(
            _job_dir(
                output_dir,
                job["simulator"],
                job["budget"],
                job["macrorep"],
            )
        )
    ]
    print(
        f"Validated {len(jobs)} jobs; {len(pending)} pending; "
        f"output={output_dir}"
    )
    if args.dry_run:
        manifest["status"] = "dry_run_validated"
        _atomic_json(manifest_path, manifest)
        return

    started = time.perf_counter()
    completed = len(jobs) - len(pending)
    if int(args.n_workers) > 1 and pending:
        executor_class = (
            ThreadPoolExecutor
            if args.executor == "thread"
            else ProcessPoolExecutor
        )
        with executor_class(max_workers=int(args.n_workers)) as pool:
            futures = {pool.submit(run_one_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                future.result()
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] {job['simulator']} "
                    f"B={job['budget']} macrorep={job['macrorep']}"
                )
    else:
        for job in pending:
            run_one_job(job)
            completed += 1
            print(
                f"[{completed}/{len(jobs)}] {job['simulator']} "
                f"B={job['budget']} macrorep={job['macrorep']}"
            )

    per_arm = _write_aggregate_outputs(output_dir)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    manifest["completed_jobs"] = int(
        per_arm[["macrorep", "simulator", "budget"]].drop_duplicates().shape[0]
    )
    manifest["expected_jobs"] = len(jobs)
    _atomic_json(manifest_path, manifest)
    print(f"Completed {len(jobs)} jobs; wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
