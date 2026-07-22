"""Run the KME/CKME feasibility experiment for M/M/1 input uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None

from .ckme_cdf import candidate_taus_from_distances, make_cdf_grid, tune_krr_cdf
from .config import (
    DEFAULT_DGP,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_FIT,
    DEFAULT_N_SEEDS,
    DEFAULT_N_TEST,
    DEFAULT_N_VAL,
    DEFAULT_R_ORACLE,
    DEFAULT_R_TRAIN,
    DEFAULT_RFF_DIM,
    DEFAULT_SEED,
)
from .dgp import generate_scenarios, scenario_outputs, training_output_reps
from .features import KMEFeatureBuilder, Standardizer, rho_hat
from .metrics import build_metrics_row, empirical_cdf, scenario_diagnostics


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_prefix(out_prefix: str) -> Path:
    path = Path(out_prefix)
    if path.is_absolute():
        return path
    return package_root() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgp", default=DEFAULT_DGP, choices=["exp_mm1"])
    parser.add_argument("--n-fit", type=int, default=DEFAULT_N_FIT)
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--r-train", type=int, default=DEFAULT_R_TRAIN)
    parser.add_argument("--r-oracle", type=int, default=DEFAULT_R_ORACLE)
    parser.add_argument("--rff-dim", type=int, default=DEFAULT_RFF_DIM)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-prefix", default="results/kme_feas_exp_mm1")
    parser.add_argument(
        "--save-grid-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one prediction row per test scenario and grid threshold.",
    )
    return parser


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_rows(
    writer: csv.DictWriter,
    seed: int,
    pred_cdf: np.ndarray,
    raw_cdf: np.ndarray,
    oracle_cdf: np.ndarray,
    t_grid: np.ndarray,
    scenarios,
) -> None:
    for scenario_id, scenario in enumerate(scenarios):
        rho_est = rho_hat(scenario.inter_arrivals, scenario.services)
        for grid_index, threshold in enumerate(t_grid):
            writer.writerow(
                {
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "grid_index": grid_index,
                    "t": float(threshold),
                    "pred_cdf": float(pred_cdf[scenario_id, grid_index]),
                    "raw_cdf": float(raw_cdf[scenario_id, grid_index]),
                    "oracle_cdf": float(oracle_cdf[scenario_id, grid_index]),
                    "n1": scenario.n_a,
                    "n2": scenario.n_s,
                    "rho_hat": float(rho_est),
                    "true_rho": float(scenario.rho),
                }
            )


def run_one_seed(args: argparse.Namespace, seed: int) -> tuple[dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fit_scenarios = generate_scenarios(args.n_fit, rng, dgp=args.dgp)
    val_scenarios = generate_scenarios(args.n_val, rng, dgp=args.dgp)
    test_scenarios = generate_scenarios(args.n_test, rng, dgp=args.dgp)

    fit_y = training_output_reps(fit_scenarios, rng, args.r_train)
    val_y = np.array([s.y for s in val_scenarios], dtype=float)

    builder = KMEFeatureBuilder(rff_dim=args.rff_dim).fit(fit_scenarios, rng)
    z_fit_raw = builder.transform(fit_scenarios)
    z_val_raw = builder.transform(val_scenarios)
    z_test_raw = builder.transform(test_scenarios)
    standardizer = Standardizer().fit(z_fit_raw)
    z_fit = standardizer.transform(z_fit_raw)
    z_val = standardizer.transform(z_val_raw)
    z_test = standardizer.transform(z_test_raw)

    t_grid = make_cdf_grid(fit_y, args.grid_size)
    tau_candidates = candidate_taus_from_distances(z_fit)
    estimator, best = tune_krr_cdf(
        z_fit=z_fit,
        y_fit=fit_y,
        z_val=z_val,
        y_val=val_y,
        t_grid=t_grid,
        tau_candidates=tau_candidates,
    )

    raw_cdf = estimator.predict_raw(z_test)
    pred_cdf = estimator.predict_cdf(z_test)
    oracle_cdf = np.empty_like(pred_cdf)
    iterator = range(len(test_scenarios))
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"oracle seed {seed}", leave=False)
    for idx in iterator:
        oracle_outputs = scenario_outputs(test_scenarios[idx], rng, args.r_oracle)
        oracle_cdf[idx] = empirical_cdf(oracle_outputs, t_grid)

    diagnostics = scenario_diagnostics(pred_cdf, raw_cdf, oracle_cdf, t_grid, test_scenarios)
    metrics_row = build_metrics_row(
        seed=seed,
        dgp=args.dgp,
        n_fit=args.n_fit,
        n_val=args.n_val,
        n_test=args.n_test,
        r_train=args.r_train,
        r_oracle=args.r_oracle,
        rff_dim=args.rff_dim,
        grid_size=args.grid_size,
        selected_tau=float(best["tau"]),
        selected_ridge=float(best["ridge"]),
        val_mse=float(best["val_mse"]),
        diagnostics=diagnostics,
    )
    hyper_row: dict[str, object] = {
        "seed": seed,
        "dgp": args.dgp,
        "selected_tau": float(best["tau"]),
        "selected_ridge": float(best["ridge"]),
        "val_mse": float(best["val_mse"]),
        "solver": best["solver"],
        "tau_candidates": json.dumps(tau_candidates),
        "ridge_candidates": json.dumps([1e-6, 1e-4, 1e-2, 1e-1, 1.0]),
        "rff_dim": args.rff_dim,
        "rff_sigma_arrival": float(builder.rff_a.sigma) if builder.rff_a is not None else np.nan,
        "rff_sigma_service": float(builder.rff_s.sigma) if builder.rff_s is not None else np.nan,
    }
    prediction_payload = {
        "pred_cdf": pred_cdf,
        "raw_cdf": raw_cdf,
        "oracle_cdf": oracle_cdf,
        "t_grid": t_grid,
        "test_scenarios": test_scenarios,
    }
    return metrics_row, hyper_row, prediction_payload


def main() -> None:
    args = build_parser().parse_args()
    prefix = resolve_prefix(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    predictions_path = prefix.with_name(prefix.name + "_test_predictions.csv")
    hyperparams_path = prefix.with_name(prefix.name + "_hyperparams.csv")
    config_path = prefix.with_name(prefix.name + "_config.json")

    config = vars(args).copy()
    config["resolved_out_prefix"] = str(prefix)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    prediction_writer = None
    prediction_handle = None
    if args.save_grid_predictions:
        prediction_handle = predictions_path.open("w", newline="", encoding="utf-8")
        prediction_writer = csv.DictWriter(
            prediction_handle,
            fieldnames=[
                "seed",
                "scenario_id",
                "grid_index",
                "t",
                "pred_cdf",
                "raw_cdf",
                "oracle_cdf",
                "n1",
                "n2",
                "rho_hat",
                "true_rho",
            ],
        )
        prediction_writer.writeheader()

    metrics_rows: list[dict[str, object]] = []
    hyper_rows: list[dict[str, object]] = []
    seeds = [args.seed + i for i in range(args.n_seeds)]
    try:
        for seed in seeds:
            metrics_row, hyper_row, payload = run_one_seed(args, seed)
            metrics_rows.append(metrics_row)
            hyper_rows.append(hyper_row)
            if prediction_writer is not None:
                write_prediction_rows(
                    prediction_writer,
                    seed,
                    payload["pred_cdf"],
                    payload["raw_cdf"],
                    payload["oracle_cdf"],
                    payload["t_grid"],
                    payload["test_scenarios"],
                )
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    write_rows(metrics_path, metrics_rows)
    write_rows(hyperparams_path, hyper_rows)
    if not args.save_grid_predictions:
        predictions_path.write_text("grid predictions disabled\n", encoding="utf-8")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote hyperparameters: {hyperparams_path}")
    print(f"Wrote config: {config_path}")


if __name__ == "__main__":
    main()
