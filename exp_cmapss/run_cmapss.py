"""
run_cmapss.py

CKME+CP experiment on C-MAPSS FD001 for RUL uncertainty prediction.

Pipeline
--------
1. Preprocess FD001 → (X_train, Y_train, X_cal, Y_cal, X_test, Y_test)
   - Train:  snapshot levels across full trajectory
   - Cal:    late-stage snapshots [0.80, 0.90, 1.0] (aligned with test)
   - Test:   last observed cycle per engine (benchmark task, 100 points)
2. Train CKMEModel on X_train, Y_train
3. CKME CP: calibrate on late-stage calibration samples
4. Evaluate on test set: overall + late-stage subset
5. Save tables to output/tables/

Usage
-----
python exp_cmapss/run_cmapss.py [--config exp_cmapss/config.txt] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CKME import CKMEModel
from CKME.parameters import Params
from CP import CP
from CP.evaluation import compute_interval_score
from exp_cmapss.preprocess import load_and_preprocess
import json


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            cfg[key.strip()] = val.split("#")[0].strip()
    return cfg


# ---------------------------------------------------------------------------
# CKME helpers
# ---------------------------------------------------------------------------

def train_ckme(X_train, Y_train, params: Params, t_grid_size: int, verbose: bool):
    model = CKMEModel(indicator_type="logistic")
    model.fit(X=X_train, Y=Y_train, params=params, verbose=verbose)
    t_min, t_max = Y_train.min(), Y_train.max()
    # Extend slightly so test RUL values near the boundary are covered
    margin = 0.05 * (t_max - t_min)
    t_grid = np.linspace(t_min - margin, t_max + margin, t_grid_size)
    return model, t_grid


def calibrate_ckme_cp(model, X_cal, Y_cal, alpha, t_grid_size=None):
    cp = CP(model=model, alpha=alpha, score_type="abs_median")
    cp.calibrate(X_cal, Y_cal)
    return cp


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_method(
    name: str,
    L: np.ndarray,
    U: np.ndarray,
    Y_test: np.ndarray,
    rul_test: np.ndarray,
    alpha: float,
    Y_pred: np.ndarray | None = None,
    late_rul: int = 30,
) -> pd.DataFrame:
    """Evaluate one method on full test set and late-stage subset."""
    rows = []
    for subset_name, mask in [
        ("all", np.ones(len(Y_test), dtype=bool)),
        (f"RUL<={late_rul}", rul_test <= late_rul),
    ]:
        n = mask.sum()
        if n == 0:
            continue
        L_s, U_s, Y_s = L[mask], U[mask], Y_test[mask]
        covered = (Y_s >= L_s) & (Y_s <= U_s)
        coverage = covered.mean()
        width = (U_s - L_s).mean()
        _, is_mean = compute_interval_score(Y_s, L_s, U_s, alpha)
        rmse = float(np.sqrt(np.mean((Y_pred[mask] - Y_s) ** 2))) if Y_pred is not None else float("nan")
        rows.append({
            "method": name,
            "subset": subset_name,
            "n": int(n),
            "coverage": round(coverage, 4),
            "width": round(width, 4),
            "interval_score": round(is_mean, 4),
            "rmse": round(rmse, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "exp_cmapss/config.txt", verbose: bool = True):
    cfg = load_config(config_path)

    data_dir       = cfg.get("data_dir", "exp_cmapss/data")
    rul_cap        = int(cfg.get("rul_cap", 125))
    window         = int(cfg.get("window_size", 20))
    train_frac     = float(cfg.get("train_frac", 0.8))
    random_state   = int(cfg.get("random_state", 42))
    t_grid_size    = int(cfg.get("t_grid_size", 300))
    alpha          = float(cfg.get("alpha", 0.1))
    late_stage_rul = int(cfg.get("late_stage_rul", 30))
    output_dir     = Path(cfg.get("output_dir", "exp_cmapss/output"))
    # Load CV-tuned params if available, else fall back to config
    pretrained_json = Path(cfg.get("output_dir", "exp_cmapss/output")).parent / "pretrained_params.json"
    if pretrained_json.exists():
        pt = json.loads(pretrained_json.read_text())
        ell_x = float(pt["ell_x"])
        lam   = float(pt["lam"])
        h     = float(pt["h"])
        if verbose:
            print(f"Using CV-tuned params from {pretrained_json}: ell_x={ell_x}, lam={lam}, h={h}")
    else:
        ell_x = float(cfg.get("ell_x", 1.0))
        lam   = float(cfg.get("lam", 0.01))
        h     = float(cfg.get("h", 0.3))
        if verbose:
            print(f"No pretrained_params.json found, using config defaults: ell_x={ell_x}, lam={lam}, h={h}")

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 0: Preprocess
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Step 0: Preprocessing FD001")
    data = load_and_preprocess(
        data_dir=data_dir,
        rul_cap=rul_cap,
        window=window,
        train_frac=train_frac,
        late_stage_rul=late_stage_rul,
        random_state=random_state,
        verbose=verbose,
    )

    X_train  = data["X_train"]
    Y_train  = data["Y_train"]
    X_cal    = data["X_cal"]
    Y_cal    = data["Y_cal"]
    X_test   = data["X_test"]
    Y_test   = data["Y_test"]
    rul_test = data["rul_test"]

    # -----------------------------------------------------------------------
    # Step 1: Train CKME model
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Step 1: Training CKME model")

    params = Params(ell_x=ell_x, lam=lam, h=h)
    model, t_grid = train_ckme(X_train, Y_train, params, t_grid_size, verbose)

    if verbose:
        print(f"  t_grid: [{t_grid.min():.1f}, {t_grid.max():.1f}], size={len(t_grid)}")

    # -----------------------------------------------------------------------
    # Step 2: CKME CP — calibrate on late-stage calibration samples
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Step 2: CKME CP (calibrate on late-stage cal samples)")

    cp_ckme = calibrate_ckme_cp(model, X_cal, Y_cal, alpha)
    L_ckme, U_ckme = cp_ckme.predict_interval(X_test, t_grid)

    if verbose:
        print(f"  CKME q_hat = {cp_ckme.q_hat:.6f}")

    # -----------------------------------------------------------------------
    # Step 3: Point prediction from CKME (median of predicted CDF)
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Step 3: CKME point prediction (CDF median)")

    F_test = model.predict_cdf(X_test, t_grid)          # (n_test, M)
    median_idx = np.argmax(F_test >= 0.5, axis=1)
    median_idx = np.clip(median_idx, 0, len(t_grid) - 1)
    Y_pred_ckme = t_grid[median_idx]

    # -----------------------------------------------------------------------
    # Step 4: Evaluate
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("Step 4: Evaluation")

    results = []
    results.append(evaluate_method(
        "CKME CP", L_ckme, U_ckme,
        Y_test, rul_test, alpha,
        Y_pred=Y_pred_ckme, late_rul=late_stage_rul,
    ))

    df_results = pd.concat(results, ignore_index=True)

    # -----------------------------------------------------------------------
    # Step 5: Save tables
    # -----------------------------------------------------------------------
    table_path = output_dir / "tables" / "cmapss_results.csv"
    df_results.to_csv(table_path, index=False)

    # Pretty print
    if verbose:
        print("\n" + "=" * 60)
        print("Results:")
        print(df_results.to_string(index=False))
        print(f"\nSaved to {table_path}")

    # Save per-sample predictions for plotting
    pred_df = pd.DataFrame({
        "rul_true": Y_test,
        "rul_pred_ckme": Y_pred_ckme,
        "L_ckme": L_ckme,
        "U_ckme": U_ckme,
    })

    pred_path = output_dir / "tables" / "cmapss_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save table split by subset for quick reference
    for subset in df_results["subset"].unique():
        sub = df_results[df_results["subset"] == subset].drop(columns=["subset"])
        sub.to_csv(output_dir / "tables" / f"cmapss_results_{subset.replace('<=', 'le')}.csv",
                   index=False)

    return df_results, pred_df, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C-MAPSS FD001 CKME+CP experiment")
    parser.add_argument("--config", default="exp_cmapss/config.txt")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    main(config_path=args.config, verbose=args.verbose)
