"""
run_returns.py

CKME + split conformal prediction on Kenneth French daily stock returns,
with R benchmark methods (DCP-QR, DCP-DR, CQR, CP-OLS, CP-loc).

Mirrors the 5-round rolling-window evaluation from Chernozhukov et al. (2021):
  * Round r trains on observations [0 .. 50%+r*10%] of the full sample
  * The last T_ho = 10% of the sample is the test window
  * The training window is split 50/50 into D0 (CKME fit) and D1 (CP calibration)

Output per round:
  exp_returns/output/round_{r}/data/        -- X0/Y0/X1/Y1/X_test/Y_test CSVs
  exp_returns/output/round_{r}/per_point.csv -- per-test-point results (all methods)
  exp_returns/output/round_{r}/binned.csv    -- conditional metrics by volatility bin
  exp_returns/output/returns_summary.csv     -- aggregated table (5 rounds + Overall)

Usage (from project root):
    python exp_returns/run_returns.py
    python exp_returns/run_returns.py --t_grid_size 300 --n_bins 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.ckme import CKMEModel
from CKME.coefficients import compute_ckme_coeffs
from CKME.parameters import Params
from CP.cp import CP
from CP.evaluation import compute_interval_score
from CP.scores import score_from_cdf
from exp_returns.preprocess import load_returns_data, get_round_splits, split_d0_d1

R_SCRIPT = _root / "exp_returns" / "run_benchmarks_returns.R"

# R benchmark method columns present in benchmarks.csv
R_METHODS = {
    "DCP-QR" : ("covered_qr",   "width_qr",   "IS_qr"),
    "DCP-DR" : ("covered_dr",   "width_dr",   "IS_dr"),
    "CQR"    : ("covered_cqr",  "width_cqr",  None),
    "CQR-m"  : ("covered_cqrm", "width_cqrm", None),
    "CQR-r"  : ("covered_cqrr", "width_cqrr", None),
    "CP-OLS" : ("covered_reg",  "width_reg",  "IS_reg"),
    "CP-loc" : ("covered_loc",  "width_loc",  "IS_loc"),
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            try:
                cfg[key] = float(val) if "." in val or "e" in val.lower() else int(val)
            except ValueError:
                cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# R benchmark helper
# ---------------------------------------------------------------------------

def _run_r_benchmarks(data_dir: Path, out_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(data_dir), str(out_csv), str(alpha), str(n_grid)]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not out_csv.exists():
        raise RuntimeError(
            f"R script did not produce {out_csv}.\n"
            f"stderr: {result.stderr or 'none'}\nstdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# Binned conditional metrics (mirrors R's binning function)
# ---------------------------------------------------------------------------

def binned_metrics(
    X_test: np.ndarray,
    df_pp: pd.DataFrame,
    n_bins: int = 20,
) -> pd.DataFrame:
    x = X_test.ravel()
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, quantiles)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    # Identify coverage and width columns for all methods
    cov_cols   = [c for c in df_pp.columns if c.startswith("covered")]
    width_cols = [c for c in df_pp.columns if c.startswith("width")]

    rows = []
    for b in range(n_bins):
        mask = (x > edges[b]) & (x <= edges[b + 1])
        if mask.sum() == 0:
            continue
        row = {
            "bin"  : b + 1,
            "x_lo" : edges[b],
            "x_hi" : edges[b + 1],
            "x_mid": 0.5 * (edges[b] + edges[b + 1]),
            "n"    : int(mask.sum()),
        }
        for col in cov_cols + width_cols:
            row[col] = float(df_pp.loc[mask, col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single round
# ---------------------------------------------------------------------------

def run_one_round(
    r: int,
    Y: np.ndarray,
    X: np.ndarray,
    ind_cp: np.ndarray,
    ind_test: np.ndarray,
    params: Params,
    alpha: float,
    t_grid_size: int,
    split_ratio: float,
    n_bins: int,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    ind_d0, ind_d1 = split_d0_d1(ind_cp, split_ratio)

    X0, Y0 = X[ind_d0], Y[ind_d0]
    X1, Y1 = X[ind_d1], Y[ind_d1]
    X_test, Y_test = X[ind_test], Y[ind_test]

    round_dir = out_dir / f"round_{r}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  D0={len(Y0)}, D1={len(Y1)}, test={len(Y_test)}")

    # t_grid: cover training Y range generously
    y_all  = np.concatenate([Y0, Y1])
    t_grid = np.linspace(np.quantile(y_all, 0.0005), np.quantile(y_all, 0.9995), t_grid_size)

    # --- CKME ---
    model = CKMEModel(indicator_type="logistic")
    model.fit(X0, Y0, params=params, verbose=verbose)

    cp = CP(model=model, alpha=alpha, score_type="abs_median")
    cp.calibrate(X1, Y1, verbose=verbose)

    L, U = cp.predict_interval(X_test, t_grid)

    # Score-based coverage (mirrors R's DCP)
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)
    G = model.indicator.g_matrix(model.Y, Y_test)
    F_test = np.clip(np.sum(C * G, axis=0), 0.0, 1.0)
    covered_ckme = (score_from_cdf(F_test, score_type="abs_median") <= cp.q_hat).astype(int)
    width_ckme   = U - L
    is_ckme, mean_is_ckme = compute_interval_score(Y_test, L, U, alpha)

    df_pp = pd.DataFrame({
        "lrealvol"   : X_test.ravel(),
        "Y"          : Y_test,
        "L_ckme"     : L,
        "U_ckme"     : U,
        "covered_ckme": covered_ckme,
        "width_ckme" : width_ckme,
        "IS_ckme"    : is_ckme,
    })

    # --- Save data CSVs for R ---
    data_dir = round_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_dir / "X0.csv",     X0,     delimiter=",")
    np.savetxt(data_dir / "Y0.csv",     Y0,     delimiter=",")
    np.savetxt(data_dir / "X1.csv",     X1,     delimiter=",")
    np.savetxt(data_dir / "Y1.csv",     Y1,     delimiter=",")
    np.savetxt(data_dir / "X_test.csv", X_test, delimiter=",")
    np.savetxt(data_dir / "Y_test.csv", Y_test, delimiter=",")

    # --- R benchmarks ---
    bench_csv = round_dir / "benchmarks.csv"
    try:
        bench_df = _run_r_benchmarks(data_dir, bench_csv, alpha, t_grid_size)
        for col in bench_df.columns:
            df_pp[col] = bench_df[col].values
        if verbose:
            for name, (cov_col, _, _) in R_METHODS.items():
                if cov_col in df_pp.columns:
                    print(f"    {name}: coverage={df_pp[cov_col].mean():.4f}")
    except RuntimeError as e:
        print(f"  Warning: R benchmarks failed; DCP/CQR cols will be absent.\n  {e}",
              file=sys.stderr)

    df_pp.to_csv(round_dir / "per_point.csv", index=False)

    # --- Binned metrics ---
    df_bin = binned_metrics(X_test, df_pp, n_bins=n_bins)
    df_bin.to_csv(round_dir / "binned.csv", index=False)

    # --- Round summary ---
    summary = {
        "round"        : r + 1,
        "n_d0"         : len(Y0),
        "n_d1"         : len(Y1),
        "n_test"       : len(Y_test),
        "CKME_coverage": float(covered_ckme.mean()),
        "CKME_width"   : float(width_ckme.mean()),
        "CKME_IS"      : mean_is_ckme,
    }
    for name, (cov_col, wid_col, is_col) in R_METHODS.items():
        key = name.replace("-", "").replace(" ", "")
        if cov_col in df_pp.columns:
            summary[f"{key}_coverage"] = float(df_pp[cov_col].mean())
        if wid_col in df_pp.columns:
            summary[f"{key}_width"] = float(df_pp[wid_col].mean())
        if is_col and is_col in df_pp.columns:
            summary[f"{key}_IS"] = float(df_pp[is_col].mean())

    if verbose:
        print(f"  CKME: coverage={summary['CKME_coverage']:.4f}  "
              f"width={summary['CKME_width']:.4f}  IS={summary['CKME_IS']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CKME + CP + R benchmarks on stock returns")
    parser.add_argument("--config",      type=str,   default="exp_returns/config.txt")
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--t_grid_size", type=int,   default=None)
    parser.add_argument("--n_bins",      type=int,   default=None)
    parser.add_argument("--verbose",     action="store_true", default=True)
    args = parser.parse_args()

    cfg = _load_config(_root / args.config)

    alpha        = float(cfg.get("alpha",        0.1))
    t_grid_size  = args.t_grid_size or int(cfg.get("t_grid_size", 500))
    n_rounds     = int(cfg.get("n_rounds",        5))
    holdout_frac = float(cfg.get("holdout_frac",  0.10))
    split_ratio  = float(cfg.get("split_ratio",   0.5))
    n_bins       = args.n_bins or int(cfg.get("n_bins", 20))

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_returns" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; benchmarks will be skipped.",
              file=sys.stderr)

    # Load pretrained params or fall back to config defaults
    pretrained_path = _root / "exp_returns" / "pretrained_params.json"
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        params = Params(**raw)
        print(f"Loaded pretrained params: {params.as_dict()}")
    else:
        params = Params(
            ell_x=float(cfg.get("ell_x", 1.0)),
            lam  =float(cfg.get("lam",   1e-2)),
            h    =float(cfg.get("h",     1.0)),
        )
        print(f"No pretrained params found; using config defaults: {params.as_dict()}")
        print("  Run 'python exp_returns/pretrain_params.py' first.\n")

    Y, X = load_returns_data()
    splits = get_round_splits(len(Y), n_rounds=n_rounds, holdout_frac=holdout_frac)
    print(f"Total observations: {len(Y)}, rounds: {len(splits)}, "
          f"T_ho={int(len(Y)*holdout_frac)}\n")

    all_summary = []
    for r, (ind_cp, ind_test) in enumerate(splits):
        print(f"--- Round {r+1} / {len(splits)} ---")
        s = run_one_round(
            r=r, Y=Y, X=X,
            ind_cp=ind_cp, ind_test=ind_test,
            params=params, alpha=alpha,
            t_grid_size=t_grid_size, split_ratio=split_ratio,
            n_bins=n_bins, out_dir=out_dir, verbose=args.verbose,
        )
        all_summary.append(s)

    # Build summary table
    df_summary = pd.DataFrame(all_summary)
    numeric_cols = df_summary.select_dtypes(include="number").columns
    overall = df_summary[numeric_cols].mean().to_dict()
    overall["round"] = "Overall"
    df_summary = pd.concat([df_summary, pd.DataFrame([overall])], ignore_index=True)

    summary_path = out_dir / "returns_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    # Print compact coverage/width table
    cov_cols = [c for c in df_summary.columns if c.endswith("_coverage")]
    wid_cols = [c for c in df_summary.columns if c.endswith("_width")]
    print("\n" + "="*70)
    print(f"Target coverage: {1-alpha:.0%}")
    print("\nCoverage:")
    print(df_summary[["round"] + cov_cols].to_string(index=False))
    print("\nWidth:")
    print(df_summary[["round"] + wid_cols].to_string(index=False))
    print(f"\nWrote {summary_path}")
    print("\nNext: python exp_returns/plot_returns.py --save")


if __name__ == "__main__":
    main()
