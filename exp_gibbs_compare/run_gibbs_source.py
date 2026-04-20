"""
run_gibbs_source.py

Re-run Gibbs et al. (2023) method on their own DGPs, saving per-macrorep
local coverage so we can compute mean ± sd.

Uses the original conditionalconformal package (Gibbs et al.'s code).

Usage:
    python exp_gibbs_compare/run_gibbs_source.py
    python exp_gibbs_compare/run_gibbs_source.py --n_rep 50 --output_dir exp_gibbs_compare/output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from conditionalconformal.synthetic_data import indicator_matrix
from conditionalconformal import CondConf

ALPHA = 0.1
X_SEQ = np.arange(-2.5, 2.5 + 0.25, 0.25)  # 21 centers
N_SEQ = len(X_SEQ)
RADII = 0.4


def simulation(n: int, setting: int, rng: np.random.Generator):
    X = rng.standard_normal(n).astype(np.float32)
    noise = rng.standard_normal(n)
    if setting == 1:
        Y = X / 2 + np.abs(np.sin(X)) * noise
    elif setting == 2:
        Y = X / 2 + 2 * norm.pdf(X, loc=0, scale=1.5) * noise
    else:
        raise ValueError(f"Unknown setting {setting}")
    return X, Y


def run_gibbs_method(
    eps: float, setting: int, n: int, n_rep: int, seed: int,
) -> np.ndarray:
    """Run Gibbs et al. method, return per-rep local coverage (n_rep, 21)."""
    disc = np.arange(-2.5, 2.5 + eps, eps)

    def phi_fn_groups(x):
        mat = indicator_matrix(x, disc)
        return np.hstack((mat, np.ones((x.shape[0], 1))))

    rng = np.random.default_rng(seed)
    per_rep = np.zeros((n_rep, N_SEQ))

    for rep in tqdm(range(n_rep), desc=f"setting={setting} eps={eps:.3f}"):
        # Use rng to set numpy global seed for compatibility with original code
        np.random.seed(rng.integers(0, 2**31))

        X_train, Y_train = simulation(n, setting, np.random.default_rng(np.random.randint(2**31)))
        X_calib, Y_calib = simulation(n, setting, np.random.default_rng(np.random.randint(2**31)))
        X_test, Y_test = simulation(n, setting, np.random.default_rng(np.random.randint(2**31)))

        X_train = X_train.reshape(n, 1)
        X_calib = X_calib.reshape(n, 1)
        X_test = X_test.reshape(n, 1)

        reg = LinearRegression().fit(X_train, Y_train)

        score_fn = lambda x, y: y - reg.predict(x)
        score_inv_fn_ub = lambda s, x: [-np.inf, reg.predict(x) + s]
        score_inv_fn_lb = lambda s, x: [reg.predict(x) + s, np.inf]

        cond_conf = CondConf(score_fn, phi_fn_groups, infinite_params={})
        cond_conf.setup_problem(X_calib, Y_calib)

        n_test = len(X_test)
        cov = np.zeros(n_test)

        for i in range(n_test):
            x_t = X_test[i, :]
            res = cond_conf.predict(ALPHA / 2, x_t, score_inv_fn_lb, exact=True, randomize=True)
            lb = res[0]
            res = cond_conf.predict(1 - ALPHA / 2, x_t, score_inv_fn_ub, exact=True, randomize=True)
            ub = res[1]
            cov[i] = float(Y_test[i] <= ub and Y_test[i] >= lb)

        # Local coverage at 21 centers
        local_ball_id = np.zeros((n_test, N_SEQ))
        for i, x_t in enumerate(X_test):
            for j, x_seq in enumerate(X_SEQ):
                if np.abs(x_t - x_seq) <= RADII:
                    local_ball_id[i, j] = 1

        denom = np.matmul(np.ones(n_test), local_ball_id)
        denom = np.maximum(denom, 1)  # avoid division by zero
        per_rep[rep] = np.matmul(cov, local_ball_id) / denom

    return per_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rep", type=int, default=50)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    _root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_gibbs_compare" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("gibbs_s1_high", 5 / 8, 1),
        ("gibbs_s1_low", 5 / 4, 1),
        ("gibbs_s2_high", 5 / 8, 2),
        ("gibbs_s2_low", 5 / 4, 2),
    ]

    all_mean = []
    for name, eps, setting in configs:
        print(f"\n--- {name} (eps={eps:.3f}, setting={setting}) ---")
        per_rep = run_gibbs_method(eps, setting, args.n, args.n_rep, args.seed)

        mean_cov = per_rep.mean(axis=0)
        sd_cov = per_rep.std(axis=0, ddof=1)
        all_mean.append(mean_cov)

        df = pd.DataFrame({
            "center": X_SEQ,
            "mean_local_cov": mean_cov,
            "sd_local_cov": sd_cov,
        })
        df.to_csv(out_dir / f"gibbs_{name}_results.csv", index=False)
        print(f"  mean coverage range: [{mean_cov.min():.3f}, {mean_cov.max():.3f}]")
        print(f"  sd range: [{sd_cov.min():.3f}, {sd_cov.max():.3f}]")

    # Also save in original 4x21 format for compatibility
    gibbs_mat = np.array(all_mean)
    np.savetxt(out_dir / "gibbs_et_al_results_with_sd.csv", gibbs_mat, delimiter=",")
    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
