"""
exp_framing_validation / Gate 2: conformity-score homogeneity across x.

Claim under test (paper framing / homogeneity theorem):
    Under FIXED h, the conformity score R(x, y) = |F_hat(y|x) - 0.5| has a
    distribution that varies with x (heterogeneous), because the effective
    smoothing h / s(x) varies with x. Under oracle adaptive h(x) = c * s(x),
    the score distribution is (approximately) x-invariant.

This is the mechanism behind Gate 1's coverage redistribution. Existing
exp2/exp4 outputs only store thresholded coverage, so this pilot re-runs
the exp2 pipeline (same seeds -> same data) and SAVES RAW SCORES.

Pipeline per (DGP, macrorep), identical seeds to run_exp2_oracle.py:
    stage1 (grid, n_0 x r_0) -> stage2 (LHS, n_1 x r_1, split-CP) -> test set;
    raw scores on test points for both arms; q_hat for both arms.

Analysis (pooled over macroreps, per DGP):
    - 10 equal-count x-bins; per bin: two-sample KS(bin scores, complement).
    - Permutation null band (random bins of same size, 95th percentile).
    - Per-bin 90% quantile of scores ("local threshold") vs global q_hat.

Outputs (exp_framing_validation/output_gate2/):
    macrorep_{k}/case_{sim}/scores.csv        (x0, y, score_fixed, score_oracle)
    macrorep_{k}/case_{sim}/meta.json         (q_hat_fixed, q_hat_oracle)
    gate2_ks_summary.csv, gate2_bin_stats.csv
    gate2_ks_profile.png, gate2_bin_q90.png

Usage (from project root):
    python exp_framing_validation/gate2_score_homogeneity.py --n_macro 10 --n_workers 4
    python exp_framing_validation/gate2_score_homogeneity.py --analyze_only
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator
from CKME.parameters import Params
from CP.scores import score_from_cdf
from exp_adaptive_h.adaptive_h_utils import ORACLE_SCALE, get_oracle_h, adaptive_recalibrate_q
from exp_adaptive_h.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.test_data import generate_test_data

SIMULATORS = ["wsc_gauss", "gibbs_s1", "exp1", "nongauss_A1L"]
ARMS = ["fixed", "oracle"]
STAGE1_DESIGN = "grid"
STAGE2_METHOD = "lhs"
BASE_SEED = 20260501  # same as run_exp2_oracle.py -> identical data per macrorep
N_BINS = 10
N_PERM = 200
ALPHA = 0.1

T_GRID_MARGIN = {
    "wsc_gauss": 0.30,
    "gibbs_s1": 0.30,
    "exp1": 0.50,
    "nongauss_A1L": 2.00,
}

OUT_DIR = _root / "exp_framing_validation" / "output_gate2"


def fixed_raw_scores(model, X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    """Raw conformity scores |F_hat(y|x) - 0.5| with the model's fixed h."""
    C = compute_ckme_coeffs(model.L, model.kx, model.X, np.atleast_2d(X_test))
    Y_flat = model.Y.ravel()
    if getattr(model, "r", 1) > 1:
        G_all = model.indicator.g_matrix(Y_flat, np.asarray(Y_test).ravel())
        G = G_all.reshape(model.n, model.r, -1).mean(axis=1)
    else:
        G = model.indicator.g_matrix(model.Y, np.asarray(Y_test).ravel())
    F_test = np.clip(np.sum(C * G, axis=0).astype(float), 0.0, 1.0)
    return score_from_cdf(F_test, score_type="abs_median")


def oracle_raw_scores(model, X_test: np.ndarray, Y_test: np.ndarray, h_test: np.ndarray) -> np.ndarray:
    """Raw conformity scores with per-point adaptive h (mirrors adaptive_score_coverage)."""
    X_test = np.atleast_2d(X_test)
    Y_test = np.asarray(Y_test).ravel()
    h_test = np.asarray(h_test).ravel()
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)

    scores = np.empty(len(Y_test))
    for i in range(len(Y_test)):
        ind_i = make_indicator(model.indicator_type, float(h_test[i]))
        g_i = ind_i.g_matrix(Y_flat, np.array([float(Y_test[i])]))[:, 0]
        if getattr(model, "r", 1) > 1:
            g_site = g_i.reshape(model.n, model.r).mean(axis=1)
        else:
            g_site = g_i
        F_i = float(np.clip(C[:, i] @ g_site, 0.0, 1.0))
        scores[i] = abs(F_i - 0.5)
    return scores


def run_one_macrorep(
    macrorep_id: int,
    config: dict,
    simulator_func: str,
    n_grid: int,
    params: Params,
    c_scale: float,
) -> dict:
    seed = BASE_SEED + macrorep_id * 10000
    alpha = config["alpha"]

    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=config["n_0"], r_0=config["r_0"],
        simulator_func=simulator_func,
        params=params,
        design_method=STAGE1_DESIGN,
        t_grid_size=n_grid,
        t_grid_margin=T_GRID_MARGIN.get(simulator_func),
        random_state=seed + 2,
        verbose=False,
    )
    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=config["n_1"], r_1=config["r_1"],
        simulator_func=simulator_func,
        method=STAGE2_METHOD,
        alpha=alpha,
        random_state=seed + 3,
        verbose=False,
    )
    X_test, Y_test = generate_test_data(
        stage2_result=stage2,
        n_test=config["n_test"],
        r_test=config["r_test"],
        X_cand=X_cand,
        simulator_func=simulator_func,
        random_state=seed + 4,
    )

    model = stage2.model
    sc_fixed = fixed_raw_scores(model, X_test, Y_test)

    h_test = get_oracle_h(simulator_func, X_test, c_scale)
    sc_oracle = oracle_raw_scores(model, X_test, Y_test, h_test)

    h_cal = get_oracle_h(simulator_func, stage2.X_stage2, c_scale)
    q_oracle = adaptive_recalibrate_q(model, stage2.X_stage2, stage2.Y_stage2, h_cal, alpha)

    df = pd.DataFrame(
        {
            "x0": np.atleast_2d(X_test)[:, 0].astype(float),
            "y": np.asarray(Y_test, dtype=float).ravel(),
            "score_fixed": sc_fixed.astype(float),
            "score_oracle": sc_oracle.astype(float),
        }
    )
    case_dir = OUT_DIR / f"macrorep_{macrorep_id}" / f"case_{simulator_func}"
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(case_dir / "scores.csv", index=False)
    meta = {
        "q_hat_fixed": float(stage2.cp.q_hat),
        "q_hat_oracle": float(q_oracle),
        "seed": seed,
        "c_scale": c_scale,
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"macrorep": macrorep_id, "simulator": simulator_func, **meta}


def _worker(args_tuple):
    return run_one_macrorep(*args_tuple)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _perm_null_band(scores: np.ndarray, n_bin: int, n_perm: int, rng: np.random.Generator) -> float:
    """95th pct of KS(bin, complement) when the bin is a random subset."""
    n = len(scores)
    stats = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.choice(n, size=n_bin, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        stats[b] = ks_2samp(scores[mask], scores[~mask]).statistic
    return float(np.quantile(stats, 0.95))


def analyze() -> None:
    rng = np.random.default_rng(20260707)
    bin_rows: list[dict] = []
    summary_rows: list[dict] = []

    fig_ks, axes_ks = plt.subplots(1, len(SIMULATORS), figsize=(4.2 * len(SIMULATORS), 3.8))
    fig_q, axes_q = plt.subplots(1, len(SIMULATORS), figsize=(4.2 * len(SIMULATORS), 3.8))
    colors = {"fixed": "tab:red", "oracle": "tab:blue"}

    for j, sim in enumerate(SIMULATORS):
        frames, q_fixed_list, q_oracle_list = [], [], []
        for mdir in sorted(OUT_DIR.glob("macrorep_*")):
            f = mdir / f"case_{sim}" / "scores.csv"
            if not f.exists():
                continue
            frames.append(pd.read_csv(f))
            meta = json.loads((mdir / f"case_{sim}" / "meta.json").read_text())
            q_fixed_list.append(meta["q_hat_fixed"])
            q_oracle_list.append(meta["q_hat_oracle"])
        if not frames:
            raise FileNotFoundError(f"No scores.csv for {sim}; run the pilot first.")
        pool = pd.concat(frames, ignore_index=True)
        n_macro = len(frames)
        q_hat = {"fixed": float(np.mean(q_fixed_list)), "oracle": float(np.mean(q_oracle_list))}

        edges = np.quantile(pool["x0"], np.linspace(0, 1, N_BINS + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        centers = 0.5 * (edges[:-1] + edges[1:])
        pool["bin"] = pd.cut(pool["x0"], bins=edges, labels=False)
        n_bin = int(pool.groupby("bin").size().mean())

        ax_ks, ax_q = axes_ks[j], axes_q[j]
        for arm in ARMS:
            scores = pool[f"score_{arm}"].to_numpy()
            null95 = _perm_null_band(scores, n_bin, N_PERM, rng)

            ks_vals, q90_vals = np.empty(N_BINS), np.empty(N_BINS)
            for b in range(N_BINS):
                mask = (pool["bin"] == b).to_numpy()
                ks_vals[b] = ks_2samp(scores[mask], scores[~mask]).statistic
                q90_vals[b] = np.quantile(scores[mask], 1 - ALPHA)
                bin_rows.append(
                    {
                        "simulator": sim, "arm": arm, "bin": b,
                        "x_center": centers[b], "ks": ks_vals[b],
                        "score_q90": q90_vals[b], "null95": null95,
                    }
                )
            summary_rows.append(
                {
                    "simulator": sim, "arm": arm,
                    "mean_ks": float(ks_vals.mean()),
                    "max_ks": float(ks_vals.max()),
                    "null95": null95,
                    "n_bins_above_null": int((ks_vals > null95).sum()),
                    "q90_range_ratio": float(q90_vals.max() / max(q90_vals.min(), 1e-12)),
                    "q_hat_mean": q_hat[arm],
                    "n_macroreps": n_macro,
                }
            )
            ax_ks.plot(centers, ks_vals, marker="o", ms=3.5, color=colors[arm], label=f"{arm} h")
            ax_ks.axhline(null95, color=colors[arm], ls=":", lw=1.0, alpha=0.8)
            ax_q.plot(centers, q90_vals, marker="o", ms=3.5, color=colors[arm], label=f"{arm} h")
            ax_q.axhline(q_hat[arm], color=colors[arm], ls="--", lw=1.0, alpha=0.8)

        s_fn = ORACLE_SCALE[sim]
        for ax in (ax_ks, ax_q):
            ax2 = ax.twinx()
            xs = np.linspace(edges[0], edges[-1], 300)
            ax2.plot(xs, s_fn(xs), color="gray", lw=1.0, alpha=0.5)
            ax2.set_yticks([])
        ax_ks.set_title(sim)
        ax_q.set_title(sim)
        ax_ks.set_xlabel("x")
        ax_q.set_xlabel("x")
        if j == 0:
            ax_ks.set_ylabel("KS(bin vs rest) of scores")
            ax_q.set_ylabel("bin score q90")
            ax_ks.legend(fontsize=8)
            ax_q.legend(fontsize=8)

    fig_ks.suptitle(
        "Gate 2: score-distribution heterogeneity across x "
        "(KS per bin; dotted = permutation null 95%)",
        fontsize=12,
    )
    fig_q.suptitle(
        "Gate 2: local score q90 vs global q_hat (dashed); gray = s(x)",
        fontsize=12,
    )
    for f, name in ((fig_ks, "gate2_ks_profile.png"), (fig_q, "gate2_bin_q90.png")):
        f.tight_layout(rect=[0, 0, 1, 0.94])
        f.savefig(OUT_DIR / name, dpi=150)
        plt.close(f)

    pd.DataFrame(bin_rows).to_csv(OUT_DIR / "gate2_bin_stats.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "gate2_ks_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'gate2_ks_profile.png'}")
    print(f"Wrote {OUT_DIR / 'gate2_bin_q90.png'}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2: score homogeneity pilot")
    parser.add_argument("--config", type=str, default="exp_adaptive_h/config.txt")
    parser.add_argument("--n_macro", type=int, default=10)
    parser.add_argument("--c_scale", type=float, default=1.0)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        config = get_config(load_config_from_file(_root / args.config), quick=False)
        n_grid = config.get("t_grid_size", 1000)
        raw = json.loads((_root / "exp_adaptive_h" / "pretrained_params.json").read_text())
        pretrained = {sim: Params(**raw[sim]) for sim in SIMULATORS}

        jobs = [
            (k, config, sim, n_grid, pretrained[sim], args.c_scale)
            for sim in SIMULATORS
            for k in range(args.n_macro)
        ]
        print(f"Gate 2 pilot: {len(jobs)} jobs ({len(SIMULATORS)} DGPs x {args.n_macro} macroreps)")
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futs = {pool.submit(_worker, j): (j[2], j[0]) for j in jobs}
                for fut in as_completed(futs):
                    sim, k = futs[fut]
                    r = fut.result()
                    print(f"  {sim:14s} macrorep {k:2d}  q_fix={r['q_hat_fixed']:.4f} q_or={r['q_hat_oracle']:.4f}")
        else:
            for j in jobs:
                r = _worker(j)
                print(f"  {j[2]:14s} macrorep {j[0]:2d}  q_fix={r['q_hat_fixed']:.4f} q_or={r['q_hat_oracle']:.4f}")

    analyze()


if __name__ == "__main__":
    main()
