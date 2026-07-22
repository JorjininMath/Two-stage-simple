"""
experiments/framing_validation / Gate 3: epistemic diagnostics vs CP interval width.

Claim under test (paper framing, PDF Sec 6.2 Exp B):
    With FLAT aleatoric noise (sigma = const) and a LOCALIZED epistemic
    hotspot (a narrow bump in f that the kernel length scale cannot
    resolve), split-CP interval width cannot localize the hotspot,
    while targeted epistemic diagnostics can:
      - cdf_l2(x)  = int (F_hat(t|x) - F_true(t|x))^2 dt   (oracle check)
      - u_tail(x)  = Var_b q_hat_0.05(x) + Var_b q_hat_0.95(x)
                     over B site-bootstrap refits             (data-only)

Design (standalone, CKMEModel directly; no simulator registration):
    x in [0, 1], sigma = 0.10 constant
    control: f0(x) = sin(2*pi*x)
    bump:    f1(x) = f0(x) + 1.5 * exp(-(x - 0.5)^2 / (2 * 0.03^2))
    Stage 1: n_0 = 100 grid sites x r_0 = 10 reps
    Params: ell_x = 0.1, lam = 1e-3, h = 0.05, logistic indicator
    Calibration: n_1 = 100 LHS sites x r_1 = 10 reps -> split-CP q_hat
    Diagnostics on a 200-point x grid; N_MACRO macroreps averaged.

Separation index per (arm, diagnostic):
    SI = max_{x in bump window} D(x) / median_{x outside window} D(x)
    (bump window = 0.5 +/- 3 * bump width). Control arm should give
    SI ~ 1 everywhere; bump arm should give SI >> 1 for cdf_l2/u_tail
    and SI ~ 1 (or mildly > 1) for width if the claim holds.

Outputs (experiments/framing_validation/output_gate3/):
    gate3_curves.csv        per-x mean +/- SE of each diagnostic
    gate3_summary.csv       separation indices, marginal coverage, q_hat
    gate3_epistemic_diagnostics.png

Usage (from project root):
    python experiments/framing_validation/gate3_epistemic_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct script execution needs both the src layout and the experiment packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _import_path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))
_root = _PROJECT_ROOT
_EXPERIMENT_DIR = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from CKME.coefficients import compute_ckme_coeffs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_SEED = 20260707
N_MACRO = 5
N_0, R_0 = 100, 10          # stage-1 grid design
N_1, R_1 = 100, 10          # calibration (LHS)
N_TEST = 1000               # marginal-coverage sanity check
N_BOOT = 30                 # site-bootstrap refits for u_tail
ALPHA = 0.1
SIGMA = 0.10                # flat aleatoric noise
BUMP_CENTER, BUMP_SD, BUMP_AMP = 0.5, 0.03, 1.5
PARAMS = Params(ell_x=0.1, lam=1e-3, h=0.05)
N_XGRID = 200
N_TGRID = 500
T_MARGIN = 0.5              # t_grid margin beyond data range
TAU_LO, TAU_HI = 0.05, 0.95  # tail quantiles for u_tail

OUT_DIR = Path(__file__).resolve().parent / "output_gate3"

ARMS = ["control", "bump"]
X_GRID = np.linspace(0.02, 0.98, N_XGRID)
BUMP_WINDOW = (BUMP_CENTER - 3 * BUMP_SD, BUMP_CENTER + 3 * BUMP_SD)


def f_true(x: np.ndarray, arm: str) -> np.ndarray:
    base = np.sin(2 * np.pi * x)
    if arm == "control":
        return base
    return base + BUMP_AMP * np.exp(-((x - BUMP_CENTER) ** 2) / (2 * BUMP_SD ** 2))


def simulate(x_sites: np.ndarray, r: int, arm: str, rng: np.random.Generator):
    """Return (X_rep, Y) with each site's x repeated r times consecutively."""
    x_rep = np.repeat(x_sites, r)
    y = f_true(x_rep, arm) + SIGMA * rng.standard_normal(len(x_rep))
    return x_rep.reshape(-1, 1), y


def lhs_1d(n: int, rng: np.random.Generator) -> np.ndarray:
    """1D Latin hypercube on [0, 1]: one uniform draw per stratum, shuffled."""
    return (rng.permutation(n) + rng.uniform(size=n)) / n


def fit_ckme(X_rep: np.ndarray, Y: np.ndarray, r: int) -> CKMEModel:
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_rep, Y, params=PARAMS, r=r)
    return model


def raw_scores(model: CKMEModel, X_eval: np.ndarray, Y_eval: np.ndarray) -> np.ndarray:
    """Fixed-h conformity scores |F_hat(y|x) - 0.5|, vectorized."""
    X_eval = np.atleast_2d(X_eval)
    Y_eval = np.asarray(Y_eval).ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)  # (n_sites, m)
    Y_flat = model.Y.ravel()
    G = model.indicator.g_matrix(Y_flat, Y_eval)                 # (n_flat, m)
    if model.r > 1:
        G = G.reshape(model.n, model.r, -1).mean(axis=1)         # (n_sites, m)
    F = np.clip(np.sum(C * G, axis=0), 0.0, 1.0)
    return np.abs(F - 0.5)


def monotone_cdf(F: np.ndarray) -> np.ndarray:
    """Enforce monotonicity along t (axis=1) and clip to [0, 1]."""
    return np.clip(np.maximum.accumulate(F, axis=1), 0.0, 1.0)


def invert_cdf(F: np.ndarray, t_grid: np.ndarray, tau: float) -> np.ndarray:
    """q_tau(x) per row via first grid crossing of a monotone CDF."""
    mask = F >= tau
    idx = np.where(mask.any(axis=1), mask.argmax(axis=1), len(t_grid) - 1)
    return t_grid[idx]


def run_one_macrorep(arm: str, k: int) -> dict:
    seed = BASE_SEED + k * 1000
    rng_train = np.random.default_rng(seed)
    rng_cal = np.random.default_rng(seed + 1)
    rng_test = np.random.default_rng(seed + 2)
    rng_boot = np.random.default_rng(seed + 3)

    # --- Stage 1: grid design ---
    x_sites = np.linspace(0.0, 1.0, N_0)
    X_rep, Y = simulate(x_sites, R_0, arm, rng_train)
    model = fit_ckme(X_rep, Y, R_0)

    t_lo = Y.min() - T_MARGIN
    t_hi = Y.max() + T_MARGIN
    t_grid = np.linspace(t_lo, t_hi, N_TGRID)

    # --- Split-CP calibration (LHS sites) ---
    x_cal_sites = lhs_1d(N_1, rng_cal)
    X_cal, Y_cal = simulate(x_cal_sites, R_1, arm, rng_cal)
    scores_cal = raw_scores(model, X_cal, Y_cal)
    n_cal = len(scores_cal)
    k_ord = min(int(np.ceil((1 - ALPHA) * (1 + n_cal))), n_cal)
    q_hat = float(np.sort(scores_cal)[k_ord - 1])

    # --- Marginal coverage sanity check ---
    x_test = rng_test.uniform(0.0, 1.0, N_TEST)
    y_test = f_true(x_test, arm) + SIGMA * rng_test.standard_normal(N_TEST)
    cov = float(np.mean(raw_scores(model, x_test.reshape(-1, 1), y_test) <= q_hat))

    # --- Diagnostic 1: CP interval width on X_GRID ---
    Xg = X_GRID.reshape(-1, 1)
    F_grid = monotone_cdf(model.predict_cdf(Xg, t_grid))         # (N_XGRID, N_TGRID)
    L = invert_cdf(F_grid, t_grid, 0.5 - q_hat)
    U = invert_cdf(F_grid, t_grid, 0.5 + q_hat)
    width = U - L

    # --- Diagnostic 2: oracle CDF L2 gap ---
    F_true_grid = norm.cdf(
        (t_grid[None, :] - f_true(X_GRID, arm)[:, None]) / SIGMA
    )
    cdf_l2 = np.trapz((F_grid - F_true_grid) ** 2, t_grid, axis=1)

    # --- Diagnostic 3: u_tail via site bootstrap ---
    q_lo_b = np.empty((N_BOOT, N_XGRID))
    q_hi_b = np.empty((N_BOOT, N_XGRID))
    Y_sites = Y.reshape(N_0, R_0)
    for b in range(N_BOOT):
        idx = rng_boot.integers(0, N_0, size=N_0)
        Xb = np.repeat(x_sites[idx], R_0).reshape(-1, 1)
        Yb = Y_sites[idx].ravel()
        mb = fit_ckme(Xb, Yb, R_0)
        Fb = monotone_cdf(mb.predict_cdf(Xg, t_grid))
        q_lo_b[b] = invert_cdf(Fb, t_grid, TAU_LO)
        q_hi_b[b] = invert_cdf(Fb, t_grid, TAU_HI)
    u_tail = q_lo_b.var(axis=0, ddof=1) + q_hi_b.var(axis=0, ddof=1)

    return {
        "arm": arm,
        "macrorep": k,
        "q_hat": q_hat,
        "coverage": cov,
        "width": width,
        "cdf_l2": cdf_l2,
        "u_tail": u_tail,
    }


def separation_index(curve: np.ndarray) -> float:
    in_win = (X_GRID >= BUMP_WINDOW[0]) & (X_GRID <= BUMP_WINDOW[1])
    bg = np.median(curve[~in_win])
    return float(curve[in_win].max() / bg) if bg > 0 else np.inf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for arm in ARMS:
        for k in range(N_MACRO):
            res = run_one_macrorep(arm, k)
            results.append(res)
            print(
                f"  {arm:8s} macrorep {k}  q_hat={res['q_hat']:.4f} "
                f"cov={res['coverage']:.3f}"
            )

    diagnostics = ["width", "cdf_l2", "u_tail"]
    curve_rows, summary_rows = [], []
    curves = {}  # (arm, diag) -> (mean, se)
    for arm in ARMS:
        arm_res = [r for r in results if r["arm"] == arm]
        for diag in diagnostics:
            mat = np.stack([r[diag] for r in arm_res])          # (N_MACRO, N_XGRID)
            mean = mat.mean(axis=0)
            se = mat.std(axis=0, ddof=1) / np.sqrt(len(arm_res))
            curves[(arm, diag)] = (mean, se)
            for i, x in enumerate(X_GRID):
                curve_rows.append(
                    {"arm": arm, "diag": diag, "x": x, "mean": mean[i], "se": se[i]}
                )
            summary_rows.append(
                {
                    "arm": arm,
                    "diag": diag,
                    "separation_index": separation_index(mean),
                    "coverage_mean": np.mean([r["coverage"] for r in arm_res]),
                    "q_hat_mean": np.mean([r["q_hat"] for r in arm_res]),
                    "n_macroreps": len(arm_res),
                }
            )

    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "gate3_curves.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "gate3_summary.csv", index=False)

    # --- Figure: DGP | width | cdf_l2 | u_tail ---
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    colors = {"control": "tab:blue", "bump": "tab:red"}

    ax = axes[0]
    xs = np.linspace(0, 1, 400)
    for arm in ARMS:
        ax.plot(xs, f_true(xs, arm), color=colors[arm], lw=1.6, label=arm)
    ax.fill_between(
        xs, f_true(xs, "bump") - 2 * SIGMA, f_true(xs, "bump") + 2 * SIGMA,
        color="tab:red", alpha=0.12,
    )
    ax.axvline(BUMP_CENTER, color="k", ls=":", lw=0.8)
    ax.set_title(f"DGP: flat noise sigma={SIGMA}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)

    titles = {
        "width": "CP interval width",
        "cdf_l2": "oracle CDF L2 gap",
        "u_tail": "u_tail (bootstrap tail-quantile var)",
    }
    for j, diag in enumerate(diagnostics):
        ax = axes[j + 1]
        for arm in ARMS:
            m, se = curves[(arm, diag)]
            ax.plot(X_GRID, m, color=colors[arm], lw=1.5, label=arm)
            ax.fill_between(X_GRID, m - se, m + se, color=colors[arm], alpha=0.25)
        ax.axvline(BUMP_CENTER, color="k", ls=":", lw=0.8)
        ax.axvspan(*BUMP_WINDOW, color="gray", alpha=0.12)
        if diag == "width":
            oracle_w = 2 * norm.ppf(1 - ALPHA / 2) * SIGMA
            ax.axhline(oracle_w, color="k", ls="--", lw=0.9, alpha=0.6)
        si = {
            arm: next(
                r["separation_index"] for r in summary_rows
                if r["arm"] == arm and r["diag"] == diag
            )
            for arm in ARMS
        }
        ax.set_title(f"{titles[diag]}\nSI ctrl={si['control']:.2f}, bump={si['bump']:.2f}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        if diag in ("cdf_l2", "u_tail"):
            ax.set_yscale("log")

    fig.suptitle(
        "Gate 3: localized epistemic hotspot — CP width vs epistemic diagnostics "
        f"({N_MACRO} macroreps, B={N_BOOT} bootstraps)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = OUT_DIR / "gate3_epistemic_diagnostics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {fig_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
