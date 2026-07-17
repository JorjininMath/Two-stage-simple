"""
exp_framing_validation / Gate 4: fix epistemic error via targeted Stage-2 sampling.

This script implements the experiment specified in spec_gate4.md:
  G4a: a data-scarcity hotspot that should be fixed by u_tail-targeted sampling.
  G4b: a capacity-limited hotspot where sampling alone is insufficient and a
       lack-of-fit statistic should route the workflow to a smaller ell_x.

Usage from project root:
    python exp_framing_validation/gate4_fix_epistemic.py --part all --n_macro 3
    python exp_framing_validation/gate4_fix_epistemic.py --part all --n_macro 20 --n_workers 4
    python exp_framing_validation/gate4_fix_epistemic.py --part a --n_stage2 20 --output_suffix n20
    python exp_framing_validation/gate4_fix_epistemic.py --part all --analyze_only
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from CKME.ckme import CKMEModel
from CKME.loss_functions import CRPSLoss
from CKME.parameters import Params
from exp_framing_validation.gate3_epistemic_diagnostics import (
    invert_cdf,
    lhs_1d,
    monotone_cdf,
    raw_scores,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_SEED = 20260708
ALPHA = 0.1
SIGMA = 0.10
BUMP_CENTER = 0.5
BUMP_AMP = 1.5

DEFAULT_PARAMS = Params(ell_x=0.1, lam=1e-3, h=0.05)
RETUNE_ELL_GRID = [0.02, 0.05, 0.1]

R_SITE = 10
N_CAL = 100
N_TEST = 1000
N_STAGE2 = 60
N_CAND = 2000
N_LOF = 40
R_LOF = 20
GAMMA = 0.2

N_XGRID = 200
N_TGRID = 500
T_MARGIN = 0.5
TAU_LO = 0.05
TAU_HI = 0.95

OUT_A = _root / "exp_framing_validation" / "output_gate4a"
OUT_B = _root / "exp_framing_validation" / "output_gate4b"
X_GRID = np.linspace(0.02, 0.98, N_XGRID)


@dataclass(frozen=True)
class PartSpec:
    part: str
    label: str
    bump_sd: float
    window: tuple[float, float]
    arms: tuple[str, ...]


PART_A = PartSpec(
    part="a",
    label="G4a: data-scarcity hotspot",
    bump_sd=0.10,
    window=(0.35, 0.65),
    arms=("lhs", "utail", "oracle"),
)

PART_B = PartSpec(
    part="b",
    label="G4b: capacity-limited hotspot",
    bump_sd=0.03,
    window=(0.41, 0.59),
    arms=("lhs_fixed", "utail_fixed", "utail_retune"),
)


# ---------------------------------------------------------------------------
# DGP, fitting, and diagnostics
# ---------------------------------------------------------------------------


def f_true(x: np.ndarray, bump_sd: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    base = np.sin(2 * np.pi * x)
    bump = BUMP_AMP * np.exp(-((x - BUMP_CENTER) ** 2) / (2 * bump_sd**2))
    return base + bump


def stage1_sites(part: str) -> np.ndarray:
    if part == "a":
        left = np.linspace(0.0, 0.34, 25)
        inside = np.linspace(0.37, 0.63, 6)
        right = np.linspace(0.66, 1.0, 25)
        return np.concatenate([left, inside, right])
    if part == "b":
        return np.linspace(0.0, 1.0, 100)
    raise ValueError(f"Unknown part: {part}")


def simulate_sites(
    x_sites: np.ndarray,
    r: int,
    bump_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return repeated X, flat Y, and site-by-rep Y."""
    x_sites = np.asarray(x_sites, dtype=float).ravel()
    x_rep = np.repeat(x_sites, r)
    y = f_true(x_rep, bump_sd) + SIGMA * rng.standard_normal(len(x_rep))
    return x_rep.reshape(-1, 1), y, y.reshape(len(x_sites), r)


def fit_ckme_with_params(X_rep: np.ndarray, Y: np.ndarray, r: int, params: Params) -> CKMEModel:
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_rep, Y, params=params, r=r)
    return model


def conformal_qhat(model: CKMEModel, X_cal: np.ndarray, Y_cal: np.ndarray) -> float:
    scores = raw_scores(model, X_cal, Y_cal)
    n_cal = len(scores)
    k_ord = min(int(np.ceil((1 - ALPHA) * (n_cal + 1))), n_cal)
    return float(np.sort(scores)[k_ord - 1])


def cdf_l2_curve(model: CKMEModel, t_grid: np.ndarray, bump_sd: float) -> tuple[np.ndarray, np.ndarray]:
    Xg = X_GRID.reshape(-1, 1)
    F_grid = monotone_cdf(model.predict_cdf(Xg, t_grid))
    F_true_grid = norm.cdf((t_grid[None, :] - f_true(X_GRID, bump_sd)[:, None]) / SIGMA)
    cdf_l2 = np.trapz((F_grid - F_true_grid) ** 2, t_grid, axis=1)
    return F_grid, cdf_l2


def interval_width_from_f(F_grid: np.ndarray, t_grid: np.ndarray, q_hat: float) -> np.ndarray:
    lo_tau = max(0.0, 0.5 - q_hat)
    hi_tau = min(1.0, 0.5 + q_hat)
    L = invert_cdf(F_grid, t_grid, lo_tau)
    U = invert_cdf(F_grid, t_grid, hi_tau)
    return U - L


def bootstrap_utail(
    x_sites: np.ndarray,
    y_sites: np.ndarray,
    t_grid: np.ndarray,
    params: Params,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_sites, r = y_sites.shape
    Xg = X_GRID.reshape(-1, 1)
    q_lo_b = np.empty((n_boot, N_XGRID))
    q_hi_b = np.empty((n_boot, N_XGRID))

    for b in range(n_boot):
        idx = rng.integers(0, n_sites, size=n_sites)
        Xb = np.repeat(x_sites[idx], r).reshape(-1, 1)
        Yb = y_sites[idx].ravel()
        mb = fit_ckme_with_params(Xb, Yb, r, params)
        Fb = monotone_cdf(mb.predict_cdf(Xg, t_grid))
        q_lo_b[b] = invert_cdf(Fb, t_grid, TAU_LO)
        q_hi_b[b] = invert_cdf(Fb, t_grid, TAU_HI)

    return q_lo_b.var(axis=0, ddof=1) + q_hi_b.var(axis=0, ddof=1)


def separation_index(curve: np.ndarray, window: tuple[float, float]) -> float:
    in_win = (X_GRID >= window[0]) & (X_GRID <= window[1])
    bg = float(np.median(curve[~in_win]))
    return float(np.max(curve[in_win]) / bg) if bg > 0 else np.inf


def summarize_window(curve: np.ndarray, window: tuple[float, float]) -> tuple[float, float]:
    in_win = (X_GRID >= window[0]) & (X_GRID <= window[1])
    return float(np.mean(curve[in_win])), float(np.median(curve[~in_win]))


def diagnostic_curves(
    model: CKMEModel,
    t_grid: np.ndarray,
    q_hat: float,
    bump_sd: float,
    x_sites: np.ndarray,
    y_sites: np.ndarray,
    params: Params,
    n_boot: int,
    rng_boot: np.random.Generator,
) -> dict[str, np.ndarray]:
    F_grid, cdf_l2 = cdf_l2_curve(model, t_grid, bump_sd)
    width = interval_width_from_f(F_grid, t_grid, q_hat)
    u_tail = bootstrap_utail(x_sites, y_sites, t_grid, params, n_boot, rng_boot)
    return {"width": width, "cdf_l2": cdf_l2, "u_tail": u_tail}


def select_stage2_sites(
    policy: str,
    cand: np.ndarray,
    pre_curves: dict[str, np.ndarray],
    rng_alloc: np.random.Generator,
) -> np.ndarray:
    if policy == "lhs":
        return lhs_1d(N_STAGE2, rng_alloc)

    if policy == "utail":
        w_grid = pre_curves["u_tail"]
    elif policy == "oracle":
        w_grid = pre_curves["cdf_l2"]
    else:
        raise ValueError(f"Unknown allocation policy: {policy}")

    w_cand = np.interp(cand, X_GRID, np.maximum(w_grid, 0.0))
    if not np.isfinite(w_cand).all() or float(w_cand.sum()) <= 0.0:
        p = np.full(len(cand), 1.0 / len(cand))
    else:
        p = GAMMA / len(cand) + (1 - GAMMA) * w_cand / w_cand.sum()
        p = p / p.sum()
    return rng_alloc.choice(cand, size=N_STAGE2, replace=False, p=p)


def stage2_policy_for_arm(arm: str) -> str:
    if arm in ("lhs", "lhs_fixed"):
        return "lhs"
    if arm in ("utail", "utail_fixed", "utail_retune"):
        return "utail"
    if arm == "oracle":
        return "oracle"
    raise ValueError(f"Unknown arm: {arm}")


def site_level_cv_ellx(
    x_sites: np.ndarray,
    y_sites: np.ndarray,
    t_grid: np.ndarray,
    seed: int,
) -> tuple[Params, list[dict]]:
    """Five-fold CRPS CV with folds split by site, keeping replicate blocks together."""
    n_sites, r = y_sites.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sites)
    folds = np.array_split(perm, 5)
    loss_fn = CRPSLoss()

    rows: list[dict] = []
    mean_losses = []
    for ell_x in RETUNE_ELL_GRID:
        params = Params(ell_x=ell_x, lam=DEFAULT_PARAMS.lam, h=DEFAULT_PARAMS.h)
        fold_losses = []
        for fold_id, val_idx in enumerate(folds):
            train_mask = np.ones(n_sites, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.flatnonzero(train_mask)

            X_train = np.repeat(x_sites[train_idx], r).reshape(-1, 1)
            Y_train = y_sites[train_idx].ravel()
            X_val = np.repeat(x_sites[val_idx], r).reshape(-1, 1)
            Y_val = y_sites[val_idx].ravel()

            model = fit_ckme_with_params(X_train, Y_train, r, params)
            F_val = monotone_cdf(model.predict_cdf(X_val, t_grid))
            fold_loss = loss_fn.compute(F_val, Y_val, t_grid)
            fold_losses.append(fold_loss)
            rows.append(
                {
                    "ell_x": ell_x,
                    "fold": fold_id,
                    "fold_loss": float(fold_loss),
                }
            )
        mean_losses.append(float(np.mean(fold_losses)))

    best_idx = int(np.argmin(mean_losses))
    best_ell = RETUNE_ELL_GRID[best_idx]
    for row in rows:
        row["mean_loss"] = mean_losses[RETUNE_ELL_GRID.index(row["ell_x"])]
        row["selected_ell_x"] = best_ell
    return Params(ell_x=best_ell, lam=DEFAULT_PARAMS.lam, h=DEFAULT_PARAMS.h), rows


def lof_profile(
    model: CKMEModel,
    t_grid: np.ndarray,
    x_lof_sites: np.ndarray,
    y_lof_sites: np.ndarray,
) -> np.ndarray:
    F_lof = monotone_cdf(model.predict_cdf(x_lof_sites.reshape(-1, 1), t_grid))
    med_hat = invert_cdf(F_lof, t_grid, 0.5)
    ybar = y_lof_sites.mean(axis=1)
    s2 = np.maximum(y_lof_sites.var(axis=1, ddof=1), 1e-8)
    return R_LOF * (ybar - med_hat) ** 2 / s2


def part_spec(part: str) -> PartSpec:
    if part == "a":
        return PART_A
    if part == "b":
        return PART_B
    raise ValueError(f"Unknown part: {part}")


# ---------------------------------------------------------------------------
# Macrorep execution
# ---------------------------------------------------------------------------


def run_one_macrorep(part: str, macrorep: int, n_boot: int) -> dict[str, list[dict]]:
    spec = part_spec(part)
    seed = BASE_SEED + macrorep * 1000
    rng_train = np.random.default_rng(seed)
    rng_cal = np.random.default_rng(seed + 1)
    rng_test = np.random.default_rng(seed + 2)
    rng_boot0 = np.random.default_rng(seed + 3)
    rng_alloc = np.random.default_rng(seed + 4)
    rng_lof = np.random.default_rng(seed + 5)

    # Shared Stage-1, calibration, test, LOF, and candidate data.
    x0_sites = stage1_sites(part)
    X0, Y0, Y0_sites = simulate_sites(x0_sites, R_SITE, spec.bump_sd, rng_train)
    model_pre = fit_ckme_with_params(X0, Y0, R_SITE, DEFAULT_PARAMS)

    x_cal_sites = lhs_1d(N_CAL, rng_cal)
    X_cal, Y_cal, _ = simulate_sites(x_cal_sites, R_SITE, spec.bump_sd, rng_cal)
    x_test = rng_test.uniform(0.0, 1.0, N_TEST)
    y_test = f_true(x_test, spec.bump_sd) + SIGMA * rng_test.standard_normal(N_TEST)
    X_test = x_test.reshape(-1, 1)
    x_lof_sites = lhs_1d(N_LOF, rng_lof)
    _, _, y_lof_sites = simulate_sites(x_lof_sites, R_LOF, spec.bump_sd, rng_lof)
    cand = lhs_1d(N_CAND, rng_alloc)

    q_hat_pre = conformal_qhat(model_pre, X_cal, Y_cal)
    t_grid_pre = np.linspace(float(Y0.min() - T_MARGIN), float(Y0.max() + T_MARGIN), N_TGRID)
    pre_curves = diagnostic_curves(
        model_pre,
        t_grid_pre,
        q_hat_pre,
        spec.bump_sd,
        x0_sites,
        Y0_sites,
        DEFAULT_PARAMS,
        n_boot,
        rng_boot0,
    )
    pre_hot_cdf_l2, pre_bg_cdf_l2 = summarize_window(pre_curves["cdf_l2"], spec.window)
    pre_si_utail = separation_index(pre_curves["u_tail"], spec.window)
    pre_cov = float(np.mean(raw_scores(model_pre, X_test, y_test) <= q_hat_pre))

    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    site_rows: list[dict] = []
    lof_rows: list[dict] = []
    cv_rows: list[dict] = []

    for diag, values in pre_curves.items():
        for x, value in zip(X_GRID, values):
            curve_rows.append(
                {
                    "part": part,
                    "macrorep": macrorep,
                    "stage": "pre",
                    "arm": "pre",
                    "diag": diag,
                    "x": float(x),
                    "value": float(value),
                }
            )

    for arm_index, arm in enumerate(spec.arms):
        rng_sim2 = np.random.default_rng(seed + 40 + arm_index)
        rng_boot1 = np.random.default_rng(seed + 60 + arm_index)
        policy = stage2_policy_for_arm(arm)
        x2_sites = select_stage2_sites(policy, cand, pre_curves, rng_alloc)
        _, _, y2_sites = simulate_sites(x2_sites, R_SITE, spec.bump_sd, rng_sim2)

        x_pool_sites = np.concatenate([x0_sites, x2_sites])
        y_pool_sites = np.vstack([Y0_sites, y2_sites])
        X_pool = np.repeat(x_pool_sites, R_SITE).reshape(-1, 1)
        Y_pool = y_pool_sites.ravel()
        t_grid_pool = np.linspace(float(Y_pool.min() - T_MARGIN), float(Y_pool.max() + T_MARGIN), N_TGRID)

        if arm == "utail_retune":
            params, rows = site_level_cv_ellx(x_pool_sites, y_pool_sites, t_grid_pool, seed + 80)
            for row in rows:
                cv_rows.append({"part": part, "macrorep": macrorep, "arm": arm, **row})
        else:
            params = DEFAULT_PARAMS

        model_post = fit_ckme_with_params(X_pool, Y_pool, R_SITE, params)
        q_hat_post = conformal_qhat(model_post, X_cal, Y_cal)
        coverage = float(np.mean(raw_scores(model_post, X_test, y_test) <= q_hat_post))
        post_curves = diagnostic_curves(
            model_post,
            t_grid_pool,
            q_hat_post,
            spec.bump_sd,
            x_pool_sites,
            y_pool_sites,
            params,
            n_boot,
            rng_boot1,
        )
        hot_cdf_l2, bg_cdf_l2 = summarize_window(post_curves["cdf_l2"], spec.window)
        hot_utail_post, _ = summarize_window(post_curves["u_tail"], spec.window)
        si_utail_post = separation_index(post_curves["u_tail"], spec.window)

        lof = lof_profile(model_post, t_grid_pool, x_lof_sites, y_lof_sites)
        in_lof_window = (x_lof_sites >= spec.window[0]) & (x_lof_sites <= spec.window[1])
        hot_lof_max = float(np.max(lof[in_lof_window]))
        bg_lof_med = float(np.median(lof[~in_lof_window]))

        in_stage2_window = (x2_sites >= spec.window[0]) & (x2_sites <= spec.window[1])
        n1_in_window = int(np.sum(in_stage2_window))

        metric_rows.append(
            {
                "part": part,
                "macrorep": macrorep,
                "arm": arm,
                "ell_x": float(params.ell_x),
                "q_hat_pre": q_hat_pre,
                "q_hat_post": q_hat_post,
                "coverage_pre": pre_cov,
                "coverage": coverage,
                "mean_width": float(np.mean(post_curves["width"])),
                "hot_cdf_l2": hot_cdf_l2,
                "bg_cdf_l2": bg_cdf_l2,
                "hot_utail_post": hot_utail_post,
                "si_utail_post": si_utail_post,
                "hot_lof_max": hot_lof_max,
                "bg_lof_med": bg_lof_med,
                "n1_in_window": n1_in_window,
                "pre_hot_cdf_l2": pre_hot_cdf_l2,
                "pre_bg_cdf_l2": pre_bg_cdf_l2,
                "pre_si_utail": pre_si_utail,
            }
        )

        for x in x2_sites:
            site_rows.append(
                {
                    "part": part,
                    "macrorep": macrorep,
                    "arm": arm,
                    "x": float(x),
                    "in_window": bool(spec.window[0] <= x <= spec.window[1]),
                }
            )
        for x, value in zip(x_lof_sites, lof):
            lof_rows.append(
                {
                    "part": part,
                    "macrorep": macrorep,
                    "arm": arm,
                    "x": float(x),
                    "lof": float(value),
                    "in_window": bool(spec.window[0] <= x <= spec.window[1]),
                }
            )
        for diag, values in post_curves.items():
            for x, value in zip(X_GRID, values):
                curve_rows.append(
                    {
                        "part": part,
                        "macrorep": macrorep,
                        "stage": "post",
                        "arm": arm,
                        "diag": diag,
                        "x": float(x),
                        "value": float(value),
                    }
                )

    return {
        "metrics": metric_rows,
        "curves_raw": curve_rows,
        "sites": site_rows,
        "lof": lof_rows,
        "cv": cv_rows,
    }


# ---------------------------------------------------------------------------
# Output aggregation and plotting
# ---------------------------------------------------------------------------


def output_dir(part: str) -> Path:
    return OUT_A if part == "a" else OUT_B


def prefix(part: str) -> str:
    return "gate4a" if part == "a" else "gate4b"


def aggregate_curves(curves_raw: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        curves_raw.groupby(["part", "stage", "arm", "diag", "x"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped.drop(columns=["std", "count"])


def write_part_outputs(part: str, rows: list[dict[str, list[dict]]]) -> None:
    out = output_dir(part)
    pref = prefix(part)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([row for res in rows for row in res["metrics"]])
    curves_raw = pd.DataFrame([row for res in rows for row in res["curves_raw"]])
    sites = pd.DataFrame([row for res in rows for row in res["sites"]])
    lof = pd.DataFrame([row for res in rows for row in res["lof"]])
    cv = pd.DataFrame([row for res in rows for row in res["cv"]])
    curves = aggregate_curves(curves_raw)

    metrics.to_csv(out / f"{pref}_metrics.csv", index=False)
    curves.to_csv(out / f"{pref}_curves.csv", index=False)
    curves_raw.to_csv(out / f"{pref}_curves_raw.csv", index=False)
    sites.to_csv(out / f"{pref}_sites.csv", index=False)
    lof.to_csv(out / f"{pref}_lof.csv", index=False)
    if part == "b":
        if not cv.empty:
            cv.to_csv(out / f"{pref}_ellx_cv.csv", index=False)
            selected = (
                cv[["macrorep", "arm", "selected_ell_x"]]
                .drop_duplicates()
                .rename(columns={"selected_ell_x": "ell_x"})
            )
        else:
            selected = metrics.loc[metrics["arm"] == "utail_retune", ["macrorep", "arm", "ell_x"]]
        selected.to_csv(out / f"{pref}_ellx_selected.csv", index=False)


def read_part_outputs(part: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = output_dir(part)
    pref = prefix(part)
    metrics = pd.read_csv(out / f"{pref}_metrics.csv")
    curves = pd.read_csv(out / f"{pref}_curves.csv")
    sites = pd.read_csv(out / f"{pref}_sites.csv")
    lof = pd.read_csv(out / f"{pref}_lof.csv")
    return metrics, curves, sites, lof


def curve_mean(curves: pd.DataFrame, stage: str, arm: str, diag: str) -> pd.DataFrame:
    return curves[(curves["stage"] == stage) & (curves["arm"] == arm) & (curves["diag"] == diag)].sort_values("x")


def _strip_box(ax: plt.Axes, data: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.25)
    rng = np.random.default_rng(123)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(i + jitter, vals, s=18, alpha=0.75, color=color, edgecolor="none")


def plot_g4a() -> Path:
    metrics, curves, sites, _ = read_part_outputs("a")
    out = OUT_A
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    axes = axes.ravel()
    colors = {"lhs": "tab:blue", "utail": "tab:orange", "oracle": "tab:green"}
    spec = PART_A

    ax = axes[0]
    xs = np.linspace(0, 1, 500)
    ax.plot(xs, f_true(xs, spec.bump_sd), color="k", lw=1.5)
    y0 = f_true(stage1_sites("a"), spec.bump_sd)
    ax.plot(stage1_sites("a"), y0 - 0.28, "|", color="tab:red", ms=10, label="Stage-1 sites")
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_title("DGP and thinned Stage-1 design")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)

    ax = axes[1]
    pre_u = curve_mean(curves, "pre", "pre", "u_tail")
    pre_c = curve_mean(curves, "pre", "pre", "cdf_l2")
    u_norm = pre_u["mean"].to_numpy() / max(float(pre_u["mean"].max()), 1e-12)
    c_norm = pre_c["mean"].to_numpy() / max(float(pre_c["mean"].max()), 1e-12)
    ax.plot(pre_u["x"], u_norm, color="tab:orange", lw=1.5, label="u_tail normalized")
    ax.plot(pre_c["x"], c_norm, color="tab:purple", lw=1.5, label="CDF L2 normalized")
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_title("PRE diagnostics: the flag")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = axes[2]
    bins = np.linspace(0, 1, 21)
    for arm in spec.arms:
        vals = sites.loc[sites["arm"] == arm, "x"].to_numpy()
        ax.hist(vals, bins=bins, histtype="step", lw=1.4, color=colors[arm], label=arm)
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_title("Stage-2 allocations: the action")
    ax.set_xlabel("x")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    ax = axes[3]
    pre = curve_mean(curves, "pre", "pre", "cdf_l2")
    ax.plot(pre["x"], pre["mean"], color="gray", lw=1.2, label="pre")
    for arm in spec.arms:
        c = curve_mean(curves, "post", arm, "cdf_l2")
        ax.plot(c["x"], c["mean"], color=colors[arm], lw=1.5, label=arm)
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("POST CDF L2: the verification")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = axes[4]
    wide = metrics.pivot(index="macrorep", columns="arm", values="hot_cdf_l2")
    ax.scatter(wide["lhs"], wide["utail"], s=30, color="tab:orange", alpha=0.85)
    mx = float(np.nanmax([wide["lhs"].max(), wide["utail"].max()]))
    ax.plot([0, mx], [0, mx], color="k", ls="--", lw=0.9)
    ax.set_title("Paired hotspot CDF L2")
    ax.set_xlabel("lhs")
    ax.set_ylabel("utail")

    ax = axes[5]
    pre_q = metrics.drop_duplicates("macrorep").sort_values("macrorep")["q_hat_pre"].to_numpy()
    data = [pre_q] + [metrics.loc[metrics["arm"] == arm, "q_hat_post"].to_numpy() for arm in spec.arms]
    labels = ["pre", "lhs", "utail", "oracle"]
    _strip_box(ax, data, labels, ["gray", colors["lhs"], colors["utail"], colors["oracle"]])
    ax.axhline(0.30, color="k", ls=":", lw=1.0, alpha=0.7)
    ax.set_title("CP q_hat width tax")
    ax.set_ylabel("q_hat")

    fig.suptitle(f"Gate 4a: u_tail-targeted sampling for a data-scarcity hotspot (N1={N_STAGE2})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / "gate4a_fix_epistemic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_g4b() -> Path:
    metrics, curves, _, lof = read_part_outputs("b")
    out = OUT_B
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    colors = {"lhs_fixed": "tab:blue", "utail_fixed": "tab:orange", "utail_retune": "tab:green"}
    spec = PART_B

    ax = axes[0]
    pre = curve_mean(curves, "pre", "pre", "cdf_l2")
    ax.plot(pre["x"], pre["mean"], color="gray", lw=1.2, label="pre")
    for arm in spec.arms:
        c = curve_mean(curves, "post", arm, "cdf_l2")
        ax.plot(c["x"], c["mean"], color=colors[arm], lw=1.5, label=arm)
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("POST CDF L2")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = axes[1]
    pre_u = curve_mean(curves, "pre", "pre", "u_tail")
    ax.plot(pre_u["x"], pre_u["mean"], color="gray", lw=1.2, label="pre")
    for arm in spec.arms:
        c = curve_mean(curves, "post", arm, "u_tail")
        ax.plot(c["x"], c["mean"], color=colors[arm], lw=1.5, label=arm)
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("POST u_tail")
    ax.set_xlabel("x")

    ax = axes[2]
    for arm in spec.arms:
        df = lof[lof["arm"] == arm]
        ax.scatter(df["x"], df["lof"], s=10, alpha=0.35, color=colors[arm], label=arm)
    ax.axhline(10.0, color="k", ls="--", lw=1.0, label="threshold 10")
    ax.axvspan(*spec.window, color="gray", alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("LOF: the second flag")
    ax.set_xlabel("x")
    ax.set_ylabel("LOF")
    ax.legend(fontsize=8)

    ax = axes[3]
    pre_q = metrics.drop_duplicates("macrorep").sort_values("macrorep")["q_hat_pre"].to_numpy()
    data = [pre_q] + [metrics.loc[metrics["arm"] == arm, "q_hat_post"].to_numpy() for arm in spec.arms]
    labels = ["pre", "lhs", "utail", "retune"]
    _strip_box(ax, data, labels, ["gray", colors["lhs_fixed"], colors["utail_fixed"], colors["utail_retune"]])
    ax.set_title("CP q_hat")
    ax.set_ylabel("q_hat")

    fig.suptitle("Gate 4b: sampling alone cannot fix a capacity-limited hotspot", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "gate4b_capacity_limit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "nan"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def summarize_g4a(metrics: pd.DataFrame) -> str:
    wide_hot = metrics.pivot(index="macrorep", columns="arm", values="hot_cdf_l2")
    wide_q = metrics.pivot(index="macrorep", columns="arm", values="q_hat_post")
    p_hot = wilcoxon(wide_hot["utail"], wide_hot["lhs"], alternative="less").pvalue
    p_q = wilcoxon(wide_q["utail"], wide_q["lhs"], alternative="less").pvalue
    p_hot_oracle = wilcoxon(wide_hot["oracle"], wide_hot["lhs"], alternative="less").pvalue
    ratio_hot = float(np.median(wide_hot["utail"] / wide_hot["lhs"]))
    ratio_hot_oracle = float(np.median(wide_hot["oracle"] / wide_hot["lhs"]))
    ratio_q = float(np.median(wide_q["utail"] / wide_q["lhs"]))
    cov_by_arm = metrics.groupby("arm")["coverage"].mean()
    n1_by_arm = metrics.groupby("arm")["n1_in_window"].mean()
    si_by_arm = metrics.groupby("arm")["si_utail_post"].median()
    pre_si = float(metrics.drop_duplicates("macrorep")["pre_si_utail"].median())

    return "\n".join(
        [
            "G4a summary:",
            f"  hot_cdf_l2 utail/lhs median ratio = {ratio_hot:.3f}, Wilcoxon p = {_fmt_p(p_hot)}",
            f"  hot_cdf_l2 oracle/lhs median ratio = {ratio_hot_oracle:.3f}, Wilcoxon p = {_fmt_p(p_hot_oracle)}",
            f"  q_hat utail/lhs median ratio = {ratio_q:.3f}, Wilcoxon p = {_fmt_p(p_q)}",
            f"  mean coverage by arm: {cov_by_arm.round(3).to_dict()}",
            f"  mean n1_in_window by arm: {n1_by_arm.round(1).to_dict()}",
            f"  median pre SI(u_tail) = {pre_si:.1f}; median post SI = {si_by_arm.round(2).to_dict()}",
        ]
    )


def summarize_g4b(metrics: pd.DataFrame, lof: pd.DataFrame | None = None) -> str:
    wide_hot = metrics.pivot(index="macrorep", columns="arm", values="hot_cdf_l2")
    ratio_fixed = float(np.median(wide_hot["utail_fixed"] / wide_hot["lhs_fixed"]))
    ratio_retune = float(np.median(wide_hot["utail_retune"] / wide_hot["utail_fixed"]))
    p_retune = wilcoxon(wide_hot["utail_retune"], wide_hot["utail_fixed"], alternative="less").pvalue
    cov_by_arm = metrics.groupby("arm")["coverage"].mean()
    lof_fixed = metrics.loc[metrics["arm"] == "utail_fixed", "hot_lof_max"].to_numpy()
    lof_retune = metrics.loc[metrics["arm"] == "utail_retune", "hot_lof_max"].to_numpy()
    fixed_high = float(np.mean(lof_fixed > 10.0))
    retune_low = float(np.mean(lof_retune < 10.0))
    ell_counts = metrics.loc[metrics["arm"] == "utail_retune", "ell_x"].value_counts().sort_index()
    q_by_arm = metrics.groupby("arm")["q_hat_post"].mean()
    bg99_line = ""
    if lof is not None and not lof.empty:
        bg99 = float(lof.loc[~lof["in_window"], "lof"].quantile(0.99))
        fixed_over_bg99 = float(np.mean(lof_fixed > bg99))
        retune_over_bg99 = float(np.mean(lof_retune > bg99))
        bg99_line = (
            f"\n  pooled background LOF 99% threshold = {bg99:.3f}; "
            f"P(fixed > threshold) = {fixed_over_bg99:.2f}; "
            f"P(retune > threshold) = {retune_over_bg99:.2f}"
        )

    return "\n".join(
        [
            "G4b summary:",
            f"  hot_cdf_l2 utail_fixed/lhs_fixed median ratio = {ratio_fixed:.3f}",
            f"  hot_cdf_l2 utail_retune/utail_fixed median ratio = {ratio_retune:.3f}, Wilcoxon p = {_fmt_p(p_retune)}",
            f"  LOF: P(hot_lof_max fixed > 10) = {fixed_high:.2f}; P(retune < 10) = {retune_low:.2f}{bg99_line}",
            f"  selected ell_x counts: {ell_counts.to_dict()}",
            f"  mean q_hat_post by arm: {q_by_arm.round(3).to_dict()}",
            f"  mean coverage by arm: {cov_by_arm.round(3).to_dict()}",
        ]
    )


def analyze_part(part: str) -> None:
    metrics, _, _, lof = read_part_outputs(part)
    if part == "a":
        fig = plot_g4a()
        print(summarize_g4a(metrics))
    else:
        fig = plot_g4b()
        print(summarize_g4b(metrics, lof))
    print(f"Wrote {fig}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_part(part: str, n_macro: int, n_boot: int, n_workers: int) -> None:
    out = output_dir(part)
    out.mkdir(parents=True, exist_ok=True)
    spec = part_spec(part)
    print(f"{spec.label}: {n_macro} macroreps, B={n_boot} bootstraps, N1={N_STAGE2}")

    results: list[dict[str, list[dict]]] = []
    if n_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = {
                    pool.submit(run_one_macrorep, part, k, n_boot): k
                    for k in range(n_macro)
                }
                for fut in as_completed(futs):
                    k = futs[fut]
                    res = fut.result()
                    results.append(res)
                    q_post = [r["q_hat_post"] for r in res["metrics"]]
                    print(f"  macrorep {k:2d} done; q_post range {min(q_post):.3f}-{max(q_post):.3f}")
            write_part_outputs(part, results)
            analyze_part(part)
            return
        except PermissionError as exc:
            print(f"  multiprocessing unavailable ({exc}); falling back to sequential execution")

    if n_workers <= 1 or not results:
        for k in range(n_macro):
            res = run_one_macrorep(part, k, n_boot)
            results.append(res)
            q_post = [r["q_hat_post"] for r in res["metrics"]]
            print(f"  macrorep {k:2d} done; q_post range {min(q_post):.3f}-{max(q_post):.3f}")

    write_part_outputs(part, results)
    analyze_part(part)


def main() -> None:
    global N_STAGE2, OUT_A, OUT_B

    parser = argparse.ArgumentParser(description="Gate 4: targeted Stage-2 sampling for epistemic hotspots")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument("--n_macro", type=int, default=20)
    parser.add_argument("--n_boot", type=int, default=30)
    parser.add_argument("--n_stage2", type=int, default=N_STAGE2)
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix for output_gate4{a,b}_<suffix>; prevents overwriting default outputs.",
    )
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    N_STAGE2 = args.n_stage2
    suffix = args.output_suffix.strip().strip("_")
    if suffix:
        OUT_A = _root / "exp_framing_validation" / f"output_gate4a_{suffix}"
        OUT_B = _root / "exp_framing_validation" / f"output_gate4b_{suffix}"

    parts = ["a", "b"] if args.part == "all" else [args.part]
    if args.analyze_only:
        for part in parts:
            analyze_part(part)
        return

    for part in parts:
        run_part(part, args.n_macro, args.n_boot, args.n_workers)


if __name__ == "__main__":
    main()
