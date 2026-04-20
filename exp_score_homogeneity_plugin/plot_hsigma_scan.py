"""
plot_hsigma_scan.py

Plot RMSE(sigma_hat) and ks_max as U-shape curves vs h_sigma
to empirically verify the bias-variance trade-off and locate
the optimal h_sigma.

Scan outputs are expected under:
  exp_score_homogeneity_plugin/output_scan/factor_<f>/
Plus the baseline factor=0.5 run under
  exp_score_homogeneity_plugin/output/

For each factor we aggregate ks_max and sigma_rmse across the
adaptive c-scan by (a) taking the min over c (best c for that
h_sigma) and (b) taking the value at a reference c=2.0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_SCAN_DIR = _HERE / "output_scan"
_BASE_DIR = _HERE / "output"
_SIMS = ["mild", "exp2", "nongauss_B2L"]

# Budget constants from config.txt
_N0 = 256
_R0 = 5
_X_LO, _X_HI = 0.0, 2 * np.pi


def _h_sigma_of(factor: float) -> float:
    return factor * (_X_HI - _X_LO) * (_N0 ** (-1.0 / 5.0))


def _parse_c(bw: str) -> float | None:
    m = re.match(r"adaptive_c([\d.]+)", bw)
    return float(m.group(1)) if m else None


def _load_factor(factor: float, sim: str) -> pd.DataFrame | None:
    if abs(factor - 0.5) < 1e-9:
        path = _BASE_DIR / f"summary_{sim}.csv"
    else:
        path = _SCAN_DIR / f"factor_{factor}" / f"summary_{sim}.csv"
    if not path.exists():
        print(f"  missing: {path}")
        return None
    df = pd.read_csv(path)
    df["c_val"] = df["bandwidth"].apply(_parse_c)
    return df[df["c_val"].notna()].copy()


def _collect(factors: list[float]):
    rows = []
    for f in factors:
        h_s = _h_sigma_of(f)
        for sim in _SIMS:
            df = _load_factor(f, sim)
            if df is None or df.empty:
                continue
            rmse = float(df["sigma_rmse_mean"].iloc[0])  # same for all c
            ks_min = float(df["ks_max_mean"].min())
            ks_best_c = float(df.loc[df["ks_max_mean"].idxmin(), "c_val"])
            df_c2 = df[np.isclose(df["c_val"], 2.0)]
            ks_c2 = float(df_c2["ks_max_mean"].iloc[0]) if len(df_c2) else np.nan
            cov_min = float(df["cov_gap_sup_mean"].min())
            rows.append(dict(factor=f, h_sigma=h_s, sim=sim,
                             sigma_rmse=rmse, ks_min=ks_min,
                             ks_best_c=ks_best_c, ks_c2=ks_c2,
                             cov_gap_min=cov_min))
    return pd.DataFrame(rows)


def _theoretical_h_star() -> float:
    # h_sigma_star ∝ (n0 * r0)^(-1/5); use same prefactor family as
    # h_sigma = factor * range * n0^(-1/5). Matching exponents gives
    # factor_star ~ range * r0^(-1/5). Just report the h_sigma value.
    return (_X_HI - _X_LO) * (_N0 * _R0) ** (-1.0 / 5.0)


def plot_scan(save_path: Path):
    factors = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 1.0]
    df = _collect(factors)
    if df.empty:
        print("no data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    h_star = _theoretical_h_star()

    colors = {"mild": "#1f77b4", "exp2": "#d62728",
              "nongauss_B2L": "#2ca02c"}

    # (a) sigma RMSE vs h_sigma
    ax = axes[0]
    for sim in _SIMS:
        sub = df[df["sim"] == sim].sort_values("h_sigma")
        if sub.empty:
            continue
        ax.plot(sub["h_sigma"], sub["sigma_rmse"], "o-",
                color=colors[sim], label=sim, linewidth=1.6)
    ax.axvline(h_star, color="k", linestyle=":", linewidth=1,
               label=rf"$h_\sigma^\star\!\propto\!(n_0 r_0)^{{-1/5}}$")
    ax.set_xlabel(r"$h_\sigma$")
    ax.set_ylabel(r"RMSE$(\hat\sigma)$")
    ax.set_title(r"(a) Plug-in error vs $h_\sigma$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (b) ks_max at best c vs h_sigma
    ax = axes[1]
    for sim in _SIMS:
        sub = df[df["sim"] == sim].sort_values("h_sigma")
        if sub.empty:
            continue
        ax.plot(sub["h_sigma"], sub["ks_min"], "s-",
                color=colors[sim], label=sim, linewidth=1.6)
    ax.axvline(h_star, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$h_\sigma$")
    ax.set_ylabel(r"KS$_{\max}$ (best $c$)")
    ax.set_title(r"(b) Homogeneity vs $h_\sigma$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (c) cov_gap_sup at best c
    ax = axes[2]
    for sim in _SIMS:
        sub = df[df["sim"] == sim].sort_values("h_sigma")
        if sub.empty:
            continue
        ax.plot(sub["h_sigma"], sub["cov_gap_min"], "^-",
                color=colors[sim], label=sim, linewidth=1.6)
    ax.axvline(h_star, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$h_\sigma$")
    ax.set_ylabel(r"sup coverage gap (best $c$)")
    ax.set_title(r"(c) Coverage gap vs $h_\sigma$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        rf"$h_\sigma$ scan ($n_0={_N0},\ r_0={_R0}$); "
        rf"theoretical $h_\sigma^\star\approx{h_star:.2f}$",
        fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")

    csv_path = save_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")


def main():
    out_dir = _HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scan(out_dir / "figD_hsigma_scan.png")


if __name__ == "__main__":
    main()
