"""Generate KME/CKME CDF-band diagnostics without conformal prediction."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

_TMP_DIR = Path(tempfile.gettempdir())
_MPL_DIR = _TMP_DIR / "ckme_mpl"
_CACHE_DIR = _TMP_DIR / "ckme_cache"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ckme_cdf import CKMECDFEstimator, pointwise_band
from .dgp import Scenario, find_representative_scenario, generate_training_scenarios
from .features import KMEFeatureBuilder, Standardizer, bootstrap_query_features, input_uncertainty, rho_hat
from .mm1 import oracle_mm1_outputs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_output(path: str, base: Path) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    return base / out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--rff-dim", type=int, default=80)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-oracle", type=int, default=20_000)
    parser.add_argument("--grid-size", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target-group", default="highU_heavy")
    parser.add_argument("--k-neighbors", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--out", default="figures/kme_ckme_cdf_bands.png")
    parser.add_argument("--data-out", default="")
    parser.add_argument("--log-dir", default="experiment_logs/ckme_dcp_mm1")
    return parser


def empirical_cdf_on_grid(values: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.mean(values[:, None] <= t_grid[None, :], axis=0)


def write_band_data(
    path: Path,
    t_grid: np.ndarray,
    oracle_cdf: np.ndarray,
    base_cdf: np.ndarray,
    without_band: tuple[np.ndarray, np.ndarray, np.ndarray],
    with_band: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t",
        "oracle_cdf",
        "base_ckme_cdf",
        "without_input_low",
        "without_input_median",
        "without_input_high",
        "with_input_low",
        "with_input_median",
        "with_input_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(t_grid):
            writer.writerow(
                {
                    "t": float(t),
                    "oracle_cdf": float(oracle_cdf[i]),
                    "base_ckme_cdf": float(base_cdf[i]),
                    "without_input_low": float(without_band[0][i]),
                    "without_input_median": float(without_band[1][i]),
                    "without_input_high": float(without_band[2][i]),
                    "with_input_low": float(with_band[0][i]),
                    "with_input_median": float(with_band[1][i]),
                    "with_input_high": float(with_band[2][i]),
                }
            )


def write_experiment_log(
    log_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    scenario: Scenario,
    figure_path: Path,
    data_path: Path,
    estimator: CKMECDFEstimator,
    diagnostics: dict[str, float],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}_kme_ckme_cdf_band.md"
    rho_est = rho_hat(scenario.inter_arrivals, scenario.services)
    content = f"""# KME/CKME CDF-Band Diagnostic Log

Date: {datetime.now().astimezone().isoformat()}

## Goal

Diagnose the KME + CKME part only, without conformal prediction. The goal is to
visualize how finite input samples become KME covariates and how input-sample
uncertainty propagates into the CKME estimate of the conditional output CDF
`F(t | Z)`.

This run is not a conformal coverage experiment and is not a Lam-style inflated
KS confidence band. It is a mechanism diagnostic for input uncertainty.

## Settings

- seed: {args.seed}
- training scenarios: {args.n_train}
- RFF dimension per input distribution: {args.rff_dim}
- bootstrap replicates: {args.n_bootstrap}
- oracle Monte Carlo outputs: {args.n_oracle}
- CDF grid size: {args.grid_size}
- pointwise band alpha: {args.alpha}
- band type: pointwise bootstrap diagnostic band
- target diagnostic group: {args.target_group}
- CKME adaptive bandwidth: `eta={args.eta}`, `beta={args.beta}`, `k={args.k_neighbors}`
- output smoothing bandwidth h_y: {estimator.h_y:.6g}
- covariate bandwidth base h_z: {estimator.fixed_h_z:.6g}

## Diagnostic Scenario

- true lambda: {scenario.lambda_rate:.6g}
- true mu: {scenario.mu_rate:.6g}
- true rho: {scenario.rho:.6g}
- finite inter-arrival sample size n_A: {scenario.n_a}
- finite service sample size n_S: {scenario.n_s}
- estimated rho_hat from finite samples: {rho_est:.6g}
- rho_hat - true rho: {rho_est - scenario.rho:.6g}

## Results

| metric | value |
| --- | ---: |
| ISE, base CKME vs oracle | {diagnostics["ise"]:.6g} |
| sup error, base CKME vs oracle | {diagnostics["sup_error"]:.6g} |
| pointwise coverage, no-IU band | {diagnostics["point_cover_noiu"]:.6g} |
| pointwise coverage, IU band | {diagnostics["point_cover_iu"]:.6g} |
| mean half-width, no-IU band | {diagnostics["mean_half_width_noiu"]:.6g} |
| mean half-width, IU band | {diagnostics["mean_half_width_iu"]:.6g} |

## Band Definitions

- Base CKME CDF: `F_hat(t | Z_hat)` using the fitted KME/RFF covariate.
- Without input uncertainty band: bootstrap training scenarios while holding
  the diagnostic scenario's finite-input KME representation fixed.
- With input uncertainty band: bootstrap training scenarios and bootstrap the
  diagnostic scenario's finite input samples before recomputing its KME
  representation.
- Oracle CDF: Monte Carlo empirical CDF from the true `(lambda, mu)` of the
  diagnostic scenario.

## Outputs

- figure: `{figure_path}`
- band data CSV: `{data_path}`
- source PDF: `{package_root() / 'mm1_ckme_dcp_input_uncertainty.pdf'}`

## Figure Panels

- Panel 1: oracle CDF, base CKME CDF, and no-IU/IU pointwise bootstrap diagnostic bands.
- Panel 2: CDF error relative to the oracle CDF, including no-IU/IU band error regions.
- Panel 3: no-IU/IU band half-widths, showing how much the input bootstrap inflates uncertainty.

## What To Look For

The with-input-uncertainty band should be wider or visibly shifted when finite
input samples make `Z_hat` unstable. A hard high-traffic, small-sample scenario
is selected by default to make this effect visible. If the horizontal zero line
in Panel 2 is outside the IU error region for long stretches, the diagnostic
band is not wide enough to cover the oracle CDF there.
"""
    log_path.write_text(content, encoding="utf-8")
    return log_path


def bootstrap_with_input_curves(
    estimator: CKMECDFEstimator,
    scenario: Scenario,
    builder: KMEFeatureBuilder,
    standardizer: Standardizer,
    t_grid: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    """Bootstrap training data and query finite input samples together."""

    if estimator.z_fit is None or estimator.y_fit is None or estimator.u_fit is None:
        raise ValueError("estimator must be fitted")
    z_query_boot, u_query_boot = bootstrap_query_features(scenario, builder, standardizer, rng, n_bootstrap)
    curves = []
    n = estimator.y_fit.size
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot = CKMECDFEstimator(
            adaptive=estimator.adaptive,
            fixed_h_z=estimator.fixed_h_z,
            k_neighbors=min(estimator.k_neighbors, n),
            eta=estimator.eta,
            beta=estimator.beta,
            h_y=estimator.h_y,
        ).fit(estimator.z_fit[idx], estimator.y_fit[idx], estimator.u_fit[idx])
        curves.append(boot.predict_cdf(z_query_boot[b : b + 1], t_grid, u_query=u_query_boot[b : b + 1])[0])
    return np.vstack(curves)


def plot_bands(
    t_grid: np.ndarray,
    oracle_cdf: np.ndarray,
    base_cdf: np.ndarray,
    without_band: tuple[np.ndarray, np.ndarray, np.ndarray],
    with_band: tuple[np.ndarray, np.ndarray, np.ndarray],
    scenario: Scenario,
    figure_path: Path,
    diagnostics: dict[str, float],
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    low_noiu, med_noiu, high_noiu = without_band
    low_iu, med_iu, high_iu = with_band

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.6, 9.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.35, 1.15]},
    )
    rho_est = rho_hat(scenario.inter_arrivals, scenario.services)

    ax = axes[0]
    ax.fill_between(t_grid, low_noiu, high_noiu, color="#4E79A7", alpha=0.16, label="no-IU band")
    ax.fill_between(t_grid, low_iu, high_iu, color="#E15759", alpha=0.18, label="IU band")
    ax.plot(t_grid, oracle_cdf, color="black", linewidth=2.2, label="oracle MC CDF")
    ax.plot(t_grid, base_cdf, color="#F28E2B", linewidth=2.0, label="base CKME CDF")
    ax.plot(t_grid, med_noiu, color="#4E79A7", linewidth=1.2, linestyle="--", alpha=0.85, label="no-IU median")
    ax.plot(t_grid, med_iu, color="#E15759", linewidth=1.2, linestyle="--", alpha=0.85, label="IU median")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(
        "Pointwise bootstrap diagnostic bands: "
        f"n_A={scenario.n_a}, n_S={scenario.n_s}, true rho={scenario.rho:.2f}, rho_hat={rho_est:.2f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    base_error = base_cdf - oracle_cdf
    ax = axes[1]
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.fill_between(t_grid, low_noiu - oracle_cdf, high_noiu - oracle_cdf, color="#4E79A7", alpha=0.15, label="no-IU band error")
    ax.fill_between(t_grid, low_iu - oracle_cdf, high_iu - oracle_cdf, color="#E15759", alpha=0.20, label="IU band error")
    ax.plot(t_grid, base_error, color="#F28E2B", linewidth=2.0, label="base CKME - oracle")
    ax.set_ylabel("CDF error")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    noiu_half_width = 0.5 * (high_noiu - low_noiu)
    iu_half_width = 0.5 * (high_iu - low_iu)
    ax = axes[2]
    ax.plot(t_grid, noiu_half_width, color="#4E79A7", linewidth=2.0, label="no-IU half-width")
    ax.plot(t_grid, iu_half_width, color="#E15759", linewidth=2.0, label="IU half-width")
    ax.set_xlabel("average sojourn time t")
    ax.set_ylabel("half-width")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    summary = (
        f"ISE={diagnostics['ise']:.4g}\n"
        f"sup error={diagnostics['sup_error']:.4g}\n"
        f"point cover no-IU={diagnostics['point_cover_noiu']:.2f}\n"
        f"point cover IU={diagnostics['point_cover_iu']:.2f}"
    )
    axes[0].text(
        0.02,
        0.98,
        summary,
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#D0D0D0"},
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(figure_path, dpi=230)
    plt.close(fig)


def compute_band_diagnostics(
    oracle_cdf: np.ndarray,
    base_cdf: np.ndarray,
    without_band: tuple[np.ndarray, np.ndarray, np.ndarray],
    with_band: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    low_noiu, _, high_noiu = without_band
    low_iu, _, high_iu = with_band
    return {
        "ise": float(np.mean((base_cdf - oracle_cdf) ** 2)),
        "sup_error": float(np.max(np.abs(base_cdf - oracle_cdf))),
        "point_cover_noiu": float(np.mean((oracle_cdf >= low_noiu) & (oracle_cdf <= high_noiu))),
        "point_cover_iu": float(np.mean((oracle_cdf >= low_iu) & (oracle_cdf <= high_iu))),
        "mean_half_width_noiu": float(np.mean(0.5 * (high_noiu - low_noiu))),
        "mean_half_width_iu": float(np.mean(0.5 * (high_iu - low_iu))),
    }


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train = generate_training_scenarios(args.n_train, rng)
    scenario = find_representative_scenario(rng, target=args.target_group)
    y_train = np.array([s.y for s in train], dtype=float)

    builder = KMEFeatureBuilder(rff_dim=args.rff_dim).fit(train, rng)
    z_train_raw = builder.transform(train)
    standardizer = Standardizer().fit(z_train_raw)
    z_train = standardizer.transform(z_train_raw)
    u_train = input_uncertainty(train)

    z_query = standardizer.transform(builder.transform([scenario]))
    u_query = np.array([scenario.n_a**-0.5 + scenario.n_s**-0.5], dtype=float)
    estimator = CKMECDFEstimator(
        adaptive=True,
        k_neighbors=min(args.k_neighbors, args.n_train),
        eta=args.eta,
        beta=args.beta,
    ).fit(z_train, y_train, u_train)

    oracle_outputs = oracle_mm1_outputs(scenario.lambda_rate, scenario.mu_rate, rng, args.n_oracle)
    lower_t = float(np.quantile(np.concatenate([oracle_outputs, y_train]), 0.001))
    upper_t = float(np.quantile(np.concatenate([oracle_outputs, y_train]), 0.995))
    t_grid = np.linspace(max(0.0, lower_t), upper_t, args.grid_size)

    oracle_cdf = empirical_cdf_on_grid(oracle_outputs, t_grid)
    base_cdf = estimator.predict_cdf(z_query, t_grid, u_query=u_query)[0]
    without_curves = estimator.bootstrap_training_curves(z_query, t_grid, rng, args.n_bootstrap, u_query=u_query)
    with_curves = bootstrap_with_input_curves(estimator, scenario, builder, standardizer, t_grid, rng, args.n_bootstrap)
    without_band = pointwise_band(without_curves, alpha=args.alpha)
    with_band = pointwise_band(with_curves, alpha=args.alpha)
    diagnostics = compute_band_diagnostics(oracle_cdf, base_cdf, without_band, with_band)

    figure_path = resolve_output(args.out, package_root())
    data_path = resolve_output(args.data_out, package_root()) if args.data_out else package_root() / "results" / f"{timestamp}_cdf_band_data.csv"
    log_dir = resolve_output(args.log_dir, repo_root())
    write_band_data(data_path, t_grid, oracle_cdf, base_cdf, without_band, with_band)
    plot_bands(t_grid, oracle_cdf, base_cdf, without_band, with_band, scenario, figure_path, diagnostics)
    log_path = write_experiment_log(log_dir, timestamp, args, scenario, figure_path, data_path, estimator, diagnostics)

    print(f"Wrote figure: {figure_path}")
    print(f"Wrote band data: {data_path}")
    print(f"Wrote experiment log: {log_path}")


if __name__ == "__main__":
    main()
