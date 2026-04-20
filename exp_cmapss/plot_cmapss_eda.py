"""
plot_cmapss_eda.py

Exploratory visualization of raw C-MAPSS FD001 data.

Figures
-------
  --mode sensors   : degradation curves for the 6 sensors most correlated with RUL
  --mode rul       : RUL distribution (training set) + engine lifetime histogram
  --mode scatter   : best sensor vs RUL scatter (shows heteroscedastic noise)

Usage
-----
python exp_cmapss/plot_cmapss_eda.py --mode sensors --save exp_cmapss/output/eda_sensors.png
python exp_cmapss/plot_cmapss_eda.py --mode rul     --save exp_cmapss/output/eda_rul.png
python exp_cmapss/plot_cmapss_eda.py --mode scatter --save exp_cmapss/output/eda_scatter.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Top-6 sensors by |correlation with RUL| (computed from FD001 training set)
TOP6 = [
    ("s11", "Fan speed (N1)"),
    ("s4",  "LPT outlet temp (T50)"),
    ("s12", "LPC outlet pressure"),
    ("s7",  "HPC inlet total pressure"),
    ("s15", "Bypass ratio (BPR)"),
    ("s21", "HPC coolant bleed"),
]

# Sensors that go DOWN as the engine degrades (flip sign for intuitive display)
DEGRADING_DOWN = {"s11", "s4", "s7", "s15", "s21"}


def _load(data_dir: Path) -> np.ndarray:
    """Return structured ndarray with columns matching _COLS."""
    path = data_dir / "train_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    data = np.genfromtxt(path)
    return data


def _add_rul(data: np.ndarray, rul_cap: int = 125) -> np.ndarray:
    """Append a RUL column to data array."""
    unit  = data[:, 0].astype(int)
    cycle = data[:, 1].astype(int)
    max_cycle = np.zeros(len(unit))
    for uid in np.unique(unit):
        mask = unit == uid
        max_cycle[mask] = cycle[mask].max()
    rul = np.clip(max_cycle - cycle, 0, rul_cap)
    return np.column_stack([data, rul])  # last column = RUL


def _col_idx(name: str) -> int:
    return _COLS.index(name)


# ---------------------------------------------------------------------------
# Fig 1: Sensor degradation curves
# ---------------------------------------------------------------------------

def plot_sensors(data_dir: Path, save: str | None = None, n_engines: int = 8):
    """
    2×3 grid of sensor degradation curves for n_engines representative engines.
    x-axis = normalized life progress (0 = birth, 1 = failure).
    """
    if save:
        matplotlib.use("Agg")

    arr = _add_rul(_load(data_dir))
    unit  = arr[:, 0].astype(int)
    cycle = arr[:, 1].astype(int)
    rul   = arr[:, -1]

    # Pick engines with varied lifetimes for visual diversity
    all_units = np.unique(unit)
    engine_life = {u: cycle[unit == u].max() for u in all_units}
    sorted_by_life = sorted(all_units, key=lambda u: engine_life[u])
    step = max(1, len(sorted_by_life) // n_engines)
    selected = sorted_by_life[::step][:n_engines]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(selected)))

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False)

    for ax, (sname, slabel) in zip(axes.flat, TOP6):
        cidx = _col_idx(sname)
        for eng, color in zip(selected, colors):
            mask = unit == eng
            cyc  = cycle[mask]
            vals = arr[mask, cidx]
            max_c = cyc.max()
            life_pct = cyc / max_c          # 0 → 1
            ax.plot(life_pct, vals, lw=0.9, alpha=0.75, color=color)

        ax.set_title(f"{sname}: {slabel}", fontsize=11)
        ax.set_xlabel("Life progress (0=new, 1=failure)")
        ax.set_ylabel("Sensor reading")
        # Add a soft degradation-direction arrow annotation
        direction = "↓ degrades" if sname in DEGRADING_DOWN else "↑ degrades"
        ax.text(0.02, 0.05, direction, transform=ax.transAxes,
                fontsize=8, color="gray", va="bottom")

    fig.suptitle(
        "C-MAPSS FD001 — Top-6 sensor degradation curves\n"
        f"({len(selected)} engines, x = normalized life progress)",
        fontsize=13,
    )
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: RUL distribution + engine lifetime histogram
# ---------------------------------------------------------------------------

def plot_rul(data_dir: Path, save: str | None = None):
    """
    Left: distribution of RUL across all training observations (capped at 125).
    Right: histogram of total engine lifetimes (cycles to failure).
    """
    if save:
        matplotlib.use("Agg")

    arr   = _add_rul(_load(data_dir))
    unit  = arr[:, 0].astype(int)
    cycle = arr[:, 1].astype(int)
    rul   = arr[:, -1]

    engine_life = np.array([cycle[unit == u].max() for u in np.unique(unit)])

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: RUL histogram ---
    ax = axes[0]
    ax.hist(rul, bins=50, color="#4daf4a", edgecolor="white", alpha=0.85)
    ax.axvline(30, color="#e41a1c", lw=1.5, ls="--", label="High-risk threshold (RUL=30)")
    ax.set_xlabel("Remaining Useful Life (RUL, capped at 125)")
    ax.set_ylabel("Number of observations")
    ax.set_title("RUL distribution (all training cycles)")
    ax.legend(fontsize=9)

    # --- Right: engine lifetime histogram ---
    ax = axes[1]
    ax.hist(engine_life, bins=20, color="#377eb8", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Total flight cycles to failure")
    ax.set_ylabel("Number of engines")
    ax.set_title(f"Engine lifetime distribution\n(100 training engines, mean={engine_life.mean():.0f} cycles)")
    ax.axvline(engine_life.mean(), color="black", lw=1.5, ls="--", label=f"Mean = {engine_life.mean():.0f}")
    ax.legend(fontsize=9)

    fig.suptitle("C-MAPSS FD001 — Training Set Overview", fontsize=13)
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Sensor vs RUL scatter (heteroscedastic noise)
# ---------------------------------------------------------------------------

def plot_scatter(data_dir: Path, save: str | None = None):
    """
    Scatter of top-3 sensors vs RUL.
    x-axis = RUL (inverted so left = near failure).
    Shows that noise / spread varies with RUL → motivates CKME.
    """
    if save:
        matplotlib.use("Agg")

    arr  = _add_rul(_load(data_dir))
    rul  = arr[:, -1]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (sname, slabel) in zip(axes, TOP6[:3]):
        cidx = _col_idx(sname)
        vals = arr[:, cidx]
        ax.scatter(rul, vals, s=1.5, alpha=0.08, color="#333333", rasterized=True)
        ax.set_xlabel("RUL (right = healthy, left = near failure)")
        ax.set_ylabel(sname)
        ax.set_title(f"{sname}: {slabel}")
        ax.invert_xaxis()   # failure is on the left

    fig.suptitle(
        "C-MAPSS FD001 — Sensor vs RUL scatter\n"
        "(heteroscedastic noise motivates conditional distribution estimation)",
        fontsize=12,
    )
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C-MAPSS EDA plots")
    parser.add_argument("--mode", choices=["sensors", "rul", "scatter"],
                        default="sensors")
    parser.add_argument("--data_dir", default="exp_cmapss/data")
    parser.add_argument("--save", default=None,
                        help="Output file path. If omitted, opens interactive window.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.mode == "sensors":
        plot_sensors(data_dir, save=args.save)
    elif args.mode == "rul":
        plot_rul(data_dir, save=args.save)
    elif args.mode == "scatter":
        plot_scatter(data_dir, save=args.save)


if __name__ == "__main__":
    main()
