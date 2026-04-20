# Experiment Code Writing Skill

## Purpose
This skill helps write, review, and extend experiment scripts for the Two-stage-simple project.
It captures the standard patterns shared across all experiment folders (`exp_nongauss`,
`exp_allocation`, `exp_conditional_coverage`, `exp_onesided`, `exp_branin`, etc.) so that
new experiments are consistent, reproducible, and maintainable.

This skill is especially useful when:
- Writing a new `run_*.py` script from scratch
- Extending an existing experiment with new simulators or baselines
- Reviewing a script for correctness before a long run
- Debugging seed, output, or R-subprocess issues

---

## 1. Standard Script Skeleton

Every `run_*.py` follows this top-to-bottom order:

```python
"""
exp_<name>: One-line description of the experiment.

Longer description: what is compared, on which simulators, with what metrics.

Usage (from project root):
  python exp_<name>/run_<name>_compare.py
  python exp_<name>/run_<name>_compare.py --n_macro 10 --n_workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --- Root path setup (must come before any project imports) ---
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from Two_stage.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from CKME.parameters import Params, ParamGrid

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
R_SCRIPT    = _root / "run_benchmarks_one_case.R"
MIXED_RATIO = 0.7

SIMULATORS = ["sim_A", "sim_B"]   # ← fill in for this experiment

# Column name maps for coverage / width / interval score.
# Column names follow the convention in evaluate_per_point (CKME) and the
# R script output (DCP-DR / hetGP). The hetGP coverage column is named
# "covered_interval_hetgp" (not "covered_score_hetgp") because R outputs
# interval-based coverage rather than score-based coverage.
METHOD_COV   = {"CKME": "covered_score",    "DCP-DR": "covered_score_dr",    "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width",             "DCP-DR": "width_dr",             "hetGP": "width_hetgp"}
METHOD_SCORE = {"CKME": "interval_score",    "DCP-DR": "interval_score_dr",    "hetGP": "interval_score_hetgp"}


# ---------------------------------------------------------------------------
# R benchmark helper
# ---------------------------------------------------------------------------
def _run_r_benchmarks(case_dir: Path, output_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(case_dir), str(output_csv), str(alpha), str(n_grid)]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not output_csv.exists():
        raise RuntimeError(
            f"R script did not produce {output_csv}.\n"
            f"stderr: {result.stderr or 'none'}\nstdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(output_csv)


# ---------------------------------------------------------------------------
# Core: one macrorep
# ---------------------------------------------------------------------------
def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    out_dir: Path,
    method: str,
    params: Params | None = None,
) -> dict:
    # Seed encodes both macrorep identity AND simulator identity to avoid
    # collisions when multiple simulators share the same macrorep_id.
    # See Section 2 for the full seed convention.
    sim_idx = SIMULATORS.index(simulator_func) if simulator_func in SIMULATORS else 0
    seed = base_seed + macrorep_id * 10000 + sim_idx * 100

    n_0    = config["n_0"]
    r_0    = config["r_0"]
    n_1    = config["n_1"]
    r_1    = config["r_1"]
    alpha  = config["alpha"]
    n_grid = config.get("t_grid_size", 500)

    if params is None:
        params = config.get("params")   # Params object built by load_config_from_file

    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=simulator_func,
        params=params,
        t_grid_size=n_grid,
        random_state=seed + 2,
        verbose=False,
    )
    X0, Y0 = stage1.X_all, stage1.Y_all

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
        simulator_func=simulator_func,
        method=method,
        alpha=alpha,
        mixed_ratio=MIXED_RATIO,
        random_state=seed + 3,
        verbose=False,
    )

    # generate_test_data uses np.random.seed internally (legacy path, see Section 2 note).
    X_test, Y_test = generate_test_data(
        stage2_result=stage2,
        n_test=config["n_test"],
        r_test=config.get("r_test", 1),
        X_cand=X_cand,
        simulator_func=simulator_func,
        random_state=seed + 4,
    )

    eval_result = evaluate_per_point(stage2, X_test, Y_test)
    rows = eval_result["rows"]

    # --- Save data for R benchmarks ---
    case_name = f"{simulator_func}_{method}"
    case_dir  = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
    rep0_dir  = case_dir / "macrorep_0"
    rep0_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(rep0_dir / "X0.csv",     X0,              delimiter=",")
    np.savetxt(rep0_dir / "Y0.csv",     Y0,              delimiter=",")
    np.savetxt(rep0_dir / "X1.csv",     stage2.X_stage2, delimiter=",")
    np.savetxt(rep0_dir / "Y1.csv",     stage2.Y_stage2, delimiter=",")
    np.savetxt(rep0_dir / "X_test.csv", X_test,          delimiter=",")
    np.savetxt(rep0_dir / "Y_test.csv", Y_test,          delimiter=",")

    # --- R benchmarks (catch failure gracefully) ---
    bench_csv = case_dir / "benchmarks.csv"
    try:
        bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
        if len(bench_df) != len(rows):
            raise RuntimeError(
                f"R output has {len(bench_df)} rows but CKME has {len(rows)} rows. "
                "Row counts must match for column merge."
            )
        for i, row in enumerate(rows):
            for col in bench_df.columns:
                row[col] = bench_df.iloc[i][col]
    except RuntimeError as e:
        print(f"  Warning: R benchmarks failed for {case_name}; DCP-DR/hetGP will be NaN.\n  {e}",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(case_dir / "per_point.csv", index=False)

    # --- Aggregate to scalar metrics ---
    out = {}
    for name, col in METHOD_COV.items():
        out[f"{name}_coverage"] = df[col].mean() if col in df.columns else float("nan")
    for name, col in METHOD_WIDTH.items():
        out[f"{name}_width"] = df[col].mean() if col in df.columns else float("nan")
    for name, col in METHOD_SCORE.items():
        out[f"{name}_interval_score"] = df[col].mean() if col in df.columns else float("nan")
    return out


# ---------------------------------------------------------------------------
# Top-level worker (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def _run_task(args_tuple):
    k, sim, base_seed, config, out_dir, params, method = args_tuple
    return run_one_macrorep(k, base_seed, config, sim, out_dir, method, params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, default="exp_<name>/config.txt")
    parser.add_argument("--output_dir",      type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Override path to pretrained_params.json")
    parser.add_argument("--n_macro",         type=int, default=5)
    parser.add_argument("--base_seed",       type=int, default=42)
    parser.add_argument("--method",          type=str, default="lhs",
                        choices=("lhs", "sampling", "mixed"))
    parser.add_argument("--n_workers",       type=int, default=1)
    parser.add_argument("--quick",           action="store_true",
                        help="Scale down sizes for end-to-end pipeline testing")
    args = parser.parse_args()

    config  = load_config_from_file(_root / args.config)
    if args.quick:
        config = get_config(config, quick=True)
        # quick=True: n_0/n_1/n_test/n_cand × 0.25 (min 10), r_0=1, r_1=2, r_test=2
        # (exact scaling defined in Two_stage/config_utils.py::get_config)
    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_<name>" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pretrained hyperparameters ---
    pretrained: dict[str, Params] = {}
    pretrained_path = Path(args.pretrained_path) if args.pretrained_path \
        else _root / "exp_<name>" / "pretrained_params.json"
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in SIMULATORS}
        print(f"Loaded pretrained params from {pretrained_path}")
    else:
        print(
            f"Warning: {pretrained_path} not found; using config defaults for all simulators.\n"
            "Run 'python exp_<name>/pretrain_params.py' first to improve accuracy.",
            file=sys.stderr,
        )

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; DCP-DR/hetGP will be missing.",
              file=sys.stderr)

    # --- Run macroreps ---
    # Two loop structures are common (see Section 8):
    #   Option A (below): flatten (macrorep, sim) pairs → one pool for all tasks.
    #   Option B: outer loop over sims, inner pool over macroreps (used in exp_nongauss)
    #             → allows per-sim summary immediately after each sim completes.
    tasks = [(k, sim) for k in range(args.n_macro) for sim in SIMULATORS]
    all_rows: list[dict] = []

    if args.n_workers > 1:
        task_args = [
            (k, sim, args.base_seed, config, out_dir, pretrained.get(sim), args.method)
            for k, sim in tasks
        ]
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_run_task, t): (t[0], t[1]) for t in task_args}
            for fut in as_completed(futures):
                k, sim = futures[fut]
                try:
                    row = fut.result()
                    row["simulator"] = sim
                    row["macrorep"] = k
                    all_rows.append(row)
                    print(f"  Done: macrorep={k}, sim={sim}")
                except Exception as exc:
                    print(f"  Error macrorep={k}, sim={sim}: {exc}", file=sys.stderr)
    else:
        for k, sim in tasks:
            print(f"\n--- macrorep={k}, sim={sim} ---")
            row = run_one_macrorep(k, args.base_seed, config, sim, out_dir, args.method,
                                   pretrained.get(sim))
            row["simulator"] = sim
            row["macrorep"] = k
            all_rows.append(row)

    # --- Save summary ---
    df_all = pd.DataFrame(all_rows)
    metric_cols = [c for c in df_all.columns if c not in ("simulator", "macrorep")]

    for sim in SIMULATORS:
        df_sim = df_all[df_all["simulator"] == sim]
        if df_sim.empty:
            continue
        agg = df_sim[metric_cols].agg(["mean", "std"])
        agg.to_csv(out_dir / f"summary_{sim}.csv")

    df_all.to_csv(out_dir / "all_macroreps.csv", index=False)
    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    # Required on macOS/Windows: the 'spawn' multiprocessing start method
    # re-imports this module in every worker process. Without this guard,
    # main() would be called again in each worker, causing infinite recursion.
    main()
```

---

## 2. Seed Rules

**The most important correctness rule in this codebase.**

```python
# CORRECT
# Encode both macrorep_id and simulator index so that different (k, sim) pairs
# never share a seed, even when run simultaneously in a process pool.
sim_idx = SIMULATORS.index(simulator_func)          # 0, 1, 2, ...
seed    = base_seed + macrorep_id * 10000 + sim_idx * 100

X_cand = get_x_cand(...,          random_state=seed + 1)
stage1 = run_stage1_train(...,    random_state=seed + 2)
stage2 = run_stage2(...,          random_state=seed + 3)
X_test, Y_test = generate_test_data(..., random_state=seed + 4)

# FORBIDDEN — never do this at module or function level
np.random.seed(42)   # does not affect default_rng; misleading
random.seed(42)      # same problem
```

**Why**: All core functions (`run_stage1_train`, `run_stage2`, `data_collection`,
simulators) use `np.random.default_rng(random_state)`, not the legacy global state.
Calling `np.random.seed()` does nothing to these functions and gives false confidence.

**Stride arithmetic**:
- `macrorep_id * 10000` — enough gap that per-macrorep offsets (seed+1 … seed+99) never
  collide across macroreps.
- `sim_idx * 100` — enough gap for per-simulator offsets within the same macrorep.
  With up to 10 simulators (100 × 10 = 1000 < 10000), no cross-sim collision occurs.

**Offset convention within one (macrorep, sim) call**:
- `seed + 1` → candidate site generation
- `seed + 2` → Stage 1 (design + observations)
- `seed + 3` → Stage 2 (site selection + observations)
- `seed + 4` → test data generation
- `seed + 5, 6, ...` → additional methods in same macrorep (e.g. multiple `method` variants)

**Known exception**: `generate_test_data` (in `Two_stage/test_data.py`) internally calls
`np.random.seed(random_state)` (legacy API). This is a known technical debt. Its state is
independent from `default_rng`-based functions; pass `random_state=seed+4` as usual, and the
call will be reproducible within this function's own logic, but it does not interact with
`default_rng` generators elsewhere.

---

## 2b. Common Random Numbers (CRN) for Method Comparison

When an experiment compares multiple methods (e.g. one-stage vs two-stage, or different
site-selection strategies), use **Common Random Numbers** so that every method sees the
same "random world" within each macrorep. This drastically reduces variance of the
paired difference, making real effects visible with fewer macroreps.

**What to share across methods in the same macrorep:**

| Component | Shared? | Why |
|-----------|---------|-----|
| Candidate sites (`seed + 1`) | Yes | Same pool of candidate locations |
| Stage 1 design + observations (`seed + 2`) | Yes | Same initial model for all methods |
| Test points + test responses (`seed + 4`) | Yes | Same evaluation set for fair comparison |
| Stage 2 site selection (`seed + 3`) | **No** | This is where methods differ |

**Implementation pattern:**

```python
# Compute the base seed for this (macrorep, simulator) pair ONCE
seed = base_seed + macrorep_id * 10000 + sim_idx * 100

# Stage 1 — shared across all methods (train once, reuse)
stage1 = run_stage1_train(..., random_state=seed + 2)
X_test, Y_test = generate_test_data(..., random_state=seed + 4)

# Stage 2 — run per method, only site selection differs
for method in methods:
    stage2 = run_stage2(stage1_result=stage1, ..., method=method,
                        random_state=seed + 3)
    evaluate_per_point(stage2, X_test, Y_test)
```

The key constraint: `stage1`, `X_cand`, `X_test`, and `Y_test` are computed **once** per
macrorep and reused by every method. Only `run_stage2` (and its internal site selection +
data collection) runs separately per method.

**Analysis: use paired differences.**

Because each macrorep produces results for all methods on the same random world, the
natural analysis unit is the **per-macrorep difference** (e.g. `IS_twostage(k) − IS_onestage(k)`),
not the raw per-method means. Report:

- Mean and SD of paired differences across macroreps
- Number of macroreps where method A beats method B (e.g. "18 out of 20")
- Paired Wilcoxon signed-rank test or paired t-test for significance

This paired approach exploits the high correlation (ρ) between methods evaluated on the
same world. Variance of the paired difference scales as `(1 − ρ)` times the unpaired
variance, often giving 3–5× effective sample size increase for free.

---

## 3. Config File Format (`config.txt`)

Use `Two_stage.config_utils.load_config_from_file` for all **new** experiments.

> **Migration note**: Older experiment folders (e.g. `exp_nongauss/`) contain their own
> `config_utils.py` that mirrors this interface. For consistency, new experiments should
> import directly from `Two_stage.config_utils`, not from a per-experiment copy.

```ini
# exp_<name>/config.txt
n_0 = 250
r_0 = 20
n_1 = 500
r_1 = 10
alpha = 0.1
t_grid_size = 500
n_cand = 1000
n_test = 1000
r_test = 1
method = lhs

# Fallback hyperparameters (used when pretrained_params.json is absent)
ell_x = 0.5
lam = 1e-3
h = 0.1

# CV grid — NOT auto-parsed; pretrain_params.py reads these manually
# ell_x_list = 0.1,0.5,1.0,2.0
# lam_list = 1e-4,1e-3,1e-2
# h_list = 0.05,0.1,0.2,0.3
# cv_folds = 5
```

**Auto-parsed keys** (see `Two_stage/config_utils.py`):

| Type | Keys |
|------|------|
| `int` | `n_0 r_0 n_1 r_1 n_test r_test n_cand n_macro n_jobs t_grid_size` |
| `float` | `ell_x lam h alpha` |
| `str` | `simulator_func method` |

Any other key remains a raw string. `load_config_from_file` also constructs a `Params`
object at `config["params"]` from `ell_x`, `lam`, `h` (defaulting to 0.1, 1e-4, 0.05 if
absent). CV grid keys (`ell_x_list`, `lam_list`, `h_list`, `cv_folds`) are **not**
auto-parsed; `pretrain_params.py` must read them manually or define them as constants.

---

## 4. Quick Mode

Every new experiment script should support `--quick` for end-to-end pipeline testing.
Use `get_config(config, quick=True)` which scales all sizes down:

```python
if args.quick:
    config = get_config(config, quick=True)
    # Actual scaling (from Two_stage/config_utils.py::get_config):
    #   n_0, n_1, n_test, n_cand → max(10, int(original * 0.25))
    #   r_0 = 1, r_1 = 2, r_test = 2
```

For additional quick-mode overrides (skip pretrained, single simulator):

```python
if args.quick:
    config = get_config(config, quick=True)
    pretrained = {sim: Params(ell_x=0.5, lam=1e-3, h=0.1) for sim in SIMULATORS}
```

> **Note**: Existing scripts such as `run_nongauss_compare.py` do not yet implement
> `--quick`. New scripts should add it; retrofitting old scripts is welcome but not required.

---

## 5. Output Directory Structure

```
exp_<name>/output/
  macrorep_{k}/
    case_{simulator}_{method}/
      macrorep_0/          ← raw data files consumed by R script
        X0.csv
        Y0.csv
        X1.csv
        Y1.csv
        X_test.csv
        Y_test.csv
      per_point.csv        ← per-test-point CKME + R results merged (one row per test point)
      benchmarks.csv       ← raw R output (kept separately for debugging / R-only re-analysis)
  summary_{simulator}.csv  ← aggregated mean/std per metric across macroreps
  all_macroreps.csv        ← flat table: one row per (macrorep, simulator)
```

`benchmarks.csv` is kept even though its columns are merged into `per_point.csv` because:
(a) it allows re-running R analysis without re-running CKME, and (b) it helps debug R output
format issues independently of the Python merge step.

**`per_point.csv` complete column list** (produced by `evaluate_per_point` + R merge):

| Column | Source | Description |
|--------|--------|-------------|
| `x0` | CKME | Test point first coordinate (multi-D: also `x1`, `x2`, ...) |
| `y` | CKME | Observed response value |
| `L`, `U` | CKME | Prediction interval bounds |
| `covered_interval` | CKME | 1 if `L ≤ Y ≤ U` (direct), else 0 |
| `covered_score` | CKME | 1 if conformity score ≤ q̂ (score-based), else 0 |
| `width` | CKME | `U − L` |
| `interval_score` | CKME | Winkler IS: `(U−L) + (2/α)[L−Y]₊ + (2/α)[Y−U]₊` |
| `status` | CKME | `"in"` / `"below"` / `"above"` |
| `covered_score_dr`, `width_dr`, `interval_score_dr` | R (DCP-DR) | DCP-DR equivalents |
| `covered_interval_hetgp`, `width_hetgp`, `interval_score_hetgp` | R (hetGP) | hetGP equivalents |

R columns are absent (→ NaN in aggregation) when R fails. `covered_interval` and
`covered_score` differ: interval coverage checks Y ∈ [L, U] directly; score coverage checks
whether the conformity score s(X, Y) falls below the calibrated quantile q̂. Both are
reported; `covered_score` is used as the primary CKME metric in summary tables.

---

## 6. Pretrained Hyperparameters

Running CV for every macrorep is slow. The `pretrain_params.py` script runs CV once and
saves results to `pretrained_params.json`. The main script loads this file at startup.

```python
# pretrain_params.py skeleton
from pathlib import Path
import json
import sys

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Two_stage.config_utils import load_config_from_file
from Two_stage import run_stage1_train
from CKME.parameters import ParamGrid

CONFIG_PATH   = _root / "exp_<name>" / "config.txt"
OUT_PATH      = _root / "exp_<name>" / "pretrained_params.json"
PRETRAIN_SEED = 0

config = load_config_from_file(CONFIG_PATH)
n_0, r_0       = config["n_0"], config["r_0"]
t_grid_size    = config.get("t_grid_size", 500)

PARAM_GRID = ParamGrid(
    ell_x_list=[0.1, 0.5, 1.0, 2.0],
    lam_list=[1e-4, 1e-3, 1e-2],
    h_list=[0.05, 0.1, 0.2],
)

results = {}
for sim in SIMULATORS:
    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=sim,
        param_grid=PARAM_GRID,
        cv_folds=5,
        t_grid_size=t_grid_size,
        random_state=PRETRAIN_SEED,
        verbose=True,
    )
    p = stage1.params
    results[sim] = {"ell_x": p.ell_x, "lam": p.lam, "h": p.h}
    print(f"{sim}: ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT_PATH}")
```

In the main script, load and use:

```python
pretrained_path = Path(args.pretrained_path) if args.pretrained_path \
    else _root / "exp_<name>" / "pretrained_params.json"
if pretrained_path.exists():
    with open(pretrained_path) as f:
        raw = json.load(f)
    pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in SIMULATORS}
```

Pass `params=pretrained.get(sim)` to `run_one_macrorep`. If `None`, fall back to
`config["params"]` (the fixed defaults from `config.txt`), which is slower-tuning
but still correct.

**Re-run `pretrain_params.py` after any simulator modification** that changes the data
distribution, since CV-optimal hyperparameters depend on the DGP.

---

## 7. R Benchmark Integration

The standard pattern for calling `run_benchmarks_one_case.R`:

```python
def _run_r_benchmarks(case_dir: Path, output_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(case_dir), str(output_csv), str(alpha), str(n_grid)]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not output_csv.exists():
        raise RuntimeError(
            f"R script did not produce {output_csv}.\n"
            f"stderr: {result.stderr or 'none'}\nstdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(output_csv)
```

**Critical**: Always use `check=False` (not `check=True`) and test for file existence
rather than return code. R may exit 0 even on soft failure, and may exit non-zero even
when partial output was written.

**Always verify row count before merging** R output into CKME rows:

```python
try:
    bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
    if len(bench_df) != len(rows):
        raise RuntimeError(
            f"R output has {len(bench_df)} rows but CKME has {len(rows)}. "
            "Row counts must match for column merge."
        )
    for i, row in enumerate(rows):
        for col in bench_df.columns:
            row[col] = bench_df.iloc[i][col]
except RuntimeError as e:
    print(f"  Warning: R benchmarks failed for {case_name}; DCP-DR/hetGP will be NaN.\n  {e}",
          file=sys.stderr)
# Experiment continues; R columns absent from per_point.csv → NaN in aggregation
```

**Catch `RuntimeError` only** (not bare `except Exception`): this ensures unexpected Python
errors (import failures, disk-full, etc.) still propagate and crash the run visibly.

**R script signature** (`run_benchmarks_one_case.R`):

```
Rscript run_benchmarks_one_case.R <case_dir> <output_csv> <alpha> <n_grid>
```

It reads `macrorep_0/{X0,Y0,X1,Y1,X_test,Y_test}.csv` from `case_dir` and writes
`output_csv` with DCP-DR and hetGP results. All six CSV files must exist before calling R.

---

## 8. Parallel Execution

Use `concurrent.futures.ProcessPoolExecutor`. The worker function **must be
module-level** (not a closure or lambda) for pickling to work on macOS/Linux.

```python
# Module-level (required for pickling across processes)
def _run_task(args_tuple):
    k, sim, base_seed, config, out_dir, params, method = args_tuple
    # out_dir must be an absolute Path — relative paths break in worker processes
    return run_one_macrorep(k, base_seed, config, sim, out_dir, method, params)
```

**Two loop structures are used in practice:**

**Option A — flat pool** (skeleton default): all `(macrorep, sim)` pairs submitted at once.
```python
tasks = [(k, sim) for k in range(n_macro) for sim in SIMULATORS]
with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futures = {pool.submit(_run_task, t): (t[0], t[1]) for t in task_args}
    for fut in as_completed(futures):
        ...
```
Best when tasks have similar runtimes and you want maximum parallelism.

**Option B — sim-outer, macrorep-inner** (used in `run_nongauss_compare.py`): loop over
simulators sequentially, submit macroreps in parallel per simulator.
```python
for sim in SIMULATORS:
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(run_one_macrorep, k, ...): k for k in range(n_macro)}
        results = {futs[f]: f.result() for f in as_completed(futs)}
    # aggregate and print summary for this sim immediately
```
Best when you want per-simulator summaries printed as the run progresses.

**`if __name__ == "__main__"` guard is required** when using `ProcessPoolExecutor` on
macOS/Windows. The `spawn` start method re-imports this module in every worker. Without
the guard, `main()` is called recursively in each worker process, causing an infinite
spawning loop.

**Worker count**: `--n_workers 4` is a good starting point on an 8-core laptop. Do not
exceed physical core count; memory pressure from loading CKME models in parallel processes
will negate the speedup.

---

## 9. Summary Aggregation

**Standard pattern** (matches skeleton and actual experiment scripts):

```python
df_all = pd.DataFrame(all_rows)
# all_rows: each dict has keys like "CKME_coverage", "DCP-DR_width", "macrorep", "simulator"

metric_cols = [c for c in df_all.columns if c not in ("simulator", "macrorep")]

# Per-simulator summary (flat CSV, not subdirectory)
for sim in SIMULATORS:
    df_sim = df_all[df_all["simulator"] == sim]
    if df_sim.empty:
        continue
    agg = df_sim[metric_cols].agg(["mean", "std"])
    agg.to_csv(out_dir / f"summary_{sim}.csv")

# All-simulator flat table for plotting
df_all.to_csv(out_dir / "all_macroreps.csv", index=False)
```

**Alternative pattern** used in `run_nongauss_compare.py` (manual mean/std, more explicit):

```python
rows_summary = []
for sim in SIMULATORS:
    # collect lists per metric across macroreps
    cov_list = [r["CKME_coverage"] for r in macrorep_results if "CKME_coverage" in r]
    rows_summary.append({
        "simulator":     sim,
        "mean_coverage": np.mean(cov_list) if cov_list else np.nan,
        "sd_coverage":   np.std(cov_list, ddof=1) if len(cov_list) > 1 else np.nan,
        ...
    })
pd.DataFrame(rows_summary).to_csv(out_dir / "summary_combined.csv", index=False)
```

Use `ddof=1` for unbiased sample standard deviation whenever computing SD by hand.

**Avoid `pd.DataFrame.agg(...).columns` MultiIndex flattening tricks** — they are
fragile across pandas versions. Use the simple `.agg(["mean", "std"])` and keep the
MultiIndex, or compute mean/std manually as above.

---

## 10. Adding a New Experiment Folder

Checklist for a new `exp_<name>/` folder:

```
exp_<name>/
  config.txt            ← experiment settings (parsed by Two_stage/config_utils.py)
  run_<name>_compare.py ← main script (follow skeleton above)
  pretrain_params.py    ← one-time CV hyperparameter tuning (run before main exp)
  plot_<name>.py        ← visualization (separate from run script)
  spec.md               ← experiment design document (write BEFORE running)
  output/               ← created at runtime (gitignored)
```

Register the experiment in `EXPERIMENTS.md` with status `[planned]` before running.
Update to `[active]` when running, `[done]` when results are recorded.

---

## 11. Common Mistakes to Avoid

| Mistake | Correct Practice |
|---------|-----------------|
| `np.random.seed(42)` at top of script | Remove; use `random_state=seed+k` per call |
| Same `seed` for all simulators in one macrorep | Include `sim_idx` in seed formula (see Section 2) |
| Hardcoding `n_0=100` in `run_one_macrorep` | Read from `config["n_0"]` |
| Using `subprocess.run(..., check=True)` for R | Use `check=False` + check file existence |
| Merging R rows without checking row count | Add `assert len(bench_df) == len(rows)` or raise RuntimeError |
| Lambda or nested function passed to `ProcessPoolExecutor` | Use module-level `_run_task` |
| Passing a relative `Path` to worker processes | Always use absolute paths (`_root / ...`) |
| Writing R columns as separate file only | Merge into `per_point.csv` so analysis is unified |
| Creating `output/` with `os.makedirs` | Use `Path.mkdir(parents=True, exist_ok=True)` |
| Importing from `exp_<name>.config_utils` | Import from `Two_stage.config_utils` in new scripts |
| Using `Y.min()/Y.max()` for `t_grid` bounds | Use `np.percentile(Y, 0.5)` / `np.percentile(Y, 99.5)` |
| Assuming `load_config_from_file` parses CV grid keys | It does not; define `ell_x_list` etc. as constants in `pretrain_params.py` |
| Omitting `if __name__ == "__main__":` guard | Required for `ProcessPoolExecutor` on macOS/Windows (spawn method) |
| Confusing `covered_interval` and `covered_score` | `covered_interval`: direct Y∈[L,U]; `covered_score`: score ≤ q̂. Use `covered_score` for CKME primary metric. |

---

## 12. t_grid Construction Rule

When constructing `t_grid` manually (rare; usually handled inside `run_stage1_train`):

```python
# CORRECT: percentile-based bounds prevent heavy-tail outliers from dominating
Y_lo = np.percentile(Y_all, 0.5)
Y_hi = np.percentile(Y_all, 99.5)
y_margin = 0.10 * (Y_hi - Y_lo)
t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

# WRONG: a single outlier can expand t_grid so much that predict_quantile
# always clips at t_grid[-1] for high τ, making sup_abs_err ≈ t_grid_max
t_grid = np.linspace(Y_all.min(), Y_all.max(), t_grid_size)
```

---

## 13. Interval Score and Coverage Metric Definitions

```python
# Winkler interval score (implemented in CP/evaluation.py::compute_interval_score)
IS = (U - L) + (2 / alpha) * np.maximum(L - Y, 0) + (2 / alpha) * np.maximum(Y - U, 0)
```

Do not implement this formula inline; call `compute_interval_score` from `CP/evaluation.py`.

**Two coverage metrics** are reported for CKME (both in `per_point.csv`):

| Metric | Column | Definition | When to use |
|--------|--------|------------|-------------|
| Interval coverage | `covered_interval` | `1{L ≤ Y ≤ U}` | Sanity check; matches hetGP/DCP-DR definition |
| Score coverage | `covered_score` | `1{s(X,Y) ≤ q̂}` | Primary CKME metric; theoretically grounded for split CP |

Use `covered_score` as the primary coverage metric in all summary tables and plots for CKME.
Use `covered_interval` when comparing directly against DCP-DR or hetGP (which only report
interval coverage). Report both in per-point CSV so downstream analysis can choose.
