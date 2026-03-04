"""
evaluation.py

Per-point evaluation: coverage (interval + score), width, interval score, penalty status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from CP.evaluation import compute_interval_score

ArrayLike = np.ndarray


def _compute_score_coverage(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    q_hat: float,
    t_grid: np.ndarray,
    score_type: str = "abs_median",
) -> np.ndarray:
    """Compute score-based coverage per point: 1{score(x,y) <= q_hat}."""
    from CKME.coefficients import compute_ckme_coeffs
    from CP.scores import score_from_cdf

    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)
    G = model.indicator.g_matrix(model.Y, Y_test)
    F_test = np.sum(C * G, axis=0).astype(float)
    np.clip(F_test, 0.0, 1.0, out=F_test)
    scores = score_from_cdf(F_test, score_type=score_type)
    return (scores <= q_hat).astype(int)


def evaluate_per_point(
    stage2_result: object,
    X_test: ArrayLike,
    Y_test: ArrayLike,
) -> dict:
    """
    Compute per-point metrics: both coverage methods, width, interval score, status.

    Returns
    -------
    dict with keys:
        - rows: list of dicts, one per (x,y), with x, y, L, U, covered_interval,
          covered_score, width, interval_score, status.
        - status: "in" | "below" | "above" (Y in [L,U], Y < L, Y > U)
        - summary: n_total, n_in, n_below, n_above, n_covered_interval, n_covered_score
    """
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    Y_test = np.asarray(Y_test, dtype=float).ravel()
    n = X_test.shape[0]
    if Y_test.shape[0] != n:
        raise ValueError("X_test and Y_test must have same length")

    L, U = stage2_result.predict_interval(X_test)
    width = U - L

    # Coverage: interval method
    covered_interval = ((Y_test >= L) & (Y_test <= U)).astype(int)

    # Coverage: score method
    covered_score = _compute_score_coverage(
        model=stage2_result.model,
        X_test=X_test,
        Y_test=Y_test,
        q_hat=stage2_result.cp.q_hat,
        t_grid=stage2_result.t_grid,
        score_type="abs_median",
    )

    # Interval score per point
    interval_scores, _ = compute_interval_score(
        Y_test, L, U, stage2_result.alpha
    )

    # Status: which points get penalty (not in [L,U])
    status = np.where(Y_test < L, "below", np.where(Y_test > U, "above", "in"))
    n_below = int(np.sum(status == "below"))
    n_above = int(np.sum(status == "above"))
    n_in = int(np.sum(status == "in"))

    d = X_test.shape[1]
    rows = []
    for i in range(n):
        row = {
            "y": float(Y_test[i]),
            "L": float(L[i]),
            "U": float(U[i]),
            "covered_interval": int(covered_interval[i]),
            "covered_score": int(covered_score[i]),
            "width": float(width[i]),
            "interval_score": float(interval_scores[i]),
            "status": str(status[i]),
        }
        for j in range(d):
            row[f"x{j}"] = float(X_test[i, j])
        rows.append(row)

    summary = {
        "n_total": n,
        "n_in": n_in,
        "n_below": n_below,
        "n_above": n_above,
        "n_covered_interval": int(covered_interval.sum()),
        "n_covered_score": int(covered_score.sum()),
        "indices_below": np.where(status == "below")[0].tolist(),
        "indices_above": np.where(status == "above")[0].tolist(),
    }

    return {"rows": rows, "summary": summary}


def save_raw_evaluation(
    result: dict,
    path: Union[str, Path],
) -> None:
    """
    Save raw evaluation to CSV and summary to JSON.

    - eval_raw.csv: one row per (x,y) with x0, x1..., y, L, U, covered_interval,
      covered_score, width, interval_score, status
    - eval_summary.json: n_total, n_in, n_below, n_above, n_covered_interval,
      n_covered_score, indices_below, indices_above
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    rows = result["rows"]
    summary = result["summary"]

    # CSV: raw per-point
    import csv
    if rows:
        keys = list(rows[0].keys())
        with open(path / "eval_raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # JSON: summary with counts and which indices
    (path / "eval_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
