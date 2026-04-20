"""
pinball.py

One-sided pinball loss and hybrid CRPS+pinball loss for CKME hyperparameter tuning.

These losses are designed for one-sided quantile estimation (tau = 0.1 or 0.9).
Both losses operate on predicted CDF matrices F_pred (shape n x M) by inverting
the CDF to obtain estimated quantiles, then evaluating the pinball loss.

CDF inversion:
    q̂_tau(x_i) = t_grid[j],  j = inf{m : F_pred[i, m] >= tau}

Pinball loss at level tau:
    ell_tau(q, y) = (tau - 1{y <= q}) * (y - q)
"""

from __future__ import annotations

from typing import List

import numpy as np

from .crps import CRPSLoss

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# CDF inversion helper
# ---------------------------------------------------------------------------

def _invert_cdf(F_pred: ArrayLike, t_grid: ArrayLike, tau: float) -> ArrayLike:
    """
    Invert estimated CDF to get quantile estimates.

    For each sample i, returns:
        q̂_tau(x_i) = t_grid[j],  j = first index where F_pred[i, j] >= tau

    If no crossing exists (all F_pred[i, :] < tau), returns t_grid[-1].

    Parameters
    ----------
    F_pred : ndarray, shape (n, M)
        Estimated CDF values.
    t_grid : ndarray, shape (M,)
        Threshold grid (sorted).
    tau : float
        Target quantile level in (0, 1).

    Returns
    -------
    q : ndarray, shape (n,)
        Estimated quantiles.
    """
    mask = F_pred >= tau          # (n, M)
    any_cross = mask.any(axis=1)  # (n,)
    first_cross = np.argmax(mask, axis=1)  # (n,) — 0 if no cross (need correction)
    first_cross[~any_cross] = len(t_grid) - 1
    return t_grid[first_cross]    # (n,)


# ---------------------------------------------------------------------------
# One-sided pinball loss
# ---------------------------------------------------------------------------

class OneSidedPinballLoss:
    """
    One-sided pinball loss via CDF inversion.

    For each tau in taus, inverts F_pred to get q̂_tau(x_i), then computes
    the average pinball loss. The final loss is the mean over all taus.

    This is the most task-aligned loss for one-sided quantile estimation:
    it directly targets the conditional quantiles used in one-sided CP scores.

    Loss definition (averaged over taus and samples):
        L = (1/|taus|) * sum_tau [ (1/n) * sum_i ell_tau(q̂_tau(x_i), y_i) ]

    where:
        ell_tau(q, y) = (tau - 1{y <= q}) * (y - q)
        q̂_tau(x_i)   = t_grid[j],  j = inf{m : F_pred[i,m] >= tau}

    Parameters
    ----------
    taus : list of float
        Target quantile levels. For one-sided lower bound use [alpha],
        for upper bound use [1-alpha], for both use [alpha, 1-alpha].
    """

    def __init__(self, taus: List[float]):
        if not taus:
            raise ValueError("taus must be a non-empty list.")
        self.taus = list(taus)

    def compute(
        self,
        F_pred: ArrayLike,
        Y_true: ArrayLike,
        t_grid: ArrayLike,
        **kwargs,
    ) -> float:
        """
        Compute average one-sided pinball loss over all taus.

        Parameters
        ----------
        F_pred : ndarray, shape (n, M)
            Predicted CDF values.
        Y_true : ndarray, shape (n,)
            True Y values.
        t_grid : ndarray, shape (M,)
            Threshold grid.

        Returns
        -------
        loss : float
            Average pinball loss across taus and samples.
        """
        F_pred = np.asarray(F_pred, dtype=float)
        Y_true = np.asarray(Y_true, dtype=float).ravel()
        t_grid = np.asarray(t_grid, dtype=float).ravel()

        losses = []
        for tau in self.taus:
            q = _invert_cdf(F_pred, t_grid, tau)
            residual = Y_true - q
            pinball = np.where(residual >= 0, tau * residual, (tau - 1.0) * residual)
            losses.append(float(np.mean(pinball)))

        return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Hybrid CRPS + one-sided pinball loss
# ---------------------------------------------------------------------------

class HybridCRPSPinballLoss:
    """
    Hybrid loss: lam * CRPS + (1 - lam) * OneSidedPinball.

    Balances global CDF fidelity (CRPS) with tail-specific quantile accuracy
    (one-sided pinball). Useful when distributional stability matters alongside
    one-sided target alignment.

    Parameters
    ----------
    taus : list of float
        Target quantile levels for the pinball component.
    lam : float, default=0.5
        Mixing weight. lam=1 reduces to pure CRPS; lam=0 to pure pinball.
    crps_weight_scheme : str, default="uniform"
        Weight scheme for the CRPS component.
    """

    def __init__(self, taus: List[float], lam: float = 0.5,
                 crps_weight_scheme: str = "uniform"):
        self.pinball = OneSidedPinballLoss(taus=taus)
        self.crps = CRPSLoss(weight_scheme=crps_weight_scheme)
        self.lam = lam

    def compute(
        self,
        F_pred: ArrayLike,
        Y_true: ArrayLike,
        t_grid: ArrayLike,
        **kwargs,
    ) -> float:
        """
        Compute hybrid loss.

        Parameters
        ----------
        F_pred : ndarray, shape (n, M)
        Y_true : ndarray, shape (n,)
        t_grid : ndarray, shape (M,)

        Returns
        -------
        loss : float
        """
        crps_val = self.crps.compute(F_pred, Y_true, t_grid)
        pin_val = self.pinball.compute(F_pred, Y_true, t_grid)
        return float(self.lam * crps_val + (1.0 - self.lam) * pin_val)


# ---------------------------------------------------------------------------
# Interval Score loss
# ---------------------------------------------------------------------------

class IntervalScoreLoss:
    """
    Interval Score (Winkler score) as a tuning loss.

    Inverts the predicted CDF at alpha/2 and 1-alpha/2 to get L, U,
    then computes:
        IS = (U - L) + (2/alpha)(L - Y)_+ + (2/alpha)(Y - U)_+

    This is a proper scoring rule for prediction intervals
    (Gneiting & Raftery, 2007).

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.1 for 90% PI).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def compute(
        self,
        F_pred: ArrayLike,
        Y_true: ArrayLike,
        t_grid: ArrayLike,
        **kwargs,
    ) -> float:
        F_pred = np.asarray(F_pred, dtype=float)
        Y_true = np.asarray(Y_true, dtype=float).ravel()
        t_grid = np.asarray(t_grid, dtype=float).ravel()

        L = _invert_cdf(F_pred, t_grid, self.alpha / 2)
        U = _invert_cdf(F_pred, t_grid, 1.0 - self.alpha / 2)

        width = U - L
        pen_lo = np.maximum(L - Y_true, 0.0)
        pen_hi = np.maximum(Y_true - U, 0.0)
        is_scores = width + (2.0 / self.alpha) * (pen_lo + pen_hi)
        return float(np.mean(is_scores))
