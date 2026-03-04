"""
parameters.py

Core hyperparameter/configuration containers for the CKME model.

This module defines lightweight data classes that describe:
- The CKME hyperparameters for a single model fit (Params).
- Optional hyperparameter search grids for cross-validation (ParamGrid).

These objects are intentionally kept simple and free of any numeric code
so they can be imported from both core model code and experiment scripts
without creating circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Params: hyperparameters for a single CKME model
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """
    Hyperparameters for the CKME conditional CDF model.

    The model uses smooth indicator functions g_t(y) with bandwidth h to estimate
    the conditional CDF F(t | x).

    Attributes
    ----------
    ell_x : float
        Bandwidth parameter for the X-kernel k_X(x, x').
        Typically used in an RBF / Gaussian kernel of the form
            k_X(x, x') = exp(-||x - x'||^2 / (2 * ell_x^2)).

    lam : float
        Ridge regularization parameter λ used in the linear system
            (K_X + n * λ I) c(x) = k_X(X, x).
        This stabilizes the inverse and controls the complexity of CKME.

    h : float
        Smoothing bandwidth for the indicator function g_t(y) that approximates
        the indicator 1{y <= t}. This is a Y-direction smoothing scale.
    """

    ell_x: float
    lam: float
    h: float

    def as_dict(self) -> Dict[str, float]:
        """
        Return a plain dictionary view of the numerical hyperparameters.

        This is useful for logging, saving to JSON, or pretty-printing.
        """
        return {
            "ell_x": float(self.ell_x),
            "lam": float(self.lam),
            "h": float(self.h),
        }

    def copy_with(
        self,
        ell_x: Optional[float] = None,
        lam: Optional[float] = None,
        h: Optional[float] = None,
    ) -> "Params":
        """
        Create a shallow copy with some fields updated.

        Example
        -------
        >>> params = Params(ell_x=0.5, lam=1e-3, h=0.2)
        >>> new_params = params.copy_with(lam=1e-4)
        """
        return Params(
            ell_x=self.ell_x if ell_x is None else ell_x,
            lam=self.lam if lam is None else lam,
            h=self.h if h is None else h,
        )


# ---------------------------------------------------------------------------
# ParamGrid: hyperparameter grids for Stage-1 cross-validation (optional)
# ---------------------------------------------------------------------------

@dataclass
class ParamGrid:
    """
    Hyperparameter search grid for CKME Stage-1 cross-validation.

    This is a simple container that collects candidate values for each
    hyperparameter. It does not perform any search by itself; instead,
    Stage-1 training code can iterate over 'iter_grid()' to obtain
    Params objects.

    Attributes
    ----------
    ell_x_list : list of float
        Candidate values for ell_x.

    lam_list : list of float
        Candidate values for lam.

    h_list : list of float
        Candidate values for indicator bandwidth h.
    """

    ell_x_list: List[float] = field(default_factory=list)
    lam_list: List[float] = field(default_factory=list)
    h_list: List[float] = field(default_factory=list)

    def is_empty(self) -> bool:
        """
        Check whether any dimension of the grid is empty.

        If any of the required lists is empty, iterating over the grid will yield no
        candidates. Stage-1 code may want to assert that the grid is non-empty.
        """
        return not (self.ell_x_list and self.lam_list and self.h_list)

    def iter_grid(self):
        """
        Iterate over all combinations of hyperparameters in the grid.

        Yields
        ------
        Params
            A Params instance for each combination in the Cartesian product.
        """
        if self.is_empty():
            return

        for ell_x in self.ell_x_list:
            for lam in self.lam_list:
                for h in self.h_list:
                    yield Params(
                        ell_x=ell_x,
                        lam=lam,
                        h=h,
                    )

