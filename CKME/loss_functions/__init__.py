"""
loss_functions package

This package contains implementations of loss functions for CKME model evaluation.

Structure
---------
loss_functions/
  ├── __init__.py    # Factory and registry functions
  └── crps.py        # CRPS loss implementation

It provides:
- Concrete loss implementations (e.g., CRPSLoss)
- Factory function (make_loss) for creating loss objects
- Registry mechanism (register_loss, get_loss) for custom loss functions

Usage
-----
# Create loss using factory
from CKME.loss_functions import make_loss
loss = make_loss("crps")

# Use loss for evaluation
crps_value = loss.compute(F_pred, Y_true, t_grid)

# Register custom loss (optional)
from CKME.loss_functions import register_loss
register_loss("custom", CustomLossClass)
"""

from __future__ import annotations

from typing import Dict, Literal, Type, Union

from .crps import CRPSLoss

ArrayLike = "np.ndarray"  # Type hint string to avoid import

LossType = Literal["crps"]

__all__ = ["CRPSLoss", "make_loss", "register_loss", "get_loss", "LossType"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_loss(loss_type: LossType, **kwargs) -> CRPSLoss:
    """
    Factory function to create loss objects.

    Parameters
    ----------
    loss_type : {"crps"}
        Type of loss function.

    **kwargs
        Additional arguments (currently unused, reserved for future loss functions).

    Returns
    -------
    loss : CRPSLoss
        Loss function object.

    Examples
    --------
    >>> # CRPS loss
    >>> loss = make_loss("crps")
    >>> crps_value = loss.compute(F_pred, Y_true, t_grid)
    """
    if loss_type == "crps":
        return CRPSLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Supported types: 'crps'"
        )


# ---------------------------------------------------------------------------
# Registry pattern (for dynamic registration of custom losses)
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: Dict[str, Type[CRPSLoss]] = {
    "crps": CRPSLoss,
}


def register_loss(name: str, loss_class: Type) -> None:
    """
    Register a new loss function type.

    This allows users to add custom loss functions that can be used
    with the factory function and tuning modules.

    Parameters
    ----------
    name : str
        Name identifier for the loss function.

    loss_class : type
        Loss class that implements the same interface as CRPSLoss
        (i.e., has compute() and compute_batch() methods).

    Examples
    --------
    >>> class CustomLoss:
    ...     def compute(self, F_pred, Y_true, t_grid):
    ...         return custom_computation()
    ...     def compute_batch(self, F_pred_list, Y_true_list, t_grid):
    ...         return self.compute(...)
    ...
    >>> register_loss("custom", CustomLoss)
    >>> loss = make_loss("custom")
    """
    _LOSS_REGISTRY[name] = loss_class


def get_loss(name: str, **kwargs) -> Union[CRPSLoss]:
    """
    Get loss function by name (alternative to make_loss).

    This function uses the registry to look up loss classes and instantiate
    them. It is an alternative to make_loss() that supports dynamically
    registered loss functions.

    Parameters
    ----------
    name : str
        Name of the loss function.

    **kwargs
        Arguments passed to the loss class constructor.

    Returns
    -------
    loss : Loss object
        Loss function object.

    Raises
    ------
    ValueError
        If the loss name is not registered.
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name}. "
            f"Registered losses: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[name](**kwargs)
