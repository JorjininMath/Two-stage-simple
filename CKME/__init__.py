"""
CKME (Conditional Kernel Mean Embedding) module.

This module provides the core CKME model and related utilities for conditional
CDF estimation.

Module Structure
----------------
CKME/
  ├── ckme.py              # Main CKMEModel class
  ├── coefficients.py      # Coefficient computation utilities
  ├── cdf.py               # CDF computation utilities
  ├── tuning.py            # Hyperparameter tuning with k-fold CV
  ├── parameters.py        # Parameter definitions (Params, ParamGrid)
  ├── kernels.py           # Kernel functions (RBF)
  ├── indicators.py        # Smooth indicator functions
  └── loss_functions/      # Loss function implementations
      ├── __init__.py      # Factory and registry
      └── crps.py          # CRPS loss implementation

Main Components
---------------
- CKMEModel: Main model class for training and prediction
- Params, ParamGrid: Hyperparameter containers
- tune_ckme_params: Parameter tuning via cross-validation
- make_loss, CRPSLoss: Loss function factory and implementations

Usage
-----
# Train model with fixed parameters
from CKME import CKMEModel, Params
model = CKMEModel(indicator_type="logistic")
model.fit(X_train, Y_train, params=Params(ell_x=1.0, lam=0.01, h=0.1))

# Train with parameter tuning
from CKME import ParamGrid
param_grid = ParamGrid(ell_x_list=[0.5, 1.0], lam_list=[0.001, 0.01], h_list=[0.05, 0.1])
model.fit(X_train, Y_train, param_grid=param_grid, t_grid=t_grid, cv_folds=5)

# Predict CDF
F = model.predict_cdf(X_test, t_grid)

# Use loss functions
from CKME import make_loss
loss = make_loss("crps")
crps_value = loss.compute(F, Y_true, t_grid)
"""

from .ckme import CKMEModel
from .parameters import Params, ParamGrid
from .tuning import tune_ckme_params, cross_validate_ckme
from .loss_functions import make_loss, CRPSLoss

__all__ = [
    "CKMEModel",
    "Params",
    "ParamGrid",
    "tune_ckme_params",
    "cross_validate_ckme",
    "make_loss",
    "CRPSLoss",
]

