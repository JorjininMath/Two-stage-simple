# Two_stage

Clean two-stage CKME pipeline. **Step 1 only** (train and store).

## Step 1: Train CKME model

**Option A: Fixed params (fast, no CV)**
```python
from Two_stage import run_stage1_train, save_stage1_train_result, load_stage1_train_result
from CKME.parameters import Params

params = Params(ell_x=0.5, lam=0.01, h=0.1)
result = run_stage1_train(
    n_0=20, r_0=5, simulator_func="exp1", params=params,
    random_state=42, verbose=True,
)
```

**Option B: Param grid (CV tuning)**
```python
from CKME.parameters import ParamGrid

param_grid = ParamGrid(ell_x_list=[0.5, 1.0], lam_list=[0.01, 0.1], h_list=[0.05, 0.1])
result = run_stage1_train(
    n_0=20, r_0=5, simulator_func="exp1", param_grid=param_grid,
    cv_folds=5, random_state=42, verbose=True,
)

# Save for later (skip retraining in experiments)
save_stage1_train_result(result, "output/stage1_model")

# Load
loaded = load_stage1_train_result("output/stage1_model")
F = loaded.model.predict_cdf(X_query, loaded.t_grid)
```

## Step 2: Compute S^0 (tail_uncertainty)

```python
from Two_stage import load_stage1_train_result, compute_s0
import numpy as np

res = load_stage1_train_result("output/stage1_model")
X_cand = np.linspace(0.1, 0.9, 100).reshape(-1, 1)
s0 = compute_s0(res, X_cand, alpha=0.1)
# s0[i] = q_{0.95}(x_i) - q_{0.05}(x_i), higher = more need for data
```

## Structure

- `stage1_train.py` – Step 1: train model, no CP
- `s0_score.py` – compute S^0 via tail_uncertainty (no CP)
- `io.py` – save/load Stage1TrainResult
- `design.py` – LHS / grid design
- `data_collection.py` – collect D_0
- `sim_functions/` – exp1 simulator (minimal)

## Experiments

Currently supports `simulator_func="exp1"`. Add more in `sim_functions/` as needed.
