"""Config load + X_cand for exp2_test (same logic as exp3_test)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from CKME.parameters import Params
from Two_stage.design import generate_space_filling_design
from Two_stage.sim_functions import get_experiment_config


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    raw: Dict[str, str] = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                raw[k.strip()] = v.strip()

    int_keys = ("n_0", "r_0", "n_1", "r_1", "n_test", "r_test", "n_cand", "n_macro", "n_jobs", "t_grid_size")
    float_keys = ("ell_x", "lam", "h", "alpha")
    str_keys = ("simulator_func", "method")

    cfg: Dict[str, Any] = {}
    for k in int_keys:
        if k in raw:
            cfg[k] = int(raw[k])
    for k in float_keys:
        if k in raw:
            cfg[k] = float(raw[k])
    for k in str_keys:
        if k in raw:
            cfg[k] = raw[k]

    cfg["params"] = Params(
        ell_x=cfg.get("ell_x", 0.1),
        lam=cfg.get("lam", 1e-4),
        h=cfg.get("h", 0.05),
    )
    return cfg


def get_config(config: Dict[str, Any], quick: bool = False) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if quick:
        scale = 0.25
        for k in ("n_0", "n_1", "n_test", "n_cand"):
            if k in cfg:
                cfg[k] = max(10, int(cfg[k] * scale))
        cfg["r_0"] = 1
        cfg["r_1"] = 2
        cfg["r_test"] = 2
    return cfg


def get_x_cand(example: str, n_cand: int, random_state: Optional[int]) -> np.ndarray:
    exp_config = get_experiment_config(example)
    bounds = exp_config["bounds"]
    d = exp_config["d"]
    return generate_space_filling_design(
        n=n_cand, d=d, method="lhs", bounds=bounds, random_state=random_state
    )

