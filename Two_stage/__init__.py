"""
Two_stage

Clean two-stage CKME pipeline.
Step 1: train and store. S^0 via tail_uncertainty.
Step 2: site selection, data collection, CP calibration.
"""

from .stage1_train import Stage1TrainResult, run_stage1_train
from .io import save_stage1_train_result, load_stage1_train_result, save_s0_scores
from .s0_score import compute_s0_tail_uncertainty, compute_s0
from .stage2 import (
    Stage2Result,
    run_stage2,
    save_stage2_result,
    load_stage2_result,
)

__all__ = [
    "Stage1TrainResult",
    "run_stage1_train",
    "save_stage1_train_result",
    "load_stage1_train_result",
    "save_s0_scores",
    "compute_s0_tail_uncertainty",
    "compute_s0",
    "Stage2Result",
    "run_stage2",
    "save_stage2_result",
    "load_stage2_result",
]
