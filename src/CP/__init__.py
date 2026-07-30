"""
CP (Conformal Prediction) module for CKME.

This module provides conformal prediction functionality for CKME models,
including score computation, calibration, and prediction interval construction.
"""

from .cp import CP

__all__ = ["CP"]

