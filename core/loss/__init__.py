#!/usr/bin/env python3
"""
Loss Functions Module
"""

from .mmd_loss import MMDLoss
from .crps_loss import CRPSLoss

__all__ = ['MMDLoss', 'CRPSLoss']

