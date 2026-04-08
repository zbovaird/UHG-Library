"""
UHG Optimization Module

This module provides optimization algorithms specifically designed for Universal Hyperbolic Geometry,
ensuring all operations preserve hyperbolic structure and invariants.
"""

from .base import UHGBaseOptimizer
from .uhg_adam import UHGAdam
from .uhg_sgd import UHGSGD

__all__ = [
    "UHGBaseOptimizer",
    "UHGAdam",
    "UHGSGD",
] 