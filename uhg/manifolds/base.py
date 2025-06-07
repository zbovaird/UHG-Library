"""
Universal Hyperbolic Geometry (UHG) Manifolds.

This module provides projective geometry implementations for UHG.
"""

import torch

class Manifold:
    """Base class for all manifolds in UHG."""
    def logmap0(self, x):
        raise NotImplementedError
    def expmap0(self, x):
        raise NotImplementedError
    def inner_product(self, x, y):
        raise NotImplementedError
    def normalize_points(self, x):
        raise NotImplementedError

class ScalingInfo:
    """Information about scaling in UHG."""
    def __init__(self, scale: float = 1.0):
        self.scale = scale 