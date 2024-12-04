"""
Universal Hyperbolic Geometry (UHG) Manifolds.

This module provides projective geometry implementations for UHG.
"""

from .base import Manifold, ScalingInfo
from ..projective import ProjectiveUHG

__all__ = ["Manifold", "ScalingInfo", "ProjectiveUHG"] 