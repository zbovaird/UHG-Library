"""
Universal Hyperbolic Geometry (UHG) Library.

This library provides a pure implementation of UHG principles using projective geometry,
without relying on differential geometry or manifold concepts.
"""

from .projective import ProjectiveUHG
from .tensor import UHGTensor, UHGParameter
from .core import UHGCore

__version__ = "0.1.17"
__all__ = ["ProjectiveUHG", "UHGTensor", "UHGParameter", "UHGCore"] 