"""
Universal Hyperbolic Geometry (UHG) Library.

This library provides a pure implementation of UHG principles using projective geometry,
without relying on differential geometry or manifold concepts.
"""

from typing import List

from .projective import ProjectiveUHG
from .tensor import UHGTensor, UHGParameter
from .core import UHGCore

__version__: str = "0.1.18"
__all__: List[str] = ["ProjectiveUHG", "UHGTensor", "UHGParameter", "UHGCore"] 