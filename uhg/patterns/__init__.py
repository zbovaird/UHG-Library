"""UHG Pattern Analysis Package.

This package provides tools for analyzing patterns in hyperbolic space,
with a focus on authorization hierarchy violation detection.
"""

from .correlation import PatternCorrelator, CorrelationPattern

__all__ = ["PatternCorrelator", "CorrelationPattern"] 