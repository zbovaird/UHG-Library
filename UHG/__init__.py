"""
Universal Hyperbolic Geometry (UHG) Library.

This library provides a pure implementation of UHG principles using projective geometry,
without relying on differential geometry or manifold concepts.
"""

from typing import List

from .projective import ProjectiveUHG
from .tensor import UHGTensor, UHGParameter
from .core import UHGCore
from .attention import UHGMultiHeadAttention, UHGAttentionConfig
from .threat_indicators import (
    ThreatIndicator,
    ThreatIndicatorType,
    ThreatCorrelation
)

__version__: str = "0.2.3"
__all__: List[str] = [
    # Core components
    "ProjectiveUHG",
    "UHGTensor",
    "UHGParameter",
    "UHGCore",
    
    # Attention mechanism
    "UHGMultiHeadAttention",
    "UHGAttentionConfig",
    
    # Threat detection
    "ThreatIndicator",
    "ThreatIndicatorType",
    "ThreatCorrelation"
] 