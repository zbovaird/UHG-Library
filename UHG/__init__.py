"""
Universal Hyperbolic Geometry Library
A PyTorch library for hyperbolic deep learning using Universal Hyperbolic Geometry principles.
All operations are performed directly in hyperbolic space without tangent space mappings.
"""

__version__: str = "0.3.6"
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