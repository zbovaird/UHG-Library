"""UHG neural network package."""

from .layers import (
    UHGLayer,
    UHGConv,
    UHGAttentionLayer,
    ProjectiveSAGEConv,
    ProjectiveHierarchicalLayer,
    HyperbolicLinear,
)

from .models import ProjectiveGraphSAGE

__all__ = [
    "UHGLayer",
    "UHGConv",
    "UHGAttentionLayer",
    "ProjectiveSAGEConv",
    "ProjectiveHierarchicalLayer",
    "HyperbolicLinear",
    "ProjectiveGraphSAGE",
]

try:
    from .models import ProjectiveHierarchicalGNN

    __all__.append("ProjectiveHierarchicalGNN")
except ImportError:
    pass
