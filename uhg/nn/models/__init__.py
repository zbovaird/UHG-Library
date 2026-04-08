"""UHG neural network models."""

# ProjectiveGraphSAGE is required for anomaly detection and Part B
from .sage import ProjectiveGraphSAGE

__all__ = ["ProjectiveGraphSAGE"]

# Optional models (may have complex deps)
try:
    from .hgnn import BaseHGNN, HGCN, HGAT

    __all__.extend(["BaseHGNN", "HGCN", "HGAT"])
except ImportError:
    pass
try:
    from .hierarchical import ProjectiveHierarchicalGNN

    __all__.append("ProjectiveHierarchicalGNN")
except ImportError:
    pass
try:
    from .hgt import HGT, HyperbolicPositionalEncoding, HyperbolicTransformerLayer

    __all__.extend(
        ["HGT", "HyperbolicPositionalEncoding", "HyperbolicTransformerLayer"]
    )
except ImportError:
    pass
