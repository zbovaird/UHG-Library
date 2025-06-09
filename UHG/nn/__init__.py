"""UHG neural network package."""

from .layers import (
    UHGLayer,
    UHGConv,
    UHGAttentionLayer,
    ProjectiveSAGEConv,
    ProjectiveHierarchicalLayer
)

from .models import (
    ProjectiveGraphSAGE,
    ProjectiveHierarchicalGNN
)

__all__ = [
    # Layers
    'UHGLayer',
    'UHGConv',
    'UHGAttentionLayer',
    'ProjectiveSAGEConv',
    'ProjectiveHierarchicalLayer',
    
    # Models
    'ProjectiveGraphSAGE',
    'ProjectiveHierarchicalGNN'
] 