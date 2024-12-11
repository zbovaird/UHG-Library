"""UHG neural network models."""

from .sage import ProjectiveGraphSAGE
from .hierarchical import ProjectiveHierarchicalGNN

__all__ = [
    'ProjectiveGraphSAGE',
    'ProjectiveHierarchicalGNN'
]
