"""
Universal Hyperbolic Geometry Library
A PyTorch library for hyperbolic deep learning using Universal Hyperbolic Geometry principles.
All operations are performed directly in hyperbolic space without tangent space mappings.
"""

# Version must be defined before imports that reference `from uhg import __version__`.
__version__: str = "0.3.8"

from typing import List

from .attention import UHGAttentionConfig, UHGMultiHeadAttention
from .core import UHGCore
from .projective import ProjectiveUHG
from .tensor import UHGParameter, UHGTensor

# Stable public API (semver): documented in docs/reference/public-api.md
from .anomaly.scores import centroid_quadrance
from .anomaly.unsupervised import UHGUnsupervisedAnomalyDetector
from .cluster.dbscan import run_dbscan
from .graph.build import build_knn_graph

__all__: List[str] = [
    "ProjectiveUHG",
    "UHGTensor",
    "UHGParameter",
    "UHGCore",
    "UHGMultiHeadAttention",
    "UHGAttentionConfig",
    "UHGUnsupervisedAnomalyDetector",
    "build_knn_graph",
    "run_dbscan",
    "centroid_quadrance",
    "__version__",
]
