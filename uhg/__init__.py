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

# Stable public API (semver): documented in docs/reference/public-api.md.
# Heavy anomaly/graph/cluster exports are resolved lazily so importing core UHG
# geometry does not import scikit-learn/SciPy or optional neighbor-search stacks.
_LAZY_EXPORTS = {
    "UHGUnsupervisedAnomalyDetector": (
        "uhg.anomaly.unsupervised",
        "UHGUnsupervisedAnomalyDetector",
    ),
    "build_knn_graph": ("uhg.graph.build", "build_knn_graph"),
    "run_dbscan": ("uhg.cluster.dbscan", "run_dbscan"),
    "centroid_quadrance": ("uhg.anomaly.scores", "centroid_quadrance"),
}

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


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        attr = getattr(import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
