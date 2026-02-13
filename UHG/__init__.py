"""
Universal Hyperbolic Geometry Library
A PyTorch library for hyperbolic deep learning using Universal Hyperbolic Geometry principles.
All operations are performed directly in hyperbolic space without tangent space mappings.
"""

__version__: str = "0.3.7"
from typing import List

from .projective import ProjectiveUHG
from .tensor import UHGTensor, UHGParameter
from .core import UHGCore

# Optional: attention and threat_indicators (may be in uhg/ on some clones)
try:
    from .attention import UHGMultiHeadAttention, UHGAttentionConfig
except ImportError:
    UHGMultiHeadAttention = None  # type: ignore
    UHGAttentionConfig = None  # type: ignore

try:
    from .threat_indicators import ThreatIndicator, ThreatIndicatorType, ThreatCorrelation
except ImportError:
    ThreatIndicator = None  # type: ignore
    ThreatIndicatorType = None  # type: ignore
    ThreatCorrelation = None  # type: ignore

# Unsupervised anomaly detection pipeline (0.3.7+)
try:
    from .anomaly.unsupervised import UHGUnsupervisedAnomalyDetector
    from .graph.build import (
        build_knn_graph,
        build_maxk_then_slice,
        load_edge_index,
        save_edge_index,
    )
    from .cluster.dbscan import run_dbscan
    from .cluster.metrics import davies_bouldin, silhouette, calinski_harabasz
    from .anomaly.scores import centroid_quadrance, neighbor_quadrance, composite_score
    _HAS_ANOMALY = True
except ImportError as _e:
    _HAS_ANOMALY = False
    _ANOMALY_IMPORT_ERROR = _e  # for diagnostics: print(uhg._ANOMALY_IMPORT_ERROR)

__all__: List[str] = [
    "ProjectiveUHG",
    "UHGTensor",
    "UHGParameter",
    "UHGCore",
]
if UHGMultiHeadAttention is not None:
    __all__.extend(["UHGMultiHeadAttention", "UHGAttentionConfig"])
if ThreatIndicator is not None:
    __all__.extend(["ThreatIndicator", "ThreatIndicatorType", "ThreatCorrelation"])

if _HAS_ANOMALY:
    __all__.extend([
        "UHGUnsupervisedAnomalyDetector",
        "build_knn_graph",
        "build_maxk_then_slice",
        "load_edge_index",
        "save_edge_index",
        "run_dbscan",
        "davies_bouldin",
        "silhouette",
        "calinski_harabasz",
        "centroid_quadrance",
        "neighbor_quadrance",
        "composite_score",
    ]) 