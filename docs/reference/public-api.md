# Public API (semver)

These names are re-exported from the top-level `uhg` package and treated as **stable** for semantic versioning: breaking changes require a **major** version bump; additions are **minor**; fixes are **patch**.

Install: `pip install uhg` or `uv pip install uhg` (PyPI distribution name: **`uhg`**).

## Stable symbols

| Symbol | Description |
|--------|-------------|
| `__version__` | Package version string |
| `ProjectiveUHG` | Core projective UHG operations |
| `UHGTensor` | Tensor wrapper |
| `UHGParameter` | Parameter wrapper |
| `UHGCore` | Core UHG object |
| `UHGMultiHeadAttention` | Attention module |
| `UHGAttentionConfig` | Attention configuration |
| `UHGUnsupervisedAnomalyDetector` | End-to-end unsupervised anomaly pipeline |
| `build_knn_graph` | kNN graph construction |
| `run_dbscan` | DBSCAN clustering helper |
| `centroid_quadrance` | Quadrance-based scoring |

Import example:

```python
from uhg import (
    UHGUnsupervisedAnomalyDetector,
    build_knn_graph,
    run_dbscan,
    centroid_quadrance,
    ProjectiveUHG,
)
```

## Everything else

Submodules (e.g. `uhg.nn`, `uhg.anomaly`, `uhg.threat_indicators`) remain importable but are **not** all covered by the same stability guarantee unless listed above. Prefer importing documented symbols from `uhg` for application code.

See [stability.md](stability.md) for deprecation policy.
