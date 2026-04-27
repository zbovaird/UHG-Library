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

## Safe import contract

These imports are expected to work without importing scikit-learn, SciPy, or the
anomaly pipeline:

```python
import uhg
import uhg.projective
import uhg.layers
import uhg.nn
import uhg.manifolds
```

The top-level anomaly, graph, and cluster helpers remain stable public exports,
but they are resolved lazily. Their heavier dependencies are imported only when
the corresponding functionality is used.

## Canonical geometry API

The primary geometry class is `uhg.projective.ProjectiveUHG`, also re-exported as
`uhg.ProjectiveUHG`. It implements projective UHG operations directly in
homogeneous coordinates: `normalize_points`, `project`, `distance`,
`inner_product`, `quadrance`, `cross_ratio`, `join`, `meet`, `transform`,
`aggregate`, and `scale`.

`uhg.manifolds.HyperbolicManifold` is a compatibility wrapper around
`ProjectiveUHG`. It delegates to the same projective operations and does not
introduce tangent-space maps. There are no canonical Lorentz or Poincare
manifold classes in this package; use `ProjectiveUHG` for native UHG work and
`HyperbolicManifold` only when a manifold-shaped API is needed.

Distance APIs:

```python
import torch
from uhg.projective import ProjectiveUHG

uhg = ProjectiveUHG()
x = torch.tensor([1.0, 0.0, 2.0])
y = torch.tensor([0.0, 1.0, 2.0])
d = uhg.distance(x, y)
```

Core geometry methods accept `torch.Tensor` inputs. NumPy arrays should be
converted explicitly with `torch.as_tensor(...)` before calling `ProjectiveUHG`
methods. Graph, clustering, and anomaly utilities may accept NumPy arrays where
documented by those modules.

Expected point shape is `(..., D+1)`: the final coordinate is the homogeneous /
time-like coordinate and the preceding `D` coordinates are spatial. Pairwise
operations such as `distance(x, y)` require the same final coordinate dimension
and broadcast-compatible batch dimensions.

`ProjectiveUHG` has no curvature parameter. `HyperbolicManifold(curvature=-1.0)`
keeps a negative curvature value for API compatibility and rejects non-negative
curvature, but the operations still delegate to the projective UHG model.

## Canonical layer API

Preferred neural-network layers live under `uhg.nn.layers`:

```python
from uhg.nn.layers import (
    UHGLayer,
    UHGConv,
    UHGAttentionLayer,
    ProjectiveSAGEConv,
    ProjectiveHierarchicalLayer,
    HyperbolicLinear,
)
```

`uhg.nn.models.ProjectiveGraphSAGE` is the canonical GraphSAGE model. The
top-level `uhg.layers` module is retained for compatibility with older layer
names such as `UHGLinear`, `UHGConv`, `UHGAttention`, `UHGTransformer`,
`UHGMultiheadAttention`, and `UHGLayerNorm`.

## Everything else

Submodules (e.g. `uhg.nn`, `uhg.anomaly`, `uhg.threat_indicators`) remain importable but are **not** all covered by the same stability guarantee unless listed above. Prefer importing documented symbols from `uhg` for application code.

See [stability.md](stability.md) for deprecation policy.
