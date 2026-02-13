# Universal Hyperbolic Geometry (UHG) Library

A PyTorch library for hyperbolic deep learning using Universal Hyperbolic Geometry principles. All operations are performed directly in hyperbolic space without tangent space mappings.

## Overview

Universal Hyperbolic Geometry provides a powerful framework for representing complex hierarchical data. This library includes:

- **Core UHG operations**: Projective geometry, quadrance, spread, cross-ratio
- **UHG-based graph neural networks**: ProjectiveGraphSAGE, HGCN, HGAT
- **Unsupervised anomaly detection**: End-to-end pipeline with clustering and scoring

## Key Features

- Pure projective UHG operations (no tangent space or exp/log maps)
- `UHGUnsupervisedAnomalyDetector`: fit, cluster, score, summarize, export in one API
- Graph building with kNN and caching (`uhg.graph.build`)
- DBSCAN clustering with grid search and quality metrics (`uhg.cluster`)
- UHG quadrance-based anomaly scoring (`uhg.anomaly.scores`)
- Programmatic reporting (JSON, entity aggregation; no visualization)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/zbovaird/UHG-Library.git
cd UHG-Library
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For editable (development) install with full anomaly pipeline:
```bash
pip install -e ".[torch]"
```

## MCP Server (Development)

The UHG MCP server provides tools that offload computation when developing. Enable it when you need to run tests, compute UHG quadrance/cross-ratio, run benchmarks, or run the anomaly pipeline smoke test.

From the repository root:
```bash
python -m mcp_server.uhg_server
```

For Cursor integration, see [mcp_server/README.md](mcp_server/README.md).

## Usage

### Unsupervised Anomaly Detection (0.3.7+)

```python
from uhg import UHGUnsupervisedAnomalyDetector
import numpy as np

X = np.random.randn(1000, 10) * 0.5  # Your feature matrix

detector = UHGUnsupervisedAnomalyDetector(hidden=64, embedding_dim=32)
detector.fit(X, k=5, epochs=50, seed=42)
detector.cluster(eps=0.5, min_samples=3)

scores = detector.score(method="centroid_quadrance")
summary = detector.summarize(topk=20)
print(summary["timings"], summary["top_entities"])

scores_new, labels = detector.predict(percentile=0.95)
detector.export("model.pt")
```

### From DataFrame

```python
import pandas as pd
detector = UHGUnsupervisedAnomalyDetector()
detector.fit_from_dataframe(df, epochs=30, seed=42)
```

### Graph, Clustering, and Scoring

```python
from uhg import build_knn_graph, run_dbscan, centroid_quadrance

edge_index = build_knn_graph(X, k=5)
result = run_dbscan(embeddings, eps=0.5, min_samples=3)
scores = centroid_quadrance(embeddings)
```

## Module Structure

| Module | Description |
|--------|-------------|
| `uhg.graph.build` | `build_knn_graph`, `save_edge_index`, `load_edge_index`, `build_maxk_then_slice` |
| `uhg.cluster` | `run_dbscan`, `eps_grid_search`, `auto_eps_kdist`, `davies_bouldin`, `silhouette`, `calinski_harabasz` |
| `uhg.anomaly` | `UHGUnsupervisedAnomalyDetector`, `centroid_quadrance`, `neighbor_quadrance`, `composite_score` |
| `uhg.utils.timing` | `time_block` context manager |
| `uhg.utils.schema` | `detect_label_column`, `enforce_numeric`, `build_entity_index` |

## Requirements

- PyTorch, torch-geometric
- scikit-learn, scipy, numpy, pandas
- Optional: `mcp[cli]` for MCP server tools

## References

- Norman J. Wildberger, "Universal Hyperbolic Geometry I: Trigonometry", Geometriae Dedicata, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry II: A pictorial overview", KoG, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry III: First Steps in Projective Triangle Geometry", KoG, 2014

## License

MIT License
