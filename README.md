# Universal Hyperbolic Geometry (UHG) Library

**PyPI:** [`uhg`](https://pypi.org/project/uhg/) — `pip install uhg` / `uv pip install uhg`

A PyTorch library for hyperbolic deep learning using **Universal Hyperbolic Geometry** (projective UHG). Operations are expressed in hyperbolic space without relying on tangent-space exp/log maps for the core geometry story.

**Audience:** Researchers and practitioners who want projective UHG primitives, hyperbolic graph models, and an **optional** end-to-end **unsupervised anomaly** pipeline built on UHG embeddings.

**Support:** [GitHub Issues](https://github.com/zbovaird/UHG-Library/issues) only — no other channels.

Stable imports are documented in [`docs/reference/public-api.md`](docs/reference/public-api.md). Repository layout: [`docs/MAP.md`](docs/MAP.md).

## What you get

- **Core UHG:** Projective geometry, quadrance, spread, cross-ratio (`ProjectiveUHG`, `UHGCore`, …)
- **GNN-style models:** e.g. ProjectiveGraphSAGE and related layers under `uhg.nn`
- **Application layer:** Unsupervised anomaly detection with clustering and quadrance-based scoring (`UHGUnsupervisedAnomalyDetector`)

## Install

From **PyPI** (pip or uv):

```bash
pip install uhg
```

```bash
uv pip install uhg
```

**From GitHub** (latest default branch):

```bash
pip install "git+https://github.com/zbovaird/UHG-Library.git"
```

**CPU PyTorch first (recommended when you use CPU wheels):** some environments need a compatible **PyTorch + NumPy** pair before other packages import cleanly. Install PyTorch, then `uhg`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install uhg
```

**From source** (contributors):

```bash
git clone https://github.com/zbovaird/UHG-Library.git
cd UHG-Library
pip install -e ".[dev]"
# or: uv pip install -e ".[dev]"
```

Runtime dependencies include PyTorch and PyTorch Geometric (see `pyproject.toml`). Optional extras: `uhg[mcp]` (MCP server), `uhg[colab]` (PyArrow/pynndescent/FAISS for IDS-style notebooks — see [`docs/development/colab-local.md`](docs/development/colab-local.md)).

**Troubleshooting:** If `import uhg` fails with NumPy/torch dtype or `_ARRAY_API` errors, align versions: upgrade PyTorch, or pin NumPy (e.g. `pip install "numpy<2"`) to match your PyTorch CPU wheel.

## Quickstart (stable API)

```python
from uhg import UHGUnsupervisedAnomalyDetector
import numpy as np

X = np.random.randn(1000, 10) * 0.5

detector = UHGUnsupervisedAnomalyDetector(hidden=64, embedding_dim=32)
detector.fit(X, k=5, epochs=50, seed=42)
detector.cluster(eps=0.5, min_samples=3)

scores = detector.score(method="centroid_quadrance")
summary = detector.summarize(topk=20)
```

```python
from uhg import build_knn_graph, run_dbscan, centroid_quadrance

edge_index = build_knn_graph(X, k=5)
result = run_dbscan(embeddings, eps=0.5, min_samples=3)
scores = centroid_quadrance(embeddings)
```

## Conceptual demos

New to UHG? See [`examples/interactive/README.md`](examples/interactive/README.md) for a **local HTML** demo (pedagogy, not the core library install).

## MCP server (development only)

Optional tooling for editor-assisted workflows — **not** part of the default `uhg` install:

```bash
pip install "uhg[mcp]"
python -m mcp_server.uhg_server
```

Details: [`docs/development/mcp.md`](docs/development/mcp.md) and [`mcp_server/README.md`](mcp_server/README.md).

## Module map (selected)

| Area | Notes |
|------|--------|
| `uhg.graph` | kNN graph build, edge index I/O |
| `uhg.cluster` | DBSCAN helpers, metrics |
| `uhg.anomaly` | Detector, scores, reporting |
| `uhg.utils` | Timing, schema helpers |

## Documentation

- **MkDocs / Read the Docs:** [uhg.readthedocs.io](https://uhg.readthedocs.io) (when built)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

## References

### In this repository (mathematics)

Core UHG reference PDFs at the repo root for implementation work and validating mathematics:

- [`UHG.pdf`](UHG.pdf) — primary reference notes
- [`UHG pictorial.pdf`](UHG%20pictorial.pdf) — pictorial overview

Other PDFs are ignored by `.gitignore` unless explicitly allowed.

### Published papers

- Norman J. Wildberger, "Universal Hyperbolic Geometry I: Trigonometry", Geometriae Dedicata, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry II: A pictorial overview", KoG, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry III: First Steps in Projective Triangle Geometry", KoG, 2014

## License

MIT License
