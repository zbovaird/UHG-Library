### uhg 3.7 Next Steps

Scope: Strengthen the library’s unsupervised (fraud/anomaly) workflow and core graph/cluster pipeline while keeping UHG math intact. Exclude visualization; users can add their own plots externally.

---

## A. Unsupervised core pipeline

Deliverable: A first-class class that trains, clusters, scores, summarizes, and exports results programmatically (no plots).

Class: `uhg.anomaly.unsupervised.UHGUnsupervisedAnomalyDetector`
- Methods
  - `fit(X: Tensor|ndarray, k: int = 2, data_fraction: float = 1.0, *, metric: str = "euclidean", edge_cache: Optional[str] = None, seed: Optional[int] = None) -> Self`
    - Standardize/validate X (CPU/GPU), sample by `data_fraction`.
    - Build kNN graph via `uhg.graph.build.build_knn_graph` (or load cache).
    - Lift to homogeneous; train GraphSAGE-lite full-graph (AMP on).
    - Store: `embeddings`, `edge_index`, `timings` (data_load_s, knn_build_s, train_s, total_s), and `config`.
  - `cluster(method: str = "dbscan", **kwargs) -> dict`
    - For `dbscan`: calls `uhg.cluster.dbscan.run_dbscan(emb, eps, min_samples)`. Returns `{labels, params, metrics: {davies_bouldin, silhouette, calinski_harabasz}}`.
  - `score(method: str = "centroid_quadrance", **kwargs) -> Tensor`
    - `centroid_quadrance(emb)`, `neighbor_quadrance(emb, k)`, `composite_score(scores, weights)`.
  - `summarize(entity_ids: Optional[ndarray] = None, topk: int = 20) -> dict`
    - Aggregates per-entity stats (mean, p95, count) over chosen score; includes timings and cluster metrics.
  - `export(path: str) -> None` / `@classmethod from_export(path: str) -> Self`

Acceptance:
- End-to-end on 5–10k rows in < 5s on CPU; < 2s on T4 (excluding kNN).
- JSON summary contains timings, cluster metrics, and top entities by score.

---

## B. Graph building & caching

Module: `uhg.graph.build`
- Functions
  - `build_knn_graph(X: Tensor|ndarray, k: int, metric: str = "euclidean", cache_key: Optional[str] = None) -> Tensor`
    - Euclidean baseline (sklearn/torch-cluster/FAISS backend selectable); CPU by default, FAISS if available.
  - `save_edge_index(path: str, edge_index: Tensor) -> None`
  - `load_edge_index(path: str) -> Tensor`
  - `build_maxk_then_slice(X, max_k: int, k: int, metric: str = "euclidean") -> Tensor`
    - Build once at `max_k`, slice to `k` for reuse.
  - `build_uhg_knn_graph(X_homog: Tensor, k: int, block_size: int = 16384) -> Tensor`
    - Optional UHG-fidelity path using Minkowski quadrance (batched); slower, for research.

Acceptance:
- Cache correctness test (same graph reloaded equals original).
- Max-k slice returns identical first-k neighbors against fresh small-k build.

---

## C. Clustering & metrics (no plots)

Module: `uhg.cluster.dbscan`
- `run_dbscan(emb: ndarray|Tensor, eps: float, min_samples: int) -> dict`
  - Returns `{labels, core_mask}`.
- `eps_grid_search(emb, eps_list: list[float], min_samples_list: list[int], score: str = "db") -> dict`
  - Picks best `eps/min_samples` by chosen metric.
- `auto_eps_kdist(emb, k: int = 4) -> float`
  - Heuristic elbow point from k-distance curve.

Module: `uhg.cluster.metrics`
- `davies_bouldin(emb, labels) -> float`
- `silhouette(emb, labels) -> float`
- `calinski_harabasz(emb, labels) -> float`

Acceptance:
- Metrics match sklearn values within tolerance for standard datasets.

---

## D. Anomaly scoring & ranking

Module: `uhg.anomaly.scores`
- `centroid_quadrance(emb: Tensor, *, eps: float = 1e-9) -> Tensor`
- `neighbor_quadrance(emb: Tensor, k: int = 5, *, eps: float = 1e-9) -> Tensor`
- `boundary_score(labels: ndarray, core_mask: ndarray, emb: Tensor) -> Tensor`
- `composite_score(scores: dict[str, Tensor], weights: dict[str, float]) -> Tensor`

Module: `uhg.anomaly.report`
- `rank_topk(scores: Tensor, k: int, ids: Optional[ndarray] = None) -> list`
- `aggregate_by_entity(scores: Tensor, entity_ids: ndarray, stats=("mean","p95","count")) -> dict`

Acceptance:
- On synthetic data with injected outliers, outliers rank higher (AUC > 0.9).

---

## E. Training/runtime QoL

- Early stopping: `uhg.nn.early_stopping.EarlyStopping(min_delta, patience)`; integrates with training loop, saves best-state.
- Configurable model: expose `hidden`, `layers`, `dropout`, `seed` in detector.
- Full‑graph vs. NeighborLoader toggle for large graphs: `training_mode={"full","neighbor"}`.
- Timers utility: `uhg.utils.timing.time_block(name)` context manager to capture stage durations.

Acceptance:
- Deterministic results under fixed seed (within floating tolerance).
- NeighborLoader path produces similar embeddings on small graphs.

---

## F. Data/schema utilities

Module: `uhg.utils.schema`
- `detect_label_column(df) -> Optional[str]`
- `detect_entity_column(df) -> Optional[str]`
- `enforce_numeric(df, fill: str = "mean", replace_inf: bool = True) -> DataFrame`
- `build_entity_index(series) -> tuple[np.ndarray, dict, dict]`

Acceptance:
- Robust to common column variants; covered by unit tests on toy frames.

---

## G. Reporting (programmatic, no images)

- `summary_to_json(summary: dict, path: str) -> None`
- `display_summary(summary: dict) -> None`
  - Prints N, E, k, timings (data_load_s, knn_build_s, train_s, total_s), DB score, cluster counts, and top entities by score.

Acceptance:
- End-to-end `fit→cluster→score→summarize→export→from_export` roundtrip equality on summary fields.

---

## H. Testing plan

- Unit tests
  - Graph: cache load/save; max-k slice; UHG‑KNN vs Euclidean sanity.
  - Clustering metrics: DB/silhouette/CH vs sklearn on `make_blobs`, `make_moons`.
  - Scores: centroid/neighbor/boundary/composite monotonicity on controlled data.
  - Schema utils: column detection across capitalizations and aliases.
- Integration tests
  - CPU-only and CUDA paths (if available) produce consistent summaries.
  - Small end-to-end smoke (N≈5k): JSON has timings, DB score, and top‑N entities.
- Performance tests
  - Timing budget targets: kNN build, training, clustering measured and asserted within envelopes.

---

## I. Performance targets (T4 reference)

- kNN build (Euclidean, k=2): N=100k in ≤ 10s CPU; with FAISS GPU: ≤ 3s
- Training (GraphSAGE‑lite, full graph): ≤ 0.08s/epoch at N=60k, E=120k (AMP on)
- Clustering (DBSCAN on 2D reduction): ≤ 5s at N=60k

---

## J. Backward compatibility & API exposure

- `uhg/__init__.py` exports
  - `UHGUnsupervisedAnomalyDetector`
  - `build_knn_graph`, `build_uhg_knn_graph`, `save_edge_index`, `load_edge_index`
  - `run_dbscan`, `davies_bouldin`, `silhouette`, `calinski_harabasz`
  - `centroid_quadrance`, `neighbor_quadrance`, `composite_score`
- Keep existing projective/UHG ops unchanged; these are layered utilities.

---

## K. Implementation roadmap

Sprint 1 (Graph & clustering foundations) — 1 week
- `uhg.graph.build` (euclidean + cache, max-k slice)
- `uhg.cluster.dbscan` (run/grid/auto_eps) and `uhg.cluster.metrics`
- Tests for build/metrics/grid search

Sprint 2 (Anomaly scoring + detector skeleton) — 1 week
- `uhg.anomaly.scores` (centroid/neighbor/boundary/composite)
- `UHGUnsupervisedAnomalyDetector.fit/cluster/score` minimal
- Add timers, early stopping, config exposure
- Tests for scores and detector fit/cluster/score

Sprint 3 (Reporting + UHG-KNN option + NeighborLoader) — 1 week
- Reporting helpers; export/from_export; `display_summary`
- `build_uhg_knn_graph` (batched Minkowski) and toggle
- NeighborLoader path and parity test

Sprint 4 (Hardening & docs) — 1 week
- CPU/GPU parity, determinism checks, performance assertions
- API docs and examples (no plots), README updates

---

## L. Risks & mitigations

- kNN scalability: add FAISS backend and caching; document trade‑offs.
- DBSCAN sensitivity: include grid search + auto‑eps helper and clear defaults.
- Reproducibility: seed all RNGs; document AMP nondeterminism caveats.

---

## M. Done criteria

- A user can: load features → detector.fit → detector.cluster → detector.score → detector.summarize → export; obtain JSON with timings, DB score, and ranked entities — all without any visualization code. 