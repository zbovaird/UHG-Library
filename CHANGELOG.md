# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- **Packaging:** PEP 621 metadata in `pyproject.toml` only; removed legacy `setup.py` / `setup.cfg`. Version is **static** in `pyproject.toml` (and mirrored in `uhg.__version__`) so `pip install` does not import `uhg` before dependencies like PyTorch are installed.
- **Python:** Supported interpreters aligned to **3.10+** (CI: 3.10, 3.11, 3.12).
- **CI:** GitHub Actions runs `pytest` with coverage floor; lint job runs `black` and `mypy`.
- **Docs:** Added `docs/MAP.md`, `docs/reference/public-api.md`, `docs/reference/stability.md`, contributor and security docs.

### Removed

- Duplicate `UHG/` package tree (canonical package is `uhg/` only).
- Non-core narrative clutter from the repo root (e.g. cybersecurity/SOAR positioning docs, installer blobs); core math PDFs (`UHG.pdf`, `UHG pictorial.pdf`) remain as references.

## [0.3.7] - 2025-02-13
### Added
- Unsupervised anomaly detection pipeline: `UHGUnsupervisedAnomalyDetector`
- Graph building: `uhg.graph.build` with `build_knn_graph`, `save_edge_index`, `load_edge_index`, `build_maxk_then_slice`
- Clustering: `uhg.cluster.dbscan` (run_dbscan, eps_grid_search, auto_eps_kdist) and `uhg.cluster.metrics` (davies_bouldin, silhouette, calinski_harabasz)
- Anomaly scoring: `uhg.anomaly.scores` (centroid_quadrance, neighbor_quadrance, boundary_score, composite_score)
- Reporting: `uhg.anomaly.report` (rank_topk, aggregate_by_entity, summary_to_json, display_summary)
- Utilities: `uhg.utils.timing` (time_block), `uhg.utils.schema` (detect_label_column, enforce_numeric, build_entity_index)
- `uhg.nn.early_stopping.EarlyStopping` for training loops
- `fit_from_dataframe`, `predict`, `score_new` for production workflows
- Export/from_export with version compatibility check

## [0.3.4] - 2024-06-19
### Changed
- Re-release of 0.3.3 with identical features and fixes, due to PyPI versioning requirements.

## [0.3.3] - 2024-06-19
### Added
- Robust UHG optimizers: UHGSGD, UHGAdam, and base optimizer with full tangent space and manifold projection logic.
- Exponential map and tangent projection utilities for hyperbolic optimization.
- Extensive unit tests for all optimizers, including convergence and geometric invariants.

### Fixed
- Numerical stability in optimizer steps and projections.
- Manifold constraint checks for all parameter updates.
- Improved test reliability and optimizer convergence.

### Changed
- Updated optimizer API to use parameter group defaults and improved extensibility.
- Documentation and code comments for optimizer logic.

## [0.2.4] - 2024-03-19

### Fixed
- Corrected quadrance calculation to match UHG.pdf definition
- Fixed spread computation for better numerical stability
- Updated cross law implementation to handle all cases correctly
- Revised cross ratio calculation to use direct hyperbolic joins
- Improved handling of special cases in geometric operations
- Enhanced numerical stability across all core operations

## [0.2.0] - 2023-12-07

### Added
- UHG Metric Learning
  - Pure projective implementation
  - Cross-ratio preservation
  - Geometric pattern matching

- UHG Attention Mechanism
  - Multi-head attention in projective space
  - Cross-ratio preserving attention weights
  - Geometric relationship learning

- Threat Correlation
  - Indicator correlation using UHG
  - Cross-type pattern recognition
  - Network-System-Behavior relationships
  - Payload analysis integration

### Changed
- Updated core UHG operations for better numerical stability
- Improved projective transformation handling
- Enhanced cross-ratio computation

### Fixed
- Shape handling in projective operations
- Homogeneous coordinate normalization
- Cross-ratio preservation in transformations

## [0.1.0] - 2023-12-01

### Added
- Initial release
- Core UHG operations
- Basic feature extraction
- Anomaly detection

## [0.3.0] - 2024-03-19

### Added
- New UHG-compliant functional operations in `uhg.nn.functional`
- Specialized loss functions in `uhg.nn.losses` including UHGLoss and UHGAnomalyLoss
- Comprehensive metrics module in `uhg.utils.metrics` for geometric evaluations
- Detailed IMPLEMENTATION.md guide with best practices and examples
- Enhanced cross-ratio preservation checks and stability improvements
- Better numerical stability in geometric operations
- Improved error handling and validation

### Changed
- Optimized UHG operations for better performance
- Enhanced documentation with practical examples
- Improved type hints and error messages