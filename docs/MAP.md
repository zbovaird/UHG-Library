# Repository map (start here)

Layer-1 navigation for the UHG-Library repo: what lives where and what is “product” vs examples vs tooling.

## Core library (PyPI package `uhg`)

| Path | Purpose |
|------|---------|
| [`uhg/`](../uhg/) | Installable package: projective UHG, GNN layers, graph/cluster/anomaly pipeline |
| [`pyproject.toml`](../pyproject.toml) | Single source of packaging metadata (PEP 621) and tool config |
| [`README.md`](../README.md) | Project entry: problem, install, stable API, links |

## Documentation

| Path | Purpose |
|------|---------|
| [`docs/index.md`](index.md) | Docs home (MkDocs) |
| [`docs/reference/public-api.md`](reference/public-api.md) | **Semver-stable** symbols exported from `uhg` |
| [`docs/reference/stability.md`](reference/stability.md) | Deprecation policy for public API |
| [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common failures |
| [`docs/development/`](development/) | Contributor-oriented notes (implementation, vectorization) |
| [`docs/development/colab-local.md`](development/colab-local.md) | Colab GPU + local venv / optional `uhg[colab]` |

## Examples and pedagogy (not the default install)

| Path | Purpose |
|------|---------|
| [`examples/legacy/`](../examples/legacy/) | Older scripts and notebooks (best-effort) |
| [`examples/interactive/`](../examples/interactive/) | Static HTML conceptual demo |
| [`colab/`](../colab/) | Colab notebooks |
| [`benchmarks/`](../benchmarks/) | Benchmark scripts |

## Development tooling (optional extras)

| Path | Purpose |
|------|---------|
| [`mcp_server/`](../mcp_server/) | MCP server for editor-assisted dev; install `uhg[mcp]` — see [`mcp_server/README.md`](../mcp_server/README.md) |
| [`scripts/`](../scripts/) | Maintainer helpers (publish, venv, etc.) — not part of `uhg` |

## Tests

| Path | Purpose |
|------|---------|
| [`tests/`](../tests/) | Pytest suite (`tests/root_legacy` ignored by default) |

## Reference PDFs (mathematics)

| Path | Purpose |
|------|---------|
| [`UHG.pdf`](https://github.com/zbovaird/UHG-Library/blob/main/UHG.pdf), [`UHG pictorial.pdf`](https://github.com/zbovaird/UHG-Library/blob/main/UHG%20pictorial.pdf) | Tracked reference PDFs for validating math against UHG literature (at repository root) |

## Support

Issues and PRs: [GitHub Issues](https://github.com/zbovaird/UHG-Library/issues) only (see [`README.md`](../README.md)).
