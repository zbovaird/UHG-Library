# Contributing to UHG-Library

Thank you for helping improve the **uhg** package. This document is the contributor contract for this repository.

## Support channel

Use **[GitHub Issues](https://github.com/zbovaird/UHG-Library/issues)** for bug reports and feature discussion. There are no other official support channels.

## Development setup

Requires **Python 3.10+** (see `pyproject.toml`).

```bash
git clone https://github.com/zbovaird/UHG-Library.git
cd UHG-Library
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional: use **uv** for faster installs:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

With coverage (matches CI):

```bash
pytest --cov=uhg --cov-report=term-missing
```

Legacy tests under `tests/root_legacy/` are ignored by default (see `pyproject.toml`).

Optional markers: `unit`, `integration`, `anomaly`, `numerical`, `slow` (see `[tool.pytest.ini_options]` in `pyproject.toml`).

## Linting and types

```bash
black uhg tests
mypy uhg
```

Tox (optional):

```bash
tox
```

## Definition of done

Pull requests should:

1. **Tests:** Add or update tests for behavior changes; keep CI green.
2. **Public API:** If you change symbols listed in [`docs/reference/public-api.md`](docs/reference/public-api.md), update that doc and [`CHANGELOG.md`](CHANGELOG.md). Breaking changes need a **major** version plan.
3. **Docs:** Update `README.md` or `docs/` when user-facing behavior changes.

## Releases and PyPI

Maintainers:

1. Bump **`version`** in **`pyproject.toml`** and **`uhg/__version__`** in `uhg/__init__.py` (keep them in sync).
2. Update **`CHANGELOG.md`** with a dated section.
3. Tag **`vX.Y.Z`** in Git and push.
4. Build: `python -m build`
5. Check: `twine check dist/*`
6. Upload: `twine upload dist/*`

### Yanking a bad release

Prefer **yanking** a broken version on PyPI rather than deleting files. Reserve **deletion** for legal or credential leaks. Document incidents in `CHANGELOG.md`.

## Security

See [`SECURITY.md`](SECURITY.md) for vulnerability reporting.
