"""CSV helpers for CIC-scale network-flow tables."""

from __future__ import annotations


def read_cic_csv(path: str, *, low_memory: bool = False):
    """Load a CSV with pandas, using the PyArrow engine when available.

    PyArrow is often faster for wide CIC-style files; falls back to the default
    pandas engine if ``pyarrow`` is not installed or the engine rejects the file.

    Args:
        path: Filesystem path to ``*.csv``.
        low_memory: Passed to ``pandas.read_csv``.

    Returns:
        A ``pandas.DataFrame``.
    """
    import pandas as pd

    try:
        import pyarrow  # noqa: F401

        return pd.read_csv(path, low_memory=low_memory, engine="pyarrow")
    except (ImportError, ValueError, OSError, TypeError):
        return pd.read_csv(path, low_memory=low_memory)
