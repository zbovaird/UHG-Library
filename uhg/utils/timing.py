"""Timing utilities for capturing stage durations."""

import time
from contextlib import contextmanager
from typing import Dict


class TimingsDict(dict):
    """Dict that accumulates timings from time_block context manager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, name: str, duration_s: float) -> None:
        """Add or overwrite a timing entry."""
        self[name] = duration_s


@contextmanager
def time_block(name: str, timings: Dict[str, float] | None = None):
    """Context manager to capture duration of a block. Accumulates into timings dict if provided.

    Args:
        name: Label for this timing (e.g. "data_load_s", "knn_build_s").
        timings: Optional dict to accumulate durations into. If None, still measures but does not store.

    Yields:
        The timings dict (for convenience).

    Example:
        timings = {}
        with time_block("data_load_s", timings):
            load_data()
    """
    start = time.perf_counter()
    out = timings if timings is not None else {}
    try:
        yield out
    finally:
        duration = time.perf_counter() - start
        if isinstance(out, TimingsDict):
            out.add(name, duration)
        else:
            out[name] = duration
