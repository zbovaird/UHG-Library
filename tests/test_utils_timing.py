"""Tests for uhg.utils.timing."""

import pytest
from uhg.utils.timing import time_block, TimingsDict


def test_time_block_records_duration():
    """Context manager records duration in dict."""
    timings = TimingsDict()
    with time_block("test_s", timings):
        pass
    assert "test_s" in timings
    assert isinstance(timings["test_s"], (int, float))
    assert timings["test_s"] >= 0


def test_time_block_dict_keys_correct():
    """Multiple blocks add correct keys."""
    timings = TimingsDict()
    with time_block("a", timings):
        pass
    with time_block("b", timings):
        pass
    assert "a" in timings
    assert "b" in timings
    assert set(timings.keys()) == {"a", "b"}


def test_time_block_with_plain_dict():
    """time_block works with plain dict."""
    d = {}
    with time_block("x", d):
        pass
    assert "x" in d
    assert d["x"] >= 0


def test_time_block_without_timings_dict():
    """time_block with timings=None still runs without error."""
    with time_block("y", None):
        pass
