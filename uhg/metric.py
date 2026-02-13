"""Compatibility shim for consolidated metrics API.

This module re-exports `UHGMetric` from `uhg.metrics` to provide a single canonical
metric class while keeping existing import paths working.
"""

from .metrics import UHGMetric  # noqa: F401 