"""Tests for uhg.graph.build."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from uhg.graph.build import (
    build_knn_graph,
    build_maxk_then_slice,
    load_edge_index,
    save_edge_index,
)


def test_build_knn_graph_undirected_covers_reverse_edges():
    """Undirected mode symmetrizes; edge count is at least the directed count."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)
    ei_dir = build_knn_graph(X, k=4, undirected=False)
    ei_und = build_knn_graph(X, k=4, undirected=True)
    assert ei_und.shape[1] >= ei_dir.shape[1]
    assert ei_und.shape[1] <= 2 * ei_dir.shape[1]


def test_build_knn_graph_basic():
    """build_knn_graph returns valid edge_index."""
    X = np.random.RandomState(42).randn(100, 5)
    edge_index = build_knn_graph(X, k=5)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 100 * 5
    assert edge_index.dtype == torch.long
    assert edge_index[0].min() >= 0
    assert edge_index[0].max() < 100
    assert edge_index[1].min() >= 0
    assert edge_index[1].max() < 100


def test_build_knn_graph_accepts_tensor():
    """build_knn_graph accepts torch.Tensor."""
    X = torch.randn(50, 4)
    edge_index = build_knn_graph(X, k=3)
    assert edge_index.shape == (2, 50 * 3)


def test_save_load_edge_index_equals_original():
    """save_edge_index then load_edge_index equals original."""
    X = np.random.RandomState(1).randn(30, 3)
    edge_index = build_knn_graph(X, k=2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_edge_index(path, edge_index)
        loaded = load_edge_index(path)
        torch.testing.assert_close(loaded, edge_index.cpu())
    finally:
        Path(path).unlink(missing_ok=True)


def test_maxk_slice_equals_fresh_small_k():
    """build_maxk_then_slice returns same first-k neighbors as fresh build."""
    X = np.random.RandomState(123).randn(40, 6)
    max_k = 10
    k = 3
    sliced = build_maxk_then_slice(X, max_k=max_k, k=k)
    fresh = build_knn_graph(X, k=k)
    assert sliced.shape == fresh.shape
    torch.testing.assert_close(sliced, fresh)


def test_cache_load_equals_original():
    """Cached graph reloaded equals original."""
    X = np.random.RandomState(99).randn(25, 4)
    with tempfile.TemporaryDirectory() as d:
        key = "test_cache_123"
        ei1 = build_knn_graph(X, k=2, cache_key=key, cache_dir=d)
        ei2 = build_knn_graph(X, k=2, cache_key=key, cache_dir=d)
        torch.testing.assert_close(ei1, ei2)


def test_build_knn_graph_raises_k_too_large():
    """build_knn_graph raises when k >= n."""
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError, match="k must be"):
        build_knn_graph(X, k=10)
    with pytest.raises(ValueError, match="k must be"):
        build_knn_graph(X, k=11)


def test_build_maxk_then_slice_raises_k_gt_maxk():
    """build_maxk_then_slice raises when k > max_k."""
    X = np.random.randn(20, 3)
    with pytest.raises(ValueError, match="k.*must be <= max_k"):
        build_maxk_then_slice(X, max_k=5, k=6)
