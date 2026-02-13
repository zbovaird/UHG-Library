"""Tests for uhg.anomaly.scores."""

import numpy as np
import pytest
import torch
from uhg.anomaly.scores import (
    boundary_score,
    centroid_quadrance,
    composite_score,
    neighbor_quadrance,
)


def _make_embedding_with_outliers(n_normal: int = 80, n_outliers: int = 20, dim: int = 4, seed: int = 42):
    """Create embeddings: normal cluster near origin + far outliers. In Euclidean terms, outliers are farther from centroid."""
    rng = np.random.RandomState(seed)
    normal = rng.randn(n_normal, dim) * 0.2
    outliers = rng.randn(n_outliers, dim) * 0.2 + 10.0
    X = np.vstack([normal, outliers])
    y = np.array([0] * n_normal + [1] * n_outliers)
    return torch.from_numpy(X).float(), y


def test_centroid_quadrance_produces_scores():
    """centroid_quadrance produces non-constant scores for varied embeddings."""
    emb, _ = _make_embedding_with_outliers()
    scores = centroid_quadrance(emb)
    assert scores.shape == (emb.size(0),)
    assert scores.min() != scores.max() or emb.size(0) <= 1


def test_neighbor_quadrance_produces_scores():
    """neighbor_quadrance produces valid scores."""
    emb, _ = _make_embedding_with_outliers()
    scores = neighbor_quadrance(emb, k=5)
    assert scores.shape == (emb.size(0),)
    assert not torch.isnan(scores).any()


def test_composite_score_weights():
    """composite_score applies weights correctly."""
    n = 10
    a = torch.rand(n)
    b = torch.rand(n)
    out = composite_score({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    expected = a * 0.5 + b * 0.5
    torch.testing.assert_close(out, expected)


def test_boundary_score_non_core_high():
    """Non-core points get score 1.0."""
    labels = np.array([0, 0, 1, 1, -1, -1])
    core_mask = np.array([True, True, True, True, False, False])
    emb = torch.randn(6, 3)
    scores = boundary_score(labels, core_mask, emb)
    assert scores[4] == 1.0
    assert scores[5] == 1.0
