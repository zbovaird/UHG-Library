"""Tests for uhg.cluster.metrics."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from uhg.cluster.metrics import calinski_harabasz, davies_bouldin, silhouette


def test_davies_bouldin_matches_sklearn_blobs():
    """davies_bouldin matches sklearn on make_blobs."""
    X, y = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    ref = davies_bouldin_score(X, y)
    ours = davies_bouldin(X, y)
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


def test_silhouette_matches_sklearn_blobs():
    """silhouette matches sklearn on make_blobs."""
    X, y = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    ref = silhouette_score(X, y)
    ours = silhouette(X, y)
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


def test_calinski_harabasz_matches_sklearn_blobs():
    """calinski_harabasz matches sklearn on make_blobs."""
    X, y = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    ref = calinski_harabasz_score(X, y)
    ours = calinski_harabasz(X, y)
    np.testing.assert_allclose(ours, ref, rtol=1e-5)


def test_metrics_accept_tensor():
    """Metrics accept torch.Tensor input."""
    X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=1)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    d = davies_bouldin(X_t, y_t)
    s = silhouette(X_t, y_t)
    c = calinski_harabasz(X_t, y_t)
    assert isinstance(d, (int, float))
    assert isinstance(s, (int, float))
    assert isinstance(c, (int, float))


def test_metrics_moons():
    """Metrics run on make_moons."""
    X, y = make_moons(n_samples=80, noise=0.05, random_state=42)
    d = davies_bouldin(X, y)
    s = silhouette(X, y)
    c = calinski_harabasz(X, y)
    assert d > 0
    assert -1 <= s <= 1
    assert c > 0
