"""Tests for uhg.cluster.metrics."""

import numpy as np
import torch

from uhg.cluster.metrics import calinski_harabasz, davies_bouldin, silhouette


def make_blobs(n_samples=100, n_features=5, centers=3, random_state=42):
    rng = np.random.RandomState(random_state)
    per_center = n_samples // centers
    X_parts = []
    y_parts = []
    for label in range(centers):
        center = rng.randn(n_features) * 4.0
        X_parts.append(center + 0.3 * rng.randn(per_center, n_features))
        y_parts.append(np.full(per_center, label))
    return np.vstack(X_parts), np.concatenate(y_parts)


def make_moons(n_samples=80, noise=0.05, random_state=42):
    rng = np.random.RandomState(random_state)
    half = n_samples // 2
    theta = np.linspace(0, np.pi, half)
    moon_a = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    moon_b = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    X = np.vstack([moon_a, moon_b]) + noise * rng.randn(n_samples, 2)
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    return X, y


def test_metrics_are_finite_on_separated_blobs():
    """Cluster metrics run without requiring scikit-learn."""
    X, y = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    assert davies_bouldin(X, y) > 0
    assert -1 <= silhouette(X, y) <= 1
    assert calinski_harabasz(X, y) > 0


def test_metrics_accept_tensor():
    """Metrics accept torch.Tensor input."""
    X, y = make_blobs(n_samples=50, n_features=4, centers=2, random_state=1)
    X_t = torch.tensor(X.tolist(), dtype=torch.float32)
    y_t = torch.tensor(y.tolist(), dtype=torch.long)
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
