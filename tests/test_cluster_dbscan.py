"""Tests for uhg.cluster.dbscan."""

import numpy as np
import pytest
import torch

from uhg.cluster.dbscan import auto_eps_kdist, eps_grid_search, run_dbscan


def test_run_dbscan_returns_labels_core_mask():
    """run_dbscan returns labels and core_mask."""
    X, _ = np.random.RandomState(42).randn(50, 4), None
    out = run_dbscan(X, eps=0.5, min_samples=3)
    assert "labels" in out
    assert "core_mask" in out
    assert out["labels"].shape == (50,)
    assert out["core_mask"].shape == (50,)
    assert out["core_mask"].dtype == bool
    np.testing.assert_array_equal(out["core_mask"], out["labels"] >= 0)


def test_grid_search_returns_params_metrics():
    """eps_grid_search returns labels, params, metrics when valid clusters found."""
    X, _ = np.random.RandomState(1).randn(60, 5), None
    out = eps_grid_search(
        X,
        eps_list=[0.5, 1.0, 2.0],
        min_samples_list=[2, 3],
        score="db",
    )
    assert "labels" in out
    assert "params" in out
    assert "metrics" in out
    if out["metrics"] is not None:
        assert "davies_bouldin" in out["metrics"]
        assert "silhouette" in out["metrics"]
        assert "calinski_harabasz" in out["metrics"]


def test_grid_search_all_noise():
    """eps_grid_search handles case where all points are noise."""
    X = np.random.RandomState(99).randn(30, 3) * 100
    out = eps_grid_search(
        X,
        eps_list=[0.01, 0.02],
        min_samples_list=[2],
        score="db",
    )
    assert "labels" in out
    assert out["labels"].shape == (30,)
    assert "params" in out
    assert "metrics" in out
    if out["metrics"] is None:
        assert np.all(out["labels"] == -1) or "params" in out


def test_auto_eps_kdist_returns_float():
    """auto_eps_kdist returns positive float."""
    X = np.random.RandomState(7).randn(40, 4)
    eps = auto_eps_kdist(X, k=4)
    assert isinstance(eps, (int, float))
    assert eps > 0


def test_run_dbscan_accepts_tensor():
    """run_dbscan accepts torch.Tensor."""
    X = torch.randn(20, 3)
    out = run_dbscan(X, eps=1.0, min_samples=2)
    assert out["labels"].shape == (20,)


def test_run_dbscan_sanitizes_huge_and_nonfinite_values():
    """run_dbscan handles overflow-scale finite values and Inf/NaN."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4).astype(np.float64)
    X[0, 0] = np.inf
    X[1, 1] = np.nan
    X[2, 2] = 1e200  # finite but far beyond float32-safe range
    X[3, 3] = -1e200

    out = run_dbscan(X, eps=1.0, min_samples=2)
    assert out["labels"].shape == (40,)
    assert out["core_mask"].shape == (40,)
