"""Tests for UHGUnsupervisedAnomalyDetector."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from uhg.anomaly.unsupervised import UHGUnsupervisedAnomalyDetector


def test_fit_cluster_score_summarize_non_empty():
    """fit -> cluster -> score -> summarize produces non-empty summary."""
    X = np.random.RandomState(42).randn(200, 5) * 0.5
    det = UHGUnsupervisedAnomalyDetector(hidden=16, embedding_dim=8)
    det.fit(X, k=5, epochs=10, seed=42)
    det.cluster(eps=0.8, min_samples=2)
    scores = det.score(method="centroid_quadrance")
    assert scores.shape == (200,)
    summary = det.summarize(topk=5)
    assert "n_nodes" in summary
    assert summary["n_nodes"] == 200
    assert "timings" in summary
    assert "top_entities" in summary
    assert len(summary["top_entities"]) <= 5


def test_fit_input_validation():
    """fit raises on invalid input."""
    det = UHGUnsupervisedAnomalyDetector()
    with pytest.raises(ValueError, match="empty"):
        det.fit(np.array([]).reshape(0, 1))
    with pytest.raises(ValueError, match="k must be"):
        det.fit(np.random.randn(10, 2), k=10)
    X = np.random.randn(10, 2)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="nan"):
        det.fit(X)


def test_export_from_export_roundtrip():
    """export and from_export roundtrip preserves summary fields."""
    X = np.random.RandomState(1).randn(100, 4)
    det = UHGUnsupervisedAnomalyDetector(hidden=8, embedding_dim=4)
    det.fit(X, k=4, epochs=5, seed=1)
    det.cluster(eps=1.0, min_samples=2)
    s1 = det.summarize(topk=3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        det.export(path)
        det2 = UHGUnsupervisedAnomalyDetector.from_export(path)
        s2 = det2.summarize(topk=3)
        assert s2["n_nodes"] == s1["n_nodes"]
        assert s2["k"] == s1["k"]
        assert s2["top_entities"][0]["rank"] == 1
    finally:
        Path(path).unlink(missing_ok=True)


def test_score_requires_fit():
    """score raises if fit not called."""
    det = UHGUnsupervisedAnomalyDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.score()


def test_cluster_requires_fit():
    """cluster raises if fit not called."""
    det = UHGUnsupervisedAnomalyDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.cluster()
