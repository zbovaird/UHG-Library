"""Tests for fit_from_dataframe and predict."""

import numpy as np
import pytest
import torch

from uhg.anomaly.unsupervised import UHGUnsupervisedAnomalyDetector

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_fit_from_dataframe():
    """fit_from_dataframe runs without error."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0] * 30,
        "b": [4.0, 5.0, 6.0] * 30,
        "label": [0, 1, 0] * 30,
    })
    det = UHGUnsupervisedAnomalyDetector(hidden=8, embedding_dim=4)
    det.fit_from_dataframe(df, epochs=5, seed=42)
    assert det.embeddings is not None


def test_predict_percentile():
    """predict returns binary labels with correct percentile cutoff."""
    X = np.random.RandomState(1).randn(100, 3)
    det = UHGUnsupervisedAnomalyDetector(hidden=8, embedding_dim=4)
    det.fit(X, k=5, epochs=5, seed=1)
    scores, labels = det.predict(percentile=0.95)
    assert labels.sum().item() == 5
    assert (labels == 1).sum().item() == 5


def test_predict_threshold():
    """predict with threshold returns correct binary labels."""
    X = np.random.RandomState(2).randn(50, 3)
    det = UHGUnsupervisedAnomalyDetector(hidden=8, embedding_dim=4)
    det.fit(X, k=4, epochs=5, seed=2)
    scores, labels = det.predict(threshold=0.5)
    assert (scores >= 0.5).sum().item() == labels.sum().item()


def test_score_new():
    """score_new returns scores for new points."""
    X = np.random.RandomState(3).randn(80, 4)
    det = UHGUnsupervisedAnomalyDetector(hidden=8, embedding_dim=4)
    det.fit(X, k=5, epochs=5, seed=3)
    X_new = np.random.RandomState(99).randn(5, 4)
    s = det.score_new(X_new)
    assert s.shape == (5,)
