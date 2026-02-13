"""Tests for uhg.anomaly.report."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from uhg.anomaly.report import aggregate_by_entity, rank_topk, summary_to_json


def test_rank_topk_order():
    """rank_topk returns correct order by score."""
    scores = torch.tensor([0.1, 0.9, 0.5, 0.8])
    out = rank_topk(scores, k=3)
    assert len(out) == 3
    assert abs(out[0]["score"] - 0.9) < 0.01
    assert abs(out[1]["score"] - 0.8) < 0.01
    assert out[2]["score"] == 0.5


def test_rank_topk_with_ids():
    """rank_topk includes ids when provided."""
    scores = np.array([0.5, 0.9])
    ids = np.array(["a", "b"])
    out = rank_topk(scores, k=2, ids=ids)
    assert out[0]["id"] == "b"
    assert out[1]["id"] == "a"


def test_aggregate_by_entity_stats():
    """aggregate_by_entity computes mean, p95, count."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    entity_ids = np.array([0, 0, 0, 1, 1])
    out = aggregate_by_entity(scores, entity_ids, stats=("mean", "p95", "count"))
    assert 0 in out
    assert 1 in out
    np.testing.assert_allclose(out[0]["mean"], 2.0)
    assert out[0]["count"] == 3
    assert out[1]["count"] == 2


def test_summary_to_json():
    """summary_to_json writes valid JSON."""
    summary = {"n_nodes": 100, "timings": {"train_s": 1.5}, "top_entities": [{"rank": 1, "score": 0.9}]}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        summary_to_json(summary, path)
        import json

        with open(path) as fp:
            loaded = json.load(fp)
        assert loaded["n_nodes"] == 100
        assert loaded["timings"]["train_s"] == 1.5
    finally:
        Path(path).unlink(missing_ok=True)
