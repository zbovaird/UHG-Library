"""Tests for uhg.utils.schema."""

import numpy as np
import pytest
from uhg.utils.schema import (
    detect_label_column,
    detect_entity_column,
    enforce_numeric,
    build_entity_index,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDetectLabelColumn:
    def test_detects_label(self):
        df = pd.DataFrame({"a": [1, 2], "label": [0, 1], "b": [3, 4]})
        assert detect_label_column(df) == "label"

    def test_detects_labels_capitalized(self):
        df = pd.DataFrame({"Labels": [0, 1, 0]})
        assert detect_label_column(df) == "Labels"

    def test_detects_target(self):
        df = pd.DataFrame({"x": [1], "target": [0]})
        assert detect_label_column(df) == "target"

    def test_detects_y(self):
        df = pd.DataFrame({"y": [0, 1]})
        assert detect_label_column(df) == "y"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert detect_label_column(df) is None


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDetectEntityColumn:
    def test_detects_entity_id(self):
        df = pd.DataFrame({"entity_id": ["e1", "e2"], "val": [1, 2]})
        assert detect_entity_column(df) == "entity_id"

    def test_detects_id(self):
        df = pd.DataFrame({"id": [1, 2], "x": [3, 4]})
        assert detect_entity_column(df) == "id"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert detect_entity_column(df) is None


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestEnforceNumeric:
    def test_converts_to_numeric(self):
        df = pd.DataFrame({"a": ["1", "2", "3"], "b": [4, 5, 6]})
        out = enforce_numeric(df)
        assert np.issubdtype(out["a"].dtype, np.number)
        np.testing.assert_allclose(out["a"], [1.0, 2.0, 3.0])

    def test_fill_mean(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = enforce_numeric(df, fill="mean")
        assert not out["a"].isna().any()
        np.testing.assert_allclose(out["a"], [1.0, 2.0, 3.0])

    def test_replace_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        out = enforce_numeric(df, replace_inf=True, fill="mean")
        assert not np.isinf(out["a"]).any()


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestBuildEntityIndex:
    def test_builds_mapping(self):
        s = pd.Series(["x", "y", "x", "z"])
        idx_arr, id2idx, idx2id = build_entity_index(s)
        assert len(id2idx) == 3
        assert id2idx["x"] == 0 or id2idx["x"] >= 0
        assert len(idx2id) == 3
        np.testing.assert_array_equal(idx_arr, [id2idx["x"], id2idx["y"], id2idx["x"], id2idx["z"]])

    def test_roundtrip(self):
        s = pd.Series(["a", "b", "a"])
        idx_arr, id2idx, idx2id = build_entity_index(s)
        for i, orig in enumerate(s):
            if orig in id2idx:
                assert idx2id[id2idx[orig]] == orig
