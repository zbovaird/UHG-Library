"""Smoke tests for focal loss, stratified sampling, read_cic_csv."""

from __future__ import annotations

import numpy as np
import torch

from uhg.nn.losses import FocalLoss
from uhg.utils.cic_io import read_cic_csv
from uhg.utils.sampling import stratified_subsample_indices


def test_focal_loss_forward():
    logits = torch.randn(16, 5)
    target = torch.randint(0, 5, (16,))
    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_stratified_subsample_indices_floor():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    idx = stratified_subsample_indices(y, n_total=6, min_per_class=2, random_state=0)
    assert len(idx) == 6
    assert len(np.unique(idx)) == 6
    subs = y[idx]
    assert np.bincount(subs, minlength=3).min() >= 2


def test_read_cic_csv_reads_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("flow_id,Label,x\n1,BENIGN,0.1\n2,Attack,0.2\n")
        path = f.name
    try:
        df = read_cic_csv(path)
        assert len(df) == 2
        assert "Label" in df.columns
    finally:
        Path(path).unlink(missing_ok=True)
