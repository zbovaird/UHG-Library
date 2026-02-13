"""Tests for uhg.nn.early_stopping."""

import torch
import torch.nn as nn

from uhg.nn.early_stopping import EarlyStopping


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


def test_early_stopping_stops_on_plateau():
    """Stops when loss plateaus for patience epochs."""
    model = DummyModel()
    es = EarlyStopping(min_delta=0.01, patience=3, mode="min")
    losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84]
    for loss in losses:
        stop = es(loss, model)
        if stop:
            break
    assert es.should_stop
    assert es.best_score == 0.85
    assert es.counter == 3


def test_early_stopping_restores_best():
    """restore_best loads best state."""
    model = DummyModel()
    es = EarlyStopping(min_delta=0.01, patience=2, mode="min")
    es(1.0, model)
    es(0.5, model)
    best_weight = model.fc.weight.data.clone()
    es(0.8, model)
    es(0.9, model)
    model.fc.weight.data.fill_(999.0)
    es.restore_best(model)
    torch.testing.assert_close(model.fc.weight.data, best_weight)
