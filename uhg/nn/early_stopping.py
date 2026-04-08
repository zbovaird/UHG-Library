"""Early stopping for training loops."""

from typing import Any, Optional

import torch.nn as nn


class EarlyStopping:
    """Early stopping that tracks best loss and restores best state.

    Stops when loss does not improve by at least min_delta for patience epochs.
    """

    def __init__(self, min_delta: float = 0.0, patience: int = 5, mode: str = "min"):
        """Initialize early stopping.

        Args:
            min_delta: Minimum change to qualify as improvement.
            patience: Number of epochs without improvement before stopping.
            mode: "min" (lower is better) or "max" (higher is better).
        """
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Record score and check if should stop. Saves best state when improved.

        Args:
            score: Current metric (e.g. loss).
            model: Model to save/restore state from.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best(self, model: nn.Module) -> None:
        """Restore model to best saved state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)
