from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass(frozen=True)
class ScalingInfo:
    """Placeholder for any scaling metadata needed by higher-level APIs."""
    eps: float = 1e-8


class Manifold(Protocol):
    """Protocol for UHG manifold-like wrappers (projective-only)."""

    def normalize_points(self, x: torch.Tensor) -> torch.Tensor: ...

    def project(self, x: torch.Tensor) -> torch.Tensor: ...

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

    def cross_ratio(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor: ...

    def join(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor: ...

    def meet(self, line: torch.Tensor, point: torch.Tensor) -> torch.Tensor: ...

    def transform(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor: ...

    def aggregate(self, points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor: ...

    def scale(self, points: torch.Tensor, factor: torch.Tensor) -> torch.Tensor: ...

    def add(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor: ... 