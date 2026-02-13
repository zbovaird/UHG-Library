from __future__ import annotations

from typing import Optional
import torch

from .base import Manifold, ScalingInfo
from ..projective import ProjectiveUHG


class HyperbolicManifold:  # projective-only wrapper
    """
    UHG-compliant manifold wrapper.
    - Delegates to `ProjectiveUHG` for all operations
    - No tangent-space, no exp/log, no Möbius ops
    - Methods are batchable and GPU-friendly
    """

    def __init__(self, curvature: float = -1.0, eps: float = 1e-8):
        if curvature >= 0:
            raise ValueError("HyperbolicManifold requires negative curvature in UHG (projective model)")
        self.curvature = curvature
        self.eps = eps
        self._uhg = ProjectiveUHG(epsilon=eps)
        self.scaling = ScalingInfo(eps=eps)

    # Core delegates
    def normalize_points(self, x: torch.Tensor) -> torch.Tensor:
        return self._uhg.normalize_points(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self._uhg.project(x)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._uhg.distance(x, y)

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._uhg.inner_product(a, b)

    def cross_ratio(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._uhg.cross_ratio(a, b, c, d)

    def join(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        return self._uhg.join(p1, p2)

    def meet(self, line: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        return self._uhg.meet(line, point)

    def transform(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self._uhg.transform(x, T)

    def aggregate(self, points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self._uhg.aggregate(points, weights)

    def scale(self, points: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        return self._uhg.scale(points, factor)

    # Compatibility helper used by some modules (projective-safe "addition")
    def add(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Projective-safe update: x + update in feature part, then re-normalize.
        Assumes inputs are homogeneous coordinates [..., D] with last component time-like.
        """
        x_features = x[..., :-1]
        x_time = x[..., -1:]
        # Case 1: update matches feature part only
        if update.size(-1) == x_features.size(-1):
            new_features = x_features + update
            # Recompute time-like coordinate to stay on hyperboloid: z = sqrt(1 + ||features||^2)
            new_time = torch.sqrt(1.0 + torch.sum(new_features * new_features, dim=-1, keepdim=True))
            return torch.cat([new_features, new_time], dim=-1)
        # Case 2: update matches full homogeneous vector
        elif update.size(-1) == x.size(-1):
            new = x + update
            new_features = new[..., :-1]
            new_time = torch.sqrt(1.0 + torch.sum(new_features * new_features, dim=-1, keepdim=True))
            return torch.cat([new_features, new_time], dim=-1)
        # Fallback: broadcast if possible, then recompute time
        new_features = x_features + update
        new_time = torch.sqrt(1.0 + torch.sum(new_features * new_features, dim=-1, keepdim=True))
        return torch.cat([new_features, new_time], dim=-1) 