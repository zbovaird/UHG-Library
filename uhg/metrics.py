"""Universal Hyperbolic Geometry metrics implementation."""

import torch
import math
from typing import Optional, Tuple

class UHGMetric:
    """
    Universal Hyperbolic Geometry (UHG) metric implementation.
    This class provides methods for computing hyperbolic distances, metrics,
    and related geometric operations in UHG space, strictly using projective/UHG principles.
    No tangent-space, exponential map, or logarithmic map methods are present.
    """
    def __init__(self, eps: float = 1e-8):
        """
        Initialize UHG metric.
        Args:
            eps: Numerical stability constant
        """
        self.eps = eps

    def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor at point x in UHG space.
        In UHG, the metric tensor is the identity matrix (projective model).
        """
        return torch.eye(x.shape[-1], device=x.device)

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points x and y using projective invariants.
        Reference: UHG.pdf, Ch. 4
        """
        # Compute Minkowski inner product
        spatial_dot = torch.sum(x[..., :-1] * y[..., :-1], dim=-1)
        time_dot = x[..., -1] * y[..., -1]
        inner_prod = spatial_dot - time_dot
        # Ensure inner product is <= -1 for acosh
        inner_prod = torch.clamp(inner_prod, max=-1.0 - self.eps)
        d = torch.acosh(-inner_prod)
        return d

def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Top-level function for hyperbolic distance for test imports."""
    return UHGMetric().hyperbolic_distance(x, y) 