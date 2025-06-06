"""
UHG Metric Operations Module

This module implements core metric operations in Universal Hyperbolic Geometry.
All operations strictly follow the mathematical principles outlined in UHG.pdf.
"""

import torch
from typing import Optional, Tuple, Union
from .points import UHGPoint
from .lines import UHGLine

class UHGMetric:
    """
    Implements metric operations in Universal Hyperbolic Geometry.
    All calculations are performed directly in hyperbolic space without tangent space approximations.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initialize the UHG metric operations.

        Args:
            eps: Small value for numerical stability in calculations.
        """
        self.eps = eps

    def quadrance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute the quadrance (squared distance) between two points in UHG.
        The quadrance is computed using the cross-ratio formula from UHG.pdf.

        Args:
            p1: First point tensor of shape (..., D+1)
            p2: Second point tensor of shape (..., D+1)

        Returns:
            Quadrance tensor of shape (...)
        """
        # Ensure points are normalized
        p1_norm = p1 / torch.norm(p1, dim=-1, keepdim=True)
        p2_norm = p2 / torch.norm(p2, dim=-1, keepdim=True)

        # Compute the cross-ratio based quadrance
        dot_product = torch.sum(p1_norm * p2_norm, dim=-1)
        quad = 1.0 - dot_product * dot_product

        # Ensure numerical stability
        return torch.clamp(quad, min=self.eps)

    def spread(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """
        Compute the spread (hyperbolic angle) between three points in UHG.
        The spread is computed using the quadrance formula from UHG.pdf.

        Args:
            p1: First point tensor of shape (..., D+1)
            p2: Second point tensor of shape (..., D+1)
            p3: Third point tensor of shape (..., D+1)

        Returns:
            Spread tensor of shape (...)
        """
        # Compute quadrances between points
        q12 = self.quadrance(p1, p2)
        q13 = self.quadrance(p1, p3)
        q23 = self.quadrance(p2, p3)

        # Compute spread using the quadrance formula
        numerator = q12 * q13
        denominator = q23

        # Ensure numerical stability
        denominator = torch.clamp(denominator, min=self.eps)
        spread = numerator / denominator

        # Ensure spread is between 0 and 1
        return torch.clamp(spread, min=0.0, max=1.0)

    def distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic distance between two points in UHG.
        The distance is computed as the square root of the quadrance.

        Args:
            p1: First point tensor of shape (..., D+1)
            p2: Second point tensor of shape (..., D+1)

        Returns:
            Distance tensor of shape (...)
        """
        return torch.sqrt(self.quadrance(p1, p2))

    def cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, 
                   p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-ratio of four points in UHG.
        The cross-ratio is a fundamental invariant in projective geometry.

        Args:
            p1: First point tensor of shape (..., D+1)
            p2: Second point tensor of shape (..., D+1)
            p3: Third point tensor of shape (..., D+1)
            p4: Fourth point tensor of shape (..., D+1)

        Returns:
            Cross-ratio tensor of shape (...)
        """
        # Compute quadrances between points
        q12 = self.quadrance(p1, p2)
        q34 = self.quadrance(p3, p4)
        q13 = self.quadrance(p1, p3)
        q24 = self.quadrance(p2, p4)

        # Compute cross-ratio
        numerator = q12 * q34
        denominator = q13 * q24

        # Ensure numerical stability
        denominator = torch.clamp(denominator, min=self.eps)
        return numerator / denominator

    def is_collinear(self, p1: torch.Tensor, p2: torch.Tensor, 
                    p3: torch.Tensor) -> torch.Tensor:
        """
        Check if three points are collinear in UHG.
        Points are collinear if the cross product of their coordinates is zero.

        Args:
            p1: First point tensor of shape (..., D+1)
            p2: Second point tensor of shape (..., D+1)
            p3: Third point tensor of shape (..., D+1)

        Returns:
            Boolean tensor indicating collinearity of shape (...)
        """
        # Compute vectors between points
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Compute cross product
        cross = torch.cross(v1, v2, dim=-1)
        
        # Points are collinear if cross product is zero
        return torch.norm(cross, dim=-1) < self.eps

    def mean(self, x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the hyperbolic mean of points in UHG.
        The mean is computed using the Frechet mean formula in hyperbolic space.

        Args:
            x: Input tensor of shape (..., D+1)
            dim: Dimension along which to compute the mean
            keepdim: Whether to keep the reduced dimension

        Returns:
            Mean tensor of shape (..., D+1) with the specified dimension reduced
        """
        # Ensure input is normalized
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        
        # Compute the sum of normalized points
        sum_points = torch.sum(x_norm, dim=dim, keepdim=keepdim)
        
        # Normalize the sum to get the mean
        mean = sum_points / torch.norm(sum_points, dim=-1, keepdim=True)
        
        # Handle zero norm case
        zero_norm = (torch.norm(sum_points, dim=-1, keepdim=True) < self.eps)
        if zero_norm.any():
            # Create a canonical point (first basis vector)
            canonical = torch.zeros_like(mean)
            canonical[..., 0] = 1.0
            mean = torch.where(zero_norm, canonical, mean)
        
        return mean 

    def variance(self, x: torch.Tensor, mean: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the variance of points in UHG space around the mean.
        Uses the squared UHG distance to the mean.

        Args:
            x: Input tensor of shape (..., D+1)
            mean: Mean tensor of shape (..., D+1)
            dim: Dimension along which to compute the variance
            keepdim: Whether to keep the reduced dimension

        Returns:
            Variance tensor
        """
        # Compute squared UHG distance to the mean
        dist2 = self.quadrance(x, mean)
        var = torch.mean(dist2, dim=dim, keepdim=keepdim)
        return var 