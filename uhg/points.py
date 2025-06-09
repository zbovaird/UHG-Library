"""
UHG Points Module

This module defines the UHGPoint class for representing points in Universal Hyperbolic Geometry.
All operations strictly follow the mathematical principles outlined in UHG.pdf.
"""

import torch
from typing import Optional, Tuple

class UHGPoint:
    """
    Represents a point in Universal Hyperbolic Geometry (UHG).
    A point is represented by its projective coordinates [x:y:z].
    """

    def __init__(self, coords: torch.Tensor):
        """
        Initialize a UHG point from its projective coordinates.

        Args:
            coords: Tensor of shape (3,) representing the point [x:y:z].
        """
        self.coords = coords

    def normalize(self) -> 'UHGPoint':
        """
        Normalize the point to ensure it lies on the unit circle.
        This is crucial for maintaining hyperbolic invariants.

        Returns:
            UHGPoint: The normalized point.
        """
        norm = torch.norm(self.coords)
        if norm > 0:
            self.coords = self.coords / norm
        return self

    def is_null(self) -> bool:
        """
        Check if the point is null (lies on the absolute conic).
        In UHG, a point [x, y, z] is null if x^2 + y^2 - z^2 == 0.
        """
        x, y, z = self.coords
        return torch.abs(x**2 + y**2 - z**2) < 1e-10

    def dual_line(self) -> torch.Tensor:
        """
        Compute the dual line of this point.
        In UHG, the dual of a point [x:y:z] is the line [x:y:z].

        Returns:
            Tensor of shape (3,) representing the dual line.
        """
        return self.coords 