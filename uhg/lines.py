"""
UHG Lines Module

This module defines the UHGLine class for representing lines in Universal Hyperbolic Geometry.
All operations strictly follow the mathematical principles outlined in UHG.pdf.
"""

import torch
from typing import Optional, Tuple

class UHGLine:
    """
    Represents a line in Universal Hyperbolic Geometry (UHG).
    A line is represented by its projective coordinates [a:b:c].
    """

    def __init__(self, coords: torch.Tensor):
        """
        Initialize a UHG line from its projective coordinates.

        Args:
            coords: Tensor of shape (3,) representing the line [a:b:c].
        """
        self.coords = coords

    @classmethod
    def from_points(cls, p1: torch.Tensor, p2: torch.Tensor) -> 'UHGLine':
        """
        Construct a UHG line from two points using the join operation.
        The line is the projective join of the two points.

        Args:
            p1: First point tensor of shape (3,).
            p2: Second point tensor of shape (3,).

        Returns:
            UHGLine: The line joining p1 and p2.
        """
        # Compute the cross product to get the line coefficients
        line_coords = torch.cross(p1, p2)
        return cls(line_coords)

    def point_lies_on_line(self, point: torch.Tensor) -> bool:
        """
        Check if a point lies on this line using the projective incidence relation.
        A point [x:y:z] lies on a line [a:b:c] if ax + by + cz = 0.

        Args:
            point: Tensor of shape (3,) representing the point.

        Returns:
            bool: True if the point lies on the line, False otherwise.
        """
        return torch.abs(torch.dot(self.coords, point)) < 1e-10

    def dual_point(self) -> torch.Tensor:
        """
        Compute the dual point of this line.
        In UHG, the dual of a line [a:b:c] is the point [a:b:c].

        Returns:
            Tensor of shape (3,) representing the dual point.
        """
        return self.coords 