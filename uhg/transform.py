"""
UHG Transformations Module

This module defines the UHGTransform class for representing projective and hyperbolic transformations in Universal Hyperbolic Geometry.
All operations strictly follow the mathematical principles outlined in UHG.pdf.
"""

import torch
from typing import Optional, Tuple

class UHGTransform:
    """
    Represents a transformation in Universal Hyperbolic Geometry (UHG).
    This class handles both projective and hyperbolic transformations.
    """

    def __init__(self, matrix: torch.Tensor):
        """
        Initialize a UHG transformation from its transformation matrix.

        Args:
            matrix: Tensor of shape (3, 3) representing the transformation matrix.
        """
        self.matrix = matrix

    def apply_to_point(self, point: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to a point.

        Args:
            point: Tensor of shape (3,) representing the point.

        Returns:
            Tensor of shape (3,) representing the transformed point.
        """
        return torch.matmul(self.matrix, point)

    def apply_to_line(self, line: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to a line.

        Args:
            line: Tensor of shape (3,) representing the line.

        Returns:
            Tensor of shape (3,) representing the transformed line.
        """
        return torch.matmul(self.matrix, line) 