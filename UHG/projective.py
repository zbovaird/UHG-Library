"""
Universal Hyperbolic Geometry implementation based on projective geometry.

This implementation strictly follows UHG principles:
- Works directly with cross-ratios
- Uses projective transformations
- No differential geometry or manifold concepts
- No curvature parameters
- Pure projective operations

References:
    - UHG.pdf Chapter 3: Projective Geometry
    - UHG.pdf Chapter 4: Cross-ratios and Invariants
    - UHG.pdf Chapter 5: The Fundamental Operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class ProjectiveUHG:
    """Universal Hyperbolic Geometry operations using projective geometry.
    
    All operations are performed using pure projective geometry,
    without any manifold concepts or tangent space mappings.
    """
    def __init__(self):
        pass
        
    def get_projective_matrix(self, dim: int) -> torch.Tensor:
        """Get a random projective transformation matrix.
        
        Args:
            dim: Dimension of the projective space
            
        Returns:
            Projective transformation matrix
        """
        # Create random matrix
        matrix = torch.randn(dim + 1, dim + 1)
        
        # Make it orthogonal (preserves cross-ratios)
        q, r = torch.linalg.qr(matrix)
        
        # Ensure determinant is positive
        if torch.det(q) < 0:
            q = -q
            
        return q
        
    def transform(self, points: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply a projective transformation to points.
        
        All transformations preserve cross-ratios as required by UHG.
        
        Args:
            points: Points to transform
            matrix: Optional projective transformation matrix
            
        Returns:
            Transformed points
        """
        if matrix is None:
            matrix = self.get_projective_matrix(points.shape[-1] - 1)
            
        # Ensure matrix has correct dimensions
        if points.shape[-1] != matrix.shape[0]:
            # Add row and column for homogeneous coordinate
            pad_size = points.shape[-1] - matrix.shape[0]
            if pad_size > 0:
                pad = torch.zeros(matrix.shape[0] + pad_size, matrix.shape[1] + pad_size, device=points.device)
                pad[:matrix.shape[0], :matrix.shape[1]] = matrix
                pad[-1, -1] = 1.0
                matrix = pad
            else:
                # Truncate matrix if needed
                matrix = matrix[:points.shape[-1], :points.shape[-1]]
            
        # Apply projective transformation
        transformed = torch.matmul(points, matrix.t())
        
        # Normalize homogeneous coordinates
        return self.normalize_points(transformed)
        
    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie in projective space.
        
        Args:
            points: Points to normalize
            
        Returns:
            Normalized points
        """
        norm = torch.norm(points, p=2, dim=-1, keepdim=True)
        return points / (norm + 1e-8)
        
    def join(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Join two points to create a line.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Line through the points
        """
        # Extract 3D coordinates for cross product
        p1_3d = p1[..., :3]
        p2_3d = p2[..., :3]
        
        # Use cross product to get line coefficients
        line = torch.linalg.cross(p1_3d, p2_3d, dim=-1)
        
        # Add homogeneous coordinate
        if p1.size(-1) > 3:
            line = torch.cat([line, torch.ones_like(line[..., :1])], dim=-1)
            
        return self.normalize_points(line)
        
    def meet(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """Meet two lines to get their intersection point.
        
        Args:
            l1: First line
            l2: Second line
            
        Returns:
            Intersection point
        """
        # Extract 3D coordinates for cross product
        l1_3d = l1[..., :3]
        l2_3d = l2[..., :3]
        
        # Use cross product to get intersection point
        point = torch.linalg.cross(l1_3d, l2_3d, dim=-1)
        
        # Add homogeneous coordinate
        if l1.size(-1) > 3:
            point = torch.cat([point, torch.ones_like(point[..., :1])], dim=-1)
            
        return self.normalize_points(point)
        
    def get_ideal_points(self, line: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the ideal points on a line.
        
        These are the points where the line intersects the absolute conic.
        
        Args:
            line: Line to find ideal points on
            
        Returns:
            Tuple of two ideal points
        """
        # Get coefficients of line
        a, b, c = line[..., :3].unbind(-1)
        
        # Solve quadratic equation ax^2 + by^2 + c = 0
        discriminant = torch.sqrt(b**2 - 4*a*c + 1e-8)
        x1 = (-b + discriminant) / (2*a + 1e-8)
        x2 = (-b - discriminant) / (2*a + 1e-8)
        
        # Create homogeneous coordinates
        p1 = torch.stack([x1, torch.ones_like(x1), torch.zeros_like(x1)], dim=-1)
        p2 = torch.stack([x2, torch.ones_like(x2), torch.zeros_like(x2)], dim=-1)
        
        # Add homogeneous coordinate if needed
        if line.size(-1) > 3:
            p1 = torch.cat([p1, torch.ones_like(p1[..., :1])], dim=-1)
            p2 = torch.cat([p2, torch.ones_like(p2[..., :1])], dim=-1)
            
        return self.normalize_points(p1), self.normalize_points(p2)
        
    def cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Compute the cross-ratio of four points.
        
        Args:
            p1: First point
            p2: Second point
            p3: Third point
            p4: Fourth point
            
        Returns:
            Cross-ratio value
        """
        # Compute distances using dot products
        d12 = torch.sum(p1 * p2, dim=-1)
        d34 = torch.sum(p3 * p4, dim=-1)
        d13 = torch.sum(p1 * p3, dim=-1)
        d24 = torch.sum(p2 * p4, dim=-1)
        
        # Compute cross-ratio
        return (d12 * d34) / (d13 * d24 + 1e-8)
        
    def projective_average(self, points: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted average in projective space.
        
        Args:
            points: Points to average (shape: [num_points, *point_dims])
            weights: Optional weights for each point (shape: [num_points])
            
        Returns:
            Weighted average point
        """
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)
            weights = weights / weights.sum()
            
        # Normalize weights
        weights = weights / weights.sum()
        
        # Reshape weights for broadcasting
        weights = weights.view(-1, *([1] * (points.dim() - 1)))
        
        # Compute weighted sum
        avg = torch.sum(points * weights, dim=0)
        
        return self.normalize_points(avg)
        
    def absolute_polar(self, line: torch.Tensor) -> torch.Tensor:
        """Get the polar of a line with respect to the absolute conic.
        
        Args:
            line: Line to find polar of
            
        Returns:
            Polar point
        """
        # For the absolute conic x^2 + y^2 = z^2, the polar is simple
        return self.normalize_points(line)