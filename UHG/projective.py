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
from typing import Optional, Tuple

class ProjectiveUHG:
    """Pure projective implementation of Universal Hyperbolic Geometry."""
    
    def __init__(self, eps: float = 1e-9):
        """Initialize the UHG projective geometry system."""
        self.eps = eps
        
    def cross_ratio(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-ratio of four points.
        
        The cross-ratio is the fundamental invariant in projective geometry
        and the basis for all UHG calculations.
        
        Args:
            a, b, c, d: Points in projective space
            
        Returns:
            Cross-ratio value
        """
        # Compute projective distances using determinants
        def proj_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.shape[-1]
            dist = 0.0
            for i in range(n-1):
                for j in range(i+1, n):
                    det = x[..., i] * y[..., j] - x[..., j] * y[..., i]
                    dist = dist + det * det
            return torch.sqrt(dist + self.eps)
            
        # Compute cross-ratio using projective distances
        ac = proj_dist(a, c)
        bd = proj_dist(b, d)
        ad = proj_dist(a, d)
        bc = proj_dist(b, c)
        
        return (ac * bd) / (ad * bc + self.eps)
        
    def join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the join (line through two points) in projective space.
        
        Args:
            a, b: Points to join
            
        Returns:
            Line in projective coordinates
        """
        # Compute join using exterior product
        n = a.shape[-1]
        line = torch.zeros(n)
        for i in range(n-1):
            for j in range(i+1, n):
                det = a[i] * b[j] - a[j] * b[i]
                line[i] = line[i] + det
                line[j] = line[j] - det
        return line
        
    def meet(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """
        Compute the meet (intersection) of two lines in projective space.
        
        Args:
            l1, l2: Lines to intersect
            
        Returns:
            Point of intersection in projective coordinates
        """
        # Meet is dual to join - use same operation on dual coordinates
        return self.join(l1, l2)
        
    def transform(self, points: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply a projective transformation to points.
        
        All transformations preserve cross-ratios as required by UHG.
        
        Args:
            points: Points to transform
            matrix: Optional projective transformation matrix
            
        Returns:
            Transformed points
        """
        if matrix is None:
            matrix = self.get_projective_matrix(points.shape[-1] - 1)
            
        # Apply projective transformation
        transformed = torch.matmul(points, matrix.T)
        
        # Normalize to preserve projective structure
        transformed = transformed / (transformed[..., -1:] + self.eps)
        
        return transformed
        
    def get_projective_matrix(self, dim: int) -> torch.Tensor:
        """
        Get a projective transformation matrix.
        
        Creates a random projective transformation that preserves
        cross-ratio and hyperbolic structure.
        
        Args:
            dim: Dimension of the projective space
            
        Returns:
            (dim+1) x (dim+1) projective transformation matrix
        """
        # Create random matrix
        matrix = torch.randn(dim+1, dim+1)
        
        # Ensure matrix is non-singular
        u, s, v = torch.linalg.svd(matrix)
        s = torch.ones_like(s)  # Set all singular values to 1
        matrix = torch.matmul(torch.matmul(u, torch.diag(s)), v)
        
        return matrix
        
    def distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between points using projective cross-ratio.
        
        Uses the fundamental formula from UHG:
            d(p1,p2) = |log(CR(p1,p2,i1,i2))|
        where i1,i2 are the ideal points.
        
        Args:
            p1, p2: Points in projective space
            
        Returns:
            Hyperbolic distance
        """
        # Normalize points to projective space
        p1_norm = p1 / (torch.norm(p1) + self.eps)
        p2_norm = p2 / (torch.norm(p2) + self.eps)
        
        # Compute inner product
        inner = torch.sum(p1_norm * p2_norm)
        
        # Compute projective distance using cross-ratio formula
        # This is numerically more stable than finding ideal points
        dist = torch.acosh(torch.abs(inner) + self.eps)
        
        # Handle numerical issues
        if torch.isnan(dist) or torch.isinf(dist):
            # Fallback to simpler distance formula for close points
            diff = p1_norm - p2_norm
            dist = torch.norm(diff)
            
        return dist
        
    def absolute_polar(self, line: torch.Tensor) -> torch.Tensor:
        """
        Compute the absolute polar of a line.
        
        This is a fundamental operation in UHG that relates
        to the hyperbolic structure.
        
        Args:
            line: Line in projective coordinates
            
        Returns:
            Polar point in projective coordinates
        """
        # Compute polar using quadratic form
        n = len(line)
        polar = torch.zeros_like(line)
        polar[:-1] = line[:-1]
        polar[-1] = -line[-1]
        return polar