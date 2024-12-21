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
from torch import Tensor

class ProjectiveUHG:
    """
    Universal Hyperbolic Geometry operations in projective space.
    All operations follow the mathematical definitions from UHG.pdf.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def wedge(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute the wedge product between two vectors.
        a∧b = a₁b₂ - a₂b₁ for 2D projective space
        
        Args:
            a: First vector tensor of shape (..., 3)
            b: Second vector tensor of shape (..., 3)
            
        Returns:
            Wedge product tensor of shape (...)
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Compute wedge product components
        w12 = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        w23 = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
        w31 = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
        
        return torch.stack([w23, w31, w12], dim=-1)
    
    def quadrance(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute quadrance between two points in projective space.
        q(a1,a2) = 1 - (x₁x₂ + y₁y₂ - z₁z₂)² / ((x₁² + y₁² - z₁²)(x₂² + y₂² - z₂²))
        
        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            
        Returns:
            Quadrance between points
            
        Notes:
            - Undefined if either point is null (x² + y² - z² = 0)
            - Equal to 1 when points are perpendicular
            - Equal to 0 when points are same or form a null line
            - For hyperbolic points, norms should be negative
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Check for null points first
        norm_a = a[..., 0]**2 + a[..., 1]**2 - a[..., 2]**2
        norm_b = b[..., 0]**2 + b[..., 1]**2 - b[..., 2]**2
        
        # Points should be non-null and have negative norms for hyperbolic space
        if torch.any(torch.abs(norm_a) < self.epsilon) or torch.any(torch.abs(norm_b) < self.epsilon):
            raise ValueError("Quadrance is undefined for null points")
            
        if torch.any(norm_a > -self.epsilon) or torch.any(norm_b > -self.epsilon):
            raise ValueError("Points must have negative norms in hyperbolic space")
        
        # Compute inner product exactly as in formula
        inner_prod = a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1] - a[..., 2]*b[..., 2]
        
        # For hyperbolic points with negative norms, we don't need absolute values
        # The denominator will be positive because it's the product of two negative norms
        return 1.0 - (inner_prod**2) / (norm_a * norm_b)
    
    def spread(self, l1: Tensor, l2: Tensor) -> Tensor:
        """
        Compute spread between two lines in projective space.
        S(l,m) = (l₁m₁ + l₂m₂ - l₃m₃)² / ((l₁² + l₂² - l₃²)(m₁² + m₂² - m₃²))
        """
        if l1.shape[-1] != 3 or l2.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Normalize inputs for numerical stability
        l1 = l1 / (torch.norm(l1, dim=-1, keepdim=True) + self.epsilon)
        l2 = l2 / (torch.norm(l2, dim=-1, keepdim=True) + self.epsilon)
        
        # Split coordinates
        l1_xy = l1[..., :2]
        l1_z = l1[..., 2]
        l2_xy = l2[..., :2]
        l2_z = l2[..., 2]
        
        # Compute numerator using hyperbolic inner product
        dot_xy = torch.sum(l1_xy * l2_xy, dim=-1)
        numerator = (dot_xy - l1_z * l2_z) ** 2
        
        # Compute denominators using hyperbolic norm
        denom_l1 = torch.clamp(torch.sum(l1_xy * l1_xy, dim=-1) - l1_z * l1_z, min=self.epsilon)
        denom_l2 = torch.clamp(torch.sum(l2_xy * l2_xy, dim=-1) - l2_z * l2_z, min=self.epsilon)
        denominator = denom_l1 * denom_l2
        
        # Compute spread with proper scaling
        s = numerator / denominator
        return 1.0 - torch.clamp(s, min=0.0, max=1.0)
    
    def cross_ratio(self, v1: Tensor, v2: Tensor, u1: Tensor, u2: Tensor) -> Tensor:
        """
        Compute cross-ratio of four vectors in projective space.
        CR(v₁,v₂;u₁,u₂) = (z₁w₂)/(w₁z₂) where u₁ = z₁v₁ + w₁v₂, u₂ = z₂v₁ + w₂v₂
        """
        if not all(t.shape[-1] == 3 for t in [v1, v2, u1, u2]):
            raise ValueError("All input tensors must have shape (..., 3)")
            
        # Normalize inputs for numerical stability
        v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + self.epsilon)
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + self.epsilon)
        u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + self.epsilon)
        u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + self.epsilon)
            
        # Create matrices for solving systems
        A = torch.stack([v1, v2], dim=-1)  # Shape: (..., 3, 2)
        b1 = u1.unsqueeze(-1)  # Shape: (..., 3, 1)
        b2 = u2.unsqueeze(-1)  # Shape: (..., 3, 1)
        
        # Solve using updated linalg.solve
        ATA = torch.matmul(A.transpose(-2, -1), A)
        ATb1 = torch.matmul(A.transpose(-2, -1), b1)
        ATb2 = torch.matmul(A.transpose(-2, -1), b2)
        
        # Add regularization for stability
        reg = self.epsilon * torch.eye(2, device=ATA.device).expand_as(ATA)
        ATA = ATA + reg
        
        # Solve systems using updated torch.linalg.solve
        coeff1 = torch.linalg.solve(ATA, ATb1)
        coeff2 = torch.linalg.solve(ATA, ATb2)
        
        z1, w1 = coeff1[..., 0, 0], coeff1[..., 1, 0]
        z2, w2 = coeff2[..., 0, 0], coeff2[..., 1, 0]
        
        # Compute cross-ratio with stabilization
        numerator = z1 * w2
        denominator = (w1 * z2).clamp(min=self.epsilon)
        
        # Normalize the result to handle large values
        result = numerator / denominator
        scale = torch.max(torch.abs(result))
        if scale > 1.0:
            result = result / scale
        
        return result
    
    def opposite_points(self, a1: Tensor, a2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute opposite points for two points in projective space.
        Returns o₁, o₂ where:
        o₁ = (a₁·a₂)a₁ - (a₁·a₁)a₂
        o₂ = (a₁·a₂)a₁ - (a₁·a₂)a₂
        """
        if a1.shape[-1] != 3 or a2.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Normalize inputs for numerical stability
        a1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + self.epsilon)
        a2 = a2 / (torch.norm(a2, dim=-1, keepdim=True) + self.epsilon)
        
        # Split coordinates
        a1_xy = a1[..., :2]
        a1_z = a1[..., 2]
        a2_xy = a2[..., :2]
        a2_z = a2[..., 2]
        
        # Compute dot products with stabilization
        dot_12 = torch.sum(a1_xy * a2_xy, dim=-1, keepdim=True) - a1_z.unsqueeze(-1) * a2_z.unsqueeze(-1)
        dot_11 = torch.sum(a1_xy * a1_xy, dim=-1, keepdim=True) - a1_z.unsqueeze(-1) * a1_z.unsqueeze(-1)
        dot_22 = torch.sum(a2_xy * a2_xy, dim=-1, keepdim=True) - a2_z.unsqueeze(-1) * a2_z.unsqueeze(-1)
        
        # Compute opposite points
        o1 = dot_12 * a1 - dot_11 * a2
        o2 = dot_22 * a1 - dot_12 * a2
        
        # Normalize outputs
        o1 = o1 / (torch.norm(o1, dim=-1, keepdim=True) + self.epsilon)
        o2 = o2 / (torch.norm(o2, dim=-1, keepdim=True) + self.epsilon)
        
        return o1, o2
    
    def normalize(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Normalize vectors to lie in projective space.
        Handles both 2D->3D projection and normalization.
        """
        if x.shape[dim] not in [2, 3]:
            raise ValueError("Input tensor must have 2 or 3 components in normalization dimension")
            
        # Project 2D points to 3D if necessary
        if x.shape[dim] == 2:
            # Add third coordinate as 1
            shape = list(x.shape)
            shape[dim] = 1
            ones = torch.ones(shape, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=dim)
            
        # Compute normalization factor with stabilization
        norm = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + self.epsilon)
        
        return x / norm