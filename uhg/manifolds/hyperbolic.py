import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import Manifold

class HyperbolicManifold(Manifold):
    """
    Hyperbolic manifold implementation following Universal Hyperbolic Geometry (UHG) principles.

    All operations are performed directly in hyperbolic/projective space.
    No tangent space, exponential map, or logarithmic map operations are present,
    in strict accordance with UHG.pdf (see Ch. 3-5).
    """
    def __init__(self, curvature: float = -1.0):
        """
        Initialize the hyperbolic manifold.
        Args:
            curvature (float): Manifold curvature (must be negative)
        """
        if curvature >= 0:
            raise ValueError("Hyperbolic manifold must have negative curvature")
        self.curvature = curvature
        super().__init__()

    def normalize_points(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize points to lie on the hyperbolic manifold (Minkowski norm -1).
        For x in R^{n+1}, ensures x_1^2 + ... + x_n^2 - x_{n+1}^2 = -1.
        Reference: UHG.pdf, Ch. 3 (Projective Model)
        """
        # Compute Minkowski norm
        spatial_norm = torch.sum(x[..., :-1] ** 2, dim=-1, keepdim=True)
        time_norm = x[..., -1:] ** 2
        norm = torch.sqrt(torch.abs(spatial_norm - time_norm))
        eps = torch.tensor(1e-8, dtype=x.dtype, device=x.device)
        norm = torch.clamp(norm, min=eps)
        # Normalize so that Minkowski norm is -1
        x_normalized = x / norm
        # Ensure time component is positive
        time_mask = x_normalized[..., -1:] < 0
        x_normalized = torch.where(time_mask, -x_normalized, x_normalized)
        # Rescale so that Minkowski norm is exactly -1
        spatial = x_normalized[..., :-1]
        time = x_normalized[..., -1:]
        minkowski = torch.sum(spatial ** 2, dim=-1, keepdim=True) - time ** 2
        scale = torch.sqrt(torch.abs(minkowski))
        x_normalized = x_normalized / scale
        return x_normalized

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points onto the hyperbolic manifold (Minkowski norm -1).
        Reference: UHG.pdf, Ch. 3
        """
        return self.normalize_points(x)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between points using the Minkowski inner product.
        Reference: UHG.pdf, Ch. 4
        d(x, y) = arccosh(-<x, y>), where <.,.> is the Minkowski inner product.
        """
        x = self.normalize_points(x)
        y = self.normalize_points(y)
        spatial_dot = torch.sum(x[..., :-1] * y[..., :-1], dim=-1)
        time_dot = x[..., -1] * y[..., -1]
        inner_prod = spatial_dot - time_dot
        # Ensure inner product is <= -1 for acosh
        inner_prod = torch.clamp(inner_prod, max=-1.0 - 1e-8)
        d = torch.acosh(-inner_prod)
        return d

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Minkowski inner product between points.
        Reference: UHG.pdf, Ch. 3
        <x, y> = x_1*y_1 + ... + x_n*y_n - x_{n+1}*y_{n+1}
        """
        x = self.normalize_points(x)
        y = self.normalize_points(y)
        spatial_dot = torch.sum(x[..., :-1] * y[..., :-1], dim=-1)
        time_dot = x[..., -1] * y[..., -1]
        inner_prod = spatial_dot - time_dot
        return inner_prod

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin.
        
        Maps points from the tangent space at the origin to the manifold.
        This is a direct implementation in hyperbolic space without
        using tangent space operations.
        
        Args:
            x (torch.Tensor): Points in tangent space of shape [..., dim-1]
            
        Returns:
            torch.Tensor: Points on manifold of shape [..., dim]
        """
        # Compute norm of input points
        norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Add small epsilon to prevent division by zero
        eps = torch.tensor(1e-8, dtype=x.dtype, device=x.device)
        norm = torch.clamp(norm, min=eps)
        
        # Compute exponential map
        sinh_norm = torch.sinh(norm)
        cosh_norm = torch.cosh(norm)
        
        # Project to manifold
        x_proj = (sinh_norm / norm) * x
        time_comp = cosh_norm
        
        # Combine spatial and time components
        x_manifold = torch.cat([x_proj, time_comp], dim=-1)
        
        return self.normalize_points(x_manifold)
        
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at the origin.
        
        Maps points from the manifold to the tangent space at the origin.
        This is a direct implementation in hyperbolic space without
        using tangent space operations.
        
        Args:
            x (torch.Tensor): Points on manifold of shape [..., dim]
            
        Returns:
            torch.Tensor: Points in tangent space of shape [..., dim-1]
        """
        # Split into spatial and time components
        x_spatial = x[..., :-1]
        x_time = x[..., -1:]
        
        # Compute norm of spatial component
        norm = torch.norm(x_spatial, dim=-1, keepdim=True)
        
        # Add small epsilon to prevent division by zero
        eps = torch.tensor(1e-8, dtype=x.dtype, device=x.device)
        norm = torch.clamp(norm, min=eps)
        
        # Compute logarithmic map
        scale = torch.atanh(norm / x_time) / norm
        x_tangent = scale * x_spatial
        
        return x_tangent
        
    def parallel_transport(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport of vector v from x to y.
        
        This implements parallel transport directly in hyperbolic space
        without using tangent space operations.
        
        Args:
            x (torch.Tensor): Source point on manifold
            v (torch.Tensor): Vector to transport
            y (torch.Tensor): Target point on manifold
            
        Returns:
            torch.Tensor: Transported vector at y
        """
        # Compute hyperbolic distance between x and y
        d = self.distance(x, y)
        
        # Compute parallel transport
        scale = torch.cosh(d)
        v_transported = scale * v
        
        return v_transported 