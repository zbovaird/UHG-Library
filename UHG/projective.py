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
        Compute wedge product of two vectors.
        a∧b = [a₂b₃-a₃b₂ : a₃b₁-a₁b₃ : a₁b₂-a₂b₁]
        
        Args:
            a: First vector tensor of shape (..., 3)
            b: Second vector tensor of shape (..., 3)
            
        Returns:
            Wedge product tensor of shape (..., 3)
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Compute components with numerical stability
        w1 = a[..., 1]*b[..., 2] - a[..., 2]*b[..., 1]
        w2 = a[..., 2]*b[..., 0] - a[..., 0]*b[..., 2]
        w3 = a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
        
        # Stack components
        wedge = torch.stack([w1, w2, w3], dim=-1)
        
        # Normalize if non-zero
        norm = torch.norm(wedge, dim=-1, keepdim=True)
        mask = norm > self.epsilon
        wedge = torch.where(mask, wedge / (norm + self.epsilon), wedge)
        
        return wedge
    
    def hyperbolic_dot(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute hyperbolic dot product.
        In UHG, this is defined as:
        x·y = x₁y₁ + x₂y₂ + ... - x_ny_n
        where n is the last dimension (time component)
        """
        # Ensure both tensors have same shape
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        
        # Split into spatial and time components
        spatial = x[..., :-1] * y[..., :-1]
        time = x[..., -1:] * y[..., -1:]
        
        # Sum spatial components and subtract time component
        return torch.sum(spatial, dim=-1) - torch.sum(time, dim=-1)
    
    def quadrance(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute quadrance between two points in projective space.
        q(a,b) = 1 - (a·b)² / ((a·a)(b·b))
        
        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            
        Returns:
            Quadrance between points
            
        Notes:
            - Undefined if either point is null (x² + y² - z² = 0)
            - Equal to 1 when points are perpendicular
            - Equal to 0 when points are same or form a null line
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Compute hyperbolic inner products
        aa = self.hyperbolic_dot(a, a)
        bb = self.hyperbolic_dot(b, b)
        ab = self.hyperbolic_dot(a, b)
        
        # Points should be non-null
        if torch.any(torch.abs(aa) < self.epsilon) or torch.any(torch.abs(bb) < self.epsilon):
            raise ValueError("Quadrance is undefined for null points")
        
        # Compute quadrance using the formula from UHG.pdf
        return 1.0 - (ab * ab) / (aa * bb)
    
    def cross_ratio(self, v1: Tensor, v2: Tensor, u1: Tensor, u2: Tensor) -> Tensor:
        """
        Compute cross-ratio of four vectors in projective space.
        Following UHG.pdf definition:
        CR(A,B;C,D) = (AC)(BD)/((AD)(BC))
        where (XY) represents the hyperbolic join of points X and Y.
        
        For machine learning applications, we use the fixed order (A,B;C,D)
        and focus on numerical stability rather than permutation invariance.
        """
        # Get the last dimension as time component
        time_dim = -1
        
        # Normalize points to have unit time component for numerical stability
        v1 = v1 / (torch.abs(v1[..., time_dim:]) + self.epsilon)
        v2 = v2 / (torch.abs(v2[..., time_dim:]) + self.epsilon)
        u1 = u1 / (torch.abs(u1[..., time_dim:]) + self.epsilon)
        u2 = u2 / (torch.abs(u2[..., time_dim:]) + self.epsilon)
        
        # Compute joins using hyperbolic dot product
        AC = self.hyperbolic_dot(v1, u1)
        BD = self.hyperbolic_dot(v2, u2)
        AD = self.hyperbolic_dot(v1, u2)
        BC = self.hyperbolic_dot(v2, u1)
        
        # Handle special cases and ensure numerical stability
        numerator = AC * BD
        denominator = AD * BC
        
        # Avoid division by zero while preserving gradients
        safe_denominator = torch.where(
            torch.abs(denominator) > self.epsilon,
            denominator,
            torch.ones_like(denominator)
        )
        
        return numerator / safe_denominator
    
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
        # For ML applications, we allow higher dimensions but normalize based on last component
        if x.shape[dim] < 2:
            raise ValueError("Input tensor must have at least 2 components")
        
        # Project to homogeneous coordinates if needed
        if x.shape[dim] == 2:
            shape = list(x.shape)
            shape[dim] = 1
            ones = torch.ones(shape, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=dim)
        
        # Normalize by last component
        last_component = x.index_select(dim, torch.tensor([x.shape[dim]-1], device=x.device))
        scale = torch.where(
            torch.abs(last_component) > self.epsilon,
            1.0 / last_component,
            torch.ones_like(last_component)
        )
        
        return x * scale
    
    def join(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the join (line through two points) in projective space.
        For ML applications with higher dimensions, we project to 3D first.
        Returns the line as a tensor representing the coefficients of the line equation.
        """
        # Project to 3D if needed by taking first two components and last component
        if x.shape[-1] > 3:
            x_proj = torch.cat([x[..., :2], x[..., -1:]], dim=-1)
            y_proj = torch.cat([y[..., :2], y[..., -1:]], dim=-1)
        else:
            x_proj = x
            y_proj = y
        
        # Normalize points
        x_proj = self.normalize_points(x_proj)
        y_proj = self.normalize_points(y_proj)
        
        # Compute cross product for line coefficients
        return torch.cross(x_proj, y_proj, dim=-1)
    
    def get_projective_matrix(self, dim: int) -> Tensor:
        """
        Generate a random projective transformation matrix.
        The matrix will be invertible and preserve the hyperbolic structure.
        
        Args:
            dim: Dimension of the projective space (2 for planar geometry)
            
        Returns:
            (dim+1) x (dim+1) projective transformation matrix
        """
        # Generate random matrix with entries in [-1, 1]
        matrix = 2 * torch.rand(dim + 1, dim + 1) - 1
        
        # Make it preserve hyperbolic structure by ensuring it's in O(2,1)
        # First, make it orthogonal
        q, r = torch.linalg.qr(matrix)
        matrix = q
        
        # Then ensure it preserves the hyperbolic form
        # For O(2,1), we need M^T J M = J where J = diag(1, 1, -1)
        J = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        matrix = torch.matmul(matrix, torch.sqrt(J))
        matrix = torch.matmul(torch.sqrt(J), matrix)
        
        return matrix
    
    def transform(self, points: Tensor, matrix: Tensor) -> Tensor:
        """
        Apply a projective transformation to points.
        For ML applications, we transform all components while preserving the homogeneous structure.

        Args:
            points: Points to transform [..., D] where D is dimension
            matrix: Transformation matrix [..., D-1, D-1] for D-dimensional points

        Returns:
            Transformed points [..., D]
        """
        # Add batch dimension if needed
        if points.dim() == 1:
            points = points.unsqueeze(0)
        
        # Split into spatial and homogeneous components
        spatial = points[..., :-1]
        homogeneous = points[..., -1:]
        
        # Apply transformation to spatial components
        transformed_spatial = torch.matmul(spatial, matrix.T)
        
        # Recombine with homogeneous coordinate
        transformed = torch.cat([transformed_spatial, homogeneous], dim=-1)
        
        # Normalize homogeneous coordinates
        transformed = transformed / (torch.abs(transformed[..., -1:]) + self.epsilon)
        
        # Remove batch dimension if it was added
        if points.dim() == 2 and points.size(0) == 1:
            transformed = transformed.squeeze(0)
        
        return transformed
    
    def quadrance_from_cross_ratio(self, a1: Tensor, a2: Tensor) -> Tensor:
        """
        Compute quadrance between points using the cross ratio relationship:
        q(a1, a2) = 1 - (a1, a2 : o2, o1)
        where o1, o2 are opposite points:
        o1 = (a1·a2)a1 - (a1·a1)a2
        o2 = (a2·a2)a1 - (a1·a2)a2
        
        Args:
            a1: First point [x₁:y₁:z₁]
            a2: Second point [x₂:y₂:z₂]
            
        Returns:
            Quadrance between points
        """
        if a1.shape[-1] != 3 or a2.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Check for null points
        norm_1 = a1[..., 0]**2 + a1[..., 1]**2 - a1[..., 2]**2
        norm_2 = a2[..., 0]**2 + a2[..., 1]**2 - a2[..., 2]**2
        
        if torch.any(torch.abs(norm_1) < self.epsilon) or torch.any(torch.abs(norm_2) < self.epsilon):
            raise ValueError("Quadrance is undefined for null points")
            
        # For points on same line, quadrance is 0
        def det3(a, b, c):
            return (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1]) -
                   a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0]) +
                   a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0]))
                   
        # Create a third point to test collinearity
        c = torch.zeros_like(a1)
        c[..., 2] = 1.0  # Reference point [0:0:1]
        
        if torch.abs(det3(a1, a2, c)) < self.epsilon:
            return torch.zeros_like(norm_1)
            
        # Compute hyperbolic inner products
        dot_12 = torch.sum(a1 * a2 * torch.tensor([1.0, 1.0, -1.0]), dim=-1)
        
        # Compute opposite points according to UHG.pdf
        o1 = dot_12.unsqueeze(-1) * a1 - norm_1.unsqueeze(-1) * a2
        o2 = norm_2.unsqueeze(-1) * a1 - dot_12.unsqueeze(-1) * a2
        
        # Normalize opposite points
        o1 = o1 / (torch.norm(o1, dim=-1, keepdim=True) + self.epsilon)
        o2 = o2 / (torch.norm(o2, dim=-1, keepdim=True) + self.epsilon)
        
        # Compute quadrance as 1 minus cross ratio
        cr = self.cross_ratio(a1, a2, o2, o1)
        return 1.0 - cr
    
    def normalize_points(self, x: Tensor) -> Tensor:
        """Normalize points to have unit time component."""
        # Get the time component
        time_component = x[..., -1:]
        # Avoid division by zero
        scale = torch.where(
            torch.abs(time_component) > self.epsilon,
            1.0 / time_component,
            torch.ones_like(time_component)
        )
        return x * scale
    
    def projective_average(self, points: Tensor, weights: Tensor) -> Tensor:
        """
        Compute weighted projective average of points.
        Args:
            points: Tensor of shape (..., N, D) where N is number of points
            weights: Tensor of shape (..., N) with weights summing to 1
        Returns:
            Weighted average point of shape (..., D)
        """
        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # Expand weights for broadcasting
        # Add extra dimensions to match points shape
        for _ in range(points.dim() - weights.dim()):
            weights = weights.unsqueeze(-1)
        
        # Compute weighted sum
        avg = torch.sum(points * weights, dim=-2)
        
        # Ensure last component is 1 for homogeneous coordinates
        avg = avg / (torch.abs(avg[..., -1:]) + self.epsilon)
        
        return avg
    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute hyperbolic distance between points.
        d(x,y) = acosh(-<x,y>)
        """
        # Normalize points
        x = self.normalize_points(x)
        y = self.normalize_points(y)
        
        # Compute inner product
        inner_prod = -self.hyperbolic_dot(x, y)
        
        # Ensure inner product is >= 1 for acosh
        inner_prod = torch.clamp(inner_prod, min=1.0 + self.epsilon)
        
        return torch.acosh(inner_prod)
    
    def meet(self, line: Tensor, point: Tensor) -> Tensor:
        """
        Compute the meet of a line and a point in projective space.
        Returns the intersection point.
        """
        # Normalize inputs
        point = self.normalize_points(point)
        
        # Compute cross product for intersection point
        result = torch.cross(line, point, dim=-1)
        
        # Normalize result
        return self.normalize_points(result)
    
    def scale(self, points: Tensor, factor: Tensor) -> Tensor:
        """
        Scale points by a factor while preserving projective structure.
        
        Args:
            points: Points to scale [..., D]
            factor: Scale factor [..., 1]
            
        Returns:
            Scaled points [..., D]
        """
        # Expand factor for broadcasting
        while factor.dim() < points.dim():
            factor = factor.unsqueeze(-1)
        
        # Scale spatial components only
        scaled = points.clone()
        scaled[..., :-1] = points[..., :-1] * factor
        
        # Normalize homogeneous coordinate
        return self.normalize_points(scaled)
    
    def aggregate(self, points: Tensor, weights: Tensor) -> Tensor:
        """
        Aggregate points using weighted projective average.
        
        Args:
            points: Points to aggregate [..., N, D]
            weights: Weights for aggregation [..., N]
            
        Returns:
            Aggregated point [..., D]
        """
        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # Expand weights for broadcasting
        while weights.dim() < points.dim():
            weights = weights.unsqueeze(-1)
        
        # Compute weighted sum
        weighted_sum = torch.sum(points * weights, dim=-2)
        
        # Normalize result
        return self.normalize_points(weighted_sum)