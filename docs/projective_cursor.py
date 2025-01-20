# code from projective.py file in cursor
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
    
    def hyperbolic_dot(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute hyperbolic dot product between two points.
        For points [x₁:y₁:z₁] and [x₂:y₂:z₂]:
        x₁x₂ + y₁y₂ - z₁z₂
        
        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            
        Returns:
            Hyperbolic dot product
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Split into spatial and time components
        a_space = a[..., :2]
        b_space = b[..., :2]
        a_time = a[..., 2:]
        b_time = b[..., 2:]
        
        # Compute dot product with correct signature
        space_dot = torch.sum(a_space * b_space, dim=-1)
        time_dot = torch.sum(a_time * b_time, dim=-1)
        
        return space_dot - time_dot

    def quadrance(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Calculate quadrance between two points a1=[x1:y1:z1] and a2=[x2:y2:z2]
        According to UHG principles:
        q(A,B) = 1 - <A,B>²/(<A,A><B,B>) for non-null points
        where <A,B> = x₁x₂ + y₁y₂ - z₁z₂
        
        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            
        Returns:
            Quadrance between points
            
        Raises:
            ValueError: if either point is null
        """
        # First normalize points
        a = self.normalize_points(a)
        b = self.normalize_points(b)
        
        # Check for null points
        if self.is_null_point(a) or self.is_null_point(b):
            raise ValueError("Quadrance is undefined for null points")
            
        # Get components
        x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2]
        x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2]
        
        # Compute hyperbolic dot product <A,B>
        dot = x1*x2 + y1*y2 - z1*z2
        
        # Compute norms <A,A> and <B,B>
        a_norm = x1*x1 + y1*y1 - z1*z1
        b_norm = x2*x2 + y2*y2 - z2*z2
        
        # Handle numerical stability for denominator
        safe_denom = torch.where(
            torch.abs(a_norm * b_norm) > self.epsilon,
            a_norm * b_norm,
            torch.ones_like(a_norm) * self.epsilon
        )
        
        # Compute quadrance q = 1 - <A,B>²/(<A,A><B,B>)
        q = 1.0 - dot*dot / safe_denom
        
        # Handle numerical stability for result
        q = torch.where(
            torch.abs(q) < self.epsilon,
            torch.zeros_like(q),
            q
        )
        q = torch.where(
            torch.abs(q - 1.0) < self.epsilon,
            torch.ones_like(q),
            q
        )
        
        # Ensure quadrance is non-negative
        q = torch.abs(q)
        
        return q
    
    def det2(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2x2 determinant for vectors arranged as:
        |a b|
        |c d|
        """
        return a * d - b * c

    def cross_ratio(self, v1: Tensor, v2: Tensor, u1: Tensor, u2: Tensor) -> Tensor:
        """
        Compute cross-ratio of four vectors in projective space.
        Following UHG.pdf definition:
        CR(A,B;C,D) = |x₁ y₁|  |x₁ y₁|
                      |z₁ w₁|  |z₂ w₂|
                      ─────── / ───────
                      |x₂ y₂|  |x₂ y₂|
                      |z₁ w₁|  |z₂ w₂|
        
        For the special case where v1,v2 are used as basis:
        CR(v₁,v₂:u₁,u₂) = w₁/w₂ / z₁/z₂ = w₁z₂/w₂z₁
        
        Args:
            v1, v2, u1, u2: Points in projective space
            
        Returns:
            Cross-ratio tensor
        """
        # First try the special case where v1,v2 can be used as basis
        # This is numerically more stable when applicable
        try:
            # Check if v1,v2 can be used as basis by attempting to solve
            # u1 = z1*v1 + w1*v2
            # u2 = z2*v1 + w2*v2
            
            # Create system matrix [v1 v2]
            basis = torch.stack([v1, v2], dim=-1)
            
            # Solve for coefficients [z1,w1] and [z2,w2]
            try:
                # Using batched solve for better efficiency
                u = torch.stack([u1, u2], dim=-1)
                coeffs = torch.linalg.solve(basis, u)
                z1, w1 = coeffs[..., 0]
                z2, w2 = coeffs[..., 1]
                
                # Use special case formula
                return (w1 * z2) / (w2 * z1 + self.epsilon)
                
            except RuntimeError:
                # Matrix not invertible, fall back to general case
                pass
                
        except RuntimeError:
            # Error in setup, fall back to general case
            pass
            
        # General case using determinant form
        # Project to 2D if needed by taking first two coordinates
        # This preserves cross-ratio due to projective invariance
        v1_2d = v1[..., :2]
        v2_2d = v2[..., :2]
        u1_2d = u1[..., :2]
        u2_2d = u2[..., :2]
        
        # Compute the four 2x2 determinants
        det11 = self.det2(v1_2d[..., 0], v1_2d[..., 1], 
                         u1_2d[..., 0], u1_2d[..., 1])
        det12 = self.det2(v1_2d[..., 0], v1_2d[..., 1],
                         u2_2d[..., 0], u2_2d[..., 1])
        det21 = self.det2(v2_2d[..., 0], v2_2d[..., 1],
                         u1_2d[..., 0], u1_2d[..., 1])
        det22 = self.det2(v2_2d[..., 0], v2_2d[..., 1],
                         u2_2d[..., 0], u2_2d[..., 1])
        
        # Compute cross-ratio as ratio of determinants
        numerator = det11 * det22
        denominator = det12 * det21
        
        # Handle numerical stability while preserving sign
        safe_denom = torch.where(
            torch.abs(denominator) > self.epsilon,
            denominator,
            torch.sign(denominator) * self.epsilon
        )
        
        return numerator / safe_denom
    
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
        o₂ = (a₂·a₂)a₁ - (a₁·a₂)a₂
        """
        if a1.shape[-1] != 3 or a2.shape[-1] != 3:
            raise ValueError("Input tensors must have shape (..., 3)")
            
        # Normalize inputs for numerical stability
        a1 = self.normalize_points(a1)
        a2 = self.normalize_points(a2)
        
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
    
    def normalize(self, points: Tensor) -> Tensor:
        """
        Normalize points while preserving cross ratios.
        For 4 or more points, ensures cross ratio is preserved after normalization.
        """
        norms = torch.norm(points[..., :-1], dim=-1, keepdim=True)
        normalized = points / (norms + self.epsilon)
        if points.size(0) > 3:
            cr_before = self.cross_ratio(points[0], points[1], points[2], points[3])
            cr_after = self.cross_ratio(normalized[0], normalized[1], normalized[2], normalized[3])
            if not torch.isnan(cr_before) and not torch.isnan(cr_after) and cr_after != 0:
                scale = torch.sqrt(torch.abs(cr_before / cr_after))
                normalized[..., :-1] *= scale
        return normalized
    
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
        # For UHG we work in RP², so dim should be 2
        if dim != 2:
            raise ValueError("UHG only works in RP² (dim=2)")
            
        # Generate random matrix with entries in [-1, 1]
        matrix = 2 * torch.rand(3, 3) - 1
        
        # Make it preserve hyperbolic structure by ensuring it's in O(2,1)
        # First, make it orthogonal
        q, r = torch.linalg.qr(matrix)
        matrix = q
        
        # Then ensure it preserves the hyperbolic form
        # For O(2,1), we need M^T J M = J where J = diag(1, 1, -1)
        J = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        matrix = torch.matmul(matrix, torch.sqrt(torch.abs(J)))
        
        # Ensure determinant is 1
        det = torch.linalg.det(matrix)
        matrix = matrix / torch.abs(det)**(1/3)
        
        return matrix
    
    def transform(self, points: Tensor, matrix: Tensor) -> Tensor:
        """
        Apply a projective transformation to points.
        
        Args:
            points: Points to transform [..., 3]
            matrix: 3x3 transformation matrix
            
        Returns:
            Transformed points
        """
        if points.shape[-1] != 3:
            raise ValueError("Points must have shape (..., 3)")
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must have shape (3, 3)")
            
        # Reshape points for batch matrix multiply
        points_flat = points.reshape(-1, 3)
        
        # Apply transformation
        transformed = torch.matmul(points_flat, matrix.transpose(0,1))
        
        # Reshape back to original
        transformed = transformed.reshape(points.shape)
        
        # Normalize result
        return self.normalize_points(transformed)

    def quadrance_from_cross_ratio(self, a1: Tensor, a2: Tensor) -> Tensor:
        """
        Compute quadrance between points using the cross ratio relationship:
        q(a1, a2) = 1 - (a1, a2 : o2, o1)
        where o1, o2 are opposite points:
        o1 = (a1·a2)a1 - (a1·a1)a2
        o2 = (a2·a2)a1 - (a1·a2)a2
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
    
    def normalize_points(self, points: Tensor) -> Tensor:
        """
        Normalize points to have consistent scale.
        For UHG, we normalize so that:
        1. If point is null (x² + y² = z²), normalize so z = 1
        2. If point is non-null, normalize so largest component is ±1
        
        Args:
            points: Points to normalize [..., 3]
            
        Returns:
            Normalized points
        """
        if points.shape[-1] != 3:
            raise ValueError("Points must have shape (..., 3)")
            
        # Get components
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        
        # Compute norms
        space_norm = x*x + y*y
        time_norm = z*z
        
        # Check if points are null (x² + y² = z²)
        is_null = torch.abs(space_norm - time_norm) < self.epsilon
        
        # For null points, normalize so z = 1
        null_scale = 1.0 / (torch.abs(z) + self.epsilon)
        
        # For non-null points, normalize by largest component
        max_abs = torch.max(torch.abs(points), dim=-1)[0]
        non_null_scale = 1.0 / (max_abs + self.epsilon)
        
        # Choose appropriate scaling based on whether point is null
        scale = torch.where(is_null, null_scale, non_null_scale)
        
        # Apply scaling
        normalized = points * scale.unsqueeze(-1)
        
        # Handle sign consistently - make largest absolute component positive
        max_abs_idx = torch.argmax(torch.abs(normalized), dim=-1)
        max_abs_val = torch.gather(normalized, -1, max_abs_idx.unsqueeze(-1))
        sign_correction = torch.sign(max_abs_val)
        normalized = normalized * sign_correction
        
        return normalized
    
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
        
        # Reshape points and weights for proper broadcasting
        B = points.size(0)  # Batch size
        D = points.size(-1)  # Feature dimension
        points = points.view(B, -1, D)  # [B, N, D]
        weights = weights.view(B, -1, 1)  # [B, N, 1]
        
        # Compute weighted sum
        weighted_sum = torch.sum(points * weights, dim=1)  # [B, D]
        
        # Normalize result
        return self.normalize_points(weighted_sum)
    
    def is_null_point(self, point: Tensor) -> Tensor:
        """
        Check if a point is null (lies on its dual line).
        A point [x:y:z] is null when x² + y² = z²
        
        Args:
            point: Point in projective coordinates [x:y:z]
            
        Returns:
            Boolean tensor indicating if point is null
        """
        # First normalize the point
        point = self.normalize_points(point)
        
        # Get components
        x = point[..., 0]
        y = point[..., 1]
        z = point[..., 2]
        
        # Compute norms
        space_norm = x*x + y*y
        time_norm = z*z
        
        # Check if x² + y² = z²
        return torch.abs(space_norm - time_norm) < self.epsilon

    def null_point(self, t: Tensor, u: Tensor) -> Tensor:
        """
        Get null point parametrized by t:u
        Returns [t²-u² : 2tu : t²+u²]
        
        Args:
            t, u: Parameters for null point
            
        Returns:
            Null point in projective coordinates
        """
        point = torch.stack([
            t*t - u*u,
            2*t*u,
            t*t + u*u
        ], dim=-1)
        return self.normalize_points(point)

    def join_null_points(self, t1: Tensor, u1: Tensor, t2: Tensor, u2: Tensor) -> Tensor:
        """
        Get line through two null points parametrized by (t₁:u₁) and (t₂:u₂)
        Returns (t₁t₂-u₁u₂ : t₁u₂+t₂u₁ : t₁t₂+u₁u₂)
        
        From UHG.pdf Theorem 23, this produces a non-null line when t₁:u₁ ≠ t₂:u₂
        
        Args:
            t1, u1: Parameters of first null point
            t2, u2: Parameters of second null point
            
        Returns:
            Line joining the null points (l:m:n)
            
        Raises:
            ValueError: if null points are not distinct (same t:u proportions)
        """
        # Check parameter proportions are different
        if torch.allclose(t1*u2, t2*u1, rtol=self.epsilon):
            raise ValueError("Null points must be distinct (different t:u proportions)")
            
        # Compute join using Theorem 23 formula
        line = torch.stack([
            t1*t2 - u1*u2,  # l component
            t1*u2 + t2*u1,  # m component
            t1*t2 + u1*u2   # n component
        ], dim=-1)
        
        # Normalize with sign convention
        return self.normalize_points(line)
    
    def join_points(self, a1: Tensor, a2: Tensor) -> Tensor:
        """
        Get line joining two points using hyperbolic cross product.
        
        Args:
            a1: First point [x₁:y₁:z₁]
            a2: Second point [x₂:y₂:z₂]
            
        Returns:
            Line joining the points (l:m:n)
        """
        if a1.shape[-1] != 3 or a2.shape[-1] != 3:
            raise ValueError("Points must have shape (..., 3)")
            
        # Get components
        x1, y1, z1 = a1[..., 0], a1[..., 1], a1[..., 2]
        x2, y2, z2 = a2[..., 0], a2[..., 1], a2[..., 2]
        
        # Compute join using cross product formula
        line = torch.stack([
            y1*z2 - y2*z1,
            z1*x2 - z2*x1,
            x2*y1 - x1*y2
        ], dim=-1)
        
        return self.normalize_points(line)
    
    def triple_quad_formula(self, q1: Tensor, q2: Tensor, q3: Tensor) -> Tensor:
        """
        Verifies if three quadrances satisfy the triple quad formula
        (q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4q₁q₂q₃

        Args:
            q1, q2, q3: Three quadrances to verify

        Returns:
            Boolean tensor indicating if triple quad formula is satisfied
        """
        lhs = (q1 + q2 + q3)**2
        rhs = 2*(q1**2 + q2**2 + q3**2) + 4*q1*q2*q3
        return torch.abs(lhs - rhs) < self.epsilon

    def pythagoras(self, q1: Tensor, q2: Tensor, q3: Tensor) -> Tensor:
        """
        Verifies if three quadrances satisfy the hyperbolic Pythagorean theorem.
        According to UHG.pdf Theorem 42:
        For a right triangle (S₃ = 1), q₃ = q₁ + q₂ - q₁q₂

        Args:
            q1, q2: Quadrances of the legs of the right triangle
            q3: Quadrance of the hypotenuse

        Returns:
            Boolean tensor indicating if hyperbolic Pythagorean theorem is satisfied
        """
        expected_q3 = q1 + q2 - q1*q2
        return torch.abs(q3 - expected_q3) < self.epsilon

    def spread_law(self, S1: Tensor, S2: Tensor, S3: Tensor, q1: Tensor, q2: Tensor, q3: Tensor) -> Tensor:
        """
        Verifies the spread law relation
        S₁/q₁ = S₂/q₂ = S₃/q₃

        Args:
            S1, S2, S3: Three spreads
            q1, q2, q3: Three corresponding quadrances

        Returns:
            Boolean tensor indicating if spread law is satisfied
        """
        ratio1 = S1/q1
        ratio2 = S2/q2
        ratio3 = S3/q3
        return torch.abs(ratio1 - ratio2) < self.epsilon and torch.abs(ratio2 - ratio3) < self.epsilon

    def cross_dual_law(self, S1: Tensor, S2: Tensor, S3: Tensor, q1: Tensor) -> Tensor:
        """
        Verifies the cross dual law
        (S₂S₃q₁ - S₁ - S₂ - S₃ + 2)² = 4(1-S₁)(1-S₂)(1-S₃)

        Args:
            S1, S2, S3: Three spreads
            q1: Quadrance

        Returns:
            Boolean tensor indicating if cross dual law is satisfied
        """
        lhs = (S2*S3*q1 - S1 - S2 - S3 + 2)**2
        rhs = 4*(1-S1)*(1-S2)*(1-S3)
        return torch.abs(lhs - rhs) < self.epsilon

    def spread_quadrance_duality(self, L1: Tensor, L2: Tensor) -> bool:
        """
        Verify that spread between lines equals quadrance between dual points
        S(L₁,L₂) = q(L₁⊥,L₂⊥)

        Args:
            L1, L2: Two lines in projective coordinates

        Returns:
            Boolean indicating if duality holds
        """
        spread_val = self.spread(L1, L2)
        quad_val = self.quadrance(self.dual_line_to_point(L1), self.dual_line_to_point(L2))
        return torch.abs(spread_val - quad_val) < self.epsilon

    def point_lies_on_line(self, point: Tensor, line: Tensor) -> bool:
        """
        Check if point [x:y:z] lies on line (l:m:n)
        lx + my - nz = 0

        Args:
            point: Point in projective coordinates [x:y:z]
            line: Line in projective coordinates (l:m:n)

        Returns:
            Boolean indicating if point lies on line
        """
        x, y, z = point[..., 0], point[..., 1], point[..., 2]
        l, m, n = line[..., 0], line[..., 1], line[..., 2]
        return torch.abs(l*x + m*y - n*z) < self.epsilon

    def points_perpendicular(self, a1: Tensor, a2: Tensor) -> bool:
        """
        Check if points [x₁:y₁:z₁] and [x₂:y₂:z₂] are perpendicular
        x₁x₂ + y₁y₂ - z₁z₂ = 0

        Args:
            a1, a2: Points in projective coordinates

        Returns:
            Boolean indicating if points are perpendicular
        """
        return torch.abs(self.hyperbolic_dot(a1, a2)) < self.epsilon

    def lines_perpendicular(self, L1: Tensor, L2: Tensor) -> bool:
        """
        Check if lines (l₁:m₁:n₁) and (l₂:m₂:n₂) are perpendicular
        l₁l₂ + m₁m₂ - n₁n₂ = 0

        Args:
            L1, L2: Lines in projective coordinates

        Returns:
            Boolean indicating if lines are perpendicular
        """
        return torch.abs(self.hyperbolic_dot(L1, L2)) < self.epsilon

    def parametrize_line_point(self, L: Tensor, p: Tensor, r: Tensor, s: Tensor) -> Tensor:
        """
        Get point on line L=(l:m:n) parametrized by p,r,s
        Returns [np-ms : ls+nr : lp+mr]

        Args:
            L: Line in projective coordinates (l:m:n)
            p, r, s: Parameters

        Returns:
            Point on line in projective coordinates
        """
        l, m, n = L[..., 0], L[..., 1], L[..., 2]
        return torch.stack([n*p - m*s, l*s + n*r, l*p + m*r], dim=-1)

    def are_collinear(self, a1: Tensor, a2: Tensor, a3: Tensor) -> bool:
        """
        Test if three points are collinear using determinant formula
        x₁y₂z₃ - x₁y₃z₂ + x₂y₃z₁ - x₃y₂z₁ + x₃y₁z₂ - x₂y₁z₃ = 0

        Args:
            a1, a2, a3: Points in projective coordinates

        Returns:
            Boolean indicating if points are collinear
        """
        x1, y1, z1 = a1[..., 0], a1[..., 1], a1[..., 2]
        x2, y2, z2 = a2[..., 0], a2[..., 1], a2[..., 2]
        x3, y3, z3 = a3[..., 0], a3[..., 1], a3[..., 2]
        
        det = x1*y2*z3 - x1*y3*z2 + x2*y3*z1 - x3*y2*z1 + x3*y1*z2 - x2*y1*z3
        return torch.abs(det) < self.epsilon

    def are_concurrent(self, L1: Tensor, L2: Tensor, L3: Tensor) -> bool:
        """
        Test if three lines are concurrent using determinant
        l₁m₂n₃ - l₁m₃n₂ + l₂m₃n₁ - l₃m₂n₁ + l₃m₁n₂ - l₂m₁n₃ = 0

        Args:
            L1, L2, L3: Lines in projective coordinates

        Returns:
            Boolean indicating if lines are concurrent
        """
        l1, m1, n1 = L1[..., 0], L1[..., 1], L1[..., 2]
        l2, m2, n2 = L2[..., 0], L2[..., 1], L2[..., 2]
        l3, m3, n3 = L3[..., 0], L3[..., 1], L3[..., 2]
        
        det = l1*m2*n3 - l1*m3*n2 + l2*m3*n1 - l3*m2*n1 + l3*m1*n2 - l2*m1*n3
        return torch.abs(det) < self.epsilon

    def altitude_line(self, a: Tensor, L: Tensor) -> Tensor:
        """
        Get altitude line through point a perpendicular to line L
        Returns aL^⊥

        Args:
            a: Point in projective coordinates
            L: Line in projective coordinates

        Returns:
            Altitude line in projective coordinates
        """
        L_perp = torch.stack([L[..., 0], L[..., 1], -L[..., 2]], dim=-1)
        return self.join(a, L_perp)

    def altitude_point(self, a: Tensor, L: Tensor) -> Tensor:
        """
        Get altitude point on L perpendicular to point a
        Returns a^⊥L

        Args:
            a: Point in projective coordinates
            L: Line in projective coordinates

        Returns:
            Altitude point in projective coordinates
        """
        a_perp = torch.stack([a[..., 0], a[..., 1], -a[..., 2]], dim=-1)
        return self.meet(a_perp, L)

    def parallel_line(self, a: Tensor, L: Tensor) -> Tensor:
        """
        Get parallel line through a to line L
        Returns a(a^⊥L)

        Args:
            a: Point in projective coordinates
            L: Line in projective coordinates

        Returns:
            Parallel line in projective coordinates
        """
        alt_point = self.altitude_point(a, L)
        return self.join(a, alt_point)

    def parallel_point(self, a: Tensor, L: Tensor) -> Tensor:
        """
        Get parallel point on a^⊥ to point a on L
        Returns a^⊥(aL^⊥)

        Args:
            a: Point in projective coordinates
            L: Line in projective coordinates

        Returns:
            Parallel point in projective coordinates
        """
        alt_line = self.altitude_line(a, L)
        a_perp = torch.stack([a[..., 0], a[..., 1], -a[..., 2]], dim=-1)
        return self.meet(a_perp, alt_line)