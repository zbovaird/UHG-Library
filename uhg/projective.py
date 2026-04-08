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
from typing import Optional, Tuple, Union, List
from torch import Tensor
from .utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class ProjectiveUHG:
    """
    Core projective operations for Universal Hyperbolic Geometry (UHG).
    All operations are performed using pure projective geometry, following UHG.pdf.
    No tangent space, exponential map, or logarithmic map operations are present.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.eps = epsilon
        
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
        mask = norm > self.eps
        wedge = torch.where(mask, wedge / (norm + self.eps), wedge)
        
        return wedge

    def hyperbolic_dot(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Compute hyperbolic dot product between two points.
        For points [x₁:y₁:z₁] and [x₂:y₂:z₂]:
        x₁x₂ + y₁y₂ - z₁z₂
        
        Args:
            a: First point [x₁:y₁:z₁] or [..., x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂] or [..., x₂:y₂:z₂]
            
        Returns:
            Hyperbolic dot product
        """
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        print("[DEBUG] hyperbolic_dot: a=", a, "b=", b)
        a_space = a[..., :2]
        b_space = b[..., :2]
        a_time = a[..., 2:]
        b_time = b[..., 2:]
        space_dot = torch.sum(a_space * b_space, dim=-1)
        time_dot = torch.sum(a_time * b_time, dim=-1)
        result = space_dot - time_dot
        print("[DEBUG] hyperbolic_dot: space_dot=", space_dot, "time_dot=", time_dot, "result=", result)
        return result

    def quadrance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute quadrance between points using UHG formula.
        Reference: UHG.pdf, Ch. 2, Section 2.3

        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]

        Returns:
            Quadrance between points

        Raises:
            ValueError: if either point is null or not on hyperboloid
        """
        # Ensure points are normalized to hyperboloid
        try:
            a_norm = self.normalize_points(a)
            b_norm = self.normalize_points(b)
        except ValueError as e:
            raise ValueError(f"Points must be on hyperboloid: {e}")

        # Compute Minkowski inner product
        a_dot_b = self.inner_product(a_norm, b_norm)

        # Compute quadrance using UHG formula
        # q = 1 - (a·b)² since points are normalized
        q = 1 - (a_dot_b * a_dot_b)

        # Handle numerical stability
        q = torch.where(
            torch.abs(q) < self.eps,
            torch.zeros_like(q),
            q
        )
        q = torch.where(
            torch.abs(q - 1.0) < self.eps,
            torch.ones_like(q),
            q
        )

        # Verify result is valid
        if torch.any(torch.isnan(q)) or torch.any(torch.isinf(q)):
            raise ValueError("Quadrance calculation resulted in NaN or Inf")

        return q

    def det2(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2x2 determinant for vectors arranged as:
        |a b|
        |c d|
        """
        return a * d - b * c

    def spread(self, L1: torch.Tensor, L2: torch.Tensor, L3: torch.Tensor) -> torch.Tensor:
        """
        Compute spread between two lines using a reference line.
        Reference: UHG.pdf, Ch. 3, Section 3.2

        Args:
            L1: First line [l₁:m₁:n₁]
            L2: Second line [l₂:m₂:n₂]
            L3: Reference line [l₃:m₃:n₃]

        Returns:
            Spread between L1 and L2

        Raises:
            ValueError: if any line is null or not properly normalized
        """
        # Ensure lines are in homogeneous coordinates
        if L1.dim() == 1:
            L1 = L1.unsqueeze(0)
        if L2.dim() == 1:
            L2 = L2.unsqueeze(0)
        if L3.dim() == 1:
            L3 = L3.unsqueeze(0)

        # Check for null lines
        L1_norm = self.inner_product(L1, L1)
        L2_norm = self.inner_product(L2, L2)
        L3_norm = self.inner_product(L3, L3)

        if torch.any(torch.abs(L1_norm) < self.eps) or \
           torch.any(torch.abs(L2_norm) < self.eps) or \
           torch.any(torch.abs(L3_norm) < self.eps):
            raise ValueError("Spread is undefined for null lines")

        # Normalize lines to unit Minkowski norm
        L1 = L1 / torch.sqrt(torch.abs(L1_norm)).unsqueeze(-1)
        L2 = L2 / torch.sqrt(torch.abs(L2_norm)).unsqueeze(-1)
        L3 = L3 / torch.sqrt(torch.abs(L3_norm)).unsqueeze(-1)

        # Compute inner products
        L1_dot_L2 = self.inner_product(L1, L2)
        L1_dot_L3 = self.inner_product(L1, L3)
        L2_dot_L3 = self.inner_product(L2, L3)

        # Compute spread using UHG formula
        # s = 1 - (L1·L2)²/((L1·L3)(L2·L3))
        numerator = L1_dot_L2 * L1_dot_L2
        denominator = L1_dot_L3 * L2_dot_L3

        # Handle numerical stability
        denominator = torch.where(
            torch.abs(denominator) < self.eps,
            torch.ones_like(denominator) * self.eps,
            denominator
        )

        s = 1 - (numerator / denominator)

        # Clamp spread to [0, 1] range
        s = torch.clamp(s, 0.0, 1.0)

        return s

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
        o1 = o1 / (torch.norm(o1, dim=-1, keepdim=True) + self.eps)
        o2 = o2 / (torch.norm(o2, dim=-1, keepdim=True) + self.eps)
        
        return o1, o2
    
    def normalize(self, points: Tensor) -> Tensor:
        """
        Normalize points while preserving cross ratios.
        For 4 or more points, ensures cross ratio is preserved after normalization.
        """
        norms = torch.norm(points[..., :-1], dim=-1, keepdim=True)
        normalized = points / (norms + self.eps)
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
        print("[DEBUG] join: x=", x, "y=", y)
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
        result = torch.cross(x_proj, y_proj, dim=-1)
        print("[DEBUG] join: x_proj=", x_proj, "y_proj=", y_proj, "result=", result)
        return result
    
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
    
    def transform(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Apply projective transformation to points.
        Reference: UHG.pdf, Ch. 4, Section 4.1

        Args:
            x: Points to transform [x:y:z]
            T: 3x3 transformation matrix

        Returns:
            Transformed points

        Raises:
            ValueError: if transformation results in null points
        """
        # Ensure points are in homogeneous coordinates
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Apply transformation
        x_transformed = torch.matmul(T, x.transpose(-2, -1)).transpose(-2, -1)

        # Normalize transformed points to hyperboloid
        try:
            return self.normalize_points(x_transformed)
        except ValueError as e:
            raise ValueError(f"Transformation resulted in invalid points: {e}")

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
        
        if torch.any(torch.abs(norm_1) < self.eps) or torch.any(torch.abs(norm_2) < self.eps):
            raise ValueError("Quadrance is undefined for null points")
            
        # For points on same line, quadrance is 0
        def det3(a, b, c):
            return (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1]) -
                   a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0]) +
                   a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0]))
                   
        # Create a third point to test collinearity
        c = torch.zeros_like(a1)
        c[..., 2] = 1.0  # Reference point [0:0:1]
        
        if torch.abs(det3(a1, a2, c)) < self.eps:
            return torch.zeros_like(norm_1)
            
        # Compute hyperbolic inner products
        dot_12 = torch.sum(a1 * a2 * torch.tensor([1.0, 1.0, -1.0]), dim=-1)
        
        # Compute opposite points according to UHG.pdf
        o1 = dot_12.unsqueeze(-1) * a1 - norm_1.unsqueeze(-1) * a2
        o2 = norm_2.unsqueeze(-1) * a1 - dot_12.unsqueeze(-1) * a2
        
        # Normalize opposite points
        o1 = o1 / (torch.norm(o1, dim=-1, keepdim=True) + self.eps)
        o2 = o2 / (torch.norm(o2, dim=-1, keepdim=True) + self.eps)
        
        # Compute quadrance as 1 minus cross ratio
        cr = self.cross_ratio(a1, a2, o2, o1)
        return 1.0 - cr
    
    def normalize_points(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize points to lie on the hyperbolic manifold (Minkowski norm = -1).
        Reference: UHG.pdf, Ch. 2, Section 2.2

        Args:
            x: Points to normalize [x:y:z]

        Returns:
            Normalized points with Minkowski norm = -1

        Raises:
            ValueError: if points are null or cannot be normalized to hyperboloid
        """
        # Ensure points are in homogeneous coordinates
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Compute Minkowski norm
        norm = self.inner_product(x, x)

        # Check for null points
        if torch.any(torch.abs(norm) < self.eps):
            raise ValueError("Cannot normalize null points (Minkowski norm = 0)")

        # Check if points are hyperbolic (negative norm)
        if torch.any(norm > 0):
            raise ValueError("Cannot normalize Euclidean points (positive norm)")

        # Normalize to unit Minkowski norm
        scale = torch.sqrt(torch.abs(norm))
        x_norm = x / scale.unsqueeze(-1)

        # Verify normalization
        final_norm = self.inner_product(x_norm, x_norm)
        if not torch.all(torch.abs(final_norm + 1.0) < self.eps):
            raise ValueError("Failed to normalize points to hyperboloid")

        return x_norm

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
        inner_prod = torch.clamp(inner_prod, max=-1.0 - self.eps)
        d = torch.acosh(-inner_prod)
        return d

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Minkowski inner product between two points or lines.
        <a,b> = x₁x₂ + y₁y₂ - z₁z₂
        """
        return torch.sum(a * b * torch.tensor([1.0, 1.0, -1.0]), dim=-1)

    def cross_ratio(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Compute cross-ratio of four points using 3x3 determinants (UHG projective geometry)."""
        # Ensure points are in homogeneous coordinates
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if d.dim() == 1:
            d = d.unsqueeze(0)

        # Stack points for batch processing
        # Each point is (..., 3), so stack to (..., 3, 3)
        def det3(p, q, r):
            # Stack last dimension to shape (..., 3, 3)
            mat = torch.stack([p, q, r], dim=-2)
            return torch.det(mat)

        # Compute determinants for cross-ratio
        det_abd = det3(a, b, d)
        det_cbd = det3(c, b, d)
        det_acd = det3(a, c, d)
        det_bcd = det3(b, c, d)

        # UHG cross-ratio formula (see UHG.pdf, Ch. 2):
        # CR(a, b; c, d) = (|a b d| * |c b d|) / (|a c d| * |b c d|)
        numer = det_abd * det_cbd
        denom = det_acd * det_bcd

        # Handle numerical stability
        if torch.any(torch.abs(denom) < self.eps):
            raise ValueError("Cross-ratio calculation unstable: denominator too small")

        return numer / denom

    def restore_cross_ratio(self, x: torch.Tensor, target_cr: torch.Tensor) -> torch.Tensor:
        """
        Restore the cross-ratio of points to a target value.
        Reference: UHG.pdf, Ch. 2
        """
        return restore_cross_ratio(x, target_cr)

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
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        # Expand weights for broadcasting
        # Add extra dimensions to match points shape
        for _ in range(points.dim() - weights.dim()):
            weights = weights.unsqueeze(-1)
        
        # Compute weighted sum
        avg = torch.sum(points * weights, dim=-2)
        
        # Ensure last component is 1 for homogeneous coordinates
        avg = avg / (torch.abs(avg[..., -1:]) + self.eps)
        
        return avg
    
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
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        # Reshape points and weights for proper broadcasting
        B = points.size(0)  # Batch size
        D = points.size(-1)  # Feature dimension
        points = points.view(B, -1, D)  # [B, N, D]
        weights = weights.view(B, -1, 1)  # [B, N, 1]
        
        # Compute weighted sum
        weighted_sum = torch.sum(points * weights, dim=1)  # [B, D]
        
        # Normalize result
        return self.normalize_points(weighted_sum)
    
    def is_null_point(self, p: torch.Tensor) -> torch.Tensor:
        """Check if point lies on the null cone."""
        # Compute Minkowski norm
        norm = self.inner_product(p, p)
        return torch.abs(norm) < self.eps
    
    def null_point(self, t: Tensor, u: Tensor) -> Tensor:
        """
        Create a null point in projective space.
        Following UHG.pdf, null points satisfy x² + y² = z².
        
        Args:
            t, u: Parameters for null point
            
        Returns:
            Null point [x:y:z]
        """
        # Ensure inputs are tensors
        t = torch.as_tensor(t)
        u = torch.as_tensor(u)
        
        # Create null point
        x = t
        y = u
        z = torch.sqrt(t**2 + u**2 + self.eps)
        
        return torch.stack([x, y, z], dim=-1)

    def join_null_points(self, t1: Tensor, u1: Tensor, t2: Tensor, u2: Tensor) -> Tensor:
        """
        Get line through two null points parametrized by (t₁:u₁) and (t₂:u₂)
        According to UHG.pdf, the join of null points is:
        (t₁t₂-u₁u₂ : t₁u₂+t₂u₁ : t₁t₂+u₁u₂)
        
        Args:
            t1, u1: Parameters of first null point
            t2, u2: Parameters of second null point
            
        Returns:
            Join line in projective coordinates
        """
        # Compute components
        x = t1*t2 - u1*u2
        y = t1*u2 + t2*u1
        z = t1*t2 + u1*u2
        
        # Stack into line
        line = torch.stack([x, y, z], dim=-1)
        
        # Normalize line
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
    
    def midpoints(self, a1: Tensor, a2: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Calculate midpoints of side a₁a₂.
        From UHG.pdf Theorem 54:
        1. Exists when p = 1 - q is a square
        2. When existing, after normalizing to equal hyperbolic norms:
           m₁ = [x₁+x₂ : y₁+y₂ : z₁+z₂]
           m₂ = [x₁-x₂ : y₁-y₂ : z₁-z₂]
        """
        # Handle special cases
        if torch.allclose(a1, a2, rtol=self.eps):
            return a1, None
            
        # Check if both points are null
        if self.is_null_point(a1) and self.is_null_point(a2):
            return None, None
            
        # If one point is null, return it as the midpoint
        if self.is_null_point(a1):
            return a1, None
        if self.is_null_point(a2):
            return a2, None
            
        # Normalize to equal hyperbolic norms
        norm1 = torch.abs(a1[..., 0]**2 + a1[..., 1]**2 - a1[..., 2]**2)
        norm2 = torch.abs(a2[..., 0]**2 + a2[..., 1]**2 - a2[..., 2]**2)
        
        # Scale points to have equal norms
        scale = torch.sqrt(norm1/norm2)
        a2_scaled = a2 * scale
        
        # Calculate quadrance and check existence
        q = self.quadrance(a1, a2_scaled)
        p = 1 - q
        
        if p < 0:
            return None, None
        
        # Construct midpoints using normalized points
        m1 = a1 + a2_scaled
        m2 = a1 - a2_scaled
        
        # Normalize resulting points
        m1 = self.normalize_points(m1)
        m2 = self.normalize_points(m2)
        
        # Handle numerical instability
        if torch.any(torch.isnan(m1)) or torch.any(torch.isnan(m2)):
            return None, None
            
        return m1, m2
    
    def verify_midpoints(self, a1: Tensor, a2: Tensor, m1: Tensor, m2: Tensor) -> bool:
        """Verify midpoint properties with detailed logging"""
        eps = 1e-5
        print("\nMidpoint Verification:")
        
        # 1. Equal quadrances to endpoints
        q11 = self.quadrance(a1, m1)
        q21 = self.quadrance(a2, m1)
        print(f"\nm1 quadrances:")
        print(f"q(a1,m1)={q11}")
        print(f"q(a2,m1)={q21}")
        print(f"Difference: {abs(q11-q21)}")
        if not torch.allclose(q11, q21, rtol=eps):
            print("❌ First midpoint quadrances not equal")
            return False
            
        q12 = self.quadrance(a1, m2)
        q22 = self.quadrance(a2, m2)
        print(f"\nm2 quadrances:")
        print(f"q(a1,m2)={q12}")
        print(f"q(a2,m2)={q22}")
        print(f"Difference: {abs(q12-q22)}")
        if not torch.allclose(q12, q22, rtol=eps):
            print("❌ Second midpoint quadrances not equal")
            return False
            
        # 2. Perpendicularity
        dot = self.hyperbolic_dot(m1, m2)
        print(f"\nPerpendicularity:")
        print(f"Hyperbolic dot product: {dot}")
        if abs(dot) > eps:
            print("❌ Midpoints not perpendicular")
            return False
            
        # 3. Cross-ratio
        cr = self.cross_ratio(a1, a2, m1, m2)
        print(f"\nCross-ratio: {cr}")
        if not torch.allclose(cr, torch.tensor(-1.0), rtol=eps):
            print("❌ Cross-ratio not -1")
            return False
        
        print("\n✅ All midpoint properties verified")
        return True
    
    def is_null_line(self, l: torch.Tensor) -> torch.Tensor:
        """Check if line is null (satisfies l² + m² = n²)."""
        # Compute line norm
        space_norm = torch.sum(l[..., :-1] * l[..., :-1], dim=-1)
        time_norm = l[..., -1] * l[..., -1]
        return torch.abs(space_norm - time_norm) < self.eps
    
    def transform_verify_midpoints(self, a1: Tensor, a2: Tensor, matrix: Tensor) -> bool:
        """
        Verify that midpoints transform correctly under projective transformation
        """
        # Get original midpoints
        m1, m2 = self.midpoints(a1, a2)
        if m1 is None or m2 is None:
            return True  # No midpoints case
            
        # Transform points
        a1_trans = self.transform(a1, matrix)
        a2_trans = self.transform(a2, matrix)
        
        # Get midpoints of transformed points
        m1_trans, m2_trans = self.midpoints(a1_trans, a2_trans)
        
        # Transform original midpoints
        m1_expected = self.transform(m1, matrix)
        m2_expected = self.transform(m2, matrix)
        
        # Verify up to projective equivalence
        eps = 1e-4
        m1_match = (torch.allclose(m1_trans, m1_expected, rtol=eps) or 
                    torch.allclose(m1_trans, -m1_expected, rtol=eps))
        m2_match = (torch.allclose(m2_trans, m2_expected, rtol=eps) or 
                    torch.allclose(m2_trans, -m2_expected, rtol=eps))
                    
        return m1_match and m2_match

    def triple_quad_formula(self, q1: Tensor, q2: Tensor, q3: Tensor) -> bool:
        """
        Verify the triple quad formula from UHG.pdf.
        For any three points, the quadrances satisfy:
        (q1 + q2 + q3)² = 2(q1² + q2² + q3²) + 4q1q2q3
        Reference: UHG.pdf, Ch. 2
        """
        # Calculate both sides of the formula
        lhs = (q1 + q2 + q3)**2
        rhs = 2*(q1**2 + q2**2 + q3**2) + 4*q1*q2*q3
        # Check if they are equal within numerical tolerance
        return torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def triple_spread_formula(self, S1: Tensor, S2: Tensor, S3: Tensor) -> bool:
        """
        Verify the triple spread formula from UHG.pdf.
        For any three points, the spreads satisfy:
        (S1 + S2 + S3)² = 2(S1² + S2² + S3²) + 4S1S2S3
        Reference: UHG.pdf, Ch. 2
        """
        # Calculate both sides of the formula
        lhs = (S1 + S2 + S3)**2
        rhs = 2*(S1**2 + S2**2 + S3**2) + 4*S1*S2*S3
        # Check if they are equal within numerical tolerance
        return torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def cross_law(self, q1: Tensor, q2: Tensor, q3: Tensor, S1: Tensor, S2: Tensor, S3: Tensor) -> bool:
        """
        Verify the cross law from UHG.pdf.
        For any three points, the quadrances and spreads satisfy:
        (q1 + q2 - q3)² = 4q1q2(1 - S3)
        Reference: UHG.pdf, Ch. 2
        """
        # Calculate both sides of the law
        lhs = (q1 + q2 - q3)**2
        rhs = 4*q1*q2*(1 - S3)
        # Check if they are equal within numerical tolerance
        return torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def cross_dual_law(self, S1: Tensor, S2: Tensor, S3: Tensor, q1: Tensor) -> bool:
        """
        Verify the dual cross law from UHG.pdf.
        For any three points, the spreads and quadrance satisfy:
        (S1 + S2 - S3)² = 4S1S2(1 - q1)
        Reference: UHG.pdf, Ch. 2
        """
        # Calculate both sides of the law
        lhs = (S1 + S2 - S3)**2
        rhs = 4*S1*S2*(1 - q1)
        # Check if they are equal within numerical tolerance
        return torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def pythagoras(self, q1: Tensor, q2: Tensor, q3: Optional[Tensor] = None) -> Tensor:
        """
        Pythagoras' theorem in UHG.
        For a right triangle (S₃ = 1), q₃ = q₁ + q₂ - q₁q₂
        
        Args:
            q1, q2: Quadrances of the legs
            q3: Optional quadrance of the hypotenuse
            
        Returns:
            If q3 is None, returns the expected q3.
            Otherwise, returns whether the theorem is satisfied.
        """
        expected_q3 = q1 + q2 - q1*q2
        
        if q3 is None:
            return expected_q3
        else:
            return torch.abs(q3 - expected_q3) < self.eps
        
    def dual_pythagoras(self, S1: Tensor, S2: Tensor, S3: Optional[Tensor] = None) -> Tensor:
        """
        Dual Pythagoras' theorem in UHG.
        For a right triangle (q₃ = 1), S₃ = S₁ + S₂ - S₁S₂
        
        Args:
            S1, S2: Spreads of the legs
            S3: Optional spread of the hypotenuse
            
        Returns:
            If S3 is None, returns the expected S3.
            Otherwise, returns whether the theorem is satisfied.
        """
        expected_S3 = S1 + S2 - S1*S2
        
        if S3 is None:
            return expected_S3
        else:
            return torch.abs(S3 - expected_S3) < self.eps

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
        return torch.abs(spread_val - quad_val) < self.eps
        
    def dual_line_to_point(self, line: Tensor) -> Tensor:
        """
        Convert a line to its dual point.
        In UHG, the dual of a line (l:m:n) is the point [l:m:n].
        Reference: UHG.pdf, Ch. 2
        """
        # In UHG, the dual of a line (l:m:n) is simply the point [l:m:n]
        # This is because we're using the same bilinear form for both points and lines
        return line.clone()
        
    def dual_point_to_line(self, point: Tensor) -> Tensor:
        """
        Convert a point to its dual line.
        In UHG, the dual of a point [x:y:z] is the line (x:y:z).
        Reference: UHG.pdf, Ch. 2
        """
        # In UHG, the dual of a point [x:y:z] is simply the line (x:y:z)
        # This is because we're using the same bilinear form for both points and lines
        return point.clone()

    def point_lies_on_line(self, point: Tensor, line: Tensor) -> bool:
        """
        Check if a point lies on a line using the incidence relation in projective geometry.
        In projective geometry, a point [x:y:z] lies on a line (a:b:c) if ax + by + cz = 0.
        Reference: UHG.pdf, Ch. 2
        """
        # Calculate the dot product
        dot_product = torch.sum(point * line)
        # Check if the dot product is close to zero (within epsilon)
        return torch.abs(dot_product) < self.eps

    def manifold(self) -> torch.Tensor:
        """
        Get the manifold structure matrix for UHG.
        In UHG, we use the hyperbolic form J = diag(1, 1, -1).
        Reference: UHG.pdf, Ch. 3
        """
        return torch.diag(torch.tensor([1.0, 1.0, -1.0]))

    def midpoint(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate the midpoint between two points on the hyperboloid.
        Reference: UHG.pdf, Ch. 2, Section 2.4

        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]

        Returns:
            Midpoint between a and b

        Raises:
            ValueError: if either point is null or not on hyperboloid
        """
        # Ensure points are normalized to hyperboloid
        try:
            a_norm = self.normalize_points(a)
            b_norm = self.normalize_points(b)
        except ValueError as e:
            raise ValueError(f"Points must be on hyperboloid: {e}")

        # Calculate midpoint using UHG formula
        # m = (a + b)/√(2(1 - a·b))
        a_dot_b = self.inner_product(a_norm, b_norm)
        scale = torch.sqrt(2 * (1 - a_dot_b))
        m = (a_norm + b_norm) / scale.unsqueeze(-1)

        # Normalize midpoint to ensure it's on hyperboloid
        m = self.normalize_points(m)

        return m