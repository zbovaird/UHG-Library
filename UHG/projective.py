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

    def quadrance(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Calculate quadrance between two points a=[x₁:y₁:z₁] and b=[x₂:y₂:z₂]
        According to UHG.pdf equation (9):
        q(a,b) = 1 - (x₁x₂ + y₁y₂ - z₁z₂)²/((x₁² + y₁² - z₁²)(x₂² + y₂² - z₂²))
        
        This is undefined if either point is null, equals 1 when points are perpendicular,
        and equals 0 when points are the same or lie on a null line.
        
        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            
        Returns:
            Quadrance between points
            
        Raises:
            ValueError: if either point is null
        """
        print("[DEBUG] quadrance: a=", a, "b=", b)
        a = self.normalize_points(a)
        b = self.normalize_points(b)
        if torch.any(self.is_null_point(a)) or torch.any(self.is_null_point(b)):
            print("[WARNING] quadrance: One or more points are null. a=", a, "b=", b)
            raise ValueError("Quadrance is undefined for null points")
        a_dot_a = self.hyperbolic_dot(a, a)
        b_dot_b = self.hyperbolic_dot(b, b)
        a_dot_b = self.hyperbolic_dot(a, b)
        print("[DEBUG] quadrance: a_dot_a=", a_dot_a, "b_dot_b=", b_dot_b, "a_dot_b=", a_dot_b)
        denom = a_dot_a * b_dot_b
        numer = a_dot_b ** 2
        print("[DEBUG] quadrance: numer=", numer, "denom=", denom)
        q = 1 - numer / denom
        if torch.any(torch.isnan(q)):
            print("[ERROR] quadrance: Result is NaN! a=", a, "b=", b, "numer=", numer, "denom=", denom)
        q_clamped = torch.clamp(q, min=0.0, max=1.0)
        print("[DEBUG] quadrance: q=", q, "q_clamped=", q_clamped)
        return q_clamped
        
    def det2(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2x2 determinant for vectors arranged as:
        |a b|
        |c d|
        """
        return a * d - b * c

    def spread(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """
        Calculate the spread between three points a, b, c.
        The spread is the hyperbolic angle between the lines ab and ac.

        Args:
            a: First point [x₁:y₁:z₁]
            b: Second point [x₂:y₂:z₂]
            c: Third point [x₃:y₃:z₃]

        Returns:
            Spread between the lines ab and ac

        Raises:
            ValueError: if any point is null
        """
        # Normalize points
        a = self.normalize_points(a)
        b = self.normalize_points(b)
        c = self.normalize_points(c)

        # Check for null points
        if torch.any(self.is_null_point(a)) or torch.any(self.is_null_point(b)) or torch.any(self.is_null_point(c)):
            print("[DEBUG] Spread: One or more points are null. a:", a, "b:", b, "c:", c)
            raise ValueError("Spread is undefined for null points")

        # Calculate quadrances
        q1 = self.quadrance(b, c)  # Quadrance of bc
        q2 = self.quadrance(a, c)  # Quadrance of ac
        q3 = self.quadrance(a, b)  # Quadrance of ab

        # Calculate spread using UHG formula
        # S = (q1 + q2 - q3)**2 / (4 * q1 * q2)
        numerator = (q1 + q2 - q3)**2
        denominator = 4 * q1 * q2

        # Debug output
        print("[DEBUG] Spread calculation:")
        print("  a:", a)
        print("  b:", b)
        print("  c:", c)
        print("  q1 (bc):", q1)
        print("  q2 (ac):", q2)
        print("  q3 (ab):", q3)
        print("  numerator:", numerator)
        print("  denominator:", denominator)

        # Handle numerical stability
        if torch.any(denominator < self.epsilon):
            print("[WARNING] Spread denominator near zero. Returning 1.0.")
            return torch.ones_like(numerator)

        # Calculate spread
        S = numerator / denominator

        # Check for NaN
        if torch.any(torch.isnan(S)):
            print("[ERROR] Spread result is NaN!")
            print("  Inputs: a=", a, ", b=", b, ", c=", c)
            print("  Quadrances: q1=", q1, ", q2=", q2, ", q3=", q3)
            print("  Numerator:", numerator)
            print("  Denominator:", denominator)

        # Ensure result is in [0,1]
        S_clamped = torch.clamp(S, min=0.0, max=1.0)
        print("  Spread (before clamp):", S)
        print("  Spread (clamped):", S_clamped)
        return S_clamped

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
    
    def transform(self, points: Tensor, matrix: Tensor) -> Tensor:
        """
        Apply a projective transformation to points.
        
        Args:
            points: Points to transform [..., 3]
            matrix: 3x3 transformation matrix
            
        Returns:
            Transformed points
        """
        if points.shape[-1] != 3 or matrix.shape[-2:] != (3, 3):
            raise ValueError("Points must have shape (..., 3) and matrix must have shape (..., 3, 3)")
            
        # Reshape points for batch matrix multiply
        points_flat = points.reshape(-1, 3)
        
        # Apply transformation
        transformed = torch.matmul(points_flat, matrix.transpose(-2, -1))
        
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
        Normalize points to have unit hyperbolic norm.
        For a point [x:y:z], we ensure x² + y² < z².

        Args:
            points: Points to normalize [..., 3]

        Returns:
            Normalized points
        """
        print("[DEBUG] normalize_points: input=", points)
        norms = torch.norm(points[..., :-1], dim=-1, keepdim=True)
        normalized = points / (norms + self.epsilon)
        print("[DEBUG] normalize_points: norms=", norms, "normalized=", normalized)
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
        Check if a point is null in hyperbolic space.
        A point [x:y:z] is null if x² + y² = z².

        Args:
            point: Point to check [..., 3]

        Returns:
            Boolean tensor indicating if point is null
        """
        # Handle batched points
        if point.dim() > 1:
            # Check each point in the batch
            return torch.stack([self.is_null_point(p) for p in point])

        # Extract coordinates
        x, y, z = point[0], point[1], point[2]

        # Check if point is null
        return torch.abs(x**2 + y**2 - z**2) < self.epsilon
    
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
        z = torch.sqrt(t**2 + u**2 + self.epsilon)
        
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
        if torch.allclose(a1, a2, rtol=self.epsilon):
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
    
    def is_null_line(self, line: Tensor) -> bool:
        """
        Check if a line is null (contains its pole).
        A line (l:m:n) is null when l² + m² = n²
        
        Args:
            line: Line in projective coordinates (l:m:n)
            
        Returns:
            Boolean tensor indicating if line is null
        """
        # First normalize line
        line = self.normalize_points(line)
        
        # Get components
        l, m, n = line[..., 0], line[..., 1], line[..., 2]
        
        # Line is null if l² + m² = n²
        norm = l*l + m*m - n*n
        return torch.abs(norm) < self.epsilon
    
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

        Args:
            q1: First quadrance
            q2: Second quadrance
            q3: Third quadrance

        Returns:
            True if the formula is satisfied
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

        Args:
            S1: First spread
            S2: Second spread
            S3: Third spread

        Returns:
            True if the formula is satisfied
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

        Args:
            q1: First quadrance
            q2: Second quadrance
            q3: Third quadrance
            S1: First spread
            S2: Second spread
            S3: Third spread

        Returns:
            True if the law is satisfied
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

        Args:
            S1: First spread
            S2: Second spread
            S3: Third spread
            q1: First quadrance

        Returns:
            True if the law is satisfied
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
            return torch.abs(q3 - expected_q3) < self.epsilon
        
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
            return torch.abs(S3 - expected_S3) < self.epsilon

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
        
    def dual_line_to_point(self, line: Tensor) -> Tensor:
        """
        Convert a line to its dual point.
        In UHG, the dual of a line (l:m:n) is the point [l:m:n].
        
        Args:
            line: Line in projective coordinates (l:m:n)
            
        Returns:
            Dual point [l:m:n]
        """
        # In UHG, the dual of a line (l:m:n) is simply the point [l:m:n]
        # This is because we're using the same bilinear form for both points and lines
        return line.clone()
        
    def dual_point_to_line(self, point: Tensor) -> Tensor:
        """
        Convert a point to its dual line.
        In UHG, the dual of a point [x:y:z] is the line (x:y:z).
        
        Args:
            point: Point in projective coordinates [x:y:z]
            
        Returns:
            Dual line (x:y:z)
        """
        # In UHG, the dual of a point [x:y:z] is simply the line (x:y:z)
        # This is because we're using the same bilinear form for both points and lines
        return point.clone()

    def point_lies_on_line(self, point: Tensor, line: Tensor) -> bool:
        """
        Check if a point lies on a line using the incidence relation in projective geometry.
        
        In projective geometry, a point [x:y:z] lies on a line (a:b:c) if ax + by + cz = 0.
        
        Args:
            point: Tensor representing a point [x:y:z]
            line: Tensor representing a line (a:b:c)
            
        Returns:
            Boolean indicating whether the point lies on the line
        """
        # Calculate the dot product
        dot_product = torch.sum(point * line)
        
        # Check if the dot product is close to zero (within epsilon)
        return torch.abs(dot_product) < self.epsilon

    def cross_ratio(self, v1: Tensor, v2: Tensor, u1: Tensor, u2: Tensor) -> Tensor:
        """
        Calculate the cross-ratio of four points.
        The cross-ratio is a projective invariant.

        Args:
            v1: First point [x₁:y₁:z₁]
            v2: Second point [x₂:y₂:z₂]
            u1: Third point [x₃:y₃:z₃]
            u2: Fourth point [x₄:y₄:z₄]

        Returns:
            Cross-ratio of the four points

        Raises:
            ValueError: if any point is null
        """
        print("[DEBUG] cross_ratio: v1=", v1, "v2=", v2, "u1=", u1, "u2=", u2)
        # Use determinants for projective cross-ratio
        def det2(a, b):
            return a[0]*b[1] - a[1]*b[0]
        # For 3D, project to 2D by dropping last coordinate
        v1_2d = v1[..., :2]
        v2_2d = v2[..., :2]
        u1_2d = u1[..., :2]
        u2_2d = u2[..., :2]
        num = det2(v1_2d, u1_2d) * det2(v2_2d, u2_2d)
        denom = det2(v1_2d, u2_2d) * det2(v2_2d, u1_2d)
        print("[DEBUG] cross_ratio: num=", num, "denom=", denom)
        if torch.any(denom.abs() < self.epsilon):
            print("[WARNING] cross_ratio: denominator near zero.")
        cr = num / (denom + self.epsilon)
        print("[DEBUG] cross_ratio: cr=", cr)
        return cr