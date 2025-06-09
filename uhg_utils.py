"""
UHG Utilities - Bridge between UHG library and application code.

This module provides a clean interface to the UHG library for use in
anomaly detection and other applications.
"""

import torch
from uhg import ProjectiveUHG

# Create a global UHG instance with default settings
_UHG = ProjectiveUHG(epsilon=1e-9)

def get_uhg_instance(epsilon: float = 1e-9) -> ProjectiveUHG:
    """
    Get a UHG instance with the specified epsilon value.
    
    Args:
        epsilon: Small value for numerical stability
        
    Returns:
        ProjectiveUHG instance
    """
    global _UHG
    if epsilon != _UHG.epsilon:
        _UHG = ProjectiveUHG(epsilon=epsilon)
    return _UHG

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute hyperbolic inner product between points in projective coordinates.
    
    Args:
        a: First point tensor of shape (..., D+1)
        b: Second point tensor of shape (..., D+1)
        
    Returns:
        Inner product tensor of shape (..., 1)
    """
    return _UHG.hyperbolic_dot(a, b).unsqueeze(-1)

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """
    Compute the UHG norm of a point.
    
    Args:
        a: Point tensor of shape (..., D+1)
        
    Returns:
        Norm tensor of shape (..., 1)
    """
    return uhg_inner_product(a, a)

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute quadrance between two points in UHG.
    Quadrance corresponds to squared distance in Euclidean geometry.
    
    Args:
        a: First point tensor of shape (..., D+1)
        b: Second point tensor of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Quadrance tensor of shape (...)
    """
    # Use the UHG library implementation with error handling
    try:
        # The library implementation might raise ValueError for null points
        result = _UHG.quadrance(a, b)
        return result
    except ValueError:
        # Fall back to manual calculation for null points
        aa = uhg_inner_product(a, a)
        bb = uhg_inner_product(b, b)
        ab = uhg_inner_product(a, b)
        
        numerator = ab * ab - aa * bb
        denominator = aa * bb
        
        # Ensure numerical stability
        safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
        safe_sign = torch.sign(denominator)
        
        quad = numerator / (safe_denominator * safe_sign)
        return quad.squeeze(-1)

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute spread between two lines in UHG.
    Spread is the dual of quadrance and measures the squared angle.
    
    Args:
        L: First line tensor of shape (..., D+1)
        M: Second line tensor of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Spread tensor of shape (...)
    """
    # Use the UHG library implementation with error handling
    try:
        # The library implementation might raise ValueError for null lines
        result = _UHG.spread(L, M)
        return result
    except ValueError:
        # Fall back to manual calculation for null lines
        LL = uhg_inner_product(L, L)
        MM = uhg_inner_product(M, M)
        LM = uhg_inner_product(L, M)
        
        numerator = LM * LM - LL * MM
        denominator = LL * MM
        
        # Ensure numerical stability
        safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
        safe_sign = torch.sign(denominator)
        
        spread = numerator / (safe_denominator * safe_sign)
        return spread.squeeze(-1)

def uhg_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, 
                   p3: torch.Tensor, p4: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute the cross-ratio of four points in UHG.
    
    Args:
        p1, p2, p3, p4: Point tensors of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Cross-ratio tensor of shape (...)
    """
    # Use the UHG library implementation with error handling
    try:
        # The library implementation might raise errors for degenerate cases
        result = _UHG.cross_ratio(p1, p2, p3, p4)
        return result
    except (ValueError, RuntimeError):
        # Fall back to quadrance-based calculation
        q_12 = uhg_quadrance(p1, p2)
        q_34 = uhg_quadrance(p3, p4)
        q_13 = uhg_quadrance(p1, p3)
        q_24 = uhg_quadrance(p2, p4)
        
        numerator = q_12 * q_34
        denominator = q_13 * q_24
        
        # Ensure numerical stability
        safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
        safe_sign = torch.sign(denominator)
        
        return (numerator / (safe_denominator * safe_sign))

def to_uhg_space(x: torch.Tensor) -> torch.Tensor:
    """
    Convert Euclidean vectors to UHG projective space by adding a homogeneous coordinate.
    
    Args:
        x: Euclidean tensor of shape (..., D)
        
    Returns:
        UHG projective tensor of shape (..., D+1)
    """
    # Add homogeneous coordinate (z=1)
    ones = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.cat([x, ones], dim=-1)

def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize points according to UHG conventions.
    
    Args:
        points: Point tensor of shape (..., D+1)
        
    Returns:
        Normalized point tensor of shape (..., D+1)
    """
    return _UHG.normalize_points(points)

def join_points(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the line joining two points.
    
    Args:
        a, b: Point tensors of shape (..., D+1)
        
    Returns:
        Line tensor of shape (..., D+1)
    """
    return _UHG.join(a, b)

def meet_line_point(line: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Compute the intersection of a line and a point.
    
    Args:
        line: Line tensor of shape (..., D+1)
        point: Point tensor of shape (..., D+1)
        
    Returns:
        Intersection point tensor of shape (..., D+1)
    """
    return _UHG.meet(line, point) 