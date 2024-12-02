from typing import Tuple, Optional, Union, List
import torch

import torch

def join(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the join (line through two points) in UHG.
    
    The join of two points is computed using the cross product
    in projective geometry.
    
    Args:
        a: First point tensor
        b: Second point tensor
        
    Returns:
        Line through points a and b
    """
    return torch.cross(a, b, dim=-1)

def meet(L: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Compute the meet (intersection) of two lines in UHG.
    
    The meet of two lines is computed using the cross product
    in projective geometry.
    
    Args:
        L: First line tensor
        M: Second line tensor
        
    Returns:
        Point of intersection of lines L and M
    """
    return torch.cross(L, M, dim=-1)

def cross_ratio(a: torch.Tensor, b: torch.Tensor, 
                c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-ratio of four points in UHG.
    
    The cross-ratio is a projective invariant and is fundamental
    to hyperbolic geometry calculations.
    
    Args:
        a, b, c, d: Point tensors
        
    Returns:
        Cross-ratio value
    """
    # Compute joins
    ab = join(a, b)
    cd = join(c, d)
    ac = join(a, c)
    bd = join(b, d)
    
    # Compute meets
    p = meet(ab, cd)
    q = meet(ac, bd)
    
    # Compute cross-ratio using dot products
    num = torch.sum(p * c, dim=-1) * torch.sum(q * d, dim=-1)
    den = torch.sum(p * d, dim=-1) * torch.sum(q * c, dim=-1)
    
    return num / den

def perpendicular(L: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute the perpendicular to a line through a point in UHG.
    
    Uses the polar principle in projective geometry to compute
    the perpendicular line.
    
    Args:
        L: Line tensor
        p: Point tensor
        
    Returns:
        Line perpendicular to L through p
    """
    # The perpendicular is computed using the polar principle
    return torch.cross(L, p, dim=-1)

def reflect(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Reflect a point in a line in UHG.
    
    Reflection is a fundamental isometry in hyperbolic geometry,
    computed using joins and meets.
    
    Args:
        x: Point to reflect
        L: Line to reflect in
        
    Returns:
        Reflected point
    """
    # Compute perpendicular through point
    perp = perpendicular(L, x)
    
    # Compute reflection point using meet
    return meet(L, perp)

def rotate(x: torch.Tensor, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate a point around a center in UHG.
    
    Rotation in hyperbolic geometry is implemented using
    cross-ratios and reflections.
    
    Args:
        x: Point to rotate
        center: Center of rotation
        angle: Rotation angle
        
    Returns:
        Rotated point
    """
    # Convert angle to cross-ratio parameter
    t = torch.tan(angle / 2)
    
    # Compute perpendicular lines through center
    L1 = perpendicular(join(center, x), center)
    L2 = rotate_line(L1, t)
    
    # Reflect point in both lines
    y = reflect(x, L1)
    return reflect(y, L2)

def rotate_line(L: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Rotate a line by a parameter t in UHG.
    
    Helper function for rotation implementation.
    
    Args:
        L: Line to rotate
        t: Rotation parameter
        
    Returns:
        Rotated line
    """
    # Compute rotation matrix in projective form
    R = torch.zeros_like(L)
    R[..., 0] = 1 - t**2
    R[..., 1] = 2*t
    R[..., 2] = -(1 + t**2)
    
    return torch.einsum('...i,...i->...', R, L)

def translate(x: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Translate a point along a vector in UHG.
    
    Translation in hyperbolic geometry is implemented using
    cross-ratios and reflections.
    
    Args:
        x: Point to translate
        v: Translation vector
        t: Translation distance
        
    Returns:
        Translated point
    """
    # Compute line of translation
    L = join(x, v)
    
    # Convert distance to cross-ratio parameter
    s = torch.tanh(t / 2)
    
    # Compute translation using cross-ratio
    return cross_ratio_point(x, v, s)

def cross_ratio_point(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute point with given cross-ratio t from a towards b.
    
    Helper function for translation implementation.
    
    Args:
        a: Starting point
        b: Direction point
        t: Cross-ratio parameter
        
    Returns:
        Point with cross-ratio t
    """
    # Compute weighted combination in projective coordinates
    num = (1 - t) * a + (1 + t) * b
    den = torch.sqrt(1 - t**2)
    return num / den.unsqueeze(-1)

def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic distance between points in UHG.
    
    Distance is computed using the cross-ratio formula with
    ideal points.
    
    Args:
        a: First point
        b: Second point
        
    Returns:
        Hyperbolic distance between a and b
    """
    # Compute line through points
    L = join(a, b)
    
    # Find ideal points (intersections with absolute)
    p, q = ideal_points(L)
    
    # Compute distance using cross-ratio
    return torch.log(torch.abs(cross_ratio(a, b, p, q))) / 2

def ideal_points(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the ideal points on a line in UHG.
    
    Helper function for distance computation.
    
    Args:
        L: Line tensor
        
    Returns:
        Tuple of two ideal points
    """
    # Solve quadratic equation for intersections with absolute
    a = L[..., 0]**2 + L[..., 1]**2
    b = 2 * L[..., 0] * L[..., 2]
    c = L[..., 2]**2
    
    disc = torch.sqrt(b**2 - 4*a*c)
    t1 = (-b + disc) / (2*a)
    t2 = (-b - disc) / (2*a)
    
    # Construct ideal points
    p = torch.stack([t1, torch.ones_like(t1), torch.ones_like(t1)], dim=-1)
    q = torch.stack([t2, torch.ones_like(t2), torch.ones_like(t2)], dim=-1)
    
    return p, q 