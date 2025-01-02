"""Cross-ratio computation utilities for UHG."""

import torch
from ..projective import ProjectiveUHG

def compute_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """Compute cross-ratio of four points in projective space.
    
    The cross-ratio is a projective invariant defined as:
    CR(A,B;C,D) = (AC)(BD)/((AD)(BC))
    where (XY) represents the hyperbolic dot product of points X and Y
    
    Args:
        p1, p2, p3, p4: Points in projective space with homogeneous coordinates
        
    Returns:
        Cross-ratio value
    """
    # Convert inputs to double precision
    p1 = p1.double() if p1.dtype != torch.float64 else p1
    p2 = p2.double() if p2.dtype != torch.float64 else p2
    p3 = p3.double() if p3.dtype != torch.float64 else p3
    p4 = p4.double() if p4.dtype != torch.float64 else p4
    
    # Compute hyperbolic dot products
    def hyperbolic_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split into spatial and time components
        x_spatial = x[..., :-1]
        x_time = x[..., -1]
        y_spatial = y[..., :-1]
        y_time = y[..., -1]
        
        # Compute dot product with Minkowski metric
        return torch.sum(x_spatial * y_spatial, dim=-1) - x_time * y_time
    
    # Compute cross-ratio using hyperbolic dot products
    AC = hyperbolic_dot(p1, p3)
    BD = hyperbolic_dot(p2, p4)
    AD = hyperbolic_dot(p1, p4)
    BC = hyperbolic_dot(p2, p3)
    
    # Add small epsilon to prevent division by zero
    eps = torch.tensor(1e-8, dtype=torch.float64, device=p1.device)
    
    # Compute cross-ratio
    return (AC * BD) / (AD * BC + eps)

def verify_cross_ratio_preservation(
    points_before: torch.Tensor,
    points_after: torch.Tensor,
    rtol: float = 1e-4
) -> bool:
    """Verify that cross-ratio is preserved between two sets of points.
    
    Args:
        points_before: Original points [N, D+1] with homogeneous coordinates
        points_after: Transformed points [N, D+1] with homogeneous coordinates
        rtol: Relative tolerance for comparison
        
    Returns:
        True if cross-ratio is preserved within tolerance
    """
    if points_before.size(0) < 4 or points_after.size(0) < 4:
        return True  # Not enough points to compute cross-ratio
        
    # Compute cross-ratios for all possible quadruples
    N = points_before.size(0)
    preserved = True
    
    for i in range(N-3):
        for j in range(i+1, N-2):
            for k in range(j+1, N-1):
                for l in range(k+1, N):
                    cr_before = compute_cross_ratio(
                        points_before[i],
                        points_before[j],
                        points_before[k],
                        points_before[l]
                    )
                    
                    cr_after = compute_cross_ratio(
                        points_after[i],
                        points_after[j],
                        points_after[k],
                        points_after[l]
                    )
                    
                    # Skip if either cross-ratio is invalid
                    if torch.isnan(cr_before) or torch.isnan(cr_after):
                        continue
                        
                    # Compare cross-ratios in log space
                    log_cr_before = torch.log(cr_before + 1e-8)
                    log_cr_after = torch.log(cr_after + 1e-8)
                    
                    if not torch.allclose(log_cr_before, log_cr_after, rtol=rtol):
                        preserved = False
                        break
                        
            if not preserved:
                break
                
        if not preserved:
            break
            
    return preserved

def restore_cross_ratio(
    points: torch.Tensor,
    target_cr: torch.Tensor,
    preserve_norm: bool = True
) -> torch.Tensor:
    """Restore cross-ratio of points to match target value.
    
    Args:
        points: Points to adjust [N, D+1] with homogeneous coordinates
        target_cr: Target cross-ratio value
        preserve_norm: Whether to preserve unit norm after scaling
        
    Returns:
        Points with restored cross-ratio
    """
    if points.size(0) < 4:
        return points
        
    # Compute current cross-ratio
    current_cr = compute_cross_ratio(points[0], points[1], points[2], points[3])
    
    # Check if adjustment is needed
    if torch.isnan(current_cr) or torch.isnan(target_cr) or current_cr == 0:
        return points
        
    # Compute scale factor in log space for better numerical stability
    log_scale = 0.5 * (torch.log(target_cr + 1e-8) - torch.log(current_cr + 1e-8))
    scale = torch.exp(log_scale)
    
    # Extract features and homogeneous coordinates
    features = points[..., :-1]
    homogeneous = points[..., -1:]
    
    # Apply scale to features
    scaled_features = features * scale
    
    # Normalize if requested
    if preserve_norm:
        norm = torch.norm(scaled_features, p=2, dim=-1, keepdim=True)
        scaled_features = scaled_features / (norm + 1e-8)
        
    # Combine with homogeneous coordinates
    return torch.cat([scaled_features, homogeneous], dim=-1)
