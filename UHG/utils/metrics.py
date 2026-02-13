"""UHG-specific metrics and evaluation utilities."""

import torch
import torch.nn.functional as F
from typing import Optional

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic inner product between two points in projective space.
    
    This function is fully vectorized and handles arbitrary batch dimensions.
    The hyperbolic inner product is defined as:
    <a,b> = a[..., :-1] @ b[..., :-1].T - a[..., -1] * b[..., -1]
    
    Args:
        a: First point tensor of shape (..., D+1) with homogeneous coordinates
        b: Second point tensor of shape (..., D+1) with homogeneous coordinates
        
    Returns:
        Hyperbolic inner product of shape (...)
    """
    # Handle broadcasting for different batch dimensions
    # Compute spatial dot product for all dimensions except the last
    spatial_dot = torch.sum(a[..., :-1] * b[..., :-1], dim=-1)
    
    # Compute time component product
    time_product = a[..., -1] * b[..., -1]
    
    # Return hyperbolic inner product
    return spatial_dot - time_product

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """
    Compute the hyperbolic norm of points in projective space.
    
    Args:
        a: Point tensor of shape (..., D+1) with homogeneous coordinates
        
    Returns:
        Hyperbolic norm of shape (...)
    """
    return torch.sqrt(torch.abs(uhg_inner_product(a, a)))

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute UHG quadrance between two points.
    
    This is a fundamental operation in UHG that measures the hyperbolic
    distance between points without using tangent space operations.
    
    Args:
        a: First point tensor of shape (..., D+1)
        b: Second point tensor of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Quadrance between points a and b of shape (...)
    """
    dot_product = uhg_inner_product(a, b)
    denom_a = uhg_inner_product(a, a)
    denom_b = uhg_inner_product(b, b)
    
    # Ensure numerical stability
    denom = torch.clamp(denom_a * denom_b, min=eps)
    
    return 1 - (dot_product * dot_product) / denom

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute UHG spread between two lines.
    
    Spread is the hyperbolic analog of angle between lines.
    
    Args:
        L: First line tensor of shape (..., D+1)
        M: Second line tensor of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Spread between lines L and M of shape (...)
    """
    dot_product = uhg_inner_product(L, M)
    denom_L = uhg_inner_product(L, L)
    denom_M = uhg_inner_product(M, M)
    
    # Ensure numerical stability
    denom = torch.clamp(denom_L * denom_M, min=eps)
    
    return 1 - (dot_product * dot_product) / denom

def check_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, 
                     threshold: float = 1e-5) -> bool:
    """Check if cross-ratio is preserved between four points.
    
    Args:
        p1, p2, p3, p4: Point tensors of shape [..., D+1]
        threshold: Tolerance for cross-ratio preservation
        
    Returns:
        Boolean indicating if cross-ratio is preserved
    """
    cr1 = compute_cross_ratio(p1, p2, p3, p4)
    cr2 = compute_cross_ratio(p4, p3, p2, p1)
    return torch.abs(cr1 * cr2 - 1.0) < threshold

def compute_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
                       eps: float = 1e-9) -> torch.Tensor:
    """Compute cross-ratio between four points.
    
    Args:
        p1, p2, p3, p4: Point tensors of shape [..., D+1]
        eps: Small value for numerical stability
        
    Returns:
        Cross-ratio value(s)
    """
    q12 = uhg_quadrance(p1, p2)
    q34 = uhg_quadrance(p3, p4)
    q13 = uhg_quadrance(p1, p3)
    q24 = uhg_quadrance(p2, p4)
    
    num = q12 * q34
    denom = q13 * q24
    denom = torch.clamp(denom, min=eps)
    
    return num / denom

def evaluate_cross_ratio_preservation(model: torch.nn.Module, data: torch.Tensor, 
                                    num_samples: int = 10) -> float:
    """Evaluate cross-ratio preservation of a model.
    
    Args:
        model: UHG-compliant model
        data: Input data tensor
        num_samples: Number of quadruples to test
        
    Returns:
        Fraction of preserved cross-ratios
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        preserved = 0
        
        for _ in range(num_samples):
            idx = torch.randperm(len(out))[:4]
            if check_cross_ratio(
                out[idx[0]], out[idx[1]],
                out[idx[2]], out[idx[3]]
            ):
                preserved += 1
                
        return preserved / num_samples 