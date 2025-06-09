"""Cross-ratio computation utilities for UHG."""

import torch
from typing import Tuple, Optional

def compute_cross_ratio(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-ratio of four points.
    Reference: UHG.pdf, Ch. 2
    """
    # Use determinants for projective cross-ratio
    def det2(a, b):
        return a[0]*b[1] - a[1]*b[0]
    # For 3D, project to 2D by dropping last coordinate
    a_2d = a[..., :2]
    b_2d = b[..., :2]
    c_2d = c[..., :2]
    d_2d = d[..., :2]
    num = det2(a_2d, c_2d) * det2(b_2d, d_2d)
    denom = det2(a_2d, d_2d) * det2(b_2d, c_2d)
    if torch.any(denom.abs() < 1e-8):
        print("[WARNING] cross_ratio: denominator near zero.")
    cr = num / (denom + 1e-8)
    return cr

def verify_cross_ratio_preservation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, transformed_a: torch.Tensor, transformed_b: torch.Tensor, transformed_c: torch.Tensor, transformed_d: torch.Tensor) -> bool:
    """
    Verify that the cross-ratio is preserved under a projective transformation.
    Reference: UHG.pdf, Ch. 2
    """
    original_cr = compute_cross_ratio(a, b, c, d)
    transformed_cr = compute_cross_ratio(transformed_a, transformed_b, transformed_c, transformed_d)
    return torch.allclose(original_cr, transformed_cr, rtol=1e-5, atol=1e-5)

def restore_cross_ratio(x: torch.Tensor, target_cr: torch.Tensor) -> torch.Tensor:
    """
    Restore the cross-ratio of points to a target value.
    Reference: UHG.pdf, Ch. 2
    """
    # Normalize points
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    # Compute current cross-ratio
    current_cr = compute_cross_ratio(x[0], x[1], x[2], x[3])
    # Compute scaling factor
    scale = torch.sqrt(target_cr / current_cr)
    # Apply scaling
    x = x * scale
    return x
