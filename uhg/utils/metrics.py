"""UHG-specific metrics and evaluation utilities."""

import torch

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute UHG inner product between points.
    
    Args:
        a: First point tensor of shape [..., D+1]
        b: Second point tensor of shape [..., D+1]
        
    Returns:
        Inner product value(s)
    """
    return torch.sum(a[..., :-1] * b[..., :-1], dim=-1) - a[..., -1] * b[..., -1]

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """Compute UHG norm of points.
    
    Args:
        a: Point tensor of shape [..., D+1]
        
    Returns:
        Norm value(s)
    """
    return torch.sum(a[..., :-1] ** 2, dim=-1) - a[..., -1] ** 2

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between points.
    
    Args:
        a: First point tensor of shape [..., D+1]
        b: Second point tensor of shape [..., D+1]
        eps: Small value for numerical stability
        
    Returns:
        Quadrance value(s)
    """
    dot_product = uhg_inner_product(a, b)
    norm_a = uhg_norm(a)
    norm_b = uhg_norm(b)
    denom = norm_a * norm_b
    denom = torch.clamp(denom.abs(), min=eps)
    quadrance = 1 - (dot_product ** 2) / denom
    return quadrance

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between lines.
    
    Args:
        L: First line tensor of shape [..., D+1]
        M: Second line tensor of shape [..., D+1]
        eps: Small value for numerical stability
        
    Returns:
        Spread value(s)
    """
    dot_product = uhg_inner_product(L, M)
    norm_L = uhg_norm(L)
    norm_M = uhg_norm(M)
    denom = norm_L * norm_M
    denom = torch.clamp(denom.abs(), min=eps)
    spread = 1 - (dot_product ** 2) / denom
    return spread

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