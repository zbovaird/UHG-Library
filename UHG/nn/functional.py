import torch
import torch.nn.functional as F

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute UHG quadrance between two points.
    
    This is a fundamental operation in UHG that measures the hyperbolic
    distance between points without using tangent space operations.
    
    Args:
        a: First point tensor
        b: Second point tensor
        eps: Small value for numerical stability
        
    Returns:
        Quadrance between points a and b
    """
    dot_product = torch.sum(a * b, dim=-1)
    denom_a = torch.sum(a ** 2, dim=-1) - a[..., -1] ** 2 + eps
    denom_b = torch.sum(b ** 2, dim=-1) - b[..., -1] ** 2 + eps
    return 1 - (dot_product ** 2) / (denom_a * denom_b)

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute UHG spread between two lines.
    
    The spread is the hyperbolic analog of the angle between lines,
    computed directly in projective space.
    
    Args:
        L: First line tensor
        M: Second line tensor
        eps: Small value for numerical stability
        
    Returns:
        Spread between lines L and M
    """
    dot_product = torch.sum(L * M, dim=-1)
    denom_L = torch.sum(L ** 2, dim=-1) - L[..., -1] ** 2 + eps
    denom_M = torch.sum(M ** 2, dim=-1) - M[..., -1] ** 2 + eps
    return 1 - (dot_product ** 2) / (denom_L * denom_M)

def uhg_weighted_midpoint(points: torch.Tensor, weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute weighted midpoint of points in UHG space.
    
    This operation performs weighted averaging directly in hyperbolic space
    using the UHG formulation, without tangent space mappings.
    
    Args:
        points: Points tensor [batch_size, num_points, dim]
        weights: Weight tensor [batch_size, num_points]
        eps: Small value for numerical stability
        
    Returns:
        Weighted midpoint in UHG space
    """
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weighted_sum = torch.sum(points * weights.unsqueeze(-1), dim=-2)
    norm = torch.sqrt(torch.sum(weighted_sum ** 2, dim=-1, keepdim=True) - weighted_sum[..., -1:] ** 2 + eps)
    return weighted_sum / norm

def uhg_attention_kernel(query: torch.Tensor, key: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute attention scores using UHG quadrance.
    
    This implements attention directly in hyperbolic space using
    the quadrance between points as the similarity measure.
    
    Args:
        query: Query tensor
        key: Key tensor
        eps: Small value for numerical stability
        
    Returns:
        Attention scores
    """
    quad = uhg_quadrance(query.unsqueeze(-2), key.unsqueeze(-3), eps=eps)
    return F.softmax(-quad, dim=-1)

def uhg_aggregate(points: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Aggregate points using UHG-aware operations.
    
    Performs attention-weighted aggregation directly in hyperbolic space
    using UHG principles.
    
    Args:
        points: Points to aggregate [batch_size, num_points, dim]
        attention: Attention weights [batch_size, num_queries, num_points]
        
    Returns:
        Aggregated points in UHG space
    """
    return uhg_weighted_midpoint(points.unsqueeze(-3), attention) 