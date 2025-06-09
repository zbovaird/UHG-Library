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

"""UHG-compliant functional operations."""

def scatter_mean_custom(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    """UHG-compliant scatter mean operation.
    
    Args:
        src: Source tensor of shape [E, D] where E is number of edges and D is feature dimension
        index: Index tensor of shape [E] indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of output tensor's scatter dimension
        
    Returns:
        Tensor of shape [N, D] where N is number of nodes
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    # Initialize output tensor
    out = torch.zeros(dim_size, src.size(1), device=src.device)
    
    # Add features
    out = out.index_add_(0, index, src)
    
    # Compute counts for averaging
    ones = torch.ones((src.size(0), 1), device=src.device)
    count = torch.zeros(dim_size, 1, device=src.device).index_add_(0, index, ones)
    count[count == 0] = 1
    
    # Average features
    out = out / count
    
    return out

def uhg_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """UHG-compliant normalization that preserves geometric structure.
    
    Args:
        x: Input tensor with shape [..., D+1] where last dimension is homogeneous coordinate
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    features = x[..., :-1]
    homogeneous = x[..., -1:]
    
    # Compute UHG norm
    norm = torch.sqrt(torch.clamp(
        torch.sum(features ** 2, dim=-1, keepdim=True) - homogeneous ** 2,
        min=eps
    ))
    
    # Normalize features
    features = features / norm
    
    # Ensure homogeneous coordinate is positive
    homogeneous = torch.sign(homogeneous) * torch.ones_like(homogeneous)
    
    return torch.cat([features, homogeneous], dim=-1)

def uhg_relu(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """UHG-compliant ReLU that preserves geometric structure.
    
    Args:
        x: Input tensor
        eps: Small value for numerical stability
        
    Returns:
        Activated tensor with same shape as input
    """
    # Apply ReLU while preserving norm ratios
    activated = F.relu(x)
    norm = torch.norm(activated, p=2, dim=-1, keepdim=True)
    return activated / norm.clamp(min=eps) 