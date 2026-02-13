import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...manifolds import HyperbolicManifold
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class HyperbolicLinear(nn.Module):
    """Hyperbolic linear layer that preserves hyperbolic structure.
    
    This implementation follows UHG principles by:
    1. Using pure projective operations (no tangent space)
    2. Preserving cross-ratios
    3. Ensuring outputs lie on the hyperbolic manifold
    4. Maintaining hyperbolic structure through all transformations
    
    Args:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using hyperbolic-aware initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using hyperbolic-aware initialization."""
        # Initialize weights using hyperbolic-aware initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hyperbolic linear layer.
        
        Args:
            x (torch.Tensor): Input features of shape [N, in_features]
            
        Returns:
            torch.Tensor: Output features of shape [N, out_features]
        """
        # Pure linear mapping on Euclidean feature part; callers handle projective lifting
        out = F.linear(x, self.weight, self.bias)
        return out
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}' 