import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from ...manifolds import HyperbolicManifold

class HyperbolicLinear(nn.Module):
    """Hyperbolic linear layer.
    
    This layer performs a linear transformation in the tangent space of the hyperbolic
    manifold, followed by an exponential map to project back onto the manifold.
    
    Attributes:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias
        weight (nn.Parameter): Learnable weight matrix
        bias (nn.Parameter): Learnable bias vector
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
        self.has_bias = bias
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize bias if present
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hyperbolic linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [..., out_features]
        """
        # Project input to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        
        # Apply linear transformation in tangent space
        out_tangent = torch.matmul(x_tangent, self.weight.t())
        
        # Add bias if present
        if self.bias is not None:
            out_tangent = out_tangent + self.bias
        
        # Project back to hyperbolic space
        return self.manifold.expmap0(out_tangent)
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}' 