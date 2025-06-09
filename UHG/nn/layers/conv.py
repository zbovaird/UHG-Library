import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .base import UHGLayer
from .attention import ProjectiveAttention

class UHGConv(UHGLayer):
    """UHG-compliant graph convolution layer.
    
    This layer implements graph convolution using pure projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize projective transformations
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Apply projective transformation
        out = self.projective_transform(x, self.weight)
        
        # Message passing
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
            
        # Normalize edge weights
        edge_weight = edge_weight / edge_weight.sum()
        
        # Aggregate messages
        out = torch.zeros_like(x)
        for i in range(len(row)):
            src, dst = row[i], col[i]
            out[dst] += edge_weight[i] * x[src]
            
        if self.bias is not None:
            # Add bias in projective space
            bias_point = torch.cat([self.bias, torch.ones_like(self.bias[:1])], dim=0)
            bias_point = bias_point / torch.norm(bias_point)
            out = self.uhg.projective_average(
                torch.stack([out, bias_point.expand_as(out)]),
                torch.tensor([0.9, 0.1], device=x.device)
            )
            
        # Return normalized feature part
        features = out[..., :-1]
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norm + 1e-8)