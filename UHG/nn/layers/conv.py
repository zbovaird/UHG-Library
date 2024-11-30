import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ...manifolds.base import Manifold
from ..message import HyperbolicMessagePassing
from ..attention import HyperbolicAttention
from ...utils.cross_ratio import compute_cross_ratio, preserve_cross_ratio

class HyperbolicGraphConv(nn.Module):
    """Hyperbolic Graph Convolution Layer using UHG principles.
    
    This layer performs graph convolution operations directly in hyperbolic space
    using projective geometric calculations without tangent space mappings.
    
    Args:
        manifold (Manifold): The hyperbolic manifold to operate on
        in_features (int): Number of input features
        out_features (int): Number of output features
        use_attention (bool): Whether to use hyperbolic attention
        heads (int): Number of attention heads if using attention
        concat (bool): Whether to concatenate attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to include bias
    """
    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        use_attention: bool = True,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.use_attention = use_attention
        
        # Initialize message passing module
        self.message_passing = HyperbolicMessagePassing(
            manifold=manifold,
            aggr='mean',
            flow='source_to_target'
        )
        
        # Initialize attention if used
        if use_attention:
            self.attention = HyperbolicAttention(
                manifold=manifold,
                in_features=in_features,
                out_features=out_features,
                heads=heads,
                concat=concat,
                dropout=dropout,
                bias=bias
            )
            
        # Initialize weight matrix in hyperbolic space
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters in hyperbolic space."""
        bound = 1 / self.in_features ** 0.5
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)
            
    def hyperbolic_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weight transformation in hyperbolic space.
        
        Args:
            x: Input features
            
        Returns:
            Transformed features
        """
        # Project weight matrix to hyperbolic space
        weight = self.manifold.expmap0(self.weight)
        
        # Transform using hyperbolic matrix multiplication
        out = self.manifold.mobius_matvec(weight, x)
        
        if self.bias is not None:
            bias = self.manifold.expmap0(self.bias)
            out = self.manifold.mobius_add(out, bias)
            
        return out
        
    def hyperbolic_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate neighborhood features in hyperbolic space.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            size: Optional output size
            
        Returns:
            Aggregated features
        """
        if self.use_attention:
            # Use attention-based aggregation
            return self.attention(x, edge_index, size)
        else:
            # Use standard message passing
            return self.message_passing.propagate(
                edge_index=edge_index,
                x=x,
                size=size
            )
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass of hyperbolic graph convolution.
        
        Args:
            x: Input node features
            edge_index: Graph connectivity
            size: Optional output size
            
        Returns:
            Convoluted node features
        """
        # Transform node features
        x = self.hyperbolic_transform(x)
        
        # Aggregate neighborhood information
        out = self.hyperbolic_aggregate(x, edge_index, size)
        
        # Ensure output satisfies hyperbolic constraints
        out = self.manifold.normalize(out)
        
        return out
        
    def extra_repr(self) -> str:
        """String representation of layer."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'use_attention={self.use_attention}')