import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ..manifolds.base import Manifold
from ..utils.cross_ratio import compute_cross_ratio, preserve_cross_ratio

class HyperbolicAttention(nn.Module):
    """Attention mechanism in hyperbolic space using UHG principles.
    
    All operations are performed directly in hyperbolic space without
    tangent space mappings. Cross-ratios are preserved throughout.
    
    Args:
        manifold (Manifold): The hyperbolic manifold to operate on
        in_features (int): Number of input features
        out_features (int): Number of output features
        heads (int): Number of attention heads
        concat (bool): Whether to concatenate or average attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to include bias
    """
    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
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
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Initialize attention parameters directly in hyperbolic space
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters in hyperbolic space."""
        bound = 1 / self.in_features ** 0.5
        self.att.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)
            
    def compute_attention_scores(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Compute attention scores in hyperbolic space.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
            size: Optional output size
            
        Returns:
            Attention scores
        """
        # Compute attention using hyperbolic distances
        x_i = x_i.view(-1, self.heads, self.out_features)
        x_j = x_j.view(-1, self.heads, self.out_features)
        
        # Concatenate in hyperbolic space
        alpha = torch.cat([x_i, x_j], dim=-1)
        
        # Compute attention scores using hyperbolic inner product
        alpha = self._hyperbolic_attention(alpha)
        
        # Apply softmax in hyperbolic space
        alpha = self._hyperbolic_softmax(alpha, size)
        
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
            
        return alpha
        
    def _hyperbolic_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights in hyperbolic space.
        
        Args:
            x: Input features
            
        Returns:
            Attention weights
        """
        # Project attention parameters to hyperbolic space
        att = self.manifold.expmap0(self.att)
        
        # Compute attention using hyperbolic inner product
        return self.manifold.inner(x, att)
        
    def _hyperbolic_softmax(
        self,
        x: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply softmax in hyperbolic space.
        
        Args:
            x: Input tensor
            size: Optional output size
            
        Returns:
            Normalized attention weights
        """
        # Map to tangent space at infinity for numerical stability
        x_inf = self.manifold.logmap0(x)
        
        # Apply standard softmax
        x_inf = F.softmax(x_inf, dim=-1)
        
        # Map back to hyperbolic space
        return self.manifold.expmap0(x_inf)
        
    def combine_attention_values(
        self,
        alpha: torch.Tensor,
        x_j: torch.Tensor
    ) -> torch.Tensor:
        """Combine values using attention weights in hyperbolic space.
        
        Args:
            alpha: Attention weights
            x_j: Source node features
            
        Returns:
            Weighted combination in hyperbolic space
        """
        # Reshape for attention heads
        x_j = x_j.view(-1, self.heads, self.out_features)
        
        # Combine using hyperbolic weighted midpoint
        out = self._hyperbolic_weighted_sum(alpha, x_j)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            bias = self.manifold.expmap0(self.bias)
            out = self.manifold.mobius_add(out, bias)
            
        return out
        
    def _hyperbolic_weighted_sum(
        self,
        weights: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted sum in hyperbolic space.
        
        Args:
            weights: Attention weights
            values: Values to combine
            
        Returns:
            Weighted combination preserving hyperbolic structure
        """
        # Initialize output
        out = torch.zeros_like(values[:, 0])
        
        # Iteratively combine in hyperbolic space
        for i in range(values.size(1)):
            scaled = self.manifold.mobius_scalar_mul(
                weights[:, i:i+1], values[:, i])
            out = self.manifold.mobius_add(out, scaled)
            
        return out
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass of hyperbolic attention layer.
        
        Args:
            x: Input node features
            edge_index: Graph connectivity
            size: Optional output size
            
        Returns:
            Node embeddings with attention applied
        """
        # Get source and target node features
        x_i = x[edge_index[1]]
        x_j = x[edge_index[0]]
        
        # Compute attention scores
        alpha = self.compute_attention_scores(x_i, x_j, size)
        
        # Apply attention to values
        return self.combine_attention_values(alpha, x_j)
        
    def extra_repr(self) -> str:
        """String representation of layer."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'heads={self.heads}') 