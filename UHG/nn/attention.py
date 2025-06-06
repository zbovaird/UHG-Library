import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..manifolds import HyperbolicManifold

class HyperbolicAttention(nn.Module):
    """Hyperbolic attention mechanism.
    
    This module implements attention in hyperbolic space by computing attention
    scores in the tangent space and then projecting back to the hyperbolic manifold.
    Supports edge features of arbitrary dimension by projecting them to the input dimension if needed.
    
    Attributes:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        concat (bool): Whether to concatenate attention heads
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_channels: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True
    ):
        super().__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        # Attention parameters
        self.query = nn.Linear(in_channels, in_channels * num_heads)
        self.key = nn.Linear(in_channels, in_channels * num_heads)
        self.value = nn.Linear(in_channels, in_channels * num_heads)
        
        # Output projection
        self.proj = nn.Linear(in_channels * num_heads, in_channels)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Edge feature projection (initialized lazily)
        self.edge_mlp = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the hyperbolic attention layer.
        
        Args:
            x (torch.Tensor): Node features of shape [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, E]
            edge_attr (torch.Tensor, optional): Edge features of shape [E, edge_dim]
            
        Returns:
            torch.Tensor: Updated node features of shape [N, in_channels]
        """
        # Project node features to tangent space
        x_tangent = self.manifold.logmap0(x)
        
        # Compute queries, keys, and values
        queries = self.query(x_tangent).view(-1, self.num_heads, self.in_channels)
        keys = self.key(x_tangent).view(-1, self.num_heads, self.in_channels)
        values = self.value(x_tangent).view(-1, self.num_heads, self.in_channels)
        
        # Get source and target nodes
        row, col = edge_index
        
        # Compute attention scores
        attn_scores = torch.sum(queries[row] * keys[col], dim=-1) / math.sqrt(self.in_channels)
        
        # Add edge features to attention scores if provided
        if edge_attr is not None:
            if edge_attr.shape[-1] != self.in_channels:
                # Lazily initialize edge_mlp if needed
                if self.edge_mlp is None or self.edge_mlp.in_features != edge_attr.shape[-1]:
                    self.edge_mlp = nn.Linear(edge_attr.shape[-1], self.in_channels).to(edge_attr.device)
                edge_proj = self.edge_mlp(edge_attr)
            else:
                edge_proj = edge_attr
            # Expand edge_proj for all heads
            edge_proj = edge_proj.unsqueeze(1).expand(-1, self.num_heads, -1)
            attn_scores = attn_scores + edge_proj.sum(dim=-1)  # sum or mean, depending on design
        
        # Apply softmax to get attention weights
        alpha = F.softmax(attn_scores, dim=1)
        
        # Apply dropout
        alpha = self.dropout_layer(alpha)
        
        # Aggregate values with attention weights
        out = torch.zeros(x.size(0), self.num_heads, self.in_channels, device=x.device)
        value_messages = values[col]
        # Add edge features to values if provided
        if edge_attr is not None:
            value_messages = value_messages + edge_proj
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.in_channels),
                        value_messages * alpha.unsqueeze(-1))
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.num_heads * self.in_channels)
        else:
            out = out.mean(dim=1)
        
        # Project to output dimension
        out = self.proj(out)
        
        # Project back to hyperbolic space
        return self.manifold.expmap0(out)
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (f'in_channels={self.in_channels}, num_heads={self.num_heads}, '
                f'dropout={self.dropout}, concat={self.concat}') 