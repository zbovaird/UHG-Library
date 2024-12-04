import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .base import ProjectiveLayer
from ...projective import ProjectiveUHG

class ProjectiveAttention(ProjectiveLayer):
    """Attention layer using projective geometry.
    
    This layer implements attention mechanisms using pure projective geometry,
    following UHG principles without any manifold concepts.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        heads: Number of attention heads
        concat: Whether to concatenate attention heads
        dropout: Dropout probability
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__(in_features, out_features, bias)
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Define output dimension based on concatenation
        self.out_dim = out_features * heads if concat else out_features
        
        # Initialize attention weights as projective transformations
        self.att_weight = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))
        self.reset_attention_parameters()
        
    def reset_attention_parameters(self):
        """Initialize attention parameters as projective transformations."""
        matrix = self.uhg.get_projective_matrix(2 * self.out_features)
        self.att_weight.data.copy_(matrix[:-1].view(1, 1, -1).repeat(1, self.heads, 1))
        
    def compute_attention_scores(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores using cross-ratios.
        
        Args:
            x_i: Source node features
            x_j: Target node features
            
        Returns:
            Attention scores
        """
        # Compute cross-ratio between points
        cross_ratio = self.uhg.cross_ratio(x_i, x_j, x_i, x_j)
        
        # Convert to attention scores
        return F.softmax(-cross_ratio, dim=-1)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass of projective attention.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            size: Optional output size
            
        Returns:
            Attention-weighted features
        """
        # Transform features using projective geometry
        x = super().forward(x)
        x = x.view(-1, self.heads, self.out_features)
        
        # Get source and target nodes
        row, col = edge_index
        
        # Compute attention scores using cross-ratios
        alpha = self.compute_attention_scores(x[row], x[col])
        
        # Apply dropout to attention scores
        if self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
            
        # Apply attention weights using projective transformations
        out = torch.zeros_like(x)
        for i in range(self.heads):
            # Create projective transformation from attention weights
            att_matrix = self.uhg.get_projective_matrix(self.out_features)
            att_matrix = att_matrix * alpha[:, i].view(-1, 1, 1)
            
            # Transform features
            head_out = self.uhg.transform(x[col, i], att_matrix)
            
            # Aggregate using projective operations
            out[:, i] = self.uhg.transform(head_out, self.uhg.get_projective_matrix(self.out_features))
            
        if not self.concat:
            # Average attention heads using projective mean
            out = torch.mean(out, dim=1)
        else:
            out = out.view(-1, self.out_dim)
            
        return out