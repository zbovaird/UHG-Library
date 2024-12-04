import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .base import ProjectiveLayer
from .attention import ProjectiveAttention
from ...projective import ProjectiveUHG

class ProjectiveGraphConv(ProjectiveLayer):
    """Graph convolution layer using projective geometry.
    
    This layer performs graph convolution operations using pure projective
    geometry, following UHG principles without any manifold concepts.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        use_attention: Whether to use projective attention
        heads: Number of attention heads if using attention
        concat: Whether to concatenate attention heads
        dropout: Dropout probability
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_attention: bool = True,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias)
        self.use_attention = use_attention
        
        # Initialize attention if used
        if use_attention:
            self.attention = ProjectiveAttention(
                in_features=in_features,
                out_features=out_features,
                heads=heads,
                concat=concat,
                dropout=dropout,
                bias=bias
            )
            
    def aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate neighborhood features using projective operations.
        
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
            # Use mean aggregation with projective operations
            row, col = edge_index
            out = torch.zeros_like(x)
            
            # Transform and aggregate using projective operations
            for i in range(len(row)):
                # Create projective transformation
                matrix = self.uhg.get_projective_matrix(self.out_features)
                
                # Transform neighbor feature
                neighbor = self.uhg.transform(x[col[i]], matrix)
                
                # Add to output
                out[row[i]] = out[row[i]] + neighbor
                
            # Average using projective mean
            counts = torch.zeros(x.size(0), device=x.device)
            counts.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            counts = torch.clamp(counts, min=1)
            
            return out / counts.view(-1, 1)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass of projective graph convolution.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            size: Optional output size
            
        Returns:
            Convoluted features
        """
        # Transform node features
        x = super().forward(x)
        
        # Aggregate neighborhood information
        out = self.aggregate_neighbors(x, edge_index, size)
        
        return out