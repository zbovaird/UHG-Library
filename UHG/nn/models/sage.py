import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from ..layers.sage import ProjectiveSAGEConv
from ...projective import ProjectiveUHG

class ProjectiveGraphSAGE(nn.Module):
    """UHG-compliant GraphSAGE model for graph learning.
    
    This model implements GraphSAGE using pure projective geometry,
    following UHG principles without any manifold concepts.
    
    Args:
        in_channels: Size of input features
        hidden_channels: Size of hidden features
        out_channels: Size of output features
        num_layers: Number of GraphSAGE layers
        dropout: Dropout probability
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create list to hold all layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            ProjectiveSAGEConv(
                in_features=in_channels,
                out_features=hidden_channels,
                bias=bias
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                ProjectiveSAGEConv(
                    in_features=hidden_channels,
                    out_features=hidden_channels,
                    bias=bias
                )
            )
            
        # Output layer
        self.layers.append(
            ProjectiveSAGEConv(
                in_features=hidden_channels,
                out_features=out_channels,
                bias=bias
            )
        )
        
    def projective_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """Apply dropout while preserving projective structure."""
        if not self.training or p == 0:
            return x
            
        # Create dropout mask
        mask = torch.bernoulli(torch.full_like(x[..., :-1], 1 - p))
        
        # Add ones for homogeneous coordinate
        mask = torch.cat([mask, torch.ones_like(mask[..., :1])], dim=-1)
        
        # Apply mask while preserving projective structure
        dropped = x * mask
        
        # Normalize to maintain projective structure
        norm = torch.norm(dropped, p=2, dim=-1, keepdim=True)
        return dropped / (norm + 1e-8)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of projective GraphSAGE.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            edge_weight: Optional edge weights
            
        Returns:
            Node embeddings
        """
        # Add homogeneous coordinate to input
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x, edge_index)
            
            # Add homogeneous coordinate back
            if i < len(self.layers) - 1:
                x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
                
                # Apply projective ReLU
                x_features = x[..., :-1]
                x_features = F.relu(x_features)
                x = torch.cat([x_features, x[..., -1:]], dim=-1)
                
                # Normalize to maintain projective structure
                norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                x = x / (norm + 1e-8)
                
                # Apply projective dropout
                x = self.projective_dropout(x, self.dropout)
                
        return x