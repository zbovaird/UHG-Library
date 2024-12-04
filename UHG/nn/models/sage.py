import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from ..layers import ProjectiveGraphConv, ProjectiveAttention
from ...projective import ProjectiveUHG

class ProjectiveGraphSAGE(nn.Module):
    """GraphSAGE model using projective geometry.
    
    This model implements GraphSAGE using pure projective geometry,
    following UHG principles without any manifold concepts.
    
    Args:
        in_channels: Size of input features
        hidden_channels: Size of hidden features
        out_channels: Size of output features
        num_layers: Number of GraphSAGE layers
        dropout: Dropout probability
        use_attention: Whether to use attention
        heads: Number of attention heads
        concat: Whether to concatenate attention heads
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_attention: bool = False,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create list to hold all layers
        self.layers = nn.ModuleList()
        
        # Input layer
        if use_attention:
            self.layers.append(
                ProjectiveAttention(
                    in_features=in_channels,
                    out_features=hidden_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    bias=bias
                )
            )
        else:
            self.layers.append(
                ProjectiveGraphConv(
                    in_features=in_channels,
                    out_features=hidden_channels,
                    use_attention=False,
                    bias=bias
                )
            )
            
        # Hidden layers
        for _ in range(num_layers - 2):
            if use_attention:
                self.layers.append(
                    ProjectiveAttention(
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        heads=heads,
                        concat=concat,
                        dropout=dropout,
                        bias=bias
                    )
                )
            else:
                self.layers.append(
                    ProjectiveGraphConv(
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        use_attention=False,
                        bias=bias
                    )
                )
                
        # Output layer
        if use_attention:
            self.layers.append(
                ProjectiveAttention(
                    in_features=hidden_channels,
                    out_features=out_channels,
                    heads=1,
                    concat=True,
                    dropout=dropout,
                    bias=bias
                )
            )
        else:
            self.layers.append(
                ProjectiveGraphConv(
                    in_features=hidden_channels,
                    out_features=out_channels,
                    use_attention=False,
                    bias=bias
                )
            )
            
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
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x, edge_index)
            
            # Apply activation for all but last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                
            # Apply dropout
            if self.training and i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=True)
                
        return x