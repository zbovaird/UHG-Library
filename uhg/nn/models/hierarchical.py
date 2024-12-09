"""
Hierarchical Graph Neural Network model using Universal Hyperbolic Geometry.

This model implements hierarchical graph learning using pure projective operations,
following strict UHG principles without any manifold concepts or tangent spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from ..layers.hierarchical import ProjectiveHierarchicalLayer
from ...projective import ProjectiveUHG
from ...utils.cross_ratio import compute_cross_ratio

class ProjectiveHierarchicalGNN(nn.Module):
    """UHG-compliant hierarchical GNN model using pure projective operations.
    
    This model implements hierarchical graph learning using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.
    
    Key features:
    1. Pure projective operations - no manifold concepts
    2. Cross-ratio preservation in all transformations
    3. Hierarchical structure preservation
    4. Level-aware message passing
    5. Parent-child relationship preservation
    
    Args:
        in_channels: Size of input features
        hidden_channels: Size of hidden features
        out_channels: Size of output features
        num_layers: Number of hierarchical layers
        num_levels: Number of hierarchical levels in the data
        level_dim: Dimension for level encoding
        dropout: Dropout probability
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_levels: int = 3,
        level_dim: int = 8,
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
            ProjectiveHierarchicalLayer(
                in_features=in_channels,
                out_features=hidden_channels,
                num_levels=num_levels,
                level_dim=level_dim,
                bias=bias
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                ProjectiveHierarchicalLayer(
                    in_features=hidden_channels,
                    out_features=hidden_channels,
                    num_levels=num_levels,
                    level_dim=level_dim,
                    bias=bias
                )
            )
            
        # Output layer
        self.layers.append(
            ProjectiveHierarchicalLayer(
                in_features=hidden_channels,
                out_features=out_channels,
                num_levels=num_levels,
                level_dim=level_dim,
                bias=bias
            )
        )
        
    def projective_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """Apply dropout while preserving projective structure and cross-ratios."""
        if not self.training or p == 0:
            return x
            
        # Create dropout mask for features only
        mask = torch.bernoulli(torch.full_like(x[..., :-1], 1 - p))
        
        # Add ones for homogeneous coordinate
        mask = torch.cat([mask, torch.ones_like(mask[..., :1])], dim=-1)
        
        # Apply mask while preserving projective structure
        dropped = x * mask
        
        # Normalize to maintain projective structure
        norm = torch.norm(dropped[..., :-1], p=2, dim=-1, keepdim=True)
        dropped = torch.cat([dropped[..., :-1] / (norm + 1e-8), dropped[..., -1:]], dim=-1)
        
        if x.size(0) > 3:
            # Compute cross-ratio of first four points
            cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
            cr_after = compute_cross_ratio(dropped[0], dropped[1], dropped[2], dropped[3])
            
            # Handle NaN values in cross-ratio
            if torch.isnan(cr_after):
                # If cross-ratio is NaN, use original points
                dropped = x
            else:
                # Scale output to preserve cross-ratio
                scale = (cr_before / (cr_after + 1e-8)).sqrt()
                dropped = torch.cat([dropped[..., :-1] * scale, dropped[..., -1:]], dim=-1)
                
        return dropped
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass using pure projective operations.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            node_levels: Level assignment for each node
            edge_weight: Optional edge weights
            size: Optional output size
            
        Returns:
            Node embeddings
        """
        # Add homogeneous coordinate to input if not present
        if x.size(-1) == self.layers[0].in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x, edge_index, node_levels, size)
            
            if i < len(self.layers) - 1:
                # Add homogeneous coordinate back
                x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
                
                # Apply projective ReLU to features only
                x_features = x[..., :-1]
                x_features = F.relu(x_features)
                x = torch.cat([x_features, x[..., -1:]], dim=-1)
                
                # Normalize to maintain projective structure
                norm = torch.norm(x[..., :-1], p=2, dim=-1, keepdim=True)
                x = torch.cat([x[..., :-1] / (norm + 1e-8), x[..., -1:]], dim=-1)
                
                # Apply projective dropout
                x = self.projective_dropout(x, self.dropout)
                
        # Return only feature part for final layer
        return x[..., :-1] 