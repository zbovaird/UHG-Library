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
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

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
            
        # Store original cross-ratio if enough points
        has_cr = x.size(0) > 3
        if has_cr:
            cr_before = self.uhg.cross_ratio(x[0], x[1], x[2], x[3])
            
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Create dropout mask for features
        mask = torch.bernoulli(torch.full_like(features, 1 - p))
        
        # Ensure at least one feature survives per node
        zero_rows = (mask.sum(dim=-1) == 0)
        if zero_rows.any():
            # For rows with all zeros, randomly keep one feature
            random_feature = torch.randint(0, features.size(-1), (zero_rows.sum(),), device=x.device)
            mask[zero_rows, random_feature] = 1
            
        # Scale mask to maintain expected value
        mask = mask / (1 - p + 1e-8)
        
        # Apply dropout to features
        dropped_features = features * mask
        
        # Normalize features using UHG
        dropped_features = self.uhg.normalize(dropped_features)
        
        # Combine with homogeneous coordinate
        dropped = torch.cat([dropped_features, homogeneous], dim=-1)
        
        # Restore cross-ratio if needed
        if has_cr:
            cr_after = self.uhg.cross_ratio(dropped[0], dropped[1], dropped[2], dropped[3])
            if not torch.isnan(cr_after) and not torch.isnan(cr_before) and cr_after != 0:
                dropped = self.uhg.scale(dropped, torch.sqrt(torch.abs(cr_before / cr_after)))
                
        return dropped
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Store initial cross-ratio if enough points
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = self.uhg.cross_ratio(x[0], x[1], x[2], x[3])
            
        # Add homogeneous coordinate to input if not present
        if x.size(-1) == self.layers[0].in_features:
            x = self.uhg.normalize(x)
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
                
                # Normalize features using UHG
                x_features = self.uhg.normalize(x_features)
                x = torch.cat([x_features, x[..., -1:]], dim=-1)
                
                # Apply projective dropout
                x = self.projective_dropout(x, self.dropout)
                
                # Restore initial cross-ratio if possible
                if has_cr:
                    cr_current = self.uhg.cross_ratio(x[0], x[1], x[2], x[3])
                    if not torch.isnan(cr_current) and not torch.isnan(cr_initial) and cr_current != 0:
                        x = self.uhg.scale(x, torch.sqrt(torch.abs(cr_initial / cr_current)))
            else:
                # For final layer, ensure output dimension matches out_channels
                if x.size(-1) != self.layers[-1].out_features + 1:
                    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
                    
        # Return normalized feature part using UHG
        features = x[..., :-1]
        features = self.uhg.normalize(features)
        
        # Ensure output has correct dimension
        if features.size(-1) != self.layers[-1].out_features:
            pad_size = self.layers[-1].out_features - features.size(-1)
            if pad_size > 0:
                # Pad with zeros if needed
                features = F.pad(features, (0, pad_size))
            else:
                # Truncate if too large
                features = features[..., :self.layers[-1].out_features]
                
        return features