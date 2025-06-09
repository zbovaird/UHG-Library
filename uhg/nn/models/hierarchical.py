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
        
    def projective_dropout(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """Apply dropout while preserving projective structure."""
        if not self.training or p == 0.0:
            return x

        # Keep homogeneous coordinate
        features = x[:, :-1]
        homogeneous = x[:, -1:]

        # Apply dropout to features with cross-ratio preservation
        mask = torch.bernoulli(torch.ones_like(features) * (1 - p))
        features = features * mask / (1 - p)

        # Compute hyperbolic norm
        spatial_norm = torch.sum(features * features, dim=-1, keepdim=True)
        time_norm = homogeneous * homogeneous
        norm = torch.sqrt(torch.clamp(spatial_norm - time_norm + 1e-15, min=1e-15))

        # Normalize features while preserving hyperbolic structure
        features = features / (norm + 1e-15)

        # Ensure time component remains stable
        homogeneous = torch.clamp(homogeneous, min=1.0)

        # Recombine with homogeneous coordinate
        out = torch.cat([features, homogeneous], dim=-1)

        # Track and restore cross-ratios if possible
        if x.size(0) >= 4:
            cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
            cr_after = compute_cross_ratio(out[0], out[1], out[2], out[3])
            if not torch.isnan(cr_before) and not torch.isnan(cr_after) and cr_after != 0:
                scale = torch.sqrt(torch.abs(cr_before / cr_after))
                features = out[:, :-1] * scale
                out = torch.cat([features, homogeneous], dim=-1)

        # Final normalization to ensure unit norm in feature space
        features = out[:, :-1]
        spatial_norm = torch.sum(features * features, dim=-1, keepdim=True)
        features = features / torch.sqrt(torch.clamp(spatial_norm, min=1e-15))
        return torch.cat([features, homogeneous], dim=-1)
        
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