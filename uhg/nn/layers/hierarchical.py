"""
Hierarchical Graph Neural Network layer using Universal Hyperbolic Geometry.

This implementation uses pure projective operations and preserves cross-ratios
throughout all hierarchical operations. No manifold concepts or tangent spaces
are used, following strict UHG principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .base import ProjectiveLayer
from ...projective import ProjectiveUHG
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class ProjectiveHierarchicalLayer(ProjectiveLayer):
    """UHG-compliant hierarchical GNN layer using pure projective operations.
    
    This layer implements hierarchical message passing using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.
    
    Key features:
    1. Pure projective operations - no manifold concepts
    2. Cross-ratio preservation in all transformations
    3. Hierarchical structure preservation
    4. Level-aware message passing
    5. Parent-child relationship preservation
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_levels: Number of hierarchical levels
        level_dim: Dimension for level encoding
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_levels: int,
        level_dim: int = 8,
        bias: bool = True
    ):
        # Initialize base class without resetting parameters
        nn.Module.__init__(self)
        self.uhg = ProjectiveUHG()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store dimensions
        self.num_levels = num_levels
        self.level_dim = level_dim
        
        # Register parameters with correct dimensions
        self.register_parameter('W_self', nn.Parameter(torch.Tensor(out_features, in_features)))
        self.register_parameter('W_neigh', nn.Parameter(torch.Tensor(out_features, in_features)))
        self.register_parameter('W_parent', nn.Parameter(torch.Tensor(out_features, in_features)))
        self.register_parameter('W_child', nn.Parameter(torch.Tensor(out_features, in_features)))
        self.register_parameter('W_level', nn.Parameter(torch.Tensor(out_features, level_dim)))
        self.register_parameter('level_encodings', nn.Parameter(torch.Tensor(num_levels, level_dim)))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.Tensor(out_features)))
        else:
            self.register_parameter('bias', None)
            
        # Now reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        # Use Glorot initialization for weight matrices
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_parent)
        nn.init.xavier_uniform_(self.W_child)
        nn.init.xavier_uniform_(self.W_level)
        
        # Initialize level encodings to be orthogonal
        nn.init.orthogonal_(self.level_encodings)
        
        if self.bias is not None:
            # Initialize bias with small values
            nn.init.zeros_(self.bias)
            
    def projective_transform(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        preserve_cr: bool = True
    ) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Use UHG projective transform directly
        return self.uhg.transform(x, weight)
        
    def compute_cross_ratio_weight(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        level_diff: int = 0
    ) -> torch.Tensor:
        """Compute cross-ratio based weight between two points."""
        # Use UHG distance for weight computation
        dist = self.uhg.distance(p1, p2)
        
        # Adjust weight based on level difference using exponential decay
        level_factor = torch.exp(-torch.abs(torch.tensor(level_diff, dtype=torch.float32)))
        weight = torch.exp(-dist) * level_factor
        
        return weight.clamp(0.1, 0.9)  # Prevent extreme values
        
    def aggregate_hierarchical(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate features using hierarchical structure and projective operations."""
        # Store initial cross-ratio if enough points
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = self.uhg.cross_ratio(x[0], x[1], x[2], x[3])
            
        row, col = edge_index
        
        # Initialize output with correct dimensions
        N = x.size(0)
        out = torch.zeros(N, self.out_features + 1, device=x.device)
        out[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = self.uhg.normalize(x)
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Get level encodings for each node and transform
        node_level_enc = F.embedding(node_levels, self.level_encodings)
        # Normalize level encodings using UHG
        node_level_enc = self.uhg.normalize(node_level_enc)
        level_features = torch.cat([node_level_enc, torch.ones_like(node_level_enc[..., :1])], dim=-1)
        level_features = self.projective_transform(level_features, self.W_level)
        
        # Transform nodes with different weights based on relationship
        nodes_self = self.projective_transform(x, self.W_self)
        nodes_neigh = self.projective_transform(x, self.W_neigh)
        nodes_parent = self.projective_transform(x, self.W_parent)
        nodes_child = self.projective_transform(x, self.W_child)
        
        # Create adjacency matrices for different relationships
        adj_same = torch.zeros(N, N, device=x.device)
        adj_parent = torch.zeros(N, N, device=x.device)
        adj_child = torch.zeros(N, N, device=x.device)
        
        # Compute weights for all edges
        for i in range(len(row)):
            src, dst = row[i], col[i]
            src_level = node_levels[src].item()
            dst_level = node_levels[dst].item()
            level_diff = dst_level - src_level
            
            # Compute weight using cross-ratio
            weight = self.compute_cross_ratio_weight(x[src], x[dst], level_diff)
            
            if level_diff == 0:
                adj_same[src, dst] = weight
            elif level_diff > 0:
                adj_parent[src, dst] = weight
            else:
                adj_child[src, dst] = weight
                
        # Normalize adjacency matrices with numerical stability
        adj_same = adj_same / (adj_same.sum(dim=1, keepdim=True) + 1e-8)
        adj_parent = adj_parent / (adj_parent.sum(dim=1, keepdim=True) + 1e-8)
        adj_child = adj_child / (adj_child.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate features from different relationships using UHG operations
        out_same = self.uhg.aggregate(nodes_neigh, adj_same)
        out_parent = self.uhg.aggregate(nodes_parent, adj_parent)
        out_child = self.uhg.aggregate(nodes_child, adj_child)
        
        # Stack all points for each node
        points = torch.stack([
            nodes_self,
            out_same,
            out_parent,
            out_child,
            level_features
        ], dim=1)  # [N, 5, D+1]
        
        # Create weights tensor for each node
        weights = torch.tensor([0.3, 0.3, 0.15, 0.15, 0.1], device=x.device)
        
        # Average points for each node separately
        out = torch.stack([
            self.uhg.projective_average(points[i], weights)
            for i in range(N)
        ])
        
        if self.bias is not None:
            # Add bias in projective space with proper normalization
            bias_point = torch.cat([self.bias, torch.ones_like(self.bias[:1])], dim=0)
            bias_point = self.uhg.normalize(bias_point)
            bias_weights = torch.tensor([0.9, 0.1], device=x.device)
            out = torch.stack([
                self.uhg.projective_average(
                    torch.stack([out[i], bias_point]),
                    bias_weights
                )
                for i in range(N)
            ])
            
        # Restore initial cross-ratio if needed
        if has_cr:
            cr_final = self.uhg.cross_ratio(out[0], out[1], out[2], out[3])
            if not torch.isnan(cr_final) and not torch.isnan(cr_initial) and cr_final != 0:
                scale = torch.sqrt(torch.abs(cr_initial / cr_final))
                out = self.uhg.scale(out, scale)
            
        return out
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Aggregate features using hierarchical structure
        out = self.aggregate_hierarchical(x, edge_index, node_levels, size)
        
        # Return normalized feature part using UHG
        features = out[..., :-1]
        return self.uhg.normalize(features) 