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
from ...utils.cross_ratio import compute_cross_ratio

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
        
        # Register parameters
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
        nn.init.orthogonal_(self.W_self)
        nn.init.orthogonal_(self.W_neigh)
        nn.init.orthogonal_(self.W_parent)
        nn.init.orthogonal_(self.W_child)
        nn.init.orthogonal_(self.W_level)
        
        # Initialize level encodings in projective space
        nn.init.orthogonal_(self.level_encodings)
        
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def projective_transform(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        preserve_cr: bool = True
    ) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply linear transformation to features
        transformed = torch.matmul(features, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize to maintain projective structure
        norm = torch.norm(out[..., :-1], p=2, dim=-1, keepdim=True)
        out = torch.cat([out[..., :-1] / (norm + 1e-8), out[..., -1:]], dim=-1)
        
        if preserve_cr and x.size(0) > 3:
            # Compute cross-ratio of first four points
            cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
            cr_after = compute_cross_ratio(out[0], out[1], out[2], out[3])
            
            # Scale output to preserve cross-ratio
            scale = (cr_before / (cr_after + 1e-8)).sqrt()
            out = torch.cat([out[..., :-1] * scale, out[..., -1:]], dim=-1)
            
        return out
        
    def compute_cross_ratio_weight(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        level_diff: int = 0
    ) -> torch.Tensor:
        """Compute cross-ratio based weight between two points."""
        # Extract features
        p1_feat = p1[:-1]
        p2_feat = p2[:-1]
        
        # Compute cosine similarity
        dot_product = torch.sum(p1_feat * p2_feat)
        norm_p1 = torch.norm(p1_feat)
        norm_p2 = torch.norm(p2_feat)
        cos_sim = dot_product / (norm_p1 * norm_p2 + 1e-8)
        
        # Adjust weight based on level difference
        level_factor = torch.exp(-torch.tensor(abs(level_diff), dtype=torch.float32))
        weight = (cos_sim + 1) / 2 * level_factor
        
        return weight.clamp(0.1, 0.9)  # Prevent extreme values
        
    def aggregate_hierarchical(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate features using hierarchical structure and projective operations."""
        row, col = edge_index
        
        # Initialize output in projective space
        out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
        out[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Get level encodings for each node
        node_level_enc = F.embedding(node_levels, self.level_encodings)
        
        # Transform nodes with different weights based on relationship
        nodes_self = self.projective_transform(x, self.W_self)
        nodes_neigh = self.projective_transform(x, self.W_neigh)
        nodes_parent = self.projective_transform(x, self.W_parent)
        nodes_child = self.projective_transform(x, self.W_child)
        
        # Create adjacency matrices for different relationships
        N = x.size(0)
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
                
        # Normalize adjacency matrices
        adj_same = adj_same / (adj_same.sum(dim=1, keepdim=True) + 1e-8)
        adj_parent = adj_parent / (adj_parent.sum(dim=1, keepdim=True) + 1e-8)
        adj_child = adj_child / (adj_child.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate features from different relationships
        out_same = torch.matmul(adj_same, nodes_neigh)
        out_parent = torch.matmul(adj_parent, nodes_parent)
        out_child = torch.matmul(adj_child, nodes_child)
        
        # Transform level encodings
        level_features = self.projective_transform(
            torch.cat([node_level_enc, torch.ones_like(node_level_enc[..., :1])], dim=-1),
            self.W_level
        )
        
        # Combine all features using projective average
        points = torch.stack([nodes_self, out_same, out_parent, out_child, level_features])
        weights = torch.tensor([0.4, 0.3, 0.15, 0.15, 0.1], device=x.device)
        out = self.uhg.projective_average(points, weights)
        
        if self.bias is not None:
            # Add bias in projective space
            bias_point = torch.cat([self.bias, torch.ones_like(self.bias[:1])], dim=0)
            bias_point = bias_point / torch.norm(bias_point)
            out = self.uhg.projective_average(
                torch.stack([out, bias_point.expand_as(out)]),
                torch.tensor([0.9, 0.1], device=x.device)
            )
            
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
        
        # Return normalized feature part
        features = out[..., :-1]
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norm + 1e-8) 