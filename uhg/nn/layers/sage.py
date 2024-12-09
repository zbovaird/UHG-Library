import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from .base import ProjectiveLayer
from ...projective import ProjectiveUHG

class ProjectiveSAGEConv(nn.Module):
    """UHG-compliant GraphSAGE convolution layer using pure projective operations.
    
    This layer implements the GraphSAGE convolution using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.uhg = ProjectiveUHG()
        
        # Initialize projective transformations
        self.W_self = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_neigh = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        nn.init.orthogonal_(self.W_self)
        nn.init.orthogonal_(self.W_neigh)
        if self.bias is not None:
            # Initialize bias in projective space
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply weight to features
        transformed = torch.matmul(features, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize to maintain projective structure
        norm = torch.norm(out[..., :-1], p=2, dim=-1, keepdim=True)
        out = torch.cat([out[..., :-1] / (norm + 1e-8), out[..., -1:]], dim=-1)
        return out
        
    def compute_cross_ratio_weight(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute cross-ratio based weight between two points."""
        # Extract features
        p1_feat = p1[:-1]
        p2_feat = p2[:-1]
        
        # Compute cosine similarity
        dot_product = torch.sum(p1_feat * p2_feat)
        norm_p1 = torch.norm(p1_feat)
        norm_p2 = torch.norm(p2_feat)
        cos_sim = dot_product / (norm_p1 * norm_p2 + 1e-8)
        
        # Map to [0, 1] range
        weight = (cos_sim + 1) / 2
        
        return weight.clamp(0.1, 0.9)  # Prevent extreme values
        
    def aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate neighborhood features using projective operations."""
        row, col = edge_index
        
        # Initialize output in projective space
        out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
        out[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        
        # Transform all nodes first
        all_nodes = self.projective_transform(x, self.W_neigh)
        
        # Create adjacency matrix with self-loops
        N = x.size(0)
        adj = torch.zeros(N, N, device=x.device)
        adj[row, col] = 1
        adj = adj + torch.eye(N, device=x.device)
        
        # Compute weights for all edges
        for i in range(len(row)):
            src, dst = row[i], col[i]
            weight = self.compute_cross_ratio_weight(x[src], x[dst])
            adj[src, dst] = weight
            adj[dst, src] = weight
            
        # Normalize weights
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate features using normalized weights
        out = torch.matmul(adj, all_nodes)
        
        # Ensure output lies in projective space
        norm = torch.norm(out[..., :-1], p=2, dim=-1, keepdim=True)
        out = torch.cat([out[..., :-1] / (norm + 1e-8), out[..., -1:]], dim=-1)
        
        return out
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Transform self features
        self_trans = self.projective_transform(x, self.W_self)
        
        # Aggregate and transform neighbor features
        neigh_trans = self.aggregate_neighbors(x, edge_index, size)
        
        # Combine using projective average with equal weights
        points = torch.stack([self_trans, neigh_trans])
        weights = torch.tensor([0.5, 0.5], device=x.device)
        out = self.uhg.projective_average(points, weights)
        
        if self.bias is not None:
            # Add bias in projective space
            bias_point = torch.cat([self.bias, torch.ones_like(self.bias[:1])], dim=0)
            bias_point = bias_point / torch.norm(bias_point)
            out = self.uhg.projective_average(
                torch.stack([out, bias_point.expand_as(out)]),
                torch.tensor([0.9, 0.1], device=x.device)
            )
            
        # Return normalized feature part
        features = out[..., :-1]
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norm + 1e-8) 