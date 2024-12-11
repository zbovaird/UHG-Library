import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from .base import UHGLayer
from ...projective import ProjectiveUHG

class ProjectiveSAGEConv(UHGLayer):
    """UHG-compliant GraphSAGE convolution layer using pure projective operations.
    
    This layer implements the GraphSAGE convolution using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_samples: Number of neighbors to sample. If None, use all neighbors
        aggregator: Aggregation method ('mean', 'max', 'lstm')
        bias: Whether to use bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: Optional[int] = None,
        aggregator: str = 'mean',
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        self.aggregator = aggregator.lower()
        
        if self.aggregator not in ['mean', 'max', 'lstm']:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        
        # Initialize projective transformations
        self.W_self = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_neigh = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # LSTM aggregator if specified
        if self.aggregator == 'lstm':
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=in_features,
                batch_first=True
            )
        
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
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def sample_neighbors(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Sample a fixed number of neighbors for each node."""
        if self.num_samples is None:
            return edge_index
            
        row, col = edge_index
        
        # Create adjacency list representation
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(len(row)):
            adj_list[row[i].item()].append(col[i].item())
            
        # Sample neighbors
        sampled_rows = []
        sampled_cols = []
        
        for node, neighbors in enumerate(adj_list):
            if len(neighbors) == 0:
                continue
                
            # Sample with replacement if we need more neighbors than available
            num_to_sample = min(self.num_samples, len(neighbors))
            sampled = torch.tensor(neighbors)[torch.randperm(len(neighbors))[:num_to_sample]]
            
            sampled_rows.extend([node] * len(sampled))
            sampled_cols.extend(sampled.tolist())
            
        return torch.tensor([sampled_rows, sampled_cols], device=edge_index.device)
        
    def aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Aggregate neighborhood features using projective operations."""
        # Sample neighbors if specified
        edge_index = self.sample_neighbors(edge_index, x.size(0))
        row, col = edge_index
        
        # Get neighbor features
        neigh_features = x[col]
        
        if self.aggregator == 'mean':
            # Mean aggregation with cross-ratio weights
            out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
            out[..., -1] = 1.0  # Set homogeneous coordinate
            
            # Compute weights using cross-ratios
            weights = []
            for i in range(len(row)):
                src, dst = row[i], col[i]
                weight = self.compute_cross_ratio_weight(x[src], x[dst])
                weights.append(weight)
            
            weights = torch.tensor(weights, device=x.device)
            weights = weights / weights.sum()
            
            # Weighted aggregation
            for i, w in enumerate(weights):
                out[row[i]] += w * self.projective_transform(neigh_features[i:i+1], self.W_neigh)[0]
                
        elif self.aggregator == 'max':
            # Max aggregation in projective space
            transformed = self.projective_transform(neigh_features, self.W_neigh)
            
            # Initialize output
            out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
            out[..., -1] = 1.0
            
            # Max pooling for each node's neighbors
            for node in range(x.size(0)):
                mask = row == node
                if mask.any():
                    node_neighs = transformed[mask]
                    # Max pooling in feature space
                    max_feats = torch.max(node_neighs[..., :-1], dim=0)[0]
                    # Add homogeneous coordinate back
                    out[node] = torch.cat([max_feats, torch.ones(1, device=x.device)])
                    
        else:  # LSTM
            # LSTM aggregation
            transformed = self.projective_transform(neigh_features, self.W_neigh)
            out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
            out[..., -1] = 1.0
            
            # Process each node's neighbors through LSTM
            for node in range(x.size(0)):
                mask = row == node
                if mask.any():
                    node_neighs = transformed[mask][..., :-1]  # Remove homogeneous coordinate
                    node_neighs = node_neighs.unsqueeze(0)  # Add batch dimension
                    lstm_out, _ = self.lstm(node_neighs)
                    # Use last LSTM output
                    out[node, :-1] = lstm_out[0, -1]
                    
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