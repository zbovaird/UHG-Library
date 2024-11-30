import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .base import HyperbolicLayer
from ..functional import uhg_quadrance, uhg_weighted_midpoint, uhg_aggregate
from ...manifolds import Manifold

class UHGGraphSAGEConv(HyperbolicLayer):
    """
    Universal Hyperbolic Geometry GraphSAGE Convolution Layer.
    
    This layer implements GraphSAGE convolution operations directly in UHG space
    using projective geometry, without any tangent space mappings. All operations
    preserve the hyperbolic structure through direct UHG computations.
    
    References:
        - Chapter 9.1: Graph Operations in UHG
        - Chapter 9.2: Direct Message Passing
        - Chapter 9.3: Neighborhood Aggregation
    """
    
    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the UHG GraphSAGE layer.
        
        Args:
            manifold: The hyperbolic manifold to operate on
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to use bias
            dropout: Dropout probability
        """
        super().__init__(manifold)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Initialize weights in UHG space
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_neigh = nn.Parameter(torch.Tensor(in_features, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset layer parameters using UHG-aware initialization.
        """
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_neigh)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of UHG GraphSAGE convolution.
        
        Implements the forward pass directly in UHG space:
        1. Transform node features using UHG operations
        2. Aggregate neighbors using UHG weighted midpoint
        3. Combine self and neighbor features in UHG space
        
        Args:
            x: Node feature matrix [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        self.check_input(x)
        
        # Get source and target nodes
        row, col = edge_index
        
        # Apply dropout in UHG space
        if self.training and self.dropout > 0:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout))
            x = x * mask
            x = self.manifold.proj_manifold(x)
        
        # Transform self features
        self_transformed = self.uhg_transform(x, self.weight_self)
        
        # Transform and aggregate neighbor features
        neigh_transformed = self.uhg_transform(x[col], self.weight_neigh)
        
        # Compute weights for neighbor aggregation
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
            
        # Aggregate neighbors using UHG weighted midpoint
        out = torch.zeros_like(self_transformed)
        for i in range(x.size(0)):
            mask = row == i
            if mask.any():
                neighbors = neigh_transformed[mask]
                weights = edge_weight[mask]
                out[i] = uhg_weighted_midpoint(neighbors.unsqueeze(0), 
                                             weights.unsqueeze(0)).squeeze(0)
        
        # Combine self and neighbor features in UHG space
        out = self.uhg_combine(self_transformed, out)
        
        # Add bias if present
        if self.bias is not None:
            bias = torch.cat([self.bias, torch.ones(1, device=self.bias.device)], dim=0)
            out = self.uhg_combine(out, bias.expand_as(out))
        
        self.check_output(out)
        return out 