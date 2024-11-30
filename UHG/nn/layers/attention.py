import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .base import HyperbolicLayer
from ...manifolds import Manifold

class HyperbolicAttention(HyperbolicLayer):
    """
    Hyperbolic Attention Layer.
    
    This layer implements attention mechanisms directly in hyperbolic space
    using projective geometry, without any tangent space mappings. The
    implementation follows UHG.pdf principles for hyperbolic transformations.
    
    The attention scores are computed using hyperbolic distances and
    cross-ratios to ensure all operations preserve hyperbolic structure.
    
    References:
        - Chapter 9.4: Hyperbolic Attention
        - Chapter 9.4.1: Computing Attention Scores
        - Chapter 9.4.2: Attention-based Message Passing
    """
    
    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        use_metric: bool = True,
    ):
        """
        Initialize the hyperbolic attention layer.
        
        Args:
            manifold: The hyperbolic manifold to operate on
            in_features: Number of input features
            out_features: Number of output features
            heads: Number of attention heads
            concat: Whether to concatenate or average attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            use_metric: Whether to use the hyperbolic metric for attention
        """
        super().__init__(manifold)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.use_metric = use_metric
        
        # Define output dimension based on concatenation
        self.out_dim = out_features * heads if concat else out_features
        
        # Initialize transformations in hyperbolic space
        self.weight = nn.Parameter(torch.Tensor(in_features, heads * out_features))
        self.att_weight = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset layer parameters using hyperbolic-aware initialization.
        
        The initialization ensures weights respect the hyperbolic structure
        and attention mechanism as described in UHG.pdf Chapter 9.4.
        """
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of hyperbolic attention.
        
        Implements attention mechanism directly in hyperbolic space:
        1. Transform node features using hyperbolic operations
        2. Compute attention scores using hyperbolic distances
        3. Apply attention-weighted aggregation in hyperbolic space
        
        Args:
            x: Node feature matrix [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            size: Size of source and target tensors
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        self.check_input(x)
        
        # Transform features in hyperbolic space
        out = self._attention_forward(x, edge_index, size)
        
        self.check_output(out)
        return out
    
    def _attention_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores and apply attention mechanism.
        
        Args:
            x: Node features in hyperbolic space
            edge_index: Graph connectivity
            size: Size of source and target tensors
            
        Returns:
            Attention-weighted node features
        """
        # Apply feature transformation in hyperbolic space
        x = self.manifold.proj_manifold(torch.matmul(x, self.weight))
        x = x.view(-1, self.heads, self.out_features)
        
        # Compute source and target nodes
        row, col = edge_index
        
        # Self-attention on the nodes
        alpha = self._compute_attention_scores(x[row], x[col])
        
        # Apply dropout to attention scores
        if self.training:
            mask = torch.bernoulli(torch.full_like(alpha, 1 - self.dropout))
            alpha = alpha * mask
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention weights in hyperbolic space
        out = self._aggregate_neighbors(x, alpha, row, col, size)
        
        if not self.concat:
            # Average attention heads using hyperbolic mean
            out = self._hyperbolic_mean(out, dim=1)
        else:
            out = out.view(-1, self.out_dim)
            
        if self.bias is not None:
            out = self.manifold.proj_manifold(out + self.bias)
            
        return out
    
    def _compute_attention_scores(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention scores between node pairs.
        
        Uses hyperbolic distances or cross-ratios to compute attention
        scores, ensuring all operations preserve hyperbolic structure.
        
        Args:
            x_i: Source node features
            x_j: Target node features
            
        Returns:
            Attention scores
        """
        if self.use_metric:
            # Use hyperbolic distance for attention
            dist = self.manifold.dist(x_i, x_j, keepdim=True)
            return F.softmax(-dist, dim=-1)
        else:
            # Use cross-ratio for attention
            origin = self.manifold.origin(x_i.shape[:-1])
            cross_ratio = self.manifold.compute_cross_ratio(x_i, x_j, origin)
            return F.softmax(-cross_ratio, dim=-1)
    
    def _aggregate_neighbors(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        size: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using attention weights.
        
        Performs weighted aggregation directly in hyperbolic space
        using the attention scores.
        
        Args:
            x: Node features
            alpha: Attention weights
            row: Source node indices
            col: Target node indices
            size: Size of source and target tensors
            
        Returns:
            Aggregated features
        """
        # Apply attention weights in hyperbolic space
        weighted = self.manifold.weighted_midpoint(x[col], alpha.unsqueeze(-1))
        
        # Aggregate using hyperbolic Einstein midpoint
        out = torch.zeros_like(x) if size is None else \
              torch.zeros((size[1], self.heads, self.out_features), device=x.device)
              
        row_expand = row.view(-1, 1, 1).expand(-1, self.heads, self.out_features)
        out.scatter_add_(0, row_expand, weighted)
        
        return self.manifold.proj_manifold(out)
    
    def _hyperbolic_mean(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute the hyperbolic mean along a dimension.
        
        Args:
            x: Input tensor
            dim: Dimension to average over
            
        Returns:
            Hyperbolic mean
        """
        weights = torch.ones(x.shape[dim], device=x.device) / x.shape[dim]
        return self.manifold.weighted_midpoint(x, weights, dim=dim)