import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from ..layers import HyperbolicGraphConv, HyperbolicAttention
from ...manifolds import Manifold

class HyperbolicGraphSAGE(nn.Module):
    """
    Hyperbolic GraphSAGE Model.
    
    This model implements GraphSAGE in hyperbolic space using Universal Hyperbolic
    Geometry principles. All operations are performed directly in hyperbolic space
    using projective geometry, without any tangent space mappings.
    
    Key features:
    - Direct hyperbolic space operations (no tangent space)
    - Neighborhood sampling and aggregation in hyperbolic space
    - Multi-layer hyperbolic message passing
    - Optional attention-based aggregation
    
    References:
        - Chapter 10.1: Graph Embedding in Hyperbolic Space
        - Chapter 10.2: Neighborhood Aggregation
        - Chapter 10.3: Multi-scale Representations
    """
    
    def __init__(
        self,
        manifold: Manifold,
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
        """
        Initialize HyperbolicGraphSAGE.
        
        Args:
            manifold: The hyperbolic manifold to operate on
            in_channels: Size of input features
            hidden_channels: Size of hidden features
            out_channels: Size of output features
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            use_attention: Whether to use attention for aggregation
            heads: Number of attention heads (if use_attention=True)
            concat: Whether to concatenate attention heads
            bias: Whether to use bias in layers
        """
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Create list to hold all layers
        self.layers = nn.ModuleList()
        
        # Input layer
        if use_attention:
            self.layers.append(
                HyperbolicAttention(
                    manifold=manifold,
                    in_features=in_channels,
                    out_features=hidden_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    bias=bias,
                )
            )
        else:
            self.layers.append(
                HyperbolicGraphConv(
                    manifold=manifold,
                    in_features=in_channels,
                    out_features=hidden_channels,
                    use_bias=bias,
                    dropout=dropout,
                )
            )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if use_attention:
                self.layers.append(
                    HyperbolicAttention(
                        manifold=manifold,
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        heads=heads,
                        concat=concat,
                        dropout=dropout,
                        bias=bias,
                    )
                )
            else:
                self.layers.append(
                    HyperbolicGraphConv(
                        manifold=manifold,
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        use_bias=bias,
                        dropout=dropout,
                    )
                )
        
        # Output layer
        if use_attention:
            self.layers.append(
                HyperbolicAttention(
                    manifold=manifold,
                    in_features=hidden_channels,
                    out_features=out_channels,
                    heads=1,
                    concat=True,
                    dropout=dropout,
                    bias=bias,
                )
            )
        else:
            self.layers.append(
                HyperbolicGraphConv(
                    manifold=manifold,
                    in_features=hidden_channels,
                    out_features=out_channels,
                    use_bias=bias,
                    dropout=dropout,
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of HyperbolicGraphSAGE.
        
        Implements the forward pass through all layers while maintaining
        the hyperbolic structure of the data. All operations are performed
        directly in hyperbolic space using projective geometry.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Node embeddings in hyperbolic space [num_nodes, out_channels]
        """
        # Initial projection to hyperbolic space if needed
        if not self.manifold.check_point_on_manifold(x)[0]:
            x = self.manifold.proj_manifold(x)
        
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x, edge_index, edge_weight)
            
            # Apply hyperbolic activation for all but last layer
            if i < len(self.layers) - 1:
                x = self.hyperbolic_activation(x)
            
            # Apply dropout directly in hyperbolic space
            if self.training and i < len(self.layers) - 1:
                mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout))
                x = x * mask
                x = self.manifold.proj_manifold(x)  # Ensure we stay on manifold
        
        return x
    
    def hyperbolic_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function directly in hyperbolic space.
        
        Instead of using tangent space mappings, we implement the activation
        directly in hyperbolic space using projective geometric operations.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Activated tensor in hyperbolic space
        """
        # Compute hyperbolic norm
        norm = self.manifold.dist(x, self.manifold.origin(x.shape[:-1]))
        
        # Apply activation directly to the hyperbolic coordinates
        # This preserves the hyperbolic structure while providing nonlinearity
        activated = x * torch.sigmoid(norm).unsqueeze(-1)
        
        # Project back to ensure we stay exactly on the manifold
        return self.manifold.proj_manifold(activated) 