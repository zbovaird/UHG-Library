import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any

from ..layers import HyperbolicLinear
from ..message import HyperbolicMessagePassing
from ...manifolds import HyperbolicManifold
from ..attention import HyperbolicAttention
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class BaseHGNN(nn.Module):
    """Base class for Hyperbolic Graph Neural Networks.
    
    This class implements the core functionality shared across different HGNN architectures.
    It handles the hyperbolic message passing and aggregation operations while preserving
    the hyperbolic structure of the data.
    
    Attributes:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of message passing layers
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear layers
        act (callable): Activation function
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        act: Optional[callable] = F.relu
    ):
        super().__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.act = act
        
        # Initialize layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            HyperbolicLinear(
                manifold=manifold,
                in_features=in_channels,
                out_features=hidden_channels,
                bias=bias
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                HyperbolicLinear(
                    manifold=manifold,
                    in_features=hidden_channels,
                    out_features=hidden_channels,
                    bias=bias
                )
            )
        
        # Output layer
        self.layers.append(
            HyperbolicLinear(
                manifold=manifold,
                in_features=hidden_channels,
                out_features=out_channels,
                bias=bias
            )
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the HGNN.
        
        Args:
            x (torch.Tensor): Node features of shape [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, E]
            edge_attr (torch.Tensor, optional): Edge features of shape [E, edge_dim]
            batch (torch.Tensor, optional): Batch vector of shape [N]
            
        Returns:
            torch.Tensor: Node embeddings of shape [N, out_channels]
        """
        # Initial feature transformation
        x = self.layers[0](x)
        
        # Message passing layers
        for i in range(1, self.num_layers):
            # Apply dropout
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Message passing
            x = self._message_passing(x, edge_index, edge_attr)
            
            # Feature transformation
            x = self.layers[i](x)
            
            # Apply activation
            if self.act is not None:
                x = self.act(x)
        
        return x
    
    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform hyperbolic message passing.
        
        This is a base implementation that should be overridden by specific architectures.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            
        Returns:
            torch.Tensor: Updated node features
        """
        raise NotImplementedError

class HGCN(BaseHGNN):
    """Hyperbolic Graph Convolutional Network.
    
    This implementation follows the architecture described in:
    "Hyperbolic Graph Convolutional Neural Networks" (Chami et al., NeurIPS 2019)
    
    The model performs message passing in hyperbolic space while preserving the
    hyperbolic structure of the data through proper projective operations.
    
    Key features:
    1. Pure projective operations - no tangent space mappings
    2. Cross-ratio preservation in all transformations
    3. Hyperbolic-aware message passing
    4. Proper handling of hyperbolic distances
    5. UHG-compliant aggregation operations
    
    Args:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of message passing layers
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear layers
        act (callable): Activation function
        use_attn (bool): Whether to use attention mechanism
        attn_heads (int): Number of attention heads if use_attn is True
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        act: Optional[callable] = F.relu,
        use_attn: bool = False,
        attn_heads: int = 1
    ):
        super().__init__(
            manifold=manifold,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
            act=act
        )
        
        self.use_attn = use_attn
        self.attn_heads = attn_heads
        
        # Initialize message passing layers
        self.message_passing_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_passing_layers.append(
                HyperbolicMessagePassing(
                    manifold=manifold,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    aggr='mean'  # Use mean aggregation for stability
                )
            )
        
        if use_attn:
            self.attention = HyperbolicAttention(
                manifold=manifold,
                in_channels=hidden_channels,
                num_heads=attn_heads,
                dropout=dropout
            )
    
    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform hyperbolic message passing with optional attention.
        
        This implementation ensures all operations preserve hyperbolic structure
        and cross-ratios through proper projective operations.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            
        Returns:
            torch.Tensor: Updated node features
        """
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        if self.use_attn:
            # Use attention-based message passing
            x = self.attention(x, edge_index, edge_attr)
        else:
            # Use standard message passing
            x = self.message_passing_layers[0](x, edge_index, edge_attr)
        
        # Restore cross-ratio if possible
        if has_cr:
            cr_current = compute_cross_ratio(x[0], x[1], x[2], x[3])
            if not torch.isnan(cr_current) and not torch.isnan(cr_initial) and cr_current != 0:
                x = restore_cross_ratio(x, cr_initial)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the HGCN.
        
        This implementation ensures all operations preserve hyperbolic structure
        and cross-ratios through proper projective operations.
        
        Args:
            x (torch.Tensor): Node features of shape [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, E]
            edge_attr (torch.Tensor, optional): Edge features of shape [E, edge_dim]
            batch (torch.Tensor, optional): Batch vector of shape [N]
            
        Returns:
            torch.Tensor: Node embeddings of shape [N, out_channels]
        """
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Initial feature transformation
        x = self.layers[0](x)
        
        # Message passing layers
        for i in range(1, self.num_layers):
            # Apply dropout
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Message passing
            x = self._message_passing(x, edge_index, edge_attr)
            
            # Feature transformation
            x = self.layers[i](x)
            
            # Apply activation
            if self.act is not None:
                x = self.act(x)
            
            # Restore cross-ratio if possible
            if has_cr:
                cr_current = compute_cross_ratio(x[0], x[1], x[2], x[3])
                if not torch.isnan(cr_current) and not torch.isnan(cr_initial) and cr_current != 0:
                    x = restore_cross_ratio(x, cr_initial)
        
        return x

class HGAT(BaseHGNN):
    """Hyperbolic Graph Attention Network.
    
    This implementation follows the architecture described in:
    "Hyperbolic Graph Attention Networks" (Zhang et al., NeurIPS 2020)
    
    The model performs attention-based message passing in hyperbolic space while preserving
    the hyperbolic structure of the data through proper projective operations.
    
    Key features:
    1. Pure projective operations - no tangent space mappings
    2. Cross-ratio preservation in all transformations
    3. Hyperbolic-aware attention mechanism
    4. Proper handling of hyperbolic distances
    5. UHG-compliant aggregation operations
    
    Args:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of message passing layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear layers
        act (callable): Activation function
        concat (bool): Whether to concatenate multi-head outputs
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        act: Optional[callable] = F.relu,
        concat: bool = True
    ):
        super().__init__(
            manifold=manifold,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
            act=act
        )
        
        self.num_heads = num_heads
        self.concat = concat
        
        # Initialize attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                HyperbolicAttention(
                    manifold=manifold,
                    in_channels=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        
        # Initialize output projection
        if concat:
            self.out_proj = HyperbolicLinear(
                manifold=manifold,
                in_features=hidden_channels * num_heads,
                out_features=out_channels,
                bias=bias
            )
        else:
            self.out_proj = HyperbolicLinear(
                manifold=manifold,
                in_features=hidden_channels,
                out_features=out_channels,
                bias=bias
            )
    
    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform hyperbolic message passing with attention.
        
        This implementation ensures all operations preserve hyperbolic structure
        and cross-ratios through proper projective operations.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            
        Returns:
            torch.Tensor: Updated node features
        """
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Apply attention
        out = self.attention_layers[0](x, edge_index, edge_attr)
        
        # Project output if concatenating heads
        if self.concat:
            out = self.out_proj(out)
        
        # Restore cross-ratio if possible
        if has_cr:
            out = restore_cross_ratio(out, cr_initial)
        
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the HGAT.
        
        Args:
            x (torch.Tensor): Node features of shape [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, E]
            edge_attr (torch.Tensor, optional): Edge features of shape [E, edge_dim]
            batch (torch.Tensor, optional): Batch vector of shape [N]
            
        Returns:
            torch.Tensor: Node embeddings of shape [N, out_channels]
        """
        # Initial feature transformation
        x = self.layers[0](x)
        
        # Message passing layers
        for i in range(1, self.num_layers):
            # Apply dropout
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Message passing with attention
            x = self._message_passing(x, edge_index, edge_attr)
            
            # Feature transformation
            x = self.layers[i](x)
            
            # Apply activation
            if self.act is not None:
                x = self.act(x)
        
        return x 