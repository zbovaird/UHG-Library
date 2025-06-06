import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any

from ..layers import HyperbolicLinear
from ..message import HyperbolicMessagePassing
from ...manifolds import HyperbolicManifold
from ..attention import HyperbolicAttention

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
    hyperbolic structure of the data through proper tangent space operations.
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
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            
        Returns:
            torch.Tensor: Updated node features
        """
        if self.use_attn:
            # Use attention-based message passing
            return self.attention(x, edge_index, edge_attr)
        else:
            # Use standard message passing
            return HyperbolicMessagePassing(
                manifold=self.manifold,
                in_channels=x.size(-1),
                out_channels=x.size(-1)
            )(x, edge_index, edge_attr)

class HGAT(BaseHGNN):
    """Hyperbolic Graph Attention Network.
    
    This implementation follows the architecture described in:
    "Hyperbolic Graph Attention Networks" (Zhang et al., NeurIPS 2021)
    
    The model uses hyperbolic attention mechanisms to perform message passing
    while preserving the hyperbolic structure of the data.
    
    Attributes:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of message passing layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear layers
        act (callable): Activation function
        concat (bool): Whether to concatenate attention heads
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
        
        # Track the input dimension for each attention layer
        attn_dims = []
        for i in range(num_layers):
            if i == 0:
                attn_dims.append(self.layers[0].out_features)
            else:
                attn_dims.append(self.layers[i].out_features)
        # Initialize attention layers with correct input dims
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.attention_layers.append(
                    HyperbolicAttention(
                        manifold=manifold,
                        in_channels=attn_dims[i],
                        num_heads=1,
                        dropout=dropout,
                        concat=False
                    )
                )
            else:
                self.attention_layers.append(
                    HyperbolicAttention(
                        manifold=manifold,
                        in_channels=attn_dims[i],
                        num_heads=num_heads,
                        dropout=dropout,
                        concat=concat
                    )
                )
    
    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform hyperbolic attention-based message passing.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            edge_attr (torch.Tensor, optional): Edge features
            
        Returns:
            torch.Tensor: Updated node features
        """
        layer_idx = self._current_layer_idx if hasattr(self, '_current_layer_idx') else 0
        return self.attention_layers[layer_idx](x, edge_index, edge_attr)
    
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
        for i in range(self.num_layers):
            # Feature transformation
            x = self.layers[i](x)
            # Apply activation
            if self.act is not None:
                x = self.act(x)
            # Apply dropout
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # Message passing
            self._current_layer_idx = i
            x = self._message_passing(x, edge_index, edge_attr)
        return x 