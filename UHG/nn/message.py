import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from ..manifolds import HyperbolicManifold

class HyperbolicMessagePassing(nn.Module):
    """Hyperbolic message passing layer.
    
    This layer implements message passing in hyperbolic space, where messages are
    computed in the tangent space and then projected back to the hyperbolic manifold.
    Supports edge features of arbitrary dimension by projecting them to the output dimension if needed.
    
    Attributes:
        manifold (HyperbolicManifold): The hyperbolic manifold instance
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        aggr (str): Aggregation method ('mean', 'sum', or 'max')
    """
    
    def __init__(
        self,
        manifold: HyperbolicManifold,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        
        # Message transformation
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update transformation
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Edge feature projection (initialized lazily)
        self.edge_mlp = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the hyperbolic message passing layer.
        
        Args:
            x (torch.Tensor): Node features of shape [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, E]
            edge_attr (torch.Tensor, optional): Edge features of shape [E, edge_dim]
            
        Returns:
            torch.Tensor: Updated node features of shape [N, out_channels]
        """
        # Project node features to tangent space
        x_tangent = self.manifold.logmap0(x)
        
        # Compute messages
        row, col = edge_index
        messages = self.msg_mlp(x_tangent[col])
        
        # Add edge features if provided
        if edge_attr is not None:
            if edge_attr.shape[-1] != self.out_channels:
                # Lazily initialize edge_mlp if needed
                if self.edge_mlp is None or self.edge_mlp.in_features != edge_attr.shape[-1]:
                    self.edge_mlp = nn.Linear(edge_attr.shape[-1], self.out_channels).to(edge_attr.device)
                edge_proj = self.edge_mlp(edge_attr)
            else:
                edge_proj = edge_attr
            messages = messages + edge_proj
        
        # Aggregate messages
        if self.aggr == 'mean':
            out = torch.zeros(x.size(0), self.out_channels, device=x.device)
            out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_channels), messages)
            count = torch.zeros(x.size(0), device=x.device)
            count.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            out = out / (count.unsqueeze(-1) + 1e-8)
        elif self.aggr == 'sum':
            out = torch.zeros(x.size(0), self.out_channels, device=x.device)
            out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_channels), messages)
        elif self.aggr == 'max':
            out = torch.zeros(x.size(0), self.out_channels, device=x.device)
            out.scatter_reduce_(0, row.unsqueeze(-1).expand(-1, self.out_channels), messages, reduce='amax')
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")
        
        # Update node features
        out = self.update_mlp(out)
        
        # Project back to hyperbolic space
        return self.manifold.expmap0(out)
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, aggr={self.aggr}' 