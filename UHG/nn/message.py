import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from ..manifolds import HyperbolicManifold
from ..utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class HyperbolicMessagePassing(nn.Module):
    """Hyperbolic message passing layer.
    
    This layer implements message passing in hyperbolic space using pure projective operations.
    All operations preserve cross-ratios and hyperbolic structure.
    
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
        
        # Message transformation using projective operations
        self.msg_transform = nn.Linear(in_channels, out_channels)
        
        # Update transformation using projective operations
        self.update_transform = nn.Linear(out_channels, out_channels)
        
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
        # Handle empty input
        if x.numel() == 0:
            return torch.empty((0, self.out_channels), device=x.device)
            
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Compute messages using projective operations
        row, col = edge_index
        messages = self.msg_transform(x[col])
        
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
        
        # Aggregate messages using projective operations
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
        
        # Update node features using projective operations
        out = self.update_transform(out)
        
        # Ensure output lies on the hyperbolic manifold
        out = self.manifold.project(out)
        
        # Restore cross-ratio if possible
        if has_cr:
            out = restore_cross_ratio(out, cr_initial)
        
        return out
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, aggr={self.aggr}' 