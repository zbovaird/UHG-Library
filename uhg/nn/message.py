import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from ..projective import ProjectiveUHG
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
        in_features: int,
        out_features: int,
        edge_features: Optional[int] = None,
        aggr: str = 'mean'
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.aggr = aggr
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + (edge_features if edge_features else 0), out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
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
        # Handle empty input
        if x.numel() == 0:
            return torch.empty((0, self.out_features), device=x.device)
            
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Compute messages using projective operations
        row, col = edge_index
        messages = self.message_mlp(torch.cat([x[col], edge_attr], dim=-1))
        
        # Aggregate messages using projective operations
        if self.aggr == 'mean':
            out = torch.zeros(x.size(0), self.out_features, device=x.device)
            out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_features), messages)
            count = torch.zeros(x.size(0), device=x.device)
            count.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            out = out / (count.unsqueeze(-1) + 1e-8)
        elif self.aggr == 'sum':
            out = torch.zeros(x.size(0), self.out_features, device=x.device)
            out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_features), messages)
        elif self.aggr == 'max':
            out = torch.zeros(x.size(0), self.out_features, device=x.device)
            out.scatter_reduce_(0, row.unsqueeze(-1).expand(-1, self.out_features), messages, reduce='amax')
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")
        
        # Update node features using projective operations
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        # Ensure output lies on the hyperbolic manifold
        out = self.uhg.project(out)
        
        # Restore cross-ratio if possible
        if has_cr:
            out = restore_cross_ratio(out, cr_initial)
        
        return out
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_features={self.in_features}, out_features={self.out_features}, aggr={self.aggr}' 