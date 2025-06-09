"""UHG-compliant loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.metrics import uhg_quadrance, uhg_spread

class UHGLoss(nn.Module):
    """UHG-compliant loss function optimized for geometric learning tasks.
    
    This loss combines quadrance-based contrastive learning with spread regularization
    to maintain UHG geometric structure during training.
    
    Args:
        spread_weight: Weight for spread regularization term
        quad_weight: Weight for quadrance-based terms
    """
    
    def __init__(self, spread_weight: float = 0.1, quad_weight: float = 1.0):
        super().__init__()
        self.spread_weight = spread_weight
        self.quad_weight = quad_weight
        
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute UHG-compliant loss.
        
        Args:
            z: Node embeddings tensor of shape [N, D+1]
            edge_index: Graph connectivity of shape [2, E]
            batch_size: Number of nodes in batch
            
        Returns:
            Combined loss value
        """
        # Get positive pairs
        mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
        pos_edge_index = edge_index[:, mask]
        
        if pos_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=z.device)
            
        # Compute quadrance-based scores with stability
        pos_quad = torch.clamp(uhg_quadrance(z[pos_edge_index[0]], z[pos_edge_index[1]]), max=10.0)
        
        # Generate and compute negative pairs
        neg_edge_index = torch.randint(0, batch_size, (2, batch_size), device=z.device)
        neg_quad = torch.clamp(uhg_quadrance(z[neg_edge_index[0]], z[neg_edge_index[1]]), max=10.0)
        
        # Compute spread for geometric structure preservation
        spread = torch.clamp(uhg_spread(z[pos_edge_index[0]], z[pos_edge_index[1]]), max=10.0)
        
        # Compute contrastive loss with UHG metrics
        pos_loss = torch.mean(pos_quad)  # Pull similar points together
        neg_loss = torch.mean(F.relu(1 - neg_quad))  # Push dissimilar points apart
        
        # Add spread regularization
        spread_loss = self.spread_weight * spread.mean()
        
        # Scale losses for better stability
        total_loss = self.quad_weight * (pos_loss + neg_loss) + spread_loss
        return torch.clamp(total_loss, min=0, max=100.0)

class UHGAnomalyLoss(UHGLoss):
    """UHG-compliant loss function specialized for anomaly detection.
    
    This loss extends the base UHG loss with additional terms for better
    anomaly detection in hyperbolic space.
    
    Args:
        spread_weight: Weight for spread regularization term
        quad_weight: Weight for quadrance-based terms
        margin: Margin for anomaly separation
    """
    
    def __init__(self, spread_weight: float = 0.1, quad_weight: float = 1.0, margin: float = 1.0):
        super().__init__(spread_weight, quad_weight)
        self.margin = margin
        
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute UHG-compliant anomaly detection loss.
        
        Args:
            z: Node embeddings tensor of shape [N, D+1]
            edge_index: Graph connectivity of shape [2, E]
            batch_size: Number of nodes in batch
            
        Returns:
            Combined loss value
        """
        # Get base loss
        base_loss = super().forward(z, edge_index, batch_size)
        
        # Add margin-based separation for anomalies
        neg_edge_index = torch.randint(0, batch_size, (2, batch_size), device=z.device)
        neg_quad = uhg_quadrance(z[neg_edge_index[0]], z[neg_edge_index[1]])
        margin_loss = torch.mean(F.relu(self.margin - neg_quad))
        
        return base_loss + margin_loss 