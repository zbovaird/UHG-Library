"""
UHG Metric Learning Module.

This module implements metric learning in Universal Hyperbolic Geometry space,
using pure projective operations and preserving cross-ratio invariance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class UHGMetricConfig:
    """Configuration for UHG metric learning."""
    feature_dim: int  # Dimension of input features
    embedding_dim: int = 32  # Dimension of learned metric space
    margin: float = 1.0  # Margin for contrastive loss
    learning_rate: float = 0.001  # Learning rate for optimization
    batch_size: int = 64  # Batch size for training
    eps: float = 1e-9  # Numerical stability

class UHGMetricLearner(nn.Module):
    """Learn optimal metrics in UHG space using pure projective operations."""
    
    def __init__(self, config: UHGMetricConfig):
        """Initialize UHG metric learner."""
        super().__init__()
        self.config = config
        
        # Initialize projective transformation parameters
        self.proj_matrix = nn.Parameter(
            torch.eye(config.feature_dim + 1, config.embedding_dim + 1)
        )
        
        # Initialize ideal points for cross-ratio computation
        self.ideal_points = nn.Parameter(
            torch.randn(2, config.embedding_dim + 1)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
        
    def _cross_ratio(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        i1: torch.Tensor,
        i2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-ratio in projective space."""
        # Compute projective distances using determinants
        def proj_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.shape[-1]
            dist = 0.0
            for i in range(n-1):
                for j in range(i+1, n):
                    det = x[..., i] * y[..., j] - x[..., j] * y[..., i]
                    dist = dist + det * det
            return torch.sqrt(dist + self.config.eps)
        
        # Compute cross-ratio using projective distances
        ac = proj_dist(a, i1)
        bd = proj_dist(b, i2)
        ad = proj_dist(a, i2)
        bc = proj_dist(b, i1)
        
        return (ac * bd) / (ad * bc + self.config.eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input features using projective transformation."""
        # Add homogeneous coordinate if needed
        if x.shape[-1] == self.config.feature_dim:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Apply projective transformation
        h = torch.matmul(x, self.proj_matrix)
        
        # Normalize to preserve cross-ratio
        h = h / (h[..., -1:] + self.config.eps)
        
        return h
        
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance using cross-ratio with ideal points."""
        # Transform points
        x_transformed = self.forward(x)
        y_transformed = self.forward(y)
        
        # Get ideal points
        i1, i2 = self.ideal_points[0], self.ideal_points[1]
        
        # Compute distance using cross-ratio
        cr = self._cross_ratio(x_transformed, y_transformed, i1, i2)
        return torch.abs(torch.log(cr + self.config.eps))
        
    def train_step(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform single training step using cross-ratio based loss."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Transform points
        anchor_t = self.forward(anchor)
        positive_t = self.forward(positive)
        negative_t = self.forward(negative)
        
        # Compute distances using cross-ratio
        pos_dist = self.compute_distance(anchor_t, positive_t)
        neg_dist = self.compute_distance(anchor_t, negative_t)
        
        # Compute mean distances for scalar loss
        pos_dist_mean = pos_dist.mean()
        neg_dist_mean = neg_dist.mean()
        
        # Compute triplet loss
        loss = F.relu(pos_dist_mean - neg_dist_mean + self.config.margin)
        
        # Compute cross-ratio preservation loss
        cr_loss = torch.tensor(0.0, device=anchor.device)
        for i in range(len(anchor_t)):
            cr_before = self._cross_ratio(
                anchor[i], positive[i], 
                negative[i], self.ideal_points[0]
            )
            cr_after = self._cross_ratio(
                anchor_t[i], positive_t[i],
                negative_t[i], self.ideal_points[0]
            )
            cr_loss = cr_loss + F.mse_loss(cr_after, cr_before)
            
        total_loss = loss + cr_loss
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "triplet_loss": loss.item(),
            "cr_loss": cr_loss.item(),
            "pos_dist": pos_dist_mean.item(),
            "neg_dist": neg_dist_mean.item()
        }
        
    def save_state(self, path: str):
        """Save model state."""
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load_state(cls, path: str) -> 'UHGMetricLearner':
        """Load model state."""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model 