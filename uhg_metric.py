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
    use_cross_ratio: bool = True  # Whether to use cross-ratio in metric

class UHGMetricLearner(nn.Module):
    """Learn optimal metrics in UHG space using pure projective operations."""
    
    def __init__(self, config: UHGMetricConfig):
        """Initialize UHG metric learner."""
        super().__init__()
        self.config = config
        
        # Initialize projective transformation parameters
        self.proj_matrix = nn.Parameter(torch.eye(config.embedding_dim + 1))
        
        # Initialize reference points for cross-ratio computation
        self.ref_points = nn.Parameter(
            torch.randn(3, config.embedding_dim + 1)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
        
    def _normalize_projective(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize points in projective space."""
        # Add homogeneous coordinate if needed
        if x.shape[-1] == self.config.embedding_dim:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Normalize last coordinate to 1 (projective normalization)
        return x / (x[..., -1:] + self.config.eps)
        
    def _cross_ratio(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-ratio in projective space."""
        # Ensure points are normalized
        a = self._normalize_projective(a)
        b = self._normalize_projective(b)
        c = self._normalize_projective(c)
        d = self._normalize_projective(d)
        
        # Compute projective distances
        ac = torch.sum(a * c, dim=-1)
        bd = torch.sum(b * d, dim=-1)
        ad = torch.sum(a * d, dim=-1)
        bc = torch.sum(b * c, dim=-1)
        
        # Compute cross-ratio
        return (ac * bd) / (ad * bc + self.config.eps)
        
    def _projective_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute projective distance using cross-ratio with reference points."""
        # Get normalized reference points
        r1, r2, r3 = [
            self._normalize_projective(r) for r in self.ref_points
        ]
        
        # Normalize input points
        x = self._normalize_projective(x)
        y = self._normalize_projective(y)
        
        # Compute cross-ratio based distance
        cr = self._cross_ratio(x, y, r1, r2)
        cr_ref = self._cross_ratio(r1, r2, r2, r3)
        
        # Convert to distance (preserving projective invariance)
        return -torch.log(torch.abs(cr / cr_ref) + self.config.eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input features using projective transformation."""
        # Add homogeneous coordinate if needed
        if x.shape[-1] == self.config.feature_dim:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Apply projective transformation
        h = torch.matmul(x, self.proj_matrix)
        
        # Normalize in projective space
        h = self._normalize_projective(h)
        
        return h
        
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between points in learned metric space."""
        # Transform points
        x_transformed = self.forward(x)
        y_transformed = self.forward(y)
        
        # Compute projective distance
        return self._projective_distance(x_transformed, y_transformed)
        
    def train_step(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform single training step using triplet and cross-ratio loss."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Transform points
        anchor_t = self.forward(anchor)
        positive_t = self.forward(positive)
        negative_t = self.forward(negative)
        
        # Compute distances
        pos_dist = self._projective_distance(anchor_t, positive_t)
        neg_dist = self._projective_distance(anchor_t, negative_t)
        
        # Compute mean distances for scalar loss
        pos_dist_mean = pos_dist.mean()
        neg_dist_mean = neg_dist.mean()
        
        # Compute triplet loss
        triplet_loss = F.relu(pos_dist_mean - neg_dist_mean + self.config.margin)
        
        # Add cross-ratio loss if enabled
        if self.config.use_cross_ratio:
            cr_anchor = self._cross_ratio(
                anchor_t,
                positive_t,
                negative_t,
                self.ref_points[0]
            )
            cr_target = torch.ones_like(cr_anchor) * 0.5
            cr_loss = F.mse_loss(cr_anchor, cr_target)
            total_loss = triplet_loss + cr_loss
        else:
            total_loss = triplet_loss
            cr_loss = torch.tensor(0.0)
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'cr_loss': cr_loss.item(),
            'pos_dist': pos_dist_mean.item(),
            'neg_dist': neg_dist_mean.item()
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