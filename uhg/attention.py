"""
UHG Attention Mechanisms Module.

This module implements attention mechanisms in Universal Hyperbolic Geometry space,
using pure projective operations and preserving cross-ratio invariance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from uhg.projective import ProjectiveUHG

class UHGAttentionConfig:
    """Configuration for UHG attention mechanisms."""
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_cross_ratio: bool = True,
        eps: float = 1e-6
    ):
        """Initialize UHG attention config."""
        assert feature_dim % num_heads == 0, \
            f"Feature dimension {feature_dim} must be divisible by num_heads {num_heads}"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dropout = dropout
        self.use_cross_ratio = use_cross_ratio
        self.eps = eps

class UHGMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism in UHG space.
    
    Uses pure projective operations to compute attention scores,
    ensuring cross-ratio preservation and geometric invariance.
    """
    
    def __init__(self, config: UHGAttentionConfig):
        """Initialize UHG attention."""
        super().__init__()
        self.config = config
        self.uhg = ProjectiveUHG()
        
        # Initialize projective transformations for each head
        q_init = torch.zeros(
            config.num_heads,
            config.feature_dim + 1,
            config.feature_dim + 1
        )
        k_init = torch.zeros(
            config.num_heads,
            config.feature_dim + 1,
            config.feature_dim + 1
        )
        
        # Initialize using proper projective transformations
        for h in range(config.num_heads):
            # Start with identity matrix
            q_init[h] = torch.eye(config.feature_dim + 1)
            k_init[h] = torch.eye(config.feature_dim + 1)
            
            # Create projective transformation that preserves cross-ratio
            for i in range(config.feature_dim):
                # Generate special linear transformation
                angle = torch.rand(1) * 2 * torch.pi
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                
                # Apply rotation in projective space
                q_init[h, i:i+2, i:i+2] = torch.tensor([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                k_init[h, i:i+2, i:i+2] = torch.tensor([
                    [cos_a, sin_a],
                    [-sin_a, cos_a]
                ])
            
            # Ensure special linear property (det = 1)
            q_det = torch.det(q_init[h])
            k_det = torch.det(k_init[h])
            q_init[h] = q_init[h] / torch.abs(q_det).pow(1/(config.feature_dim + 1))
            k_init[h] = k_init[h] / torch.abs(k_det).pow(1/(config.feature_dim + 1))
            
            # Ensure last row preserves homogeneous coordinate
            q_init[h, -1] = torch.zeros(config.feature_dim + 1)
            q_init[h, -1, -1] = 1.0
            k_init[h, -1] = torch.zeros(config.feature_dim + 1)
            k_init[h, -1, -1] = 1.0
        
        self.W_q = nn.Parameter(q_init)
        self.W_k = nn.Parameter(k_init)
        
        # Value transformation using same principle
        v_init = torch.zeros(
            config.num_heads,
            config.feature_dim + 1,
            config.feature_dim + 1
        )
        for h in range(config.num_heads):
            v_init[h] = torch.eye(config.feature_dim + 1)
            for i in range(config.feature_dim):
                angle = torch.rand(1) * 2 * torch.pi
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                v_init[h, i:i+2, i:i+2] = torch.tensor([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
            v_det = torch.det(v_init[h])
            v_init[h] = v_init[h] / torch.abs(v_det).pow(1/(config.feature_dim + 1))
            v_init[h, -1] = torch.zeros(config.feature_dim + 1)
            v_init[h, -1, -1] = 1.0
        
        self.W_v = nn.Parameter(v_init)
        
        # Output projection using same principle
        o_init = torch.eye(config.feature_dim + 1)
        for i in range(config.feature_dim):
            angle = torch.rand(1) * 2 * torch.pi
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            o_init[i:i+2, i:i+2] = torch.tensor([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
        o_det = torch.det(o_init)
        o_init = o_init / torch.abs(o_det).pow(1/(config.feature_dim + 1))
        o_init[-1] = torch.zeros(config.feature_dim + 1)
        o_init[-1, -1] = 1.0
        
        self.W_o = nn.Parameter(o_init)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def _ensure_projective(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is in projective form."""
        if x.shape[-1] == self.config.feature_dim:
            # Add homogeneous coordinate
            ones = torch.ones(*x.shape[:-1], 1, device=x.device)
            x = torch.cat([x, ones], dim=-1)
        return x
    
    def _normalize_projective(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates using pure projective operations."""
        # First normalize using cross-ratio with standard basis points
        e1 = torch.zeros_like(x)
        e1[..., 0] = 1.0
        e1[..., -1] = 1.0
        
        e2 = torch.zeros_like(x)
        e2[..., 1] = 1.0
        e2[..., -1] = 1.0
        
        e3 = torch.zeros_like(x)
        e3[..., 2] = 1.0
        e3[..., -1] = 1.0
        
        # Compute cross-ratio based normalization
        cr = self.uhg.cross_ratio(x, e1, e2, e3)
        x = x / (cr.unsqueeze(-1) + self.config.eps)
        
        # Ensure homogeneous coordinate is 1
        x = x / (x[..., -1:] + self.config.eps)
        
        return x
    
    def _get_ideal_points(self, line: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get ideal points of a line in projective space."""
        # Create ideal points directly in projective space
        i1 = torch.zeros_like(line)
        i1[..., 0] = 1.0  # First ideal point
        i1[..., -1] = 0.0  # On the absolute
        
        i2 = torch.zeros_like(line)
        i2[..., 1] = 1.0  # Second ideal point
        i2[..., -1] = 0.0  # On the absolute
        
        return i1, i2
    
    def _compute_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute attention scores using pure projective operations."""
        batch_size = q.shape[0]
        num_queries = q.shape[1]
        num_keys = k.shape[1]
        
        # Initialize attention scores
        scores = torch.zeros(batch_size, num_queries, num_keys, device=q.device)
        
        if self.config.use_cross_ratio:
            # Use cross-ratio based attention
            for b in range(batch_size):
                for i in range(num_queries):
                    for j in range(num_keys):
                        # Get ideal points
                        i1, i2 = self._get_ideal_points(q[b, i])
                        
                        # Compute cross-ratio based score
                        cr = self.uhg.cross_ratio(q[b, i], k[b, j], i1, i2)
                        scores[b, i, j] = 1 / (1 + torch.abs(torch.log(cr + self.config.eps)))
        else:
            # Use projective distance based attention
            for i in range(num_queries):
                for j in range(num_keys):
                    # Get ideal points
                    i1, i2 = self._get_ideal_points(q[:, i])
                    
                    # Compute projective distance using cross-ratio
                    cr = self.uhg.cross_ratio(q[:, i], k[:, j], i1, i2)
                    scores[:, i, j] = 1 / (1 + torch.abs(torch.log(cr + self.config.eps)))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        return scores
    
    def _projective_transform(
        self,
        x: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratio."""
        # Ensure input is projective
        x = self._ensure_projective(x)
        
        # Get dimensions
        n = x.shape[-1]
        
        # Ensure weight has correct shape
        if weight.shape[-1] != n:
            # Create new weight matrix of correct size
            new_weight = torch.eye(n, device=weight.device)
            # Copy over the relevant part of the weight matrix
            min_dim = min(weight.shape[-1], n)
            new_weight[:min_dim, :min_dim] = weight[:min_dim, :min_dim]
            weight = new_weight
        
        # Ensure weight is special linear and preserves cross-ratio
        # First make special linear
        weight = weight / torch.abs(torch.det(weight)).pow(1/n)
        
        # Then ensure it preserves cross-ratio by making it orthogonal
        u, s, v = torch.linalg.svd(weight[:-1, :-1])
        weight_spatial = torch.matmul(u, v)
        
        # Reconstruct full weight matrix
        weight_last_row = torch.zeros_like(weight[-1])
        weight_last_row[-1] = 1.0
        weight = torch.cat([
            torch.cat([weight_spatial, torch.zeros_like(weight[:-1, -1:])], dim=1),
            weight_last_row.unsqueeze(0)
        ], dim=0)
        
        # Apply transformation
        transformed = torch.matmul(x, weight.transpose(-2, -1))
        
        # Normalize while preserving cross-ratio
        # First normalize homogeneous coordinate
        transformed = transformed / (transformed[..., -1:] + self.config.eps)
        
        # Then normalize spatial part using cross-ratio with standard basis
        e1 = torch.zeros_like(transformed[..., 0, :])
        e1[..., 0] = 1.0
        e1[..., -1] = 1.0
        
        e2 = torch.zeros_like(transformed[..., 0, :])
        e2[..., 1] = 1.0
        e2[..., -1] = 1.0
        
        e3 = torch.zeros_like(transformed[..., 0, :])
        e3[..., 2] = 1.0
        e3[..., -1] = 1.0
        
        # Compute cross-ratio based normalization for each point
        normalized = []
        for i in range(transformed.shape[-2]):
            point = transformed[..., i, :]
            cr = self.uhg.cross_ratio(point, e1, e2, e3)
            point = point / (cr.unsqueeze(-1) + self.config.eps)
            # Ensure homogeneous coordinate is 1
            point = point / (point[..., -1:] + self.config.eps)
            normalized.append(point)
        
        transformed = torch.stack(normalized, dim=-2)
        
        return transformed
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using pure projective operations."""
        batch_size = query.shape[0]
        num_queries = query.shape[1]
        num_keys = key.shape[1]
        
        # Initialize outputs
        outputs = []
        all_weights = []
        
        # Process each attention head
        for h in range(self.config.num_heads):
            # Apply projective transformations
            q = self._projective_transform(query, self.W_q[h])
            k = self._projective_transform(key, self.W_k[h])
            v = self._projective_transform(value, self.W_v[h])
            
            # Compute attention scores
            weights = self._compute_attention_scores(q, k, mask)
            all_weights.append(weights)
            
            # Apply attention using projective operations
            head_output = []
            for i in range(num_queries):
                # Weighted combination using projective operations
                point = torch.zeros_like(v[:, 0])
                point[..., -1] = 1.0  # Initialize with proper homogeneous coordinate
                
                for j in range(num_keys):
                    weight = weights[:, i, j].unsqueeze(-1)
                    # Join weighted points
                    line = self.uhg.join(point, v[:, j])
                    point = self.uhg.meet(line, weight * v[:, j])
                    # Ensure homogeneous coordinate is 1
                    point = point / (point[..., -1:] + self.config.eps)
                head_output.append(point)
            
            head_output = torch.stack(head_output, dim=1)
            outputs.append(head_output)
        
        # Combine heads using projective operations
        outputs = torch.cat(outputs, dim=-1)
        outputs = self._projective_transform(outputs, self.W_o)
        
        # Return spatial coordinates for compatibility
        outputs = outputs[..., :self.config.feature_dim]
        
        # Stack attention weights
        attention_weights = torch.stack(all_weights, dim=1)
        
        return outputs, attention_weights 