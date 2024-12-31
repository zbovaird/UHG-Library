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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention scores using pure projective operations."""
        batch_size, num_heads, num_queries, _ = q.shape
        num_keys = k.shape[2]
        
        # Initialize attention scores
        scores = torch.zeros(batch_size, num_heads, num_queries, num_keys, device=q.device)
        
        # Normalize inputs
        q = self._normalize_projective(q)
        k = self._normalize_projective(k)
        
        if self.config.use_cross_ratio:
            # Use vectorized cross-ratio computation
            # Compute pairwise cross-ratios
            for b in range(batch_size):
                for h in range(num_heads):
                    for i in range(num_queries):
                        for j in range(num_keys):
                            # Get ideal points
                            i1, i2 = self._get_ideal_points(q[b, h, i])
                            
                            # Compute cross-ratio based score
                            cr = self.uhg.cross_ratio(q[b, h, i], k[b, h, j], i1, i2)
                            scores[b, h, i, j] = 1 / (1 + torch.abs(torch.log(torch.abs(cr) + self.config.eps)))
        else:
            # Use hyperbolic distance based attention
            # Compute pairwise distances in hyperbolic space
            q_norm = self.uhg.normalize_points(q)  # [B, H, Q, D]
            k_norm = self.uhg.normalize_points(k)  # [B, H, K, D]
            
            # Compute hyperbolic inner products
            for b in range(batch_size):
                for h in range(num_heads):
                    inner_prod = torch.einsum('qd,kd->qk', q_norm[b, h, :, :-1], k_norm[b, h, :, :-1])
                    scores[b, h] = torch.softmax(-inner_prod / torch.sqrt(torch.tensor(q.size(-1))), dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
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
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of UHG attention."""
        batch_size = query.shape[0]
        
        # Add homogeneous coordinates if needed
        query = self._ensure_projective(query)
        key = self._ensure_projective(key)
        value = self._ensure_projective(value)
        
        # Transform inputs for each head
        q = []
        k = []
        v = []
        for h in range(self.config.num_heads):
            q_h = self._projective_transform(query, self.W_q[h])
            k_h = self._projective_transform(key, self.W_k[h])
            v_h = self._projective_transform(value, self.W_v[h])
            q.append(q_h)
            k.append(k_h)
            v.append(v_h)
        
        # Stack heads
        q = torch.stack(q, dim=1)  # [B, H, N, D]
        k = torch.stack(k, dim=1)  # [B, H, N, D]
        v = torch.stack(v, dim=1)  # [B, H, N, D]
        
        # Compute attention scores
        attention_weights = self._compute_attention_scores(q, k, mask)  # [B, H, Q, K]
        
        # Apply attention to values
        out = torch.zeros_like(q)
        for b in range(batch_size):
            for h in range(self.config.num_heads):
                for i in range(q.size(2)):  # For each query
                    # Weighted sum of values
                    weighted_sum = torch.zeros_like(v[b, h, 0])
                    for j in range(k.size(2)):  # For each key
                        weighted_sum += attention_weights[b, h, i, j] * v[b, h, j]
                    out[b, h, i] = self.uhg.normalize_points(weighted_sum)
        
        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
        out = out.view(batch_size, -1, self.config.feature_dim + 1)  # [B, N, D]
        
        # Final output projection
        out = self._projective_transform(out, self.W_o)
        
        # Return output and attention weights
        return out[..., :-1], attention_weights  # Remove homogeneous coordinate from output 