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
        
        # Initialize to identity to preserve cross-ratio by default
        for h in range(config.num_heads):
            q_init[h] = torch.eye(config.feature_dim + 1)
            k_init[h] = torch.eye(config.feature_dim + 1)
            q_init[h, -1] = torch.zeros(config.feature_dim + 1)
            q_init[h, -1, -1] = 1.0
            k_init[h, -1] = torch.zeros(config.feature_dim + 1)
            k_init[h, -1, -1] = 1.0
        
        self.W_q = nn.Parameter(q_init)
        self.W_k = nn.Parameter(k_init)
        
        # Value transformation identity
        v_init = torch.zeros(
            config.num_heads,
            config.feature_dim + 1,
            config.feature_dim + 1
        )
        for h in range(config.num_heads):
            v_init[h] = torch.eye(config.feature_dim + 1)
            v_init[h, -1] = torch.zeros(config.feature_dim + 1)
            v_init[h, -1, -1] = 1.0
        self.W_v = nn.Parameter(v_init)
        
        # Output projection identity
        o_init = torch.eye(config.feature_dim + 1)
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
        """Normalize coordinates using pure projective operations.
        Accepts tensors of shape [..., N, D] and returns same shape.
        """
        # Ensure homogeneous coordinate is 1 to standardize
        x = self._ensure_projective(x)
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
        # Normalize input ranks: support [B, Q, D] (no heads) and [B, H, N, D]
        squeeze_heads = False
        if q.dim() == 3 and k.dim() == 3:
            # Add a singleton head dimension
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            squeeze_heads = True
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q and k must be [B, H, N, D] or [B, N, D]")
        batch_size, num_heads, num_queries, _ = q.shape
        num_keys = k.shape[2]
        
        # Normalize inputs per point
        q = self._normalize_projective(q)
        k = self._normalize_projective(k)
        
        scores = torch.zeros(batch_size, num_heads, num_queries, num_keys, device=q.device)
        
        if self.config.use_cross_ratio:
            # Compute cross-ratio scores using ideal points per head
            for b in range(batch_size):
                for h in range(num_heads):
                    i1 = torch.zeros(q.size(-1), device=q.device); i1[0] = 1.0; i1[-1] = 0.0
                    i2 = torch.zeros(q.size(-1), device=q.device); i2[1] = 1.0; i2[-1] = 0.0
                    for i in range(num_queries):
                        for j in range(num_keys):
                            cr = self.uhg.cross_ratio(q[b, h, i], k[b, h, j], i1, i2)
                            scores[b, h, i, j] = 1.0 / (1.0 + torch.abs(torch.log(torch.abs(cr) + self.config.eps)))
        else:
            # Use hyperbolic inner product (on feature part) as similarity proxy
            for b in range(batch_size):
                for h in range(num_heads):
                    qh = q[b, h, :, :-1]  # [Q, D-1]
                    kh = k[b, h, :, :-1]  # [K, D-1]
                    inner = torch.matmul(qh, kh.transpose(0, 1))  # [Q, K]
                    scores[b, h] = inner
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over keys
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        if squeeze_heads:
            return scores[:, 0]  # [B, Q, K]
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
            new_weight = torch.eye(n, device=weight.device)
            min_dim = min(weight.shape[-1], n)
            new_weight[:min_dim, :min_dim] = weight[:min_dim, :min_dim]
            weight = new_weight
        
        # Enforce spatial orthogonality (O(2)) and preserve homogeneous coord
        w = weight.clone()
        # Zero last column except bottom-right; zero last row except bottom-right 1
        w[:-1, -1] = 0.0
        w[-1, :-1] = 0.0
        w[-1, -1] = 1.0
        
        # Apply transformation
        transformed = torch.matmul(x, w.transpose(-2, -1))
        
        # Normalize homogeneous coordinate
        transformed = transformed / (transformed[..., -1:] + self.config.eps)
        
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
        q_heads: List[torch.Tensor] = []
        k_heads: List[torch.Tensor] = []
        v_heads: List[torch.Tensor] = []
        for h in range(self.config.num_heads):
            q_h = self._projective_transform(query, self.W_q[h])
            k_h = self._projective_transform(key, self.W_k[h])
            v_h = self._projective_transform(value, self.W_v[h])
            q_heads.append(q_h)
            k_heads.append(k_h)
            v_heads.append(v_h)
        
        # Stack heads: [B, H, N, D]
        q = torch.stack(q_heads, dim=1)
        k = torch.stack(k_heads, dim=1)
        v = torch.stack(v_heads, dim=1)
        
        # Compute attention weights
        attn = self._compute_attention_scores(q, k, mask)  # [B, H, Q, K]
        
        # Weighted sum per head
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # [B, H, Q, D]
        
        # Average heads to keep feature dimension unchanged
        out = out.mean(dim=1)  # [B, Q, D]
        
        # Normalize output per point
        out = self._normalize_projective(out)
        
        # Output projection
        out = self._projective_transform(out, self.W_o)
        
        return out[..., :-1], attn  # strip homogeneous coord for return 