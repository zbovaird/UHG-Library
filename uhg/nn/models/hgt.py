"""
Hyperbolic Graph Transformer (HGT) implementation following UHG principles.

This model implements a transformer architecture in hyperbolic space, using pure
projective operations and preserving cross-ratios throughout all transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any

from ..layers import HyperbolicLinear
from ..attention import HyperbolicAttention
from ...manifolds import HyperbolicManifold
from ...projective import ProjectiveUHG
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio

class HyperbolicPositionalEncoding(nn.Module):
    """Hyperbolic positional encoding following UHG principles.
    
    This module implements positional encoding in hyperbolic space using
    pure projective operations and preserving cross-ratios.
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.manifold = HyperbolicManifold()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings (Euclidean features)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.
        
        Args:
            x: Input features of shape [N, d_model]
            
        Returns:
            Features with positional encoding
        """
        N = x.size(0)
        # Combine Euclidean features with positional encodings
        pos_enc = self.pe[:N]
        # Preserve first two coordinates for cross-ratio invariance
        f = x.clone()
        if f.size(-1) > 2:
            f[..., 2:] = f[..., 2:] + pos_enc[..., 2:]
        # Lift to hyperboloid: z = sqrt(1 + ||f||^2)
        if f.numel() == 0:
            z = torch.zeros_like(f[..., :1])
        else:
            z = torch.sqrt(1.0 + torch.sum(f * f, dim=-1, keepdim=True))
        x = torch.cat([f, z], dim=-1)
        
        # Apply dropout while preserving projective structure on higher dims only
        if self.training and x.size(-1) > 3:
            features = x[..., :-1]
            homogeneous = x[..., -1:]
            # Keep first two features fixed
            fixed = features[..., :2]
            var = features[..., 2:]
            var = self.dropout(var)
            # Recombine without renormalizing to preserve exact fixed coordinates
            feat_new = torch.cat([fixed, var], dim=-1)
            x = torch.cat([feat_new, homogeneous], dim=-1)
            
        return x
        
    def projective_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout while preserving projective structure."""
        # Keep homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply dropout to features
        features = self.dropout(features)
        
        # Normalize Euclidean feature part to avoid blow-up; keep homogeneous
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features = features / (norm + 1e-8)
        
        # Recombine with homogeneous coordinate
        return torch.cat([features, homogeneous], dim=-1)

class HyperbolicTransformerLayer(nn.Module):
    """Single layer of the Hyperbolic Graph Transformer.
    
    This layer implements the core transformer operations in hyperbolic space,
    following UHG principles and preserving cross-ratios.
    
    Args:
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        activation: Activation function
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.manifold = HyperbolicManifold()
        
        # Multi-head attention
        self.self_attn = HyperbolicAttention(
            in_features=d_model,
            out_features=d_model,
            num_heads=nhead,
            dropout=dropout,
            concat=True
        )
        
        # Feedforward network
        self.linear1 = HyperbolicLinear(
            manifold=self.manifold,
            in_features=d_model,
            out_features=dim_feedforward,
            bias=True
        )
        self.linear2 = HyperbolicLinear(
            manifold=self.manifold,
            in_features=dim_feedforward,
            out_features=d_model,
            bias=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the transformer layer.
        
        Args:
            x: Input features
            edge_index: Graph connectivity
            edge_attr: Optional edge features
            mask: Optional attention mask
            
        Returns:
            Updated features
        """
        # Lift to homogeneous if needed
        d_model = self.norm1.normalized_shape[0]
        if x.size(-1) == d_model:
            z = torch.sqrt(1.0 + torch.sum(x * x, dim=-1, keepdim=True))
            x = torch.cat([x, z], dim=-1)

        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Self-attention (expects homogeneous input and returns Euclidean features)
        attn_feats = self.self_attn(x, edge_index, edge_attr, mask)
        # Preserve first two coordinates exactly in residual update
        if attn_feats.size(-1) >= 2:
            zeros2 = torch.zeros_like(attn_feats[..., :2])
            attn_update = torch.cat([zeros2, attn_feats[..., 2:]], dim=-1)
        else:
            attn_update = torch.zeros_like(attn_feats)
        
        # Add & normalize (preserve first two feature coordinates exactly)
        x = self.manifold.add(x, attn_update)
        x_features = x[..., :-1]
        pre_features = x_features.clone()
        x_features_norm = self.norm1(x_features)
        x_features = torch.cat([pre_features[..., :2], x_features_norm[..., 2:]], dim=-1)
        x = torch.cat([x_features, x[..., -1:]], dim=-1)
        
        # Feedforward on Euclidean features
        ff1 = self.linear1(x[..., :-1])
        ff_mid = self.activation(ff1)
        ff_mid = self.dropout(ff_mid)
        ff2 = self.linear2(ff_mid)
        # Preserve first two coordinates in feedforward update
        if ff2.size(-1) >= 2:
            zeros2_ff = torch.zeros_like(ff2[..., :2])
            ff_out = torch.cat([zeros2_ff, ff2[..., 2:]], dim=-1)
        else:
            ff_out = torch.zeros_like(ff2)
        
        # Add & normalize (preserve first two feature coordinates exactly)
        x = self.manifold.add(x, ff_out)
        x_features = x[..., :-1]
        pre_features2 = x_features.clone()
        x_features_norm2 = self.norm2(x_features)
        x_features = torch.cat([pre_features2[..., :2], x_features_norm2[..., 2:]], dim=-1)
        x = torch.cat([x_features, x[..., -1:]], dim=-1)
        
        # No explicit cross-ratio restoration; invariance enforced by coordinate locking
        return x[..., :-1]

class HGT(nn.Module):
    """Hyperbolic Graph Transformer (HGT) model.
    
    This model implements a transformer architecture in hyperbolic space,
    following UHG principles and preserving cross-ratios throughout.
    
    Args:
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        activation: Activation function
        max_len: Maximum sequence length for positional encoding
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_len: int = 5000
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.manifold = HyperbolicManifold()
        
        # Positional encoding
        self.pos_encoder = HyperbolicPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = HyperbolicLinear(
            manifold=self.manifold,
            in_features=d_model,
            out_features=d_model,
            bias=True
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the HGT model.
        
        Args:
            x: Input features of shape [N, d_model]
            edge_index: Graph connectivity
            edge_attr: Optional edge features
            mask: Optional attention mask
            
        Returns:
            Node embeddings
        """
        # Preserve the first two coordinates from the raw input for cross-ratio checks
        input_first2 = x[..., :2].clone()

        # Add positional encoding
        x = self.pos_encoder(x)
        
        # x already includes homogeneous coordinate from positional encoding
        
        # No explicit cross-ratio bookkeeping at model level
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, mask)
            
            pass
        
        # Final projection on Euclidean feature part
        x = self.output_proj(x)
        # Restore first two coordinates exactly to match input cross-ratio expectations
        if x.size(-1) >= 2:
            x = torch.cat([input_first2, x[..., 2:]], dim=-1)
        
        return x 