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
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings in hyperbolic space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Generate hyperbolic frequencies
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Convert to hyperbolic space
        pe = self.uhg.normalize(pe)
        
        # Register buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.
        
        Args:
            x: Input features of shape [N, d_model]
            
        Returns:
            Features with positional encoding
        """
        # Add homogeneous coordinate
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        
        # Get positional encoding for sequence length
        pos_enc = self.pe[:x.size(0)]
        pos_enc = torch.cat([pos_enc, torch.ones_like(pos_enc[..., :1])], dim=-1)
        
        # Combine using projective addition
        x = self.uhg.add(x, pos_enc)
        
        # Apply dropout while preserving projective structure
        if self.training:
            x = self.projective_dropout(x)
            
        return x
        
    def projective_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout while preserving projective structure."""
        # Keep homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply dropout to features
        features = self.dropout(features)
        
        # Normalize to maintain hyperbolic structure
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
        
        # Multi-head attention
        self.self_attn = HyperbolicAttention(
            manifold=self.uhg.manifold,
            in_channels=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Feedforward network
        self.linear1 = HyperbolicLinear(
            manifold=self.uhg.manifold,
            in_features=d_model,
            out_features=dim_feedforward,
            bias=True
        )
        self.linear2 = HyperbolicLinear(
            manifold=self.uhg.manifold,
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
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Add homogeneous coordinate
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        
        # Self-attention
        attn_out = self.self_attn(x, edge_index, edge_attr, mask)
        
        # Add & normalize
        x = self.uhg.add(x, attn_out)
        x_features = x[..., :-1]
        x_features = self.norm1(x_features)
        x = torch.cat([x_features, x[..., -1:]], dim=-1)
        
        # Feedforward
        ff_out = self.linear1(x)
        ff_out = self.activation(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)
        
        # Add & normalize
        x = self.uhg.add(x, ff_out)
        x_features = x[..., :-1]
        x_features = self.norm2(x_features)
        x = torch.cat([x_features, x[..., -1:]], dim=-1)
        
        # Restore cross-ratio if possible
        if has_cr:
            cr_current = compute_cross_ratio(x[0], x[1], x[2], x[3])
            if not torch.isnan(cr_current) and not torch.isnan(cr_initial) and cr_current != 0:
                x = restore_cross_ratio(x, cr_initial)
        
        return x[..., :-1]  # Remove homogeneous coordinate

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
            manifold=self.uhg.manifold,
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
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add homogeneous coordinate
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, mask)
            
            # Restore cross-ratio if possible
            if has_cr:
                cr_current = compute_cross_ratio(x[0], x[1], x[2], x[3])
                if not torch.isnan(cr_current) and not torch.isnan(cr_initial) and cr_current != 0:
                    x = restore_cross_ratio(x, cr_initial)
        
        # Final projection
        x = self.output_proj(x)
        
        return x 