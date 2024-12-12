"""
Gjallarhorn1: UHG-Based Intrusion Detection System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
import os
from torch import Tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from uhg.projective import ProjectiveUHG

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProjectiveBaseLayer(nn.Module):
    """Base layer for UHG-compliant neural network operations."""
    
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]  # Remove homogeneous coordinate for transformation
        homogeneous = x[..., -1:]
        
        # Verify dimensions
        if features.size(1) != weight.size(1):
            raise ValueError(f"Feature dimension {features.size(1)} does not match weight dimension {weight.size(1)}")
        
        # Apply weight to features
        transformed = torch.matmul(features, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize to maintain projective structure
        return self.normalize_points(out)
        
    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie in projective space."""
        features = points[..., :-1]
        homogeneous = points[..., -1:]
        
        # Handle zero features
        zero_mask = torch.all(features == 0, dim=-1, keepdim=True)
        features = torch.where(zero_mask, torch.ones_like(features), features)
        
        # Normalize features
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        normalized_features = features / torch.clamp(norm, min=1e-8)
        
        # Handle homogeneous coordinate
        normalized = torch.cat([normalized_features, homogeneous], dim=-1)
        sign = torch.sign(normalized[..., -1:])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        
        return normalized * sign

class UHGSAGEConv(ProjectiveBaseLayer):
    """UHG-compliant GraphSAGE convolution using custom scatter."""
    
    def __init__(self, in_channels, out_channels, append_uhg=True):
        super().__init__()
        self.append_uhg = append_uhg
        # Simple linear transformation on concatenated features
        self.linear = nn.Linear(in_channels * 2, out_channels)
        
        # Initialize weights to preserve cross-ratios
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Scale weights for stability
        with torch.no_grad():
            self.linear.weight.div_(
                torch.norm(self.linear.weight, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            )
        
    def uhg_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """UHG-compliant normalization."""
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Compute UHG norm
        norm = torch.sqrt(torch.clamp(uhg_norm(x), min=1e-8))
        
        # Normalize features
        features = features / norm.unsqueeze(-1)
        
        # Ensure homogeneous coordinate is positive
        homogeneous = torch.sign(homogeneous) * torch.ones_like(homogeneous)
        
        return torch.cat([features, homogeneous], dim=-1)
        
    def forward(self, x, edge_index):
        """Forward pass with UHG-compliant operations."""
        # Get source and target nodes
        source_nodes, target_nodes = edge_index
        
        # Get source features for aggregation
        source_features = x[source_nodes]
        
        # Aggregate using custom scatter
        agg = scatter_mean_custom(source_features, target_nodes, dim=0, dim_size=x.size(0))
        
        # Concatenate self and aggregated features
        out = torch.cat([x, agg], dim=1)
        
        # Transform concatenated features
        out = self.linear(out)
        
        # Apply UHG-aware ReLU
        out = F.relu(out)
        
        # Add UHG coordinate if needed
        if self.append_uhg:
            ones = torch.ones((out.size(0), 1), device=out.device)
            out = torch.cat([out, ones], dim=1)
            
            # UHG-compliant normalization
            out = self.uhg_normalize(out)
            
            # Ensure cross-ratio preservation
            out = self.normalize_points(out)
        
        return out

class UHGModel(nn.Module):
    """UHG-compliant graph neural network model for anomaly detection."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer (input to hidden)
        self.convs.append(UHGSAGEConv(in_channels, hidden_channels, append_uhg=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(UHGSAGEConv(hidden_channels + 1, hidden_channels, append_uhg=True))
            
        # Output layer
        self.convs.append(UHGSAGEConv(hidden_channels + 1, out_channels, append_uhg=True))
        
    def forward(self, x, edge_index):
        """Forward pass through the model."""
        for conv in self.convs:
            x = conv(x, edge_index)
            
        return x

class UHGLoss(nn.Module):
    """UHG-compliant loss function optimized for anomaly detection."""
    
    def __init__(self, spread_weight: float = 0.1, quad_weight: float = 1.0):
        super().__init__()
        self.spread_weight = spread_weight
        self.quad_weight = quad_weight
        
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute UHG-compliant loss with anomaly detection focus."""
        # Get positive pairs
        mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
        pos_edge_index = edge_index[:, mask]
        
        if pos_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=z.device)
            
        # Compute quadrance-based scores with stability
        pos_quad = torch.clamp(uhg_quadrance(z[pos_edge_index[0]], z[pos_edge_index[1]]), max=10.0)
        
        # Generate and compute negative pairs (potential anomalies)
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

def scatter_mean_custom(src, index, dim=0, dim_size=None):
    """Custom scatter mean operation."""
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    # Initialize output tensor
    out = torch.zeros(dim_size, src.size(1), device=src.device)
    
    # Add features
    out = out.index_add_(0, index, src)
    
    # Compute counts for averaging
    ones = torch.ones((src.size(0), 1), device=src.device)
    count = torch.zeros(dim_size, 1, device=src.device).index_add_(0, index, ones)
    count[count == 0] = 1
    
    # Average features
    out = out / count
    
    return out

def prepare_features(data):
    """Prepare features for UHG processing with anomaly detection focus."""
    # Convert to tensor
    features = torch.tensor(data, dtype=torch.float32)
    
    # Normalize features for better anomaly detection
    norm = torch.norm(features, p=2, dim=-1, keepdim=True)
    features = features / norm.clamp(min=1e-8)
    
    # Add homogeneous coordinate
    features = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
    
    return features

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute UHG inner product."""
    return torch.sum(a[..., :-1] * b[..., :-1], dim=-1) - a[..., -1] * b[..., -1]

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """Compute UHG norm."""
    return torch.sum(a[..., :-1] ** 2, dim=-1) - a[..., -1] ** 2

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between points."""
    dot_product = uhg_inner_product(a, b)
    norm_a = uhg_norm(a)
    norm_b = uhg_norm(b)
    denom = norm_a * norm_b
    denom = torch.clamp(denom.abs(), min=eps)
    quadrance = 1 - (dot_product ** 2) / denom
    return quadrance

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between lines."""
    dot_product = uhg_inner_product(L, M)
    norm_L = uhg_norm(L)
    norm_M = uhg_norm(M)
    denom = norm_L * norm_M
    denom = torch.clamp(denom.abs(), min=eps)
    spread = 1 - (dot_product ** 2) / denom
    return spread

def train_step(model, optimizer, data):
    """Perform a single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    z = model(data.x, data.edge_index)
    
    # Remove homogeneous coordinate for loss computation
    z_features = z[..., :-1]
    
    # Compute loss
    loss = UHGLoss()(z_features, data.edge_index, data.batch_size)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(model, optimizer, data, num_epochs=100):
    """Train the model."""
    losses = []
    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, data)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')