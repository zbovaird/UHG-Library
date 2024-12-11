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
    """UHG-compliant GraphSAGE convolution layer using pure projective operations."""
    
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.in_features = in_features - 1  # Account for homogeneous coordinate
        self.out_features = out_features
        self.aggregator = aggregator.lower()
        
        # Initialize projective transformations using UHG principles
        self.W_self = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.W_neigh = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        
        if self.aggregator == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.in_features,
                hidden_size=self.in_features,
                batch_first=True
            )
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        # Initialize with special orthogonal matrices that preserve cross-ratios
        nn.init.orthogonal_(self.W_self)
        nn.init.orthogonal_(self.W_neigh)
        
        # Scale weights to improve numerical stability
        with torch.no_grad():
            self.W_self.div_(torch.norm(self.W_self, p=2, dim=1, keepdim=True).clamp(min=1e-8))
            self.W_neigh.div_(torch.norm(self.W_neigh, p=2, dim=1, keepdim=True).clamp(min=1e-8))
            
    def uhg_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply UHG-compliant transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]  # Remove homogeneous coordinate
        homogeneous = x[..., -1:]
        
        # Apply weight to features with numerical stability
        transformed = torch.matmul(features, weight.t())
        norm = torch.norm(transformed, p=2, dim=-1, keepdim=True)
        transformed = transformed / torch.clamp(norm, min=1e-8)
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize while preserving cross-ratios
        return self.normalize_points(out)
            
    def uhg_weighted_average(self, p1: torch.Tensor, p2: torch.Tensor, weight: float) -> torch.Tensor:
        """Compute UHG-compliant weighted average between two points."""
        # Ensure points are normalized
        p1 = self.normalize_points(p1)
        p2 = self.normalize_points(p2)
        
        # Get line through points
        line = self.uhg.join(p1, p2)
        
        # Get ideal points
        i1, i2 = self.uhg.get_ideal_points(line)
        
        # Compute cross-ratio based weighted average with stability
        t = torch.tensor(weight / (1 - weight + 1e-8), device=p1.device)  # Convert weight to cross-ratio parameter
        t = torch.clamp(t, min=1e-8, max=1e8)    # Prevent extreme values
        
        # Compute weighted point using cross-ratio
        num = p1 * t + p2
        denom = torch.ones_like(p1) * t + torch.ones_like(p2)
        avg = num / torch.clamp(denom, min=1e-8)
        
        # Ensure result preserves cross-ratio
        return self.normalize_points(avg)
    
    def forward(self, x, edge_index):
        """Forward pass using pure projective operations."""
        # Ensure input is normalized
        x = self.normalize_points(x)
        
        row, col = edge_index
        
        # Transform self features using UHG principles
        self_trans = self.uhg_transform(x, self.W_self)
        
        # Get and transform neighbor features
        neigh_features = x[col]
        neigh_trans = self.uhg_transform(neigh_features, self.W_neigh)
        
        # Initialize output tensor with correct shape
        out = torch.zeros(x.size(0), self.out_features + 1, device=x.device)
        out[..., -1] = 1  # Set homogeneous coordinate
        
        # Aggregate neighbor features using UHG-compliant operations
        if self.aggregator == 'mean':
            # Sum neighbor features
            out.index_add_(0, row, neigh_trans)
            
            # Compute mean while preserving cross-ratios
            count = torch.zeros(x.size(0), 1, device=x.device)
            count.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
            count = torch.clamp(count, min=1)
            
            # Apply normalization that preserves cross-ratios
            features = out[..., :-1] / count
            out = torch.cat([features, out[..., -1:]], dim=-1)
            out = self.normalize_points(out)
            
        elif self.aggregator == 'max':
            # Use UHG-compliant max operation
            for i in range(x.size(0)):
                mask = row == i
                if mask.any():
                    # Get neighbors for this node
                    neighbors = neigh_trans[mask]
                    neighbors = self.normalize_points(neighbors)
                    
                    # Compute cross-ratios with ideal points
                    line = self.uhg.join(neighbors[0], neighbors[-1])
                    i1, i2 = self.uhg.get_ideal_points(line)
                    
                    # Select point with maximum cross-ratio
                    cross_ratios = torch.stack([
                        self.uhg.cross_ratio(n, i1, i2, n)
                        for n in neighbors
                    ])
                    max_idx = torch.argmax(cross_ratios)
                    out[i] = neighbors[max_idx]
            
        else:  # lstm
            # Group neighbor features preserving UHG structure
            grouped_features = []
            for i in range(x.size(0)):
                mask = row == i
                if mask.any():
                    node_feats = neigh_trans[mask]
                    node_feats = self.normalize_points(node_feats)
                    grouped_features.append(node_feats)
                else:
                    # Initialize with identity element in projective space
                    zero_feat = torch.zeros(1, self.out_features + 1, device=x.device)
                    zero_feat[..., -1] = 1
                    grouped_features.append(zero_feat)
            
            # Process through LSTM
            packed = nn.utils.rnn.pack_sequence(grouped_features, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            out = self.normalize_points(unpacked)
        
        # Combine self and neighbor features using UHG-compliant weighted average
        batch_size = x.size(0)
        combined = torch.zeros_like(self_trans)
        
        # Process each point pair using UHG operations
        for i in range(batch_size):
            # Get points for this batch element
            p1 = self_trans[i]
            p2 = out[i]
            
            # Compute weighted average using cross-ratio
            combined[i] = self.uhg_weighted_average(p1, p2, 0.5)
        
        return self.normalize_points(combined)

class UHGModel(nn.Module):
    """UHG-compliant graph neural network model."""
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        
        # Account for homogeneous coordinate
        self.in_features = in_features - 1
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        # UHG-compliant layers
        self.conv1 = UHGSAGEConv(self.in_features + 1, self.hidden_features)
        self.conv2 = UHGSAGEConv(self.hidden_features + 1, self.out_features)
        
    def forward(self, x, edge_index):
        """Forward pass through the model."""
        # First convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x[..., :-1])  # Apply ReLU to features only
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)  # Add homogeneous coordinate back
        
        # Second convolution
        x = self.conv2(x, edge_index)
        
        return x

class UHGLoss(nn.Module):
    """UHG-compliant loss function using cross-ratios."""
    
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
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
        
    def compute_stable_cross_ratio(self, p1, p2, p3, p4):
        """Compute cross-ratio with numerical stability."""
        # Ensure all points are normalized
        points = [self.normalize_points(p) for p in [p1, p2, p3, p4]]
        p1, p2, p3, p4 = points
        
        # Compute distances with stability
        d12 = torch.norm(p1 - p2, p=2)
        d34 = torch.norm(p3 - p4, p=2)
        d13 = torch.norm(p1 - p3, p=2)
        d24 = torch.norm(p2 - p4, p=2)
        
        # Compute cross-ratio
        num = d12 * d34
        denom = d13 * d24
        
        # Add stability term
        epsilon = 1e-8
        cr = num / (denom + epsilon)
        
        return torch.clamp(cr, min=-5, max=5)
        
    def forward(self, z, edge_index, batch_size):
        """Compute UHG-compliant loss using cross-ratios."""
        # Add homogeneous coordinate for projective operations
        z_proj = torch.cat([z, torch.ones_like(z[..., :1])], dim=-1)
        z_proj = self.normalize_points(z_proj)
        
        # Get positive pairs from edge_index
        mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
        pos_edge_index = edge_index[:, mask]
        
        if pos_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=z.device)
            
        # Generate negative pairs using UHG principles
        neg_edge_index = torch.randint(0, batch_size, (2, batch_size), device=z.device)
        
        # Initialize losses
        pos_loss = torch.tensor(0.0, device=z.device)
        neg_loss = torch.tensor(0.0, device=z.device)
        
        # Process positive pairs with numerical stability
        for i in range(pos_edge_index.size(1)):
            src, dst = pos_edge_index[:, i]
            # Get line through points
            line = self.uhg.join(z_proj[src], z_proj[dst])
            # Get ideal points
            i1, i2 = self.uhg.get_ideal_points(line)
            # Compute stable cross-ratio
            cr = self.compute_stable_cross_ratio(z_proj[src], z_proj[dst], i1, i2)
            target = torch.ones_like(cr)
            pos_loss += F.binary_cross_entropy_with_logits(cr, target)
            
        # Process negative pairs with numerical stability
        for i in range(neg_edge_index.size(1)):
            src, dst = neg_edge_index[:, i]
            line = self.uhg.join(z_proj[src], z_proj[dst])
            i1, i2 = self.uhg.get_ideal_points(line)
            cr = self.compute_stable_cross_ratio(z_proj[src], z_proj[dst], i1, i2)
            target = torch.zeros_like(cr)
            neg_loss += F.binary_cross_entropy_with_logits(cr, target)
            
        # Normalize losses
        pos_loss = pos_loss / pos_edge_index.size(1)
        neg_loss = neg_loss / neg_edge_index.size(1)
        
        # Add regularization term for cross-ratio preservation with stability
        reg_loss = torch.tensor(0.0, device=z.device)
        num_samples = min(10, batch_size)
        indices = torch.randperm(batch_size)[:num_samples]
        
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                line = self.uhg.join(z_proj[indices[i]], z_proj[indices[j]])
                i1, i2 = self.uhg.get_ideal_points(line)
                cr1 = self.compute_stable_cross_ratio(z_proj[indices[i]], z_proj[indices[j]], i1, i2)
                cr2 = self.compute_stable_cross_ratio(i1, i2, z_proj[indices[i]], z_proj[indices[j]])
                reg_loss += torch.abs(cr1 * cr2 - 1.0)
                
        reg_loss = reg_loss / (num_samples * (num_samples - 1) / 2)
        
        # Combine losses with stability
        total_loss = pos_loss + neg_loss + 0.1 * reg_loss
        return torch.clamp(total_loss, min=0, max=5)  # Final stability check

def prepare_features(data):
    """Prepare features for UHG processing."""
    # Convert to tensor
    features = torch.tensor(data, dtype=torch.float32, device=device)
    # Add homogeneous coordinate
    features = torch.cat([features, torch.ones(features.size(0), 1, device=device)], dim=1)
    # Normalize in projective space
    uhg = ProjectiveUHG()
    return uhg.normalize_points(features) 

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