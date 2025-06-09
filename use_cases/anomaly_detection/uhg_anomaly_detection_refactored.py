"""
UHG-based Anomaly Detection - Refactored to use UHG library.

This module implements anomaly detection using Universal Hyperbolic Geometry (UHG)
principles, leveraging the UHG library for core geometric operations.
"""

!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
!pip install tqdm geoopt matplotlib scikit-learn
!pip install torch-geometric
!pip install uhg

import os
import sys
import time
import math
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from tqdm.auto import tqdm
import traceback

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Import UHG utilities
from uhg_utils import (
    uhg_inner_product,
    uhg_norm,
    uhg_quadrance,
    uhg_spread,
    uhg_cross_ratio,
    to_uhg_space,
    normalize_points,
    get_uhg_instance
)

# Constants
FILE_PATH = "network_traffic.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom scatter operations for efficient aggregation
def scatter_add(src, index, dim=0, dim_size=None):
    """Custom scatter add operation for efficient aggregation."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def load_and_preprocess_data(file_path: str, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load and preprocess network traffic data for anomaly detection.
    
    Args:
        file_path: Path to the CSV file containing network traffic data
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (features, labels, feature_info)
    """
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Extract features and labels
    if 'label' in df.columns:
        labels = df['label'].values
        features = df.drop(['label'], axis=1).values
    else:
        # If no label column, assume all normal (for training)
        features = df.values
        labels = np.zeros(len(features))
    
    # Normalize features if requested
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Create feature info dictionary
    feature_info = {
        'num_features': features.shape[1],
        'num_samples': features.shape[0],
        'num_anomalies': int(labels.sum()),
        'anomaly_ratio': float(labels.sum() / len(labels)),
        'feature_names': df.columns.drop(['label']).tolist() if 'label' in df.columns else df.columns.tolist()
    }
    
    return features_tensor, labels_tensor, feature_info

def create_graph_data(node_features: torch.Tensor, 
                     labels: torch.Tensor, 
                     k: int = 2, 
                     batch_size: int = 4096,
                     use_uhg_distance: bool = True) -> Data:
    """
    Create a graph from node features using k-nearest neighbors.
    
    Args:
        node_features: Node feature tensor
        labels: Node label tensor
        k: Number of nearest neighbors
        batch_size: Batch size for processing large datasets
        use_uhg_distance: Whether to use UHG distance for kNN
        
    Returns:
        PyTorch Geometric Data object
    """
    n_samples = node_features.shape[0]
    
    # For large datasets, process in batches
    if n_samples > batch_size:
        print(f"Processing {n_samples} samples in batches of {batch_size}...")
        
        # Initialize edge index
        edge_index = []
        
        # Process in batches
        for i in tqdm(range(0, n_samples, batch_size)):
            batch_end = min(i + batch_size, n_samples)
            batch_features = node_features[i:batch_end]
            
            # Compute pairwise distances
            if use_uhg_distance:
                # Convert to UHG space
                uhg_features = to_uhg_space(batch_features)
                
                # Compute pairwise distances for this batch to all nodes
                distances = []
                for j in range(0, n_samples, batch_size):
                    j_end = min(j + batch_size, n_samples)
                    j_features = to_uhg_space(node_features[j:j_end])
                    
                    # Compute quadrance (UHG distance) between all pairs
                    batch_distances = []
                    for idx in range(len(uhg_features)):
                        point = uhg_features[idx:idx+1].expand(len(j_features), -1)
                        quad = uhg_quadrance(point, j_features)
                        batch_distances.append(quad)
                    
                    batch_distances = torch.cat(batch_distances, dim=0)
                    distances.append(batch_distances)
                
                # Combine all distances
                batch_all_distances = torch.cat(distances, dim=1)
                
                # Get k+1 nearest neighbors (including self)
                _, indices = torch.topk(batch_all_distances, k+1, dim=1, largest=False)
                
                # Create edge index
                for idx, neighbors in enumerate(indices):
                    source = torch.full((len(neighbors),), i + idx, dtype=torch.long)
                    batch_edge_index = torch.stack([source, neighbors], dim=0)
                    edge_index.append(batch_edge_index)
            else:
                # Use Euclidean distance with scikit-learn
                knn = NearestNeighbors(n_neighbors=k+1)
                knn.fit(node_features.numpy())
                distances, indices = knn.kneighbors(batch_features.numpy())
                
                # Create edge index
                for idx, neighbors in enumerate(indices):
                    source = torch.full((len(neighbors),), i + idx, dtype=torch.long)
                    target = torch.tensor(neighbors, dtype=torch.long)
                    batch_edge_index = torch.stack([source, target], dim=0)
                    edge_index.append(batch_edge_index)
        
        # Combine all edge indices
        edge_index = torch.cat(edge_index, dim=1)
    else:
        # For smaller datasets, process all at once
        if use_uhg_distance:
            # Convert to UHG space
            uhg_features = to_uhg_space(node_features)
            
            # Compute pairwise UHG distances
            distances = []
            for i in range(n_samples):
                point = uhg_features[i:i+1].expand(n_samples, -1)
                quad = uhg_quadrance(point, uhg_features)
                distances.append(quad)
            
            distances = torch.stack(distances)
            
            # Get k+1 nearest neighbors (including self)
            _, indices = torch.topk(distances, k+1, dim=1, largest=False)
            
            # Create edge index
            rows = torch.arange(n_samples).view(-1, 1).repeat(1, k+1).view(-1)
            cols = indices.view(-1)
            edge_index = torch.stack([rows, cols], dim=0)
        else:
            # Use Euclidean distance with scikit-learn
            knn = NearestNeighbors(n_neighbors=k+1)
            knn.fit(node_features.numpy())
            distances, indices = knn.kneighbors(node_features.numpy())
            
            # Create edge index
            rows = torch.arange(n_samples).view(-1, 1).repeat(1, k+1).view(-1)
            cols = torch.tensor(indices.flatten(), dtype=torch.long)
            edge_index = torch.stack([rows, cols], dim=0)
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels
    )
    
    return data

class UHGMessagePassing(MessagePassing):
    """
    UHG-based message passing layer for graph neural networks.
    Implements message passing using UHG principles.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(UHGMessagePassing, self).__init__(aggr='add')
        
        # Layer parameters
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UHG message passing layer.
        
        Args:
            x: Node feature tensor of shape [N, in_features]
            edge_index: Graph connectivity of shape [2, E]
            
        Returns:
            Updated node features of shape [N, out_features]
        """
        # Transform node features
        transformed_x = F.linear(x, self.weight, self.bias)
        
        # Add homogeneous coordinate for UHG space
        uhg_x = to_uhg_space(x)
        
        # Propagate messages
        aggr_out = self.propagate(edge_index, x=uhg_x, size=None)
        
        # Update node representations
        return self.update(aggr_out, transformed_x)
        
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        Message function for UHG-based message passing.
        
        Args:
            x_j: Features of neighboring nodes
            
        Returns:
            Messages from neighbors
        """
        # Normalize points in UHG space
        x_j_norm = normalize_points(x_j)
        
        # Return normalized messages
        return x_j_norm
        
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Aggregation function for UHG-based message passing.
        
        Args:
            inputs: Messages from neighbors
            index: Node indices for aggregation
            
        Returns:
            Aggregated messages
        """
        # Get number of nodes
        dim_size = index.max().item() + 1
        
        # Compute weights based on UHG principles
        # Here we use the norm of each message as a weight
        weights = uhg_norm(inputs).squeeze(-1)
        weights = F.softmax(weights, dim=0)
        
        # Weighted aggregation
        weighted_inputs = inputs * weights.unsqueeze(-1)
        
        # Sum aggregation
        aggr_out = scatter_add(weighted_inputs, index, dim=0, dim_size=dim_size)
        
        return aggr_out
        
    def update(self, aggr_out: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """
        Update function for UHG-based message passing.
        
        Args:
            aggr_out: Aggregated messages
            transformed_x: Transformed node features
            
        Returns:
            Updated node features
        """
        # Extract spatial components (remove homogeneous coordinate)
        aggr_out = aggr_out[..., :-1]
        
        # Combine with transformed features
        return transformed_x + aggr_out

class UHGGraphNN(nn.Module):
    """
    UHG-based Graph Neural Network for anomaly detection.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                embedding_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super(UHGGraphNN, self).__init__()
        
        # Model parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_layer = UHGMessagePassing(in_channels, hidden_channels)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.layers.append(UHGMessagePassing(hidden_channels, hidden_channels))
            
        # Output layer
        self.output_layer = UHGMessagePassing(hidden_channels, embedding_dim)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UHG Graph Neural Network.
        
        Args:
            x: Node feature tensor of shape [N, in_channels]
            edge_index: Graph connectivity of shape [2, E]
            
        Returns:
            Node embeddings of shape [N, embedding_dim]
        """
        # Input layer
        x = self.input_layer(x, edge_index)
        x = F.relu(x)
        x = self.batch_norms[0](x)
        x = self.dropout_layer(x)
        
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.batch_norms[i+1](x)
            x = self.dropout_layer(x)
            
        # Output layer
        x = self.output_layer(x, edge_index)
        
        # Add homogeneous coordinate for UHG space
        x = to_uhg_space(x)
        
        return x

class UHGAnomalyLoss(nn.Module):
    """
    UHG-based loss function for anomaly detection.
    Combines UHG geometric principles with contrastive learning.
    """
    
    def __init__(self, spread_weight: float = 0.1, quad_weight: float = 1.0, margin: float = 1.0):
        super(UHGAnomalyLoss, self).__init__()
        self.spread_weight = spread_weight
        self.quad_weight = quad_weight
        self.margin = margin
        
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute UHG-based anomaly detection loss.
        
        Args:
            z: Node embeddings in UHG space
            edge_index: Graph connectivity
            
        Returns:
            Loss value
        """
        # Extract source and target nodes
        src, dst = edge_index
        
        # Compute quadrance (UHG distance) between connected nodes
        quad = uhg_quadrance(z[src], z[dst])
        
        # Compute spread (UHG angle) between node lines
        # First create lines by joining each node with the origin
        origin = torch.zeros_like(z[0])
        origin[-1] = 1.0  # Set homogeneous coordinate to 1
        
        src_lines = torch.stack([z[src], origin.expand_as(z[src])], dim=1)
        dst_lines = torch.stack([z[dst], origin.expand_as(z[dst])], dim=1)
        
        # Compute spread between lines
        spread = uhg_spread(src_lines, dst_lines)
        
        # Compute contrastive loss
        # Positive pairs: minimize quadrance
        pos_loss = quad.mean()
        
        # Negative pairs: maximize quadrance up to margin
        # Create negative pairs by random permutation
        neg_src = src[torch.randperm(src.size(0))]
        neg_dst = dst[torch.randperm(dst.size(0))]
        
        # Compute quadrance for negative pairs
        neg_quad = uhg_quadrance(z[neg_src], z[neg_dst])
        
        # Hinge loss: max(0, margin - neg_quad)
        neg_loss = F.relu(self.margin - neg_quad).mean()
        
        # Combine losses
        quad_loss = pos_loss + neg_loss
        
        # Spread regularization: encourage perpendicular lines (spread = 1)
        spread_loss = F.mse_loss(spread, torch.ones_like(spread))
        
        # Total loss
        total_loss = self.quad_weight * quad_loss + self.spread_weight * spread_loss
        
        return total_loss

# Main function for testing
def main():
    """Main function to test the UHG-based anomaly detection."""
    print("Testing UHG-based anomaly detection with library integration...")
    
    # Load and preprocess data
    features, labels, feature_info = load_and_preprocess_data(FILE_PATH)
    print(f"Loaded data: {features.shape}, {labels.shape}")
    print(f"Feature info: {feature_info}")
    
    # Create graph data
    graph_data = create_graph_data(features, labels, k=5)
    print(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    
    # Create model
    model = UHGGraphNN(
        in_channels=features.shape[1],
        hidden_channels=64,
        embedding_dim=32,
        num_layers=2,
        dropout=0.2
    )
    print(f"Created model: {model}")
    
    # Create loss function
    criterion = UHGAnomalyLoss(spread_weight=0.1, quad_weight=1.0, margin=1.0)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train for a few epochs
    model.train()
    for epoch in range(5):
        # Forward pass
        optimizer.zero_grad()
        z = model(graph_data.x, graph_data.edge_index)
        
        # Compute loss
        loss = criterion(z, graph_data.edge_index)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print("Done!")

if __name__ == "__main__":
    main() 