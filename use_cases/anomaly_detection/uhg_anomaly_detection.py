#!/usr/bin/env python3
"""
UHG-Based Anomaly Detection System

This implementation uses Universal Hyperbolic Geometry principles to create
an effective anomaly detection system for security log analysis.
"""

# For Google Colab Users:
# Copy and run this cell first to install dependencies
'''
!pip install torch-geometric
!pip install torch-scatter
'''

# Set up Google Drive access with this code if needed
'''
from google.colab import drive
drive.mount('/content/drive')
'''

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, degree
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from typing import Tuple, Dict, List, Optional, Set, Union, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm.auto import tqdm
from dataclasses import dataclass
import warnings
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
# Try to import scatter_add from torch_scatter, otherwise use a workaround
try:
    from torch_scatter import scatter_add
except ImportError:
    # Define a workaround using torch.scatter
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        output = torch.zeros(dim_size, src.shape[1], device=src.device)
        return output.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Constants
VISUALIZATION_DIR = "/content/outputs/visualizations"
MODEL_DIR = "/content/outputs/models"
RESULTS_DIR = "results"

# Create directories if they don't exist
for directory in [VISUALIZATION_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

###########################################
# UHG Core Operations
###########################################

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hyperbolic inner product between points in projective coordinates."""
    # Extract spatial and time components
    a_spatial, a_time = a[..., :-1], a[..., -1:]
    b_spatial, b_time = b[..., :-1], b[..., -1:]
    
    # Compute the inner product: -<a_spatial, b_spatial> + a_time * b_time
    return -torch.sum(a_spatial * b_spatial, dim=-1, keepdim=True) + a_time * b_time

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """Compute the UHG norm of a point."""
    return uhg_inner_product(a, a)

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute quadrance between two points in UHG.
    Quadrance corresponds to squared distance in Euclidean geometry.
    """
    # Get the inner products
    aa = uhg_inner_product(a, a)
    bb = uhg_inner_product(b, b)
    ab = uhg_inner_product(a, b)
    
    # Compute quadrance using the formula from UHG
    numerator = ab * ab - aa * bb
    denominator = aa * bb
    
    # Ensure numerical stability
    safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
    safe_sign = torch.sign(denominator)
    
    # Compute the final quadrance with proper sign
    quad = numerator / (safe_denominator * safe_sign)
    
    return quad.squeeze(-1)

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute spread between two lines in UHG.
    Spread is the dual of quadrance and measures the squared angle.
    """
    # Get the inner products
    LL = uhg_inner_product(L, L)
    MM = uhg_inner_product(M, M)
    LM = uhg_inner_product(L, M)
    
    # Compute spread
    numerator = LM * LM - LL * MM
    denominator = LL * MM
    
    # Ensure numerical stability
    safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
    safe_sign = torch.sign(denominator)
    
    # Compute the final spread with proper sign
    spread = numerator / (safe_denominator * safe_sign)
    
    return spread.squeeze(-1)

def uhg_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, 
                   p3: torch.Tensor, p4: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute the cross-ratio of four points in UHG."""
    # Compute the quadrances
    q_12 = uhg_quadrance(p1, p2)
    q_34 = uhg_quadrance(p3, p4)
    q_13 = uhg_quadrance(p1, p3)
    q_24 = uhg_quadrance(p2, p4)
    
    # Calculate cross-ratio
    numerator = q_12 * q_34
    denominator = q_13 * q_24
    
    # Ensure numerical stability
    safe_denominator = torch.clamp_min(torch.abs(denominator), eps)
    safe_sign = torch.sign(denominator)
    
    # Return the cross-ratio
    return (numerator / (safe_denominator * safe_sign)).squeeze(-1)

def to_uhg_space(x: torch.Tensor) -> torch.Tensor:
    """Convert feature vectors to UHG space with homogeneous coordinates."""
    # Get the feature dimension
    feature_dim = x.shape[-1]
    
    # Special case for using with torch.cross:
    # If this is being prepared for operations that need exactly 3 dimensions
    # (like torch.cross), we should ensure the resulting tensor has 3 dimensions
    if feature_dim != 2:
        # If dimension is 1, pad to 2D before adding homogeneous coordinate
        if feature_dim == 1:
            # Pad with zeros to make it 2D + homogeneous = 3D
            padding = torch.zeros(*x.shape[:-1], 1, device=x.device)
            return torch.cat([x, padding, torch.ones(*x.shape[:-1], 1, device=x.device)], dim=-1)
        # If dimension is > 3, truncate to 2D before adding homogeneous coordinate
        elif feature_dim > 2:
            # Take first 2 dimensions and add homogeneous coordinate
            x_truncated = x[..., :2]
            return torch.cat([x_truncated, torch.ones(*x.shape[:-1], 1, device=x.device)], dim=-1)
    
    # Standard case: append a homogeneous coordinate of 1.0
    homogeneous = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.cat([x, homogeneous], dim=-1)

###########################################
# Data Loading and Preprocessing
###########################################

def load_and_preprocess_data(file_path: str, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load and preprocess the data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        normalize: Whether to normalize the features
        
    Returns:
        node_features: Tensor of node features
        labels: Tensor of labels (0 for normal, 1 for anomaly)
        feature_info: Dictionary with information about features
    """
    print(f"Loading data from {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Extract features and labels
        labels = df.pop('label').values if 'label' in df.columns else np.zeros(len(df))
        
        # Store feature information
        feature_info = {
            'names': list(df.columns),
            'original_shape': df.shape
        }
        
        # Convert to tensors
        features = torch.tensor(df.values, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Normalize features if requested
        if normalize:
            # Calculate statistics on normal samples only
            normal_mask = (labels == 0)
            normal_features = features[normal_mask]
            
            # Calculate mean and std
            mean = normal_features.mean(dim=0, keepdim=True)
            std = normal_features.std(dim=0, keepdim=True) + 1e-6  # Add epsilon to avoid division by zero
            
            # Normalize all features
            features = (features - mean) / std
            
            # Store normalization parameters
            feature_info['mean'] = mean
            feature_info['std'] = std
        
        print(f"Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"Normal samples: {(labels == 0).sum().item()}, Anomalous samples: {(labels == 1).sum().item()}")
        
        return features, labels, feature_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_graph_data(node_features: torch.Tensor, 
                     labels: torch.Tensor, 
                     k: int = 2, 
                     batch_size: int = 4096,
                     use_uhg_distance: bool = True) -> Data:
    """
    Create a graph from node features using k-nearest neighbors.
    
    Args:
        node_features: Tensor of node features
        labels: Tensor of labels
        k: Number of nearest neighbors
        batch_size: Batch size for KNN computation to avoid OOM
        use_uhg_distance: Whether to use UHG distance for KNN
        
    Returns:
        graph_data: PyTorch Geometric Data object
    """
    print(f"Creating graph with k={k} nearest neighbors")
    start_time = time.time()
    
    # Get number of nodes and dimension
    num_nodes = node_features.shape[0]
    
    # Initialize edge index
    edge_index = []
    
    # Process in batches to avoid OOM
    for i in tqdm(range(0, num_nodes, batch_size), desc="Computing KNN graph"):
        batch_end = min(i + batch_size, num_nodes)
        batch_features = node_features[i:batch_end]
        
        # Convert to UHG space if requested
        if use_uhg_distance:
            source = to_uhg_space(batch_features)
            target = to_uhg_space(node_features)
            
            # Compute pairwise quadrances (UHG distances)
            distances = []
            for j in range(len(source)):
                src = source[j:j+1]  # Keep batch dimension
                # Compute quadrance with all targets
                quad = uhg_quadrance(src, target)
                distances.append(quad)
            
            # Stack distances
            distances = torch.stack(distances)
            
        else:
            # Use Euclidean distance
            batch_expanded = batch_features.unsqueeze(1)  # [batch, 1, dim]
            nodes_expanded = node_features.unsqueeze(0)  # [1, num_nodes, dim]
            
            # Compute pairwise Euclidean distances
            distances = torch.sum((batch_expanded - nodes_expanded) ** 2, dim=2)  # [batch, num_nodes]
        
        # For each node, find k nearest neighbors
        _, indices = torch.topk(distances, k=min(k+1, num_nodes), dim=1, largest=False)
        
        # Convert to edge index format
        for j in range(len(indices)):
            node_idx = i + j
            # Skip self-loop (first element if using UHG)
            start_idx = 1 if use_uhg_distance and indices[j, 0] == node_idx else 0
            for neighbor_pos in range(start_idx, len(indices[j])):
                neighbor_idx = indices[j, neighbor_pos].item()
                if neighbor_idx != node_idx:  # Skip self-loops
                    edge_index.append([node_idx, neighbor_idx])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Remove self-loops to be safe
    edge_index, _ = remove_self_loops(edge_index)
    
    # Create the graph data object
    graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
    
    print(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    print(f"Graph creation took {time.time() - start_time:.2f} seconds")
    
    return graph_data

def split_data(graph_data: Data, val_ratio: float = 0.15, test_ratio: float = 0.15, 
              stratify: bool = True) -> Tuple[Data, Data, Data, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the graph data into training, validation, and test sets.
    For anomaly detection, we only use normal samples for training.
    
    Args:
        graph_data: PyTorch Geometric Data object
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        stratify: Whether to maintain the same anomaly ratio in val/test sets
        
    Returns:
        train_data: Training data (normal samples only)
        val_data: Validation data 
        test_data: Test data
        train_mask: Mask for training nodes
        val_mask: Mask for validation nodes
        test_mask: Mask for test nodes
    """
    num_nodes = graph_data.num_nodes
    labels = graph_data.y
    
    # Create indices and shuffle
    indices = torch.randperm(num_nodes)
    
    if stratify:
        # Split maintaining anomaly ratio in each set
        normal_indices = indices[labels[indices] == 0]
        anomaly_indices = indices[labels[indices] == 1]
        
        # Calculate split sizes
        n_val_normal = int(len(normal_indices) * val_ratio)
        n_test_normal = int(len(normal_indices) * test_ratio)
        n_train_normal = len(normal_indices) - n_val_normal - n_test_normal
        
        n_val_anomaly = int(len(anomaly_indices) * val_ratio)
        n_test_anomaly = int(len(anomaly_indices) * test_ratio)
        
        # Create masks
        train_indices = normal_indices[:n_train_normal]
        val_indices = torch.cat([
            normal_indices[n_train_normal:n_train_normal+n_val_normal],
            anomaly_indices[:n_val_anomaly]
        ])
        test_indices = torch.cat([
            normal_indices[n_train_normal+n_val_normal:],
            anomaly_indices[n_val_anomaly:n_val_anomaly+n_test_anomaly]
        ])
    else:
        # Simple random split
        n_test = int(num_nodes * test_ratio)
        n_val = int(num_nodes * val_ratio)
        n_train = num_nodes - n_test - n_val
        
        # Only use normal samples for training
        train_indices = indices[labels[indices] == 0][:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Data split: {train_mask.sum()} train, {val_mask.sum()} validation, {test_mask.sum()} test")
    print(f"Train set anomalies: {labels[train_mask].sum()}/{train_mask.sum()}")
    print(f"Validation set anomalies: {labels[val_mask].sum()}/{val_mask.sum()}")
    print(f"Test set anomalies: {labels[test_mask].sum()}/{test_mask.sum()}")
    
    # Apply masks to create subgraphs
    train_data = Data(x=graph_data.x[train_mask], 
                     edge_index=None,  # We'll reconstruct the edges
                     y=graph_data.y[train_mask])
    
    val_data = Data(x=graph_data.x[val_mask],
                   edge_index=None,
                   y=graph_data.y[val_mask])
    
    test_data = Data(x=graph_data.x[test_mask],
                    edge_index=None,
                    y=graph_data.y[test_mask])
    
    return train_data, val_data, test_data, train_mask, val_mask, test_mask

###########################################
# UHG Neural Network Components
###########################################

class UHGMessagePassing(MessagePassing):
    """Message passing layer for UHG-based graph neural networks."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize the UHG message passing layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
        """
        super().__init__(aggr='add')  # We'll implement custom aggregation
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Linear transformation for messages
        self.message_linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.message_linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.message_linear.bias is not None:
            nn.init.zeros_(self.message_linear.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UHG message passing layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Add self-loops to include the node's own features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform the node features with linear layer
        transformed_x = self.linear(x)
        
        # Execute the message passing scheme
        return self.propagate(edge_index, x=x, transformed_x=transformed_x)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        Message function for edge (i,j).
        Creates messages from source node j to target node i.
        
        Args:
            x_j: Source node features
            
        Returns:
            Messages from source nodes
        """
        # Transform message with UHG-aware weights
        return self.message_linear(x_j)
    
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Aggregation function that performs a weighted sum of messages.
        
        Args:
            inputs: Messages to aggregate
            index: Target node indices
            
        Returns:
            Aggregated messages
        """
        # Implement UHG-aware aggregation
        # Convert to UHG space for weighted aggregation
        uhg_inputs = to_uhg_space(inputs)
        
        # Compute weight importance based on UHG quadrance from origin
        origin = torch.zeros(1, uhg_inputs.shape[-1], device=uhg_inputs.device)
        origin[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Compute importance weights (inverse of quadrance)
        quad = uhg_quadrance(uhg_inputs, origin)
        weights = 1.0 / (quad + 1.0)  # Add 1 to avoid division by zero
        
        # Apply weights
        weighted_inputs = inputs * weights.unsqueeze(-1)
        
        # Perform weighted aggregation
        out = torch_geometric.utils.scatter(weighted_inputs, index, dim=0, reduce='sum')
        
        return out
    
    def update(self, aggr_out: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """
        Update function for node features after aggregation.
        
        Args:
            aggr_out: Aggregated messages
            transformed_x: Transformed node features
            
        Returns:
            Updated node features
        """
        # Combine node's transformed features with aggregated messages
        return transformed_x + aggr_out

class UHGGraphNN(nn.Module):
    """UHG-based graph neural network for anomaly detection."""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                embedding_dim: int, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the UHG graph neural network.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            embedding_dim: Dimension of final embeddings
            num_layers: Number of message passing layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_layer = UHGMessagePassing(in_channels, hidden_channels)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(UHGMessagePassing(hidden_channels, hidden_channels))
        
        # Output projection to embedding dimension
        self.proj = nn.Linear(hidden_channels, embedding_dim)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UHG graph neural network.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Input layer
        x = self.input_layer(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.batch_norms[i+1](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project to embedding dimension
        x = self.proj(x)
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=-1)
        
        return x

###########################################
# Anomaly Detection Components
###########################################

class UHGAnomalyLoss(nn.Module):
    """
    UHG-based anomaly detection loss function.
    Combines reconstruction, compactness, and geometric properties.
    """
    
    def __init__(self, spread_weight: float = 0.1, quad_weight: float = 1.0, margin: float = 1.0):
        """
        Initialize the UHG anomaly loss.
        
        Args:
            spread_weight: Weight for spread term
            quad_weight: Weight for quadrance term
            margin: Margin for contrastive term
        """
        super().__init__()
        self.spread_weight = spread_weight
        self.quad_weight = quad_weight
        self.margin = margin
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the anomaly detection loss.
        
        Args:
            z: Node embeddings [num_nodes, dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Loss value
        """
        # Convert to UHG space
        z_uhg = to_uhg_space(z)
        
        # Get source and target nodes
        src, dst = edge_index
        z_src = z_uhg[src]
        z_dst = z_uhg[dst]
        
        # 1. Proximity loss: connected nodes should be close
        quad = uhg_quadrance(z_src, z_dst)
        proximity_loss = quad.mean()
        
        # 2. Compactness loss: all nodes should be close to the origin
        origin = torch.zeros(1, z_uhg.shape[-1], device=z_uhg.device)
        origin[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Compute quadrance from origin
        compactness_quad = uhg_quadrance(z_uhg, origin)
        compactness_loss = compactness_quad.mean()
        
        # 3. Spread loss: maintain geometric structure
        # Compute lines between each node and the origin
        lines = []
        
        # Check if we can use torch.cross (requires exactly 3D vectors)
        if z_uhg.shape[-1] == 3:
            for i in range(z_uhg.shape[0]):
                # Join node with origin
                line = torch.cross(z_uhg[i], origin.squeeze(0))
                lines.append(line)
        else:
            # For higher dimensions, use generalized outer product
            # (p₁ ⊗ p₂ - p₂ ⊗ p₁) for each pair of components
            for i in range(z_uhg.shape[0]):
                # Manual outer product calculation (simplified join operation)
                p1 = z_uhg[i]
                p2 = origin.squeeze(0)
                
                # Create an empty tensor for the line
                line = torch.zeros_like(p1)
                
                # Only compute the relevant components we need for spread
                # This is a simplified version that focuses on the key components
                # that affect the spread calculation
                line[..., :-1] = p1[..., -1:] * p2[..., :-1] - p2[..., -1:] * p1[..., :-1]
                line[..., -1] = torch.sum(p1[..., :-1] * p2[..., :-1], dim=-1)
                
                lines.append(line)
        
        lines = torch.stack(lines)
        
        # Compute spreads between neighboring lines
        spreads = []
        for i in range(min(10, len(src))):  # Limit to avoid excessive computation
            line_src = lines[src[i]]
            line_dst = lines[dst[i]]
            sp = uhg_spread(line_src, line_dst)
            spreads.append(sp)
        
        if spreads:
            spread_loss = torch.stack(spreads).mean()
        else:
            spread_loss = torch.tensor(0.0, device=z.device)
        
        # Combine losses
        loss = self.quad_weight * (proximity_loss + compactness_loss) + self.spread_weight * spread_loss
        
        return loss

class UHGAnomalyDetector:
    """
    UHG-based anomaly detector that trains on normal data and detects anomalies.
    """
    
    def __init__(self, in_features: int, hidden_dim: int = 64, embedding_dim: int = 32,
                num_layers: int = 2, dropout: float = 0.2, lr: float = 0.001,
                k: int = 10, device = None):
        """
        Initialize the UHG anomaly detector.
        
        Args:
            in_features: Number of input features
            hidden_dim: Hidden dimension
            embedding_dim: Embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            lr: Learning rate
            k: Number of nearest neighbors for graph construction
            device: Device to use (cpu or cuda)
        """
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.k = k
        self.device = device if device is not None else DEVICE
        
        # Initialize model
        self.model = UHGGraphNN(in_features, hidden_dim, embedding_dim, num_layers, dropout)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize loss
        self.criterion = UHGAnomalyLoss()
        
        # Initialize baseline statistics
        self.baseline_embeddings = None
        self.baseline_stats = {}
        self.threshold = None
        
        # Training history
        self.history = {'loss': [], 'val_loss': [], 'thresholds': []}
    
    def create_dataloader(self, data: Data, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create a dataloader for the given data.
        
        Args:
            data: PyTorch Geometric Data object
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader
        """
        return DataLoader([data], batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, train_data: Data, val_data: Data = None, 
                  epochs: int = 100, batch_size: int = 32, 
                  early_stopping: int = 10, verbose: int = 1) -> Dict:
        """
        Train the anomaly detection model on normal data. For unsupervised learning,
        we assume all training data represents normal behavior.
        
        Args:
            train_data: Training data with precomputed edge_index
            val_data: Validation data with precomputed edge_index
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Number of epochs to wait for improvement
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print(f"Training model on {train_data.num_nodes} samples (unsupervised learning)")
        
        # Create dataloader
        train_loader = self.create_dataloader(train_data, batch_size=batch_size)
        
        # Best validation loss
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self._train_epoch(train_loader)
            
            # Evaluate on validation set if provided
            val_loss = float('inf')
            if val_data is not None:
                # Compute embeddings
                embeddings = self.compute_embeddings(train_data.x, train_data.edge_index)
                
                # Compute baseline statistics on training embeddings
                self._compute_baseline_stats(embeddings)
                
                # Compute validation loss
                val_loss = self._compute_validation_loss(val_data)
                
                # For unsupervised learning, set threshold based on percentile
                # of anomaly scores on validation data
                val_scores = self.compute_anomaly_scores(val_data.x, val_data.edge_index)
                self.threshold = torch.quantile(val_scores, 0.95).item()  # 95th percentile
            
            # Save history
            self.history['loss'].append(train_loss)
            self.history['val_loss'] = self.history.get('val_loss', []) + [val_loss]
            self.history['thresholds'].append(self.threshold)
            
            # Print progress
            if verbose >= 1 and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}", end="")
                if val_data is not None:
                    print(f" - Val Loss: {val_loss:.4f} - Threshold: {self.threshold:.4f}")
                else:
                    print("")
            
            # Early stopping based on validation loss
            if val_data is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Save best model
                self._save_model("best_model.pt")
            elif epoch - best_epoch >= early_stopping and early_stopping > 0:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                self._load_model("best_model.pt")
                break
        
        # After training, compute final baseline on training data
        embeddings = self.compute_embeddings(train_data.x, train_data.edge_index)
        self._compute_baseline_stats(embeddings)
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            loader: DataLoader
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            z = self.model(batch.x, batch.edge_index)
            
            # Compute loss
            loss = self.criterion(z, batch.edge_index)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def compute_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for the given data.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Node embeddings
        """
        self.model.eval()
        
        # Move data to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Compute embeddings
        with torch.no_grad():
            z = self.model(x, edge_index)
        
        return z
    
    def _compute_baseline_stats(self, embeddings: torch.Tensor) -> Dict:
        """
        Compute baseline statistics on normal embeddings.
        
        Args:
            embeddings: Normal embeddings
            
        Returns:
            Baseline statistics
        """
        # Save baseline embeddings
        self.baseline_embeddings = embeddings
        
        # Convert to UHG space
        uhg_embeddings = to_uhg_space(embeddings)
        
        # Compute center (mean) of normal embeddings
        center = uhg_embeddings.mean(dim=0, keepdim=True)
        
        # Compute quadrances from center
        quadrances = uhg_quadrance(uhg_embeddings, center)
        
        # Compute statistics
        self.baseline_stats = {
            'center': center,
            'mean_quadrance': quadrances.mean().item(),
            'std_quadrance': quadrances.std().item(),
            'max_quadrance': quadrances.max().item(),
            'min_quadrance': quadrances.min().item(),
            'percentiles': {
                '50': quadrances.quantile(0.5).item(),
                '75': quadrances.quantile(0.75).item(),
                '90': quadrances.quantile(0.9).item(),
                '95': quadrances.quantile(0.95).item(),
                '99': quadrances.quantile(0.99).item()
            }
        }
        
        # Set default threshold at 95th percentile if not set
        if self.threshold is None:
            self.threshold = self.baseline_stats['percentiles']['95']
        
        return self.baseline_stats
    
    def compute_anomaly_scores(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Compute anomaly scores for the given data.
        
        Args:
            x: Node features
            edge_index: Graph connectivity (optional, will be created if not provided)
            
        Returns:
            Anomaly scores
        """
        # If edge_index is not provided and we need to create a k-NN graph
        # This case should be rare as we're now passing precomputed edge_index
        if edge_index is None:
            print("Warning: edge_index not provided, creating KNN graph. Consider precomputing the graph.")
            # Create dummy labels
            dummy_labels = torch.zeros(x.shape[0])
            # Create graph
            graph_data = create_graph_data(x, dummy_labels, k=self.k)
            edge_index = graph_data.edge_index
        
        # Compute embeddings
        embeddings = self.compute_embeddings(x, edge_index)
        
        # Convert to UHG space
        uhg_embeddings = to_uhg_space(embeddings)
        
        # Get baseline center
        center = self.baseline_stats['center']
        
        # Compute quadrances from center (UHG distance)
        quadrances = uhg_quadrance(uhg_embeddings, center)
        
        # Normalize scores
        mean_quad = self.baseline_stats['mean_quadrance']
        std_quad = self.baseline_stats['std_quadrance']
        z_scores = (quadrances - mean_quad) / (std_quad + 1e-10)
        
        # Convert to probability using sigmoid
        anomaly_scores = torch.sigmoid(z_scores)
        
        return anomaly_scores
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor = None, 
               return_scores: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict anomalies for the given data.
        
        Args:
            x: Node features
            edge_index: Graph connectivity (optional)
            return_scores: Whether to return anomaly scores
            
        Returns:
            Predictions (0 for normal, 1 for anomaly)
            Anomaly scores (if return_scores is True)
        """
        # Compute anomaly scores
        scores = self.compute_anomaly_scores(x, edge_index)
        
        # Make predictions
        predictions = (scores > self.threshold).float()
        
        if return_scores:
            return predictions, scores
        else:
            return predictions
    
    def _compute_validation_loss(self, val_data: Data) -> float:
        """
        Compute validation loss for unsupervised learning.
        This evaluates how well the model captures the normal behavior.
        
        Args:
            val_data: Validation data
            
        Returns:
            Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            x = val_data.x.to(self.device)
            edge_index = val_data.edge_index.to(self.device)
            
            # Forward pass
            z = self.model(x, edge_index)
            
            # Compute loss
            loss = self.criterion(z, edge_index)
            
        return loss.item()
    
    def _save_model(self, path: str):
        """Save model and baseline statistics."""
        model_dir = os.path.join(MODEL_DIR, path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_stats': self.baseline_stats,
            'threshold': self.threshold,
            'history': self.history,
            'config': {
                'in_features': self.in_features,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'k': self.k
            }
        }, model_dir)
        print(f"Model saved to {model_dir}")
    
    def _load_model(self, path: str):
        """Load model and baseline statistics."""
        model_dir = os.path.join(MODEL_DIR, path)
        try:
            checkpoint = torch.load(model_dir, map_location=self.device)
            
            # Load model parameters
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load baseline statistics
            self.baseline_stats = checkpoint['baseline_stats']
            
            # Load threshold
            self.threshold = checkpoint['threshold']
            
            # Load history
            self.history = checkpoint['history']
            
            print(f"Model loaded from {model_dir}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    @classmethod
    def from_pretrained(cls, path: str, device=None):
        """
        Load a pretrained anomaly detector.
        
        Args:
            path: Path to the model
            device: Device to use
            
        Returns:
            Pretrained anomaly detector
        """
        model_dir = os.path.join(MODEL_DIR, path)
        checkpoint = torch.load(model_dir, map_location=device if device else DEVICE)
        
        # Extract config
        config = checkpoint['config']
        
        # Create instance
        detector = cls(
            in_features=config['in_features'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            k=config['k'],
            device=device
        )
        
        # Load model parameters
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        detector.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load baseline statistics
        detector.baseline_stats = checkpoint['baseline_stats']
        
        # Load threshold
        detector.threshold = checkpoint['threshold']
        
        # Load history
        detector.history = checkpoint['history']
        
        return detector

    def save(self, filename: str):
        """Save the detector to a file."""
        self._save_model(filename)
    
    def load(self, filename: str):
        """Load the detector from a file."""
        self._load_model(filename)

###########################################
# Evaluation and Visualization
###########################################

def evaluate_model(model, data, threshold=None, device='cuda', compute_stats=True):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained anomaly detection model
        data: Graph data object
        threshold: Anomaly score threshold (if None, only returns scores)
        device: Device to use (cuda/cpu)
        compute_stats: Whether to compute distributional statistics
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Move to device
    data = data.to(device)
    
    # Get predictions
    with torch.no_grad():
        pred, embeddings = model(data.x, data.edge_index)
        anomaly_scores = model.compute_anomaly_score(pred)
    
    results = {
        'scores': anomaly_scores,
        'embeddings': embeddings,
    }
    
    # Compute distributional statistics if requested
    if compute_stats:
        scores_np = anomaly_scores.cpu().numpy()
        results['stats'] = {
            'min': np.min(scores_np),
            'max': np.max(scores_np),
            'mean': np.mean(scores_np),
            'median': np.median(scores_np),
            'std': np.std(scores_np),
            'q1': np.percentile(scores_np, 25),
            'q3': np.percentile(scores_np, 75),
            'p95': np.percentile(scores_np, 95),
            'p99': np.percentile(scores_np, 99),
        }
    
    # Classify points as normal/anomalous if threshold is provided
    if threshold is not None:
        predicted_labels = (anomaly_scores > threshold).float()
        results['predicted_labels'] = predicted_labels
        
        # Count predicted anomalies
        anomaly_count = torch.sum(predicted_labels).item()
        anomaly_percentage = 100 * anomaly_count / len(predicted_labels)
        results['anomaly_count'] = anomaly_count
        results['anomaly_percentage'] = anomaly_percentage
        
        # If ground truth labels are available (which is not the case for unsupervised learning),
        # we could compute metrics like precision, recall, and F1 score here
        
    return results

def determine_threshold(model, val_data, method='percentile', percentile=95, z_score=3, device='cuda'):
    """
    Determine an anomaly detection threshold based on the validation set.
    
    Args:
        model: Trained anomaly detection model
        val_data: Validation data
        method: Method to use for threshold determination ('percentile', 'zscore', or 'iqr')
        percentile: Percentile to use if method is 'percentile'
        z_score: Z-score threshold if method is 'zscore'
        device: Device to use (cuda/cpu)
        
    Returns:
        Selected threshold value
    """
    # Evaluate model on validation data to get scores
    eval_results = evaluate_model(model, val_data, threshold=None, device=device)
    scores = eval_results['scores'].cpu().numpy()
    
    # Determine threshold based on selected method
    if method == 'percentile':
        # Use a percentile of the score distribution
        threshold = np.percentile(scores, percentile)
        print(f"Using {percentile}th percentile threshold: {threshold:.6f}")
        
    elif method == 'zscore':
        # Use mean + (z_score * std) as threshold
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean + (z_score * std)
        print(f"Using Z-score threshold (μ + {z_score}σ): {threshold:.6f}")
        
    elif method == 'iqr':
        # Use IQR method: Q3 + 1.5*IQR
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        threshold = q3 + (1.5 * iqr)
        print(f"Using IQR threshold (Q3 + 1.5*IQR): {threshold:.6f}")
        
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return threshold

def train_model(model, train_data, val_data=None, epochs=100, lr=0.001, weight_decay=1e-5, 
               device='cuda', log_interval=10, patience=20, visualization_dir=None):
    """
    Train the anomaly detection model.
    
    Args:
        model: Model to train
        train_data: Training data
        val_data: Validation data (optional)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to use (cuda/cpu)
        log_interval: Interval for logging training progress
        patience: Early stopping patience
        visualization_dir: Directory to save visualizations
        
    Returns:
        Dictionary with training history
    """
    # Move model and data to device
    model = model.to(device)
    train_data = train_data.to(device)
    if val_data is not None:
        val_data = val_data.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_data is not None else None,
        'best_val_loss': float('inf') if val_data is not None else None,
    }
    
    # Initialize early stopping variables
    counter = 0
    best_val_loss = float('inf') if val_data is not None else None
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        pred, embeddings = model(train_data.x, train_data.edge_index)
        loss = model.loss(pred)
        loss.backward()
        optimizer.step()
        history['train_loss'].append(loss.item())
        
        # Validation step
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_pred, val_embeddings = model(val_data.x, val_data.edge_index)
                val_loss = model.loss(val_pred)
                history['val_loss'].append(val_loss.item())
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                    history['best_val_loss'] = best_val_loss
                    counter = 0
                else:
                    counter += 1
        
        # Logging
        if epoch % log_interval == 0:
            log_message = f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.6f}"
            if val_data is not None:
                log_message += f", Val Loss: {val_loss.item():.6f}"
            print(log_message)
            
            # Save model visualization at log intervals
            if visualization_dir and epoch % (log_interval * 5) == 0:
                if val_data is not None:
                    # Use validation data for visualization during training
                    with torch.no_grad():
                        val_scores = model.compute_anomaly_score(val_pred)
                    visualize_embeddings(val_embeddings, predicted_anomalies=None,
                                      title=f"Epoch {epoch} Embeddings", save_path=f"embeddings_epoch_{epoch}.png")
                    visualize_anomaly_scores(val_scores, labels=None, 
                                          title=f"Epoch {epoch} Anomaly Scores", save_path=f"scores_epoch_{epoch}.png")
        
        # Early stopping
        if val_data is not None and counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return history

def visualize_roc_curve(results: Dict, save_path: str = None):
    """
    Visualize ROC curve.
    
    Args:
        results: Evaluation results
        save_path: Path to save the figure
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Get ROC curve data
    fpr = results['roc_curve']['fpr']
    tpr = results['roc_curve']['tpr']
    roc_auc = results['roc_auc']
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    # Save if path is provided
    if save_path:
        plt.savefig(os.path.join(VISUALIZATION_DIR, save_path), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_pr_curve(results: Dict, save_path: str = None):
    """
    Visualize precision-recall curve.
    
    Args:
        results: Evaluation results
        save_path: Path to save the figure
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Get PR curve data
    precision = results['pr_curve']['precision']
    recall = results['pr_curve']['recall']
    pr_auc = results['pr_auc']
    
    # Plot PR curve
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    # Save if path is provided
    if save_path:
        plt.savefig(os.path.join(VISUALIZATION_DIR, save_path), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_embeddings(embeddings: torch.Tensor, labels: torch.Tensor = None, 
                       predicted_anomalies: torch.Tensor = None,
                       title: str = "Embeddings Visualization",
                       mode: str = '3d', save_path: str = None):
    """
    Visualize embeddings in 2D or 3D.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels (optional for unsupervised learning)
        predicted_anomalies: Binary tensor of predicted anomalies (1 for anomaly)
        title: Plot title
        mode: Visualization mode ('2d' or '3d')
        save_path: Path to save the figure
    """
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    # Use t-SNE for dimensionality reduction if needed
    if embeddings_np.shape[1] > 3:
        print("Reducing dimensionality with t-SNE...")
        if mode == '3d':
            tsne = TSNE(n_components=3, random_state=42)
        else:
            tsne = TSNE(n_components=2, random_state=42)
        embeddings_np = tsne.fit_transform(embeddings_np)
    
    # Determine point colors based on available information
    if labels is not None and torch.unique(labels).numel() > 1:
        # Supervised case - use actual labels
        labels_np = labels.cpu().numpy()
        colors = np.array(['blue', 'red'])
        point_colors = colors[labels_np.astype(int)]
        normal_mask = (labels_np == 0)
        anomaly_mask = (labels_np == 1)
        anomaly_label = 'True Anomaly'
    elif predicted_anomalies is not None:
        # Unsupervised case with predictions
        pred_np = predicted_anomalies.cpu().numpy()
        colors = np.array(['blue', 'red'])
        point_colors = colors[pred_np.astype(int)]
        normal_mask = (pred_np == 0)
        anomaly_mask = (pred_np == 1)
        anomaly_label = 'Predicted Anomaly'
    else:
        # Completely unsupervised case - no color differentiation
        point_colors = np.array(['blue'] * len(embeddings_np))
        normal_mask = np.ones(len(embeddings_np), dtype=bool)
        anomaly_mask = np.zeros(len(embeddings_np), dtype=bool)
        anomaly_label = 'Anomaly'

    # Create figure based on mode
    if mode == '3d' and embeddings_np.shape[1] >= 3:
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot normal points
        if np.any(normal_mask):
            ax.scatter(embeddings_np[normal_mask, 0], 
                      embeddings_np[normal_mask, 1], 
                      embeddings_np[normal_mask, 2], 
                      c='blue', marker='o', s=30, alpha=0.7, label='Normal')
        
        # Plot anomalies
        if np.any(anomaly_mask):
            ax.scatter(embeddings_np[anomaly_mask, 0], 
                      embeddings_np[anomaly_mask, 1], 
                      embeddings_np[anomaly_mask, 2], 
                      c='red', marker='*', s=60, alpha=0.9, label=anomaly_label)
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.legend()
        
    else:
        # Create 2D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot normal points
        if np.any(normal_mask):
            ax.scatter(embeddings_np[normal_mask, 0], 
                      embeddings_np[normal_mask, 1], 
                      c='blue', marker='o', s=30, alpha=0.7, label='Normal')
        
        # Plot anomalies
        if np.any(anomaly_mask):
            ax.scatter(embeddings_np[anomaly_mask, 0], 
                      embeddings_np[anomaly_mask, 1], 
                      c='red', marker='*', s=60, alpha=0.9, label=anomaly_label)
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        plt.legend()
    
    # Save if path is provided
    if save_path:
        plt.savefig(os.path.join(VISUALIZATION_DIR, save_path), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_embeddings_interactive(embeddings: torch.Tensor, labels: torch.Tensor = None, 
                                   scores: torch.Tensor = None,
                                   predicted_anomalies: torch.Tensor = None,
                                   title: str = "Interactive Embeddings Visualization",
                                   save_path: str = None):
    """
    Create an interactive 3D visualization of embeddings using Plotly.
    
    Args:
        embeddings: Node embeddings
        labels: Node labels (optional for unsupervised learning)
        scores: Anomaly scores (optional)
        predicted_anomalies: Binary tensor of predicted anomalies (1 for anomaly)
        title: Plot title
        save_path: Path to save the figure
    """
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    # Use t-SNE for dimensionality reduction if needed
    if embeddings_np.shape[1] > 3:
        print("Reducing dimensionality with t-SNE...")
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_np = tsne.fit_transform(embeddings_np)
    
    # Determine point classifications
    if labels is not None and torch.unique(labels).numel() > 1:
        # Supervised case - use actual labels
        labels_np = labels.cpu().numpy()
        normal_idx = np.where(labels_np == 0)[0]
        anomaly_idx = np.where(labels_np == 1)[0]
        anomaly_label = 'True Anomaly'
    elif predicted_anomalies is not None:
        # Unsupervised case with predictions
        pred_np = predicted_anomalies.cpu().numpy()
        normal_idx = np.where(pred_np == 0)[0]
        anomaly_idx = np.where(pred_np == 1)[0]
        anomaly_label = 'Predicted Anomaly'
    else:
        # Completely unsupervised case - no differentiation
        normal_idx = np.arange(len(embeddings_np))
        anomaly_idx = np.array([], dtype=int)
        anomaly_label = 'Anomaly'
    
    # Prepare hover text
    hover_text = []
    for i in range(len(embeddings_np)):
        text = f"ID: {i}"
        if labels is not None and torch.unique(labels).numel() > 1:
            text += f"<br>Label: {'Anomaly' if labels.cpu().numpy()[i] else 'Normal'}"
        if predicted_anomalies is not None:
            text += f"<br>Prediction: {'Anomaly' if predicted_anomalies.cpu().numpy()[i] else 'Normal'}"
        if scores is not None:
            text += f"<br>Score: {scores.cpu().numpy()[i]:.4f}"
        hover_text.append(text)
    
    # Create figure
    fig = go.Figure()
    
    # Add normal points
    if len(normal_idx) > 0:
        fig.add_trace(go.Scatter3d(
            x=embeddings_np[normal_idx, 0],
            y=embeddings_np[normal_idx, 1],
            z=embeddings_np[normal_idx, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.7
            ),
            text=[hover_text[i] for i in normal_idx],
            hoverinfo='text',
            name='Normal'
        ))
    
    # Add anomalous points
    if len(anomaly_idx) > 0:
        fig.add_trace(go.Scatter3d(
            x=embeddings_np[anomaly_idx, 0],
            y=embeddings_np[anomaly_idx, 1],
            z=embeddings_np[anomaly_idx, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=0.9
            ),
            text=[hover_text[i] for i in anomaly_idx],
            hoverinfo='text',
            name=anomaly_label
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save if path is provided
    if save_path:
        fig.write_html(os.path.join(VISUALIZATION_DIR, save_path))
    
    # Show figure
    fig.show()

def visualize_anomaly_scores(scores: torch.Tensor, labels: torch.Tensor = None, 
                           threshold: float = None,
                           title: str = "Anomaly Score Distribution",
                           save_path: str = None):
    """
    Visualize the distribution of anomaly scores.
    
    Args:
        scores: Anomaly scores
        labels: Ground truth labels (optional for unsupervised learning)
        threshold: Anomaly threshold
        title: Plot title
        save_path: Path to save the figure
    """
    # Convert to numpy
    scores_np = scores.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with KDE
    if labels is not None and torch.unique(labels).numel() > 1:
        # If we have labels with more than one class (supervised case)
        labels_np = labels.cpu().numpy()
        normal_scores = scores_np[labels_np == 0]
        anomaly_scores = scores_np[labels_np == 1]
        
        # Create histograms for normal and anomalous points
        sns.histplot(normal_scores, color='blue', alpha=0.5, label='Normal', kde=True, ax=ax)
        sns.histplot(anomaly_scores, color='red', alpha=0.5, label='Anomaly', kde=True, ax=ax)
    else:
        # Unsupervised case - just show the score distribution
        sns.histplot(scores_np, color='blue', alpha=0.7, kde=True, ax=ax)
        
        # If we have a threshold, show "predicted" anomalies in a different color
        if threshold is not None:
            potential_anomalies = scores_np[scores_np > threshold]
            normal_samples = scores_np[scores_np <= threshold]
            
            # Clear the previous plot
            ax.clear()
            
            # Plot with two colors based on threshold
            sns.histplot(normal_samples, color='blue', alpha=0.5, label='Normal', kde=True, ax=ax)
            sns.histplot(potential_anomalies, color='red', alpha=0.5, label='Potential Anomalies', kde=True, ax=ax)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', 
                  label=f'Threshold: {threshold:.4f}')
        
        # Add text with anomaly percentage if applicable
        if labels is None:
            anomaly_percentage = 100 * np.sum(scores_np > threshold) / len(scores_np)
            ax.text(0.98, 0.95, f'Anomalies: {anomaly_percentage:.2f}%', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set plot properties
    ax.set_title(title)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    plt.legend()
    
    # Save if path is provided
    if save_path:
        plt.savefig(os.path.join(VISUALIZATION_DIR, save_path), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_training_history(history: Dict, save_path: str = None):
    """
    Visualize training history.
    
    Args:
        history: Training history
        save_path: Path to save the figure
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot training loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(history['loss'], color='blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Loss', color='red')
    
    # Plot validation loss if available
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax2.plot(history['val_loss'], color='red', label='Val Loss')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot threshold if available
    if 'thresholds' in history and len(history['thresholds']) > 0:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.set_ylabel('Threshold', color='green')
        ax3.plot(history['thresholds'], color='green', linestyle='--', label='Threshold')
        ax3.tick_params(axis='y', labelcolor='green')
    
    # Set plot properties
    plt.title('Training History')
    fig.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(os.path.join(VISUALIZATION_DIR, save_path), dpi=300, bbox_inches='tight')
    
    plt.show()

# Import and mount Google Drive if available and not already mounted
def mount_google_drive():
    """
    Attempts to mount Google Drive if running in Colab.
    Returns True if successful, False otherwise.
    """
    try:
        from google.colab import drive
        # Check if already mounted
        import os
        if os.path.exists('/content/drive'):
            print("Google Drive is already mounted.")
            return True
            
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        return True
    except (ImportError, ModuleNotFoundError):
        print("Google Colab environment not detected. Skipping Drive mount.")
        return False

def run_anomaly_detection(train_file, val_file, test_file, 
                      k=10, hidden_dim=64, latent_dim=32, epochs=100,
                      lr=0.001, weight_decay=1e-5, threshold_method='percentile',
                      percentile=95):
    """
    Run the complete anomaly detection pipeline with custom parameters.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        test_file: Path to test data file
        k: Number of nearest neighbors for graph construction
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        threshold_method: Method for threshold determination ('percentile', 'zscore', 'iqr')
        percentile: Percentile for threshold if using 'percentile' method
        
    Returns:
        Dictionary with results and model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading and preparing graph data...")
    train_data = prepare_graph_data(train_file, k=k)
    val_data = prepare_graph_data(val_file, k=k)
    test_data = prepare_graph_data(test_file, k=k)
    
    # Create model
    input_dim = train_data.x.shape[1]
    model = UHGEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.2,
        activation='relu',
        num_layers=2,
        k=k
    )
    
    # Train model
    print("Training model...")
    history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        log_interval=10,
        patience=20,
        visualization_dir=VISUALIZATION_DIR
    )
    
    # Determine threshold
    print(f"Determining threshold using {threshold_method} method...")
    threshold = determine_threshold(
        model=model,
        val_data=val_data,
        method=threshold_method,
        percentile=percentile,
        device=device
    )
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_results = evaluate_model(
        model=model,
        data=test_data,
        threshold=threshold,
        device=device,
        compute_stats=True
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_embeddings(
        embeddings=test_results['embeddings'],
        predicted_anomalies=test_results.get('predicted_labels'),
        title="Test Data Embeddings",
        mode='3d',
        save_path='test_embeddings.png'
    )
    
    visualize_embeddings_interactive(
        embeddings=test_results['embeddings'],
        scores=test_results['scores'],
        predicted_anomalies=test_results.get('predicted_labels'),
        title="Interactive Test Data Embeddings",
        save_path='test_embeddings_interactive.html'
    )
    
    visualize_anomaly_scores(
        scores=test_results['scores'],
        threshold=threshold,
        title="Test Data Anomaly Scores",
        save_path='test_anomaly_scores.png'
    )
    
    # Return results
    results = {
        'model': model,
        'threshold': threshold,
        'test_results': test_results,
        'training_history': history
    }
    
    return results

def setup_colab():
    """Set up the Google Colab environment for UHG anomaly detection."""
    print("Setting up Google Colab environment for UHG anomaly detection...")
    
    # Create required directories
    data_dir = '/content/data'
    output_dir = '/content/outputs'
    visualization_dir = os.path.join(output_dir, 'visualizations')
    model_dir = os.path.join(output_dir, 'models')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Update global variables
    global VISUALIZATION_DIR, MODEL_DIR
    VISUALIZATION_DIR = visualization_dir
    MODEL_DIR = model_dir
    
    # Check if required packages are installed
    try:
        import torch_geometric
        print("PyTorch Geometric is already installed.")
    except ImportError:
        print("Installing PyTorch Geometric...")
        os.system("pip install torch-geometric")
        
    try:
        import torch_scatter
        print("torch_scatter is already installed.")
    except ImportError:
        print("Installing torch_scatter...")
        os.system("pip install torch-scatter")
    
    # Mount Google Drive if needed
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    except ImportError:
        print("Not running in Google Colab, skipping Google Drive mount.")
    
    # Check for required data files
    if not check_required_files():
        print("Required data files not found in local directory.")
        print("Checking Google Drive...")
        if check_gdrive_files():
            print("Files found in Google Drive.")
        else:
            print("Please upload the required data files (train.csv, val.csv, test.csv) to continue.")
            upload_data_to_colab()
    
    print("Colab environment setup complete!")
    return True

def check_gdrive_files():
    """
    Check if required files exist in Google Drive and copy them to the data directory if found.
    
    Returns:
        Boolean indicating if all files were found and copied
    """
    try:
        # Common paths for Google Drive data files
        gdrive_data_paths = [
            '/content/drive/MyDrive/UHG-Library/data',
            '/content/drive/MyDrive/data',
            '/content/drive/My Drive/UHG-Library/data',
            '/content/drive/My Drive/data',
            '/content/drive/Shareddrives/UHG-Library/data'
        ]
        
        data_dir = '/content/data'
        required_files = ['train.csv', 'val.csv', 'test.csv']
        
        # Check each possible path in Google Drive
        for gdrive_path in gdrive_data_paths:
            if not os.path.exists(gdrive_path):
                continue
                
            print(f"Checking {gdrive_path} for data files...")
            found_files = []
            
            for filename in required_files:
                source_path = os.path.join(gdrive_path, filename)
                if os.path.exists(source_path):
                    target_path = os.path.join(data_dir, filename)
                    
                    # Copy file from Google Drive to local data directory
                    import shutil
                    shutil.copy2(source_path, target_path)
                    print(f"Copied {filename} from Google Drive to {target_path}")
                    found_files.append(filename)
            
            # If we found all required files in this directory, we're done
            if len(found_files) == len(required_files):
                print(f"All required files found in {gdrive_path} and copied to local data directory.")
                return True
                
        # If we get here, we didn't find all required files
        if found_files:
            print(f"Found some files in Google Drive: {', '.join(found_files)}")
            print(f"Still missing: {', '.join(set(required_files) - set(found_files))}")
        else:
            print("No required files found in Google Drive.")
            
        return False
        
    except Exception as e:
        print(f"Error checking Google Drive: {str(e)}")
        return False

def main():
    """Main function to run the anomaly detection pipeline."""
    print("Starting UHG-based Anomaly Detection Pipeline")
    
    # Setup Colab environment
    is_setup = setup_colab()
    if not is_setup:
        print("Failed to set up environment. Exiting.")
        return
    
    # Paths for data and outputs
    data_dir = '/content/data'
    train_file = os.path.join(data_dir, 'train.csv')
    val_file = os.path.join(data_dir, 'val.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    # Check if required files exist
    if not check_required_files([train_file, val_file, test_file]):
        print("Required data files are missing. Please upload them and run again.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    train_data = prepare_graph_data(train_file, k=10)
    val_data = prepare_graph_data(val_file, k=10)
    test_data = prepare_graph_data(test_file, k=10)
    
    # Print dataset statistics
    print(f"Train set: {train_data.num_nodes} nodes")
    print(f"Validation set: {val_data.num_nodes} nodes")
    print(f"Test set: {test_data.num_nodes} nodes")
    
    # Create model
    input_dim = train_data.x.shape[1]
    hidden_dim = 64
    latent_dim = 32
    
    model = UHGEncoder(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     latent_dim=latent_dim,
                     activation='relu')
    
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("Training model...")
    history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=100,
        lr=0.001,
        weight_decay=1e-5,
        device=device,
        log_interval=10,
        patience=20,
        visualization_dir=VISUALIZATION_DIR
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
    plt.show()
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, 'uhg_anomaly_detection_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Determine threshold using validation data
    print("Determining anomaly threshold using validation data...")
    threshold = determine_threshold(
        model=model,
        val_data=val_data,
        method='percentile',
        percentile=95,
        device=device
    )
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_results = evaluate_model(
        model=model,
        data=test_data,
        threshold=threshold,
        device=device,
        compute_stats=True
    )
    
    # Print test results
    print("Test Results:")
    print(f"  Threshold: {threshold:.6f}")
    if 'anomaly_count' in test_results:
        print(f"  Detected anomalies: {test_results['anomaly_count']} ({test_results['anomaly_percentage']:.2f}%)")
    
    print("  Score statistics:")
    for stat, value in test_results['stats'].items():
        print(f"    {stat}: {value:.6f}")
    
    # Visualize embeddings
    print("Generating visualizations...")
    visualize_embeddings(
        embeddings=test_results['embeddings'],
        predicted_anomalies=test_results.get('predicted_labels'),
        title="Test Data Embeddings",
        mode='3d',
        save_path='test_embeddings.png'
    )
    
    # Visualize embeddings (interactive)
    visualize_embeddings_interactive(
        embeddings=test_results['embeddings'],
        scores=test_results['scores'],
        predicted_anomalies=test_results.get('predicted_labels'),
        title="Interactive Test Data Embeddings",
        save_path='test_embeddings_interactive.html'
    )
    
    # Visualize anomaly scores
    visualize_anomaly_scores(
        scores=test_results['scores'],
        threshold=threshold,
        title="Test Data Anomaly Scores",
        save_path='test_anomaly_scores.png'
    )
    
    print("UHG-based Anomaly Detection Pipeline completed successfully")

def prepare_graph_data(file_path, k=10):
    """
    Load data from CSV file and prepare graph data object with KNN graph.
    
    Args:
        file_path: Path to CSV data file
        k: Number of nearest neighbors for KNN graph
        
    Returns:
        torch_geometric.data.Data object with features and edge_index
    """
    # Load data
    print(f"Loading data from {file_path}")
    try:
        # First try with default settings
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV with default settings: {str(e)}")
        # Try with different settings
        try:
            print("Trying to read CSV with different settings...")
            df = pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e2:
            print(f"Failed to read CSV: {str(e2)}")
            # One last attempt with more flexible options
            try:
                print("Trying with more flexible settings...")
                df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1, header=None)
            except Exception as e3:
                raise ValueError(f"Could not read CSV file {file_path}. Please check the format. Error: {str(e3)}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty. Please check the file at {file_path}")
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Check if any column contains non-numeric data (except for IDs or labels)
    non_numeric_cols = []
    for col in df.columns:
        # Skip obvious ID or label columns
        if col.lower() in ['id', 'label', 'target', 'class', 'anomaly']:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns: {non_numeric_cols}")
        print("These columns will be dropped for graph construction")
        df = df.drop(columns=non_numeric_cols)
    
    # Extract features (assuming any ID or label column is at beginning or end)
    # Try to detect if there's any obvious ID or label column
    possible_id_or_label_cols = [col for col in df.columns if col.lower() in ['id', 'label', 'target', 'class', 'anomaly']]
    
    if possible_id_or_label_cols:
        print(f"Detected potential ID or label columns: {possible_id_or_label_cols}")
        print(f"These columns will be excluded from features")
        feature_cols = [col for col in df.columns if col not in possible_id_or_label_cols]
        features = df[feature_cols].values
    else:
        # If no obvious ID or label columns found, use all columns
        features = df.values
    
    # Check for NaN values
    if np.isnan(features).any():
        print("Warning: NaN values found in features. Replacing with zeros.")
        features = np.nan_to_num(features, nan=0.0)
    
    # Convert to tensor
    features = torch.tensor(features, dtype=torch.float)
    
    # Calculate k-nearest neighbors (KNN) graph
    print(f"Calculating KNN graph with k={k}")
    # Convert to numpy for sklearn
    features_np = features.numpy()
    
    # Check for potential issues
    if features_np.shape[0] <= k:
        print(f"Warning: Number of samples ({features_np.shape[0]}) is less than or equal to k ({k})")
        print(f"Reducing k to {features_np.shape[0] - 1}")
        k = features_np.shape[0] - 1
    
    # Use NearestNeighbors to compute the KNN graph
    try:
        knn = NearestNeighbors(n_neighbors=k+1)  # +1 to include self
        knn.fit(features_np)
        
        # Get distances and indices of nearest neighbors
        distances, indices = knn.kneighbors(features_np)
        
        # Create edge indices
        rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1]-1)
        cols = indices[:, 1:].flatten()  # Skip the first neighbor (self)
        
        # Create COO format edge index
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        
        # Create Data object
        data = Data(x=features, edge_index=edge_index)
        
        print(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
        print(f"Feature dimensions: {data.x.shape}")
        return data
        
    except Exception as e:
        print(f"Error creating KNN graph: {str(e)}")
        print("Falling back to a simple fully connected graph")
        
        # Create a fully connected graph as fallback
        n = features.shape[0]
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:  # Exclude self-loops
                    rows.append(i)
                    cols.append(j)
        
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        data = Data(x=features, edge_index=edge_index)
        
        print(f"Created fallback fully connected graph with {data.num_nodes} nodes and {data.num_edges} edges")
        print(f"Feature dimensions: {data.x.shape}")
        return data

def upload_data_to_colab():
    """
    Helper function to upload data files to Google Colab.
    
    Returns:
        List of uploaded file paths
    """
    try:
        from google.colab import files
        print("Please upload your CSV data files (train.csv, val.csv, test.csv):")
        uploaded = files.upload()
        
        data_dir = '/content/data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Move uploaded files to data directory
        uploaded_files = []
        for filename in uploaded.keys():
            source_path = filename
            target_path = os.path.join(data_dir, filename)
            
            # Read the uploaded file and write to target path
            with open(source_path, 'rb') as f_src:
                with open(target_path, 'wb') as f_tgt:
                    f_tgt.write(f_src.read())
            
            print(f"Moved {filename} to {target_path}")
            uploaded_files.append(target_path)
            
            # Remove source file if it exists in the current directory
            if os.path.exists(source_path) and source_path != target_path:
                os.remove(source_path)
        
        return uploaded_files
    
    except ImportError:
        print("Not running in Google Colab, skipping file upload.")
        return []

def check_required_files(required_files=None):
    """
    Check if required data files exist, and prompt for upload if not.
    
    Args:
        required_files: List of required file paths
        
    Returns:
        Boolean indicating if all files are available
    """
    if required_files is None:
        data_dir = '/content/data'
        required_files = [
            os.path.join(data_dir, 'train.csv'),
            os.path.join(data_dir, 'val.csv'),
            os.path.join(data_dir, 'test.csv')
        ]
    
    # Check if files exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing required files: {', '.join([os.path.basename(f) for f in missing_files])}")
        
        try:
            from google.colab import files
            print("Please upload the missing files:")
            uploaded_files = upload_data_to_colab()
            
            # Check again if files exist after upload
            still_missing = [f for f in required_files if not os.path.exists(f)]
            if still_missing:
                print(f"Still missing required files: {', '.join([os.path.basename(f) for f in still_missing])}")
                return False
            else:
                print("All required files are now available.")
                return True
                
        except ImportError:
            print("Not running in Google Colab, skipping file upload.")
            return False
    
    print("All required files are available.")
    return True

if __name__ == "__main__":
    main()

def colab_wrapper(k=10, custom_paths=None):
    """
    Convenience function for running the UHG anomaly detection from Google Colab.
    
    Args:
        k: Number of nearest neighbors for the KNN graph
        custom_paths: Dictionary with custom paths for data files
            Example: {'train': '/path/to/train.csv', 'val': '/path/to/val.csv', 'test': '/path/to/test.csv'}
            
    Returns:
        Dictionary with results
    """
    print("Starting UHG-based Anomaly Detection from Google Colab")
    
    # Setup environment
    is_setup = setup_colab()
    if not is_setup:
        print("Environment setup failed. Please check errors above.")
        return None
    
    # Determine data paths
    data_dir = '/content/data'
    if custom_paths is not None:
        train_file = custom_paths.get('train', os.path.join(data_dir, 'train.csv'))
        val_file = custom_paths.get('val', os.path.join(data_dir, 'val.csv'))
        test_file = custom_paths.get('test', os.path.join(data_dir, 'test.csv'))
    else:
        train_file = os.path.join(data_dir, 'train.csv')
        val_file = os.path.join(data_dir, 'val.csv')
        test_file = os.path.join(data_dir, 'test.csv')
    
    # Check for required files
    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}")
        return None
    if not os.path.exists(val_file):
        print(f"Validation file not found: {val_file}")
        return None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    try:
        train_data = prepare_graph_data(train_file, k=k)
        val_data = prepare_graph_data(val_file, k=k)
    except Exception as e:
        print(f"Error preparing graph data: {str(e)}")
        return None
    
    # Create and train model
    try:
        # Create model
        input_dim = train_data.x.shape[1]
        hidden_dim = 64
        latent_dim = 32
        
        model = UHGEncoder(input_dim=input_dim, 
                         hidden_dim=hidden_dim, 
                         latent_dim=latent_dim,
                         activation='relu')
        
        print(f"Model architecture:\n{model}")
        
        # Train model
        print("Training model...")
        history = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            epochs=100,
            lr=0.001,
            weight_decay=1e-5,
            device=device,
            log_interval=10,
            patience=20,
            visualization_dir=VISUALIZATION_DIR
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
        plt.show()
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, 'uhg_anomaly_detection_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Determine threshold using validation data
        print("Determining anomaly threshold using validation data...")
        threshold = determine_threshold(
            model=model,
            val_data=val_data,
            method='percentile',
            percentile=95,
            device=device
        )
        print(f"Anomaly threshold: {threshold:.4f}")
        
        # Test on test data if available
        results = {}
        if os.path.exists(test_file):
            print("Evaluating on test data...")
            test_data = prepare_graph_data(test_file, k=k)
            predictions, scores = predict_anomalies(
                model=model,
                data=test_data,
                threshold=threshold,
                device=device
            )
            
            # Return results
            results['model'] = model
            results['history'] = history
            results['threshold'] = threshold
            results['test_predictions'] = predictions
            results['test_scores'] = scores
            
            # Visualize results
            print("Visualizing results...")
            visualize_embeddings(
                model=model,
                data=test_data,
                predictions=predictions,
                device=device,
                save_path=os.path.join(VISUALIZATION_DIR, 'embeddings.png')
            )
            
            print("Anomaly detection complete. Check visualization results.")
            
        else:
            print("Test file not found. Skipping evaluation.")
            results['model'] = model
            results['history'] = history
            results['threshold'] = threshold
            
        return results
        
    except Exception as e:
        print(f"Error during model training or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# For running in Google Colab, copy and run this code:
'''
# Import the module
import sys
sys.path.append('/content')
from uhg_anomaly_detection import colab_wrapper

# Run the model with custom paths if needed
results = colab_wrapper(
    k=10,  # Number of nearest neighbors
    custom_paths={
        'train': '/content/drive/MyDrive/data/train.csv',  # Change to your path
        'val': '/content/drive/MyDrive/data/val.csv',      # Change to your path
        'test': '/content/drive/MyDrive/data/test.csv'     # Change to your path
    }
)
'''