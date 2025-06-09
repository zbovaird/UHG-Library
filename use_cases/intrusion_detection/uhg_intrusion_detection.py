"""
Intrusion Detection using Universal Hyperbolic Geometry (UHG).
This implementation leverages UHG's pure projective operations for better hierarchical learning.
Designed to run in Google Colab with GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from tqdm import tqdm
import uhg
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import os
from typing import Tuple, Optional, Dict
from uhg.projective import ProjectiveUHG
import time
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# Mount Google Drive
print("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Device configuration - prioritize GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU - Warning: Training may be slow")

# File paths
FILE_PATH = '/content/drive/MyDrive/CIC_data.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/uhg_ids_model.pth'
RESULTS_PATH = '/content/drive/MyDrive/uhg_ids_results'

# Create results directory
os.makedirs(RESULTS_PATH, exist_ok=True)

###########################################
# UHG Core Operations
###########################################

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Hyperbolic inner product between points in projective coordinates.
    
    Efficiently computes <a,b> = -a_spatial·b_spatial + a_time·b_time
    with proper broadcasting across arbitrary batch dimensions.
    
    Args:
        a: First point(s) with shape [..., dim]
        b: Second point(s) with shape [..., dim]
        
    Returns:
        Inner product with shape [..., 1]
    """
    # Extract spatial and time components with proper broadcasting
    a_spatial, a_time = a[..., :-1], a[..., -1:]
    b_spatial, b_time = b[..., :-1], b[..., -1:]
    
    # Compute the Minkowski inner product using efficient batch operations
    # -<a_spatial, b_spatial> + a_time * b_time
    # Use efficient batch dot product
    spatial_product = torch.sum(a_spatial * b_spatial, dim=-1, keepdim=True)
    time_product = a_time * b_time
    
    return -spatial_product + time_product

def load_and_preprocess_data(file_path: str = FILE_PATH) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Load and preprocess the CIC dataset from Google Drive."""
    print(f"\nLoading data from: {file_path}")
    
    # Load data
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = data.columns.str.strip()
    data['Label'] = data['Label'].str.strip()
    
    # Print initial statistics
    unique_labels = data['Label'].unique()
    print(f"\nUnique labels in the dataset: {unique_labels}")
    label_counts = data['Label'].value_counts()
    print("\nLabel distribution in the dataset:")
    print(label_counts)
    
    # Sample data (10%)
    data_sampled = data.sample(frac=0.10, random_state=42)
    
    # Convert to numeric and handle missing values
    data_numeric = data_sampled.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with column means
    data_filled = data_numeric.fillna(data_numeric.mean())
    data_filled = data_filled.replace([np.inf, -np.inf], np.nan)
    data_filled = data_filled.fillna(data_filled.max())
    
    # Handle any remaining NaNs
    if data_filled.isnull().values.any():
        data_filled = data_filled.fillna(0)
    
    # Extract labels and features
    labels = data_sampled['Label']
    features = data_filled.drop(columns=['Label'])
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Convert to tensors
    node_features = torch.tensor(features_scaled, dtype=torch.float32)
    
    # Convert labels to numeric
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numeric = labels.map(label_mapping).values
    labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)
    
    print("\nPreprocessing complete.")
    print(f"Feature shape: {node_features.shape}")
    print(f"Number of unique labels: {len(unique_labels)}")
    
    return node_features, labels_tensor, label_mapping

def create_graph_data(node_features: torch.Tensor, 
                     labels: torch.Tensor, 
                     k: int = 2, 
                     batch_size: int = 4096,
                     use_uhg_distance: bool = True) -> Data:
    """
    Create a graph from node features using k-nearest neighbors with optimized vectorization.
    
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
    
    # Initialize edge index storage with pre-allocated capacity
    edge_indices = []
    
    # Move data to device once
    node_features = node_features.to(device)
    
    # Pre-compute UHG space conversion if needed
    target = to_uhg_space(node_features) if use_uhg_distance else node_features
    
    # Process in batches to avoid OOM
    for i in tqdm(range(0, num_nodes, batch_size), desc="Computing KNN graph"):
        batch_end = min(i + batch_size, num_nodes)
        batch_features = node_features[i:batch_end]
        
        # Convert to UHG space if requested (only for source batch)
        if use_uhg_distance:
            source = to_uhg_space(batch_features)
            
            # Compute pairwise quadrances efficiently (vectorized)
            distances = torch.zeros((batch_end - i, num_nodes), device=device)
            
            # Compute in sub-batches if needed for very large datasets
            sub_batch_size = 128  # Smaller batches for pairwise calculations
            for j in range(0, batch_end - i, sub_batch_size):
                j_end = min(j + sub_batch_size, batch_end - i)
                src_batch = source[j:j_end]
                
                # Compute quadrance with all targets at once using broadcasting
                # This avoids the explicit loop over each source point
                aa = uhg_inner_product(src_batch.unsqueeze(1), src_batch.unsqueeze(1))  # [sub_batch, 1, 1]
                bb = uhg_inner_product(target.unsqueeze(0), target.unsqueeze(0))  # [1, num_nodes, 1]
                ab = uhg_inner_product(src_batch.unsqueeze(1), target.unsqueeze(0))  # [sub_batch, num_nodes, 1]
                
                # Compute quadrance using formula
                numerator = ab * ab - aa * bb  # Broadcasting handles the dimensions
                denominator = aa * bb
                
                # Ensure numerical stability
                safe_denominator = denominator.abs().clamp_min(1e-9)
                quad = numerator / (safe_denominator * denominator.sign())
                
                distances[j:j_end] = quad.squeeze(-1)
            
        else:
            # Use Euclidean distance with efficient vectorization
            batch_expanded = batch_features.unsqueeze(1)  # [batch, 1, dim]
            nodes_expanded = node_features.unsqueeze(0)  # [1, num_nodes, dim]
            
            # Compute pairwise Euclidean distances efficiently
            distances = torch.sum((batch_expanded - nodes_expanded) ** 2, dim=2)  # [batch, num_nodes]
        
        # For each node, find k nearest neighbors (use efficient topk operation)
        _, indices = torch.topk(distances, k=min(k+1, num_nodes), dim=1, largest=False)
        
        # Convert to edge index format efficiently
        source_indices = torch.arange(i, batch_end, device=device).view(-1, 1).repeat(1, indices.size(1))
        valid_mask = (indices != source_indices) & (indices < num_nodes)  # Filter self-loops and out-of-bounds
        
        sources = source_indices[valid_mask].view(-1)
        targets = indices[valid_mask].view(-1)
        
        new_edges = torch.stack([sources, targets], dim=0)
        edge_indices.append(new_edges)
    
    # Concatenate all edge indices
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Move back to CPU for PyG compatibility if needed
    edge_index = edge_index.cpu()
    
    # Create the graph data object
    graph_data = Data(x=node_features.cpu(), edge_index=edge_index, y=labels)
    
    print(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    print(f"Graph creation took {time.time() - start_time:.2f} seconds")
    
    return graph_data

# UHG Operations
def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute quadrance between two points in UHG.
    Quadrance corresponds to squared distance in Euclidean geometry.
    
    This implementation is fully vectorized to handle arbitrary batch dimensions.
    """
    # Get the inner products - ensure proper broadcasting across batch dimensions
    aa = uhg_inner_product(a, a)
    bb = uhg_inner_product(b, b)
    ab = uhg_inner_product(a, b)
    
    # Compute quadrance using the formula from UHG
    numerator = ab * ab - aa * bb
    denominator = aa * bb
    
    # Ensure numerical stability using a more efficient approach
    safe_denominator = denominator.abs().clamp_min(eps)
    quad = numerator / (safe_denominator * denominator.sign())
    
    return quad.squeeze(-1)

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute spread between two lines in UHG.
    Spread is the dual of quadrance and measures the squared angle.
    
    This implementation is fully vectorized to handle arbitrary batch dimensions.
    """
    # Get the inner products with optimized broadcasting
    LL = uhg_inner_product(L, L)
    MM = uhg_inner_product(M, M)
    LM = uhg_inner_product(L, M)
    
    # Compute spread
    numerator = LM * LM - LL * MM
    denominator = LL * MM
    
    # Ensure numerical stability using a more efficient approach
    safe_denominator = denominator.abs().clamp_min(eps)
    spread = numerator / (safe_denominator * denominator.sign())
    
    return spread.squeeze(-1)

def uhg_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, 
                   p3: torch.Tensor, p4: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute the cross-ratio of four points in UHG with vectorized operations.
    
    The cross-ratio is a projective invariant and fundamental in UHG.
    
    Args:
        p1, p2, p3, p4: Points in UHG space
        eps: Small value for numerical stability
        
    Returns:
        Cross-ratio value(s)
    """
    # Compute the quadrances in a vectorized way
    # This is more efficient than computing each quadrance separately
    
    # Compute all inner products at once
    p1p1 = uhg_inner_product(p1, p1)
    p2p2 = uhg_inner_product(p2, p2)
    p3p3 = uhg_inner_product(p3, p3)
    p4p4 = uhg_inner_product(p4, p4)
    
    p1p2 = uhg_inner_product(p1, p2)
    p3p4 = uhg_inner_product(p3, p4)
    p1p3 = uhg_inner_product(p1, p3)
    p2p4 = uhg_inner_product(p2, p4)
    
    # Compute quadrances using the formula
    q_12_num = p1p2 * p1p2 - p1p1 * p2p2
    q_12_den = p1p1 * p2p2
    q_12 = q_12_num / (q_12_den.abs().clamp_min(eps) * q_12_den.sign())
    
    q_34_num = p3p4 * p3p4 - p3p3 * p4p4
    q_34_den = p3p3 * p4p4
    q_34 = q_34_num / (q_34_den.abs().clamp_min(eps) * q_34_den.sign())
    
    q_13_num = p1p3 * p1p3 - p1p1 * p3p3
    q_13_den = p1p1 * p3p3
    q_13 = q_13_num / (q_13_den.abs().clamp_min(eps) * q_13_den.sign())
    
    q_24_num = p2p4 * p2p4 - p2p2 * p4p4
    q_24_den = p2p2 * p4p4
    q_24 = q_24_num / (q_24_den.abs().clamp_min(eps) * q_24_den.sign())
    
    # Calculate cross-ratio
    numerator = q_12 * q_34
    denominator = q_13 * q_24
    
    # Ensure numerical stability
    safe_denominator = denominator.abs().clamp_min(eps)
    cross_ratio = numerator / (safe_denominator * denominator.sign())
    
    return cross_ratio.squeeze(-1)

class UHGMessagePassing(MessagePassing):
    """
    Optimized Message passing layer for UHG-based graph neural networks.
    """
    
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
        
        # Cache for UHG space conversion
        self.register_buffer('origin', None)
    
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
        Forward pass of the UHG message passing layer with optimized processing.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Add self-loops to include the node's own features (precompute once)
        if self.training or edge_index._indices().size(1) == 0:  # Only recompute when necessary
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform the node features with linear layer (do this only once)
        transformed_x = self.linear(x)
        
        # Execute the message passing scheme
        return self.propagate(edge_index, x=x, transformed_x=transformed_x)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """
        Message function for edge (i,j).
        Creates messages from source node j to target node i with optimized transformation.
        
        Args:
            x_j: Source node features
            
        Returns:
            Messages from source nodes
        """
        # Transform message in one batch operation
        return self.message_linear(x_j)
    
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Aggregation function that performs a weighted sum of messages.
        Optimized to reuse computation and minimize memory usage.
        
        Args:
            inputs: Messages to aggregate
            index: Target node indices
            
        Returns:
            Aggregated messages
        """
        # Convert to UHG space for weighted aggregation (efficiently)
        uhg_inputs = to_uhg_space(inputs)
        
        # Initialize or reuse origin point
        if self.origin is None or self.origin.device != uhg_inputs.device:
            self.origin = torch.zeros(1, uhg_inputs.shape[-1], device=uhg_inputs.device)
            self.origin[..., -1] = 1.0  # Set homogeneous coordinate to 1
        
        # Compute importance weights efficiently using vectorized operations
        # Reuse the uhg_quadrance function we optimized
        quad = uhg_quadrance(uhg_inputs, self.origin.expand(uhg_inputs.size(0), -1))
        
        # Apply softmax for normalized weights that preserve the UHG structure
        # This gives more emphasis to the UHG geometry while maintaining stability
        weights = torch.exp(-quad)
        
        # Apply weights and aggregate in a single operation
        weighted_inputs = inputs * weights.unsqueeze(-1)
        
        # Use scatter_add for efficient aggregation
        out = torch.zeros(torch.max(index)+1, inputs.size(-1), device=inputs.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand(-1, inputs.size(-1)), weighted_inputs)
        
        return out
    
    def update(self, aggr_out: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """
        Update function for node features after aggregation.
        Combines the node's transformed features with aggregated messages.
        
        Args:
            aggr_out: Aggregated messages
            transformed_x: Transformed node features
            
        Returns:
            Updated node features
        """
        # Combine node's transformed features with aggregated messages
        # Skip connection helps preserve structural information
        return transformed_x + aggr_out

class UHGGraphSAGELayer(nn.Module):
    """UHG-enhanced GraphSAGE layer using projective geometry."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights for feature dimensions (excluding homogeneous coordinate)
        self.weight_neigh = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_self = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_neigh)
        nn.init.xavier_uniform_(self.weight_self)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # Split features and homogeneous coordinate
        features = x[:, :-1]  # All but last column
        homogeneous = x[:, -1:]  # Last column
        
        # Neighbor aggregation using UHG distances
        neigh_sum = torch.zeros_like(features)
        neigh_weights = torch.zeros(features.size(0), device=features.device)
        
        for i in range(len(row)):
            src, dst = row[i], col[i]
            # Compute UHG-based weight using full UHG coordinates
            weight = torch.exp(-uhg_quadrance(x[src], x[dst]))
            neigh_sum[src] += weight * features[dst]
            neigh_weights[src] += weight
        
        # Normalize by weights
        neigh_weights = torch.clamp(neigh_weights.unsqueeze(1), min=1e-6)
        neigh_features = neigh_sum / neigh_weights

        # Apply transformations to features only
        neigh_transformed = torch.matmul(neigh_features, self.weight_neigh.t())
        self_transformed = torch.matmul(features, self.weight_self.t())

        # Combine features and add homogeneous coordinate back
        combined = neigh_transformed + self_transformed
        output = torch.cat([combined, homogeneous], dim=1)
        
        return F.relu(output)

class UHGGraphSAGE(nn.Module):
    """UHG-enhanced GraphSAGE model for intrusion detection."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input features are one less than x.shape[1] due to homogeneous coordinate
        actual_in_channels = in_channels - 1
        
        # Input layer
        self.layers.append(UHGGraphSAGELayer(actual_in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(UHGGraphSAGELayer(hidden_channels, hidden_channels))
            
        # Output layer
        self.layers.append(UHGGraphSAGELayer(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Process through layers
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.dropout(x)
        
        # Final layer
        x = self.layers[-1](x, edge_index)
        
        # Use only feature part for classification, not homogeneous coordinate
        return x[:, :-1]

def train_epoch(model: nn.Module, graph_data: Data, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, batch_size: int = 16, accumulation_steps: int = 4) -> float:
    """Train for one epoch using gradient accumulation."""
    model.train()
    total_loss = 0
    train_loader = DataLoader(
        range(graph_data.train_mask.sum()),
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch = batch.to(device)
        
        # Get batch data
        x = graph_data.x[graph_data.train_mask][batch]
        y = graph_data.y[graph_data.train_mask][batch]
        
        # Create subgraph
        batch_node_ids = graph_data.train_mask.nonzero(as_tuple=True)[0][batch]
        edge_mask = torch.isin(graph_data.edge_index[0], batch_node_ids) & \
                   torch.isin(graph_data.edge_index[1], batch_node_ids)
        batch_edge_index = graph_data.edge_index[:, edge_mask]
        
        # Relabel nodes
        node_idx = torch.unique(batch_edge_index)
        idx_map = {int(idx): i for i, idx in enumerate(node_idx)}
        mapped_edge_index = torch.tensor(
            [[idx_map[int(i)] for i in batch_edge_index[0]],
             [idx_map[int(i)] for i in batch_edge_index[1]]],
            dtype=torch.long,
            device=device
        )
        
        # Forward pass
        out = model(x, mapped_edge_index)
        loss = criterion(out, y) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model: nn.Module, graph_data: Data, mask: torch.Tensor) -> float:
    """Evaluate model on given mask."""
    model.eval()
    
    # Get masked data
    node_indices = mask.nonzero(as_tuple=True)[0]
    sub_x = graph_data.x[node_indices]
    sub_y = graph_data.y[node_indices]
    
    # Create subgraph
    edge_mask = torch.isin(graph_data.edge_index[0], node_indices) & \
                torch.isin(graph_data.edge_index[1], node_indices)
    sub_edge_index = graph_data.edge_index[:, edge_mask]
    
    # Relabel nodes
    node_idx = torch.unique(sub_edge_index)
    idx_map = {int(idx): i for i, idx in enumerate(node_idx)}
    mapped_edge_index = torch.tensor(
        [[idx_map[int(i)] for i in sub_edge_index[0]],
         [idx_map[int(i)] for i in sub_edge_index[1]]],
        dtype=torch.long,
        device=device
    )
    
    # Forward pass
    out = model(sub_x, mapped_edge_index)
    pred = out.argmax(dim=1)
    
    # Calculate accuracy
    correct = (pred == sub_y).sum().item()
    accuracy = correct / len(node_indices)
    
    return accuracy

def to_uhg_space(x: torch.Tensor) -> torch.Tensor:
    """
    Convert feature vectors to UHG space with homogeneous coordinates.
    Efficiently handles arbitrary batch dimensions and prevents redundant computations.
    
    Args:
        x: Input tensor of shape [..., dim]
        
    Returns:
        Tensor in UHG space with homogeneous coordinate appended, shape [..., dim+1]
    """
    # Check if the last dimension is already the appropriate size for UHG space
    # This prevents redundant conversions
    if x.size(-1) > 0 and x[..., -1].eq(1.0).all():
        # If the last dimension is already 1.0 for all elements, it's likely already in UHG space
        return x
    
    # Create homogeneous coordinate efficiently by reshaping for any batch dimension
    ones_shape = list(x.shape[:-1]) + [1]
    homogeneous = torch.ones(ones_shape, dtype=x.dtype, device=x.device)
    
    # Concatenate along the last dimension
    return torch.cat([x, homogeneous], dim=-1)

def main():
    """Main training function."""
    # Load and preprocess data
    node_features, labels, label_mapping = load_and_preprocess_data()
    graph_data = create_graph_data(node_features, labels)
    
    # Model parameters
    in_channels = graph_data.x.size(1)
    hidden_channels = 128
    out_channels = len(label_mapping)
    num_layers = 2
    
    # Initialize model and optimizer
    model = UHGGraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 400
    best_val_acc = 0
    patience = 20
    counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        try:
            # Train
            loss = train_epoch(model, graph_data, optimizer, criterion)
            
            # Evaluate
            val_acc = evaluate(model, graph_data, graph_data.val_mask)
            test_acc = evaluate(model, graph_data, graph_data.test_mask)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"\nNew best model saved! Validation accuracy: {val_acc:.4f}")
            else:
                counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                      f'Test Accuracy: {test_acc:.4f}')
            
            # Early stopping
            if counter >= patience:
                print("Early stopping triggered!")
                break
                
        except RuntimeError as e:
            print(f"\nError in epoch {epoch}: {str(e)}")
            break
    
    # Final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        final_test_acc = evaluate(model, graph_data, graph_data.test_mask)
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main()