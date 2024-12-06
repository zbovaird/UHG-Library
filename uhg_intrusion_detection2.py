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
from typing import Tuple, Optional
from uhg.projective import ProjectiveUHG

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

def create_graph_data(node_features: torch.Tensor, labels: torch.Tensor, k: int = 2) -> Data:
    """Create graph structure using UHG principles."""
    print("\nCreating graph structure...")
    
    # Convert features to numpy for sklearn
    features_np = node_features.cpu().numpy()
    
    # Create k-nearest neighbors graph using sklearn (memory efficient)
    print("Computing KNN graph...")
    knn_graph = kneighbors_graph(
        features_np, 
        k, 
        mode='connectivity', 
        include_self=False
    )
    
    # Convert to COO format
    knn_graph_coo = coo_matrix(knn_graph)
    
    # Create edge index
    edge_index = torch.from_numpy(
        np.array([knn_graph_coo.row, knn_graph_coo.col])
    ).long().to(device)
    
    print(f"Edge index shape: {edge_index.shape}")
    
    # Add homogeneous coordinate to features
    node_features_uhg = torch.cat([
        node_features, 
        torch.ones(node_features.size(0), 1, device=node_features.device)
    ], dim=1)
    
    print(f"Feature shape with homogeneous coordinate: {node_features_uhg.shape}")
    
    # Create train/val/test split
    total_samples = len(node_features_uhg)
    indices = torch.randperm(total_samples)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    print(f"\nTrain size: {train_mask.sum()}, Val size: {val_mask.sum()}, Test size: {test_mask.sum()}")
    
    return Data(
        x=node_features_uhg,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    ).to(device)

# UHG Operations
def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between two points."""
    dot_product = torch.sum(a * b)  # For vectors
    return 1 - (dot_product ** 2) / (
        (torch.sum(a ** 2) - a[-1] ** 2 + eps) * 
        (torch.sum(b ** 2) - b[-1] ** 2 + eps)
    )

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between two lines."""
    dot_product = torch.sum(L * M)  # For vectors
    return 1 - (dot_product ** 2) / (
        (torch.sum(L ** 2) - L[-1] ** 2 + eps) * 
        (torch.sum(M ** 2) - M[-1] ** 2 + eps)
    )

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
    
    # Learning rate setup
    initial_lr = 0.01
    min_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # More sophisticated learning rate scheduling
    warmup_epochs = 5
    num_epochs = 400
    batch_size = 16
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        epochs=num_epochs,
        steps_per_epoch=len(range(graph_data.train_mask.sum())) // batch_size,
        pct_start=warmup_epochs/num_epochs,  # Warmup phase
        final_div_factor=initial_lr/min_lr,  # Minimum LR
        div_factor=10.0,  # Initial LR division factor
        three_phase=True  # Use three-phase learning
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop parameters
    best_val_acc = 0
    patience = 20
    counter = 0
    
    # Learning rate monitoring setup
    lr_history = []
    val_history = []
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        try:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            loss = train_epoch(model, graph_data, optimizer, criterion)
            
            # Step the scheduler
            scheduler.step()
            
            # Evaluate
            val_acc = evaluate(model, graph_data, graph_data.val_mask)
            test_acc = evaluate(model, graph_data, graph_data.test_mask)
            
            # Store learning rate and validation accuracy
            lr_history.append(current_lr)
            val_history.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"\nNew best model saved! Validation accuracy: {val_acc:.4f}")
            else:
                counter += 1
            
            # Print progress with learning rate
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                      f'Test Accuracy: {test_acc:.4f}, Learning Rate: {current_lr:.6f}')
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered! Final learning rate: {current_lr:.6f}")
                break
                
        except RuntimeError as e:
            print(f"\nError in epoch {epoch}: {str(e)}")
            break
    
    # Print learning rate statistics
    print("\nLearning Rate Statistics:")
    print(f"Initial LR: {lr_history[0]:.6f}")
    print(f"Final LR: {lr_history[-1]:.6f}")
    print(f"Min LR: {min(lr_history):.6f}")
    print(f"Max LR: {max(lr_history):.6f}")
    
    # Final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        final_test_acc = evaluate(model, graph_data, graph_data.test_mask)
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main() 