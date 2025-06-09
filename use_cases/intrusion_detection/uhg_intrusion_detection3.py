"""
Intrusion Detection using Universal Hyperbolic Geometry (UHG) with vectorized operations.
This implementation uses PyTorch Geometric's efficient message passing and scatter operations
for better performance and reduced training time.
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
from torch_scatter import scatter_add, scatter_mean
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import MessagePassing

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
    """Create graph structure using UHG principles with efficient edge construction."""
    print("\nCreating graph structure...")
    
    # Convert features to numpy for sklearn
    features_np = node_features.cpu().numpy()
    
    # Create k-nearest neighbors graph using sklearn (memory efficient)
    print("Computing KNN graph...")
    knn_graph = kneighbors_graph(
        features_np, 
        k, 
        mode='connectivity', 
        include_self=False,
        n_jobs=-1  # Use all available cores
    )
    
    # Convert to COO format efficiently
    knn_graph_coo = coo_matrix(knn_graph)
    
    # Create edge index efficiently
    edge_index = torch.from_numpy(
        np.vstack((knn_graph_coo.row, knn_graph_coo.col))
    ).long().to(device)
    
    print(f"Edge index shape: {edge_index.shape}")
    
    # Add homogeneous coordinate to features efficiently
    node_features_uhg = torch.cat([
        node_features, 
        torch.ones(node_features.size(0), 1, device=node_features.device)
    ], dim=1)
    
    print(f"Feature shape with homogeneous coordinate: {node_features_uhg.shape}")
    
    # Create train/val/test split efficiently
    total_samples = len(node_features_uhg)
    indices = torch.randperm(total_samples)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    # Create masks efficiently using indexing
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

def uhg_quadrance_vectorized(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between batches of points efficiently."""
    # Compute dot products for all pairs at once
    dot_products = torch.sum(x * y, dim=-1)
    
    # Compute squared norms efficiently
    x_norm_sq = torch.sum(x * x, dim=-1) - x[..., -1] * x[..., -1]
    y_norm_sq = torch.sum(y * y, dim=-1) - y[..., -1] * y[..., -1]
    
    # Compute quadrance
    return 1 - (dot_products * dot_products) / (
        torch.clamp(x_norm_sq * y_norm_sq, min=eps)
    )

def uhg_spread_vectorized(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between batches of lines efficiently."""
    # Compute dot products for all pairs at once
    dot_products = torch.sum(L * M, dim=-1)
    
    # Compute squared norms efficiently
    L_norm_sq = torch.sum(L * L, dim=-1) - L[..., -1] * L[..., -1]
    M_norm_sq = torch.sum(M * M, dim=-1) - M[..., -1] * M[..., -1]
    
    # Compute spread
    return 1 - (dot_products * dot_products) / (
        torch.clamp(L_norm_sq * M_norm_sq, min=eps)
    )

class UHGMessagePassing(MessagePassing):
    """Efficient message passing layer for UHG operations."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(aggr='add')  # Use efficient built-in aggregation
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights for feature transformations
        self.weight_msg = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_node = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_msg)
        nn.init.xavier_uniform_(self.weight_node)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Split features and homogeneous coordinate
        features = x[:, :-1]
        homogeneous = x[:, -1:]
        
        # Transform node features
        transformed_features = torch.matmul(features, self.weight_node)
        
        # Compute messages and aggregate
        out = self.propagate(edge_index, x=features, size=None)
        
        # Combine with transformed features
        out = out + transformed_features
        
        # Add homogeneous coordinate back
        return torch.cat([out, homogeneous], dim=1)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Compute UHG-based weights efficiently
        weights = torch.exp(-uhg_quadrance_vectorized(x_i, x_j))
        
        # Transform neighbor features
        messages = torch.matmul(x_j, self.weight_msg)
        
        # Weight the messages
        return messages * weights.view(-1, 1)
    
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # Use efficient scatter operations for aggregation
        # First, sum the weighted messages
        numerator = scatter_add(inputs, index, dim=0)
        
        # Then, sum the weights
        weights_sum = scatter_add(torch.ones_like(inputs), index, dim=0)
        
        # Normalize
        return numerator / torch.clamp(weights_sum, min=1e-6)

class UHGGraphSAGE(nn.Module):
    """UHG-enhanced GraphSAGE model with efficient message passing."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input features are one less than x.shape[1] due to homogeneous coordinate
        actual_in_channels = in_channels - 1
        
        # Input layer
        self.layers.append(UHGMessagePassing(actual_in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(UHGMessagePassing(hidden_channels, hidden_channels))
            
        # Output layer
        self.layers.append(UHGMessagePassing(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Process through layers efficiently
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.layers[-1](x, edge_index)
        
        # Return only feature part
        return x[:, :-1]

def train_epoch(model: nn.Module, graph_data: Data, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, batch_size: int = 32) -> float:
    """Train for one epoch using efficient batching and gradient accumulation."""
    model.train()
    total_loss = 0
    
    # Create efficient dataloader
    train_loader = DataLoader(
        range(graph_data.train_mask.sum()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    
    # Process batches efficiently
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Get batch data efficiently
        batch = batch.to(device)
        batch_mask = graph_data.train_mask.clone()
        batch_mask[graph_data.train_mask.nonzero(as_tuple=True)[0][batch]] = True
        
        # Forward pass
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[batch_mask], graph_data.y[batch_mask])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model: nn.Module, graph_data: Data, mask: torch.Tensor) -> float:
    """Evaluate model efficiently."""
    model.eval()
    
    # Forward pass on full graph
    out = model(graph_data.x, graph_data.edge_index)
    pred = out[mask].argmax(dim=1)
    
    # Compute accuracy efficiently
    correct = (pred == graph_data.y[mask]).sum()
    total = mask.sum()
    
    return correct.item() / total.item()

def main():
    """Main training function with efficient batch processing and monitoring."""
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
    batch_size = 32  # Increased batch size for efficiency
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
            loss = train_epoch(model, graph_data, optimizer, criterion, batch_size)
            
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