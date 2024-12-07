"""
Intrusion Detection using Universal Hyperbolic Geometry (UHG) with neighbor sampling.
This implementation uses efficient subgraph sampling while preserving UHG geometric properties.
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
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.loader import NeighborLoader
import os
from typing import Tuple, Optional, List
from uhg.projective import ProjectiveUHG
from torch_scatter import scatter_add, scatter_mean
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import subgraph

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
    # [Previous implementation remains the same]
    pass

def create_graph_data(node_features: torch.Tensor, labels: torch.Tensor, k: int = 2) -> Data:
    """Create graph structure using UHG principles with efficient edge construction."""
    # [Previous implementation remains the same]
    pass

def uhg_quadrance_vectorized(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between batches of points efficiently."""
    # [Previous implementation remains the same]
    pass

def uhg_spread_vectorized(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between batches of lines efficiently."""
    # [Previous implementation remains the same]
    pass

class UHGNeighborSampler:
    """UHG-aware neighbor sampling that preserves geometric properties."""
    def __init__(self, edge_index: torch.Tensor, sizes: List[int], 
                 node_features: torch.Tensor, num_hops: int = 2):
        self.edge_index = edge_index
        self.sizes = sizes
        self.num_hops = num_hops
        self.node_features = node_features
    
    def sample_neighbors(self, batch_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample neighbors while preserving UHG structure."""
        sampled_nodes = [batch_nodes]
        sampled_edges = []
        
        # Multi-hop sampling
        for hop in range(self.num_hops):
            # Get current hop's nodes
            current_nodes = sampled_nodes[-1]
            
            # Find connected edges
            mask = torch.isin(self.edge_index[0], current_nodes)
            hop_edges = self.edge_index[:, mask]
            
            # Sample neighbors based on UHG distances
            neighbors = hop_edges[1]
            if len(neighbors) > self.sizes[hop]:
                # Compute UHG distances
                source_features = self.node_features[hop_edges[0]]
                target_features = self.node_features[neighbors]
                distances = uhg_quadrance_vectorized(source_features, target_features)
                
                # Sample based on distances (prefer closer neighbors in UHG space)
                weights = torch.exp(-distances)
                probs = weights / weights.sum()
                sampled_idx = torch.multinomial(probs, self.sizes[hop], replacement=False)
                neighbors = neighbors[sampled_idx]
            
            sampled_nodes.append(neighbors)
            sampled_edges.append(hop_edges[:, sampled_idx])
        
        # Combine all sampled nodes and edges
        all_nodes = torch.cat(sampled_nodes)
        all_edges = torch.cat(sampled_edges, dim=1)
        
        # Create node mapping for the subgraph
        node_idx = torch.unique(all_nodes)
        idx_map = {int(idx): i for i, idx in enumerate(node_idx)}
        
        # Remap edges to new indices
        mapped_edges = torch.tensor([
            [idx_map[int(i)] for i in all_edges[0]],
            [idx_map[int(i)] for i in all_edges[1]]
        ], device=device)
        
        return node_idx, mapped_edges, batch_nodes

class UHGMessagePassing(MessagePassing):
    """Efficient message passing layer for UHG operations with neighbor sampling support."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
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
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Compute UHG-based weights efficiently
        weights = torch.exp(-uhg_quadrance_vectorized(x_i, x_j))
        
        # Transform neighbor features
        messages = torch.matmul(x_j, self.weight_msg)
        
        # Weight the messages
        return messages * weights.view(-1, 1)

class UHGGraphSAGE(nn.Module):
    """UHG-enhanced GraphSAGE model with neighbor sampling support."""
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
                criterion: nn.Module, batch_size: int = 32, num_neighbors: List[int] = [25, 10]) -> float:
    """Train for one epoch using neighbor sampling."""
    model.train()
    total_loss = 0
    
    # Create neighbor sampler
    neighbor_sampler = UHGNeighborSampler(
        edge_index=graph_data.edge_index,
        sizes=num_neighbors,
        node_features=graph_data.x,
        num_hops=len(num_neighbors)
    )
    
    # Create batches of root nodes
    train_nodes = graph_data.train_mask.nonzero(as_tuple=True)[0]
    train_loader = DataLoader(
        train_nodes,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Process batches
    for batch_nodes in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Sample subgraph
        node_idx, sampled_edges, batch_nodes = neighbor_sampler.sample_neighbors(batch_nodes)
        
        # Get subgraph data
        sub_x = graph_data.x[node_idx]
        sub_y = graph_data.y[batch_nodes]
        
        # Forward pass
        out = model(sub_x, sampled_edges)
        loss = criterion(out[batch_nodes], sub_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model: nn.Module, graph_data: Data, mask: torch.Tensor, 
            batch_size: int = 128, num_neighbors: List[int] = [25, 10]) -> float:
    """Evaluate model using neighbor sampling."""
    model.eval()
    total_correct = 0
    total_nodes = 0
    
    # Create neighbor sampler for evaluation
    neighbor_sampler = UHGNeighborSampler(
        edge_index=graph_data.edge_index,
        sizes=num_neighbors,
        node_features=graph_data.x,
        num_hops=len(num_neighbors)
    )
    
    # Create batches of nodes to evaluate
    eval_nodes = mask.nonzero(as_tuple=True)[0]
    eval_loader = DataLoader(
        eval_nodes,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Process batches
    for batch_nodes in eval_loader:
        # Sample subgraph
        node_idx, sampled_edges, batch_nodes = neighbor_sampler.sample_neighbors(batch_nodes)
        
        # Get subgraph data
        sub_x = graph_data.x[node_idx]
        
        # Forward pass
        out = model(sub_x, sampled_edges)
        pred = out[batch_nodes].argmax(dim=1)
        
        # Compute accuracy
        correct = (pred == graph_data.y[batch_nodes]).sum().item()
        total_correct += correct
        total_nodes += len(batch_nodes)
    
    return total_correct / total_nodes

def main():
    """Main training function with neighbor sampling."""
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
    
    # Training parameters
    num_epochs = 400
    batch_size = 64  # Increased batch size due to memory efficiency
    num_neighbors = [25, 10]  # Number of neighbors to sample per hop
    
    # Learning rate scheduling
    warmup_epochs = 5
    scheduler = torch.optim.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        epochs=num_epochs,
        steps_per_epoch=len(range(graph_data.train_mask.sum())) // batch_size,
        pct_start=warmup_epochs/num_epochs,
        final_div_factor=initial_lr/min_lr,
        div_factor=10.0,
        three_phase=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop parameters
    best_val_acc = 0
    patience = 20
    counter = 0
    
    # Monitoring setup
    lr_history = []
    val_history = []
    
    print("\nStarting training with neighbor sampling...")
    for epoch in range(1, num_epochs + 1):
        try:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            loss = train_epoch(
                model, graph_data, optimizer, criterion, 
                batch_size=batch_size, num_neighbors=num_neighbors
            )
            
            # Step the scheduler
            scheduler.step()
            
            # Evaluate
            val_acc = evaluate(
                model, graph_data, graph_data.val_mask,
                batch_size=batch_size, num_neighbors=num_neighbors
            )
            test_acc = evaluate(
                model, graph_data, graph_data.test_mask,
                batch_size=batch_size, num_neighbors=num_neighbors
            )
            
            # Store metrics
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
            
            # Print progress
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
        final_test_acc = evaluate(
            model, graph_data, graph_data.test_mask,
            batch_size=batch_size, num_neighbors=num_neighbors
        )
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main() 