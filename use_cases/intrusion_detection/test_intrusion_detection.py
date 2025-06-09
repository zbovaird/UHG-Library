"""
Test implementation of intrusion detection using UHG library.
This file is for testing purposes and is not tracked by Git.
"""`

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class UHGGraphSAGE(nn.Module):
    """UHG-based GraphSAGE implementation for intrusion detection."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.2):
        super(UHGGraphSAGE, self).__init__()
        
        # Initialize UHG manifold
        self.manifold = uhg.ProjectiveUHG()
        
        # Create layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        self.layers.append(uhg.nn.layers.HyperbolicGraphConv(
            manifold=self.manifold,
            in_features=in_channels,
            out_features=hidden_channels
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(uhg.nn.layers.HyperbolicGraphConv(
                manifold=self.manifold,
                in_features=hidden_channels,
                out_features=hidden_channels
            ))
        
        # Output layer
        self.layers.append(uhg.nn.layers.HyperbolicGraphConv(
            manifold=self.manifold,
            in_features=hidden_channels,
            out_features=out_channels
        ))

    def forward(self, x, edge_index):
        """Forward pass using UHG operations."""
        # Project input to hyperbolic space
        x = self.manifold.proj_manifold(x)
        
        # Pass through layers
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x, edge_index)))
            x = self.manifold.proj_manifold(x)  # Ensure we stay on manifold
            
        x = self.layers[-1](x, edge_index)
        return x

def load_and_preprocess_data(file_path, sample_frac=0.10):
    """Load and preprocess the CIC dataset."""
    # Load data
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = data.columns.str.strip()
    data['Label'] = data['Label'].str.strip()
    
    # Sample data
    data_sampled = data.sample(frac=sample_frac, random_state=42)
    
    # Convert to numeric and handle missing values
    data_numeric = data_sampled.apply(pd.to_numeric, errors='coerce')
    data_filled = data_numeric.fillna(data_numeric.mean())
    data_filled = data_filled.replace([np.inf, -np.inf], np.nan)
    data_filled = data_filled.fillna(data_filled.max())
    data_filled = data_filled.fillna(0)
    
    # Extract labels
    labels = data_sampled['Label']
    features = data_filled.drop(columns=['Label'])
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, labels

def create_graph_data(features, labels, k=2):
    """Create graph data structure with UHG features."""
    # Convert to UHG tensors
    node_features = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Create k-nearest neighbors graph
    knn_graph = kneighbors_graph(features, k, mode='connectivity', include_self=False)
    knn_graph_coo = coo_matrix(knn_graph)
    edge_index = torch.from_numpy(
        np.array([knn_graph_coo.row, knn_graph_coo.col])
    ).long().to(device)
    
    # Convert labels
    unique_labels = labels.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numeric = labels.map(label_mapping).values
    labels_tensor = torch.tensor(labels_numeric, dtype=torch.long).to(device)
    
    # Create train/val/test masks
    total_samples = len(features)
    indices = torch.randperm(total_samples)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        y=labels_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    ).to(device)

def train_epoch(model, graph_data, optimizer, criterion, batch_size=16, accumulation_steps=4):
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
        # Get batch data
        batch = batch.to(device)
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
def evaluate(model, graph_data, mask):
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
    file_path = 'CIC_data.csv'  # Update with your file path
    features, labels = load_and_preprocess_data(file_path)
    graph_data = create_graph_data(features, labels)
    
    # Model parameters
    in_channels = graph_data.x.size(1)
    hidden_channels = 128
    out_channels = len(labels.unique())
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
    best_model_path = 'best_uhg_model.pth'
    
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
                torch.save(model.state_dict(), best_model_path)
            else:
                counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            
            # Early stopping
            if counter >= patience:
                print("Early stopping triggered")
                break
                
        except RuntimeError as e:
            print(f"Error in epoch {epoch}:", str(e))
            break
    
    # Final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        final_test_acc = evaluate(model, graph_data, graph_data.test_mask)
        print(f"Final Test Accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main() 