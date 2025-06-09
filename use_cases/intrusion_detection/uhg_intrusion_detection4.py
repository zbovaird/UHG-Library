"""
Intrusion Detection using Universal Hyperbolic Geometry (UHG) with pure projective operations.
This implementation strictly follows UHG principles and leverages the full capabilities of the UHG library.

Key improvements:
1. Pure projective operations - no manifold concepts
2. Cross-ratio preservation in all operations
3. UHG-compliant message passing
4. Projective feature transformations
5. Cross-ratio based attention
6. Geometric pattern recognition
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
import os
from typing import Tuple, Optional
from torch_geometric.data import Data

# Import UHG components
from uhg.projective import ProjectiveUHG
from uhg.nn.models.sage import ProjectiveGraphSAGE
from uhg.nn.layers.sage import ProjectiveSAGEConv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU - Warning: Training may be slow")

# File paths
FILE_PATH = 'CIC_data.csv'
MODEL_SAVE_PATH = 'uhg_ids_model.pth'
RESULTS_PATH = 'uhg_ids_results'

def load_and_preprocess_data() -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Load and preprocess network traffic data using UHG principles."""
    print("\nLoading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(FILE_PATH)
    
    # Separate features and labels
    labels = df['Label'].values
    features = df.drop(['Label'], axis=1).values
    
    # Create label mapping
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_mapping[label] for label in labels])
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Convert to tensors
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    
    # Add homogeneous coordinate for projective space
    features = torch.cat([
        features,
        torch.ones(features.size(0), 1, device=features.device)
    ], dim=1)
    
    # Normalize in projective space
    uhg = ProjectiveUHG()
    features = uhg.normalize_points(features)
    
    print(f"Processed {len(features)} samples with {features.size(1)} features")
    print(f"Found {len(label_mapping)} unique classes")
    
    return features, labels, label_mapping

def create_graph_data(node_features: torch.Tensor, labels: torch.Tensor, k: int = 5) -> Data:
    """Create graph structure using UHG principles and cross-ratio preservation."""
    print("\nCreating graph structure...")
    uhg = ProjectiveUHG()
    
    # Convert features to numpy for KNN computation
    features_np = node_features[:, :-1].cpu().numpy()  # Exclude homogeneous coordinate
    
    # Create k-nearest neighbors graph
    print("Computing KNN graph...")
    knn_graph = kneighbors_graph(
        features_np,
        k,
        mode='connectivity',
        include_self=False,
        n_jobs=-1
    )
    
    # Convert to COO format
    knn_graph_coo = coo_matrix(knn_graph)
    edge_index = torch.from_numpy(
        np.vstack((knn_graph_coo.row, knn_graph_coo.col))
    ).long().to(device)
    
    print(f"Edge index shape: {edge_index.shape}")
    
    # Create train/val/test split
    total_samples = len(node_features)
    indices = torch.randperm(total_samples)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create edge weights using cross-ratios
    edge_weights = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        # Use cross-ratio with ideal points as reference
        line = uhg.join(node_features[src], node_features[dst])
        p1, p2 = uhg.get_ideal_points(line)
        weight = uhg.cross_ratio(
            node_features[src],
            node_features[dst],
            p1, p2
        )
        edge_weights.append(weight.item())
    
    edge_weights = torch.tensor(edge_weights, device=device)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_weight=edge_weights,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    print(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
    return data

class UHGIntrustionDetection(nn.Module):
    """UHG-based Intrusion Detection model using pure projective operations."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
        # Use UHG-compliant GraphSAGE
        self.graph_sage = ProjectiveGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Apply UHG-compliant graph convolutions
        out = self.graph_sage(data.x, data.edge_index)
        return F.log_softmax(out, dim=1)

def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> float:
    """Train for one epoch using UHG principles."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor
) -> Tuple[float, float]:
    """Evaluate model using UHG metrics."""
    model.eval()
    
    # Get predictions
    out = model(data)
    pred = out.argmax(dim=1)
    
    # Calculate accuracy
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    
    # Calculate cross-ratio based confidence
    uhg = ProjectiveUHG()
    confidences = []
    for i in range(len(pred[mask])):
        if correct[i]:
            # Use cross-ratio between prediction and true class
            p1 = out[mask][i, pred[mask][i]]
            p2 = out[mask][i, data.y[mask][i]]
            line = uhg.join(
                torch.tensor([p1, 1.0], device=device),
                torch.tensor([p2, 1.0], device=device)
            )
            i1, i2 = uhg.get_ideal_points(line)
            conf = uhg.cross_ratio(
                torch.tensor([p1, 1.0], device=device),
                torch.tensor([p2, 1.0], device=device),
                i1, i2
            )
            confidences.append(conf.item())
    
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return acc, avg_confidence

def main():
    """Main training function."""
    print("\nInitializing UHG Intrusion Detection System...")
    
    # Load and preprocess data
    node_features, labels, label_mapping = load_and_preprocess_data()
    graph_data = create_graph_data(node_features, labels)
    
    # Model parameters
    in_channels = graph_data.x.size(1)
    hidden_channels = 128
    out_channels = len(label_mapping)
    num_layers = 2
    
    # Initialize model
    model = UHGIntrustionDetection(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, 401):
        # Train
        loss = train_epoch(model, graph_data, optimizer, criterion)
        
        # Evaluate
        train_acc, train_conf = evaluate(model, graph_data, graph_data.train_mask)
        val_acc, val_conf = evaluate(model, graph_data, graph_data.val_mask)
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'Train Loss: {loss:.4f}, Acc: {train_acc:.4f}, Conf: {train_conf:.4f}')
            print(f'Val Acc: {val_acc:.4f}, Conf: {val_conf:.4f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_acc, test_conf = evaluate(model, graph_data, graph_data.test_mask)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Confidence: {test_conf:.4f}')
    
    # Save results
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    
    results = {
        'test_accuracy': test_acc,
        'test_confidence': test_conf,
        'label_mapping': label_mapping
    }
    
    torch.save(results, os.path.join(RESULTS_PATH, 'results.pt'))
    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main() 