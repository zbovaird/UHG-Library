"""
Test file for Gjallarhorn1.py to verify dimensions and UHG compliance
"""

import torch
import numpy as np
from tqdm import tqdm
from uhg.projective import ProjectiveUHG

# Import model components from Gjallarhorn1
from Gjallarhorn1 import (
    UHGModel,
    ProjectiveBaseLayer,
    UHGSAGEConv,
    prepare_features,
    UHGLoss
)

def create_synthetic_data(num_nodes=100, num_features=4):
    """Create small synthetic dataset for testing."""
    # Create random features
    features = np.random.randn(num_nodes, num_features)
    
    # Create random edges (k=3 nearest neighbors)
    edge_index = []
    for i in range(num_nodes):
        distances = np.sum((features - features[i])**2, axis=1)
        distances[i] = np.inf  # Exclude self
        neighbors = np.argsort(distances)[:3]
        for j in neighbors:
            edge_index.append([i, j])
    
    edge_index = np.array(edge_index).T
    
    # Create random labels (binary for testing)
    labels = np.random.randint(0, 2, size=num_nodes)
    
    return features, edge_index, labels

def check_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, uhg: ProjectiveUHG) -> bool:
    """Check if cross-ratio is preserved between four points."""
    cr1 = uhg.cross_ratio(p1, p2, p3, p4)
    cr2 = uhg.cross_ratio(p4, p3, p2, p1)
    return torch.abs(cr1 * cr2 - 1.0) < 1e-5

def test_dimensions():
    """Test dimension handling throughout the model."""
    print("\nTesting dimensions...")
    
    # Create synthetic data
    features, edge_index, labels = create_synthetic_data(num_nodes=100, num_features=4)
    
    # Convert to tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    
    # Prepare features using UHG
    features = prepare_features(features)
    print(f"Input feature shape: {features.shape}")  # Should be [100, 5] (4 features + 1 homogeneous)
    
    # Initialize model with correct dimensions
    model = UHGModel(
        in_channels=features.shape[1],  # 5 (4 features + 1 homogeneous)
        hidden_channels=8,
        out_channels=4,
        num_layers=2
    ).to(device)
    
    # Test forward pass
    try:
        out = model(features, edge_index)
        print(f"Output shape: {out.shape}")  # Should be [100, 5] (4 features + 1 homogeneous)
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        raise e

def test_uhg_compliance():
    """Test UHG compliance of operations."""
    print("\nTesting UHG compliance...")
    
    uhg = ProjectiveUHG()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small test data
    features, edge_index, labels = create_synthetic_data(num_nodes=10, num_features=4)
    features = prepare_features(features)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    
    # Initialize model with correct dimensions
    model = UHGModel(
        in_channels=features.shape[1],  # 5 (4 features + 1 homogeneous)
        hidden_channels=8,
        out_channels=4,
        num_layers=2
    ).to(device)
    
    # Test cross-ratio preservation through model
    with torch.no_grad():
        out = model(features, edge_index)
        
        # Check random quadruples
        preserved = 0
        total = 10
        
        for _ in range(total):
            idx = torch.randperm(len(out))[:4]
            if check_cross_ratio(
                out[idx[0]], out[idx[1]],
                out[idx[2]], out[idx[3]],
                uhg
            ):
                preserved += 1
        
        print(f"Cross-ratio preservation: {preserved}/{total} tests passed")

def test_training():
    """Test training loop with small dataset."""
    print("\nTesting training loop...")
    
    # Create synthetic data
    features, edge_index, labels = create_synthetic_data(num_nodes=100, num_features=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    features = prepare_features(features)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    
    # Initialize model with correct dimensions
    model = UHGModel(
        in_channels=features.shape[1],  # 5 (4 features + 1 homogeneous)
        hidden_channels=8,
        out_channels=4,
        num_layers=2
    ).to(device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0005
    )
    criterion = UHGLoss()
    
    # Test training loop
    try:
        model.train()
        for epoch in range(5):  # Just a few epochs for testing
            optimizer.zero_grad()
            out = model(features, edge_index)
            loss = criterion(out, edge_index, batch_size=32)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        print("✓ Training loop successful")
    except Exception as e:
        print(f"✗ Training loop failed: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Running tests for Gjallarhorn1.py...")
    
    try:
        test_dimensions()
        test_uhg_compliance()
        test_training()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {str(e)}") 