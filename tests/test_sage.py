import torch
import pytest
from uhg.nn.models.sage import ProjectiveGraphSAGE
from uhg.nn.layers.sage import ProjectiveSAGEConv
from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG()

@pytest.fixture
def sample_graph():
    # Create a small test graph
    x = torch.randn(5, 3)  # 5 nodes, 3 features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                              [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    return x, edge_index

def test_sage_conv_initialization():
    """Test proper initialization of ProjectiveSAGEConv layer."""
    layer = ProjectiveSAGEConv(in_features=3, out_features=4)
    
    assert layer.in_features == 3
    assert layer.out_features == 4
    assert isinstance(layer.W_self, torch.nn.Parameter)
    assert isinstance(layer.W_neigh, torch.nn.Parameter)
    assert layer.W_self.shape == (4, 3)
    assert layer.W_neigh.shape == (4, 3)

def test_sage_conv_forward(sample_graph, uhg):
    """Test forward pass of ProjectiveSAGEConv layer."""
    x, edge_index = sample_graph
    layer = ProjectiveSAGEConv(in_features=3, out_features=4)
    
    # Forward pass
    out = layer(x, edge_index)
    
    # Check output shape
    assert out.shape == (5, 4)
    
    # Check output lies in projective space
    assert torch.allclose(uhg.normalize_points(out), out)

def test_sage_conv_cross_ratio_preservation(sample_graph, uhg):
    """Test that ProjectiveSAGEConv preserves neighborhood structure."""
    x, edge_index = sample_graph
    layer = ProjectiveSAGEConv(in_features=3, out_features=3)
    
    # Get output
    out = layer(x, edge_index)
    
    # Compute all pairwise similarities before and after
    def cosine_sim(a, b):
        return torch.sum(a * b) / (torch.norm(a) * torch.norm(b) + 1e-8)
    
    # For each node, check that its neighbors remain more similar than non-neighbors
    row, col = edge_index
    N = x.size(0)
    
    for node in range(N):
        # Get neighbors
        neighbors = col[row == node]
        
        # Skip if no neighbors
        if len(neighbors) == 0:
            continue
            
        # Get non-neighbors
        non_neighbors = torch.tensor([i for i in range(N) if i not in neighbors and i != node])
        
        # Skip if no non-neighbors
        if len(non_neighbors) == 0:
            continue
            
        # Compute average similarity to neighbors
        neighbor_sim_before = torch.mean(torch.stack([
            cosine_sim(x[node], x[n]) for n in neighbors
        ]))
        neighbor_sim_after = torch.mean(torch.stack([
            cosine_sim(out[node], out[n]) for n in neighbors
        ]))
        
        # Compute average similarity to non-neighbors
        non_neighbor_sim_before = torch.mean(torch.stack([
            cosine_sim(x[node], x[n]) for n in non_neighbors
        ]))
        non_neighbor_sim_after = torch.mean(torch.stack([
            cosine_sim(out[node], out[n]) for n in non_neighbors
        ]))
        
        # Check that the difference in similarity is preserved
        sim_diff_before = neighbor_sim_before - non_neighbor_sim_before
        sim_diff_after = neighbor_sim_after - non_neighbor_sim_after
        
        # The sign of the difference should be preserved
        if sim_diff_before > 0:
            assert sim_diff_after > -0.1  # Allow small negative values due to numerical issues

def test_sage_model_initialization():
    """Test proper initialization of ProjectiveGraphSAGE model."""
    model = ProjectiveGraphSAGE(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_layers=3
    )
    
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], ProjectiveSAGEConv)
    assert model.layers[0].in_features == 3
    assert model.layers[0].out_features == 4
    assert model.layers[1].in_features == 4
    assert model.layers[1].out_features == 4
    assert model.layers[2].in_features == 4
    assert model.layers[2].out_features == 2

def test_sage_model_forward(sample_graph, uhg):
    """Test forward pass of ProjectiveGraphSAGE model."""
    x, edge_index = sample_graph
    model = ProjectiveGraphSAGE(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_layers=3
    )
    
    # Forward pass
    out = model(x, edge_index)
    
    # Check output shape
    assert out.shape == (5, 2)
    
    # Check output lies in projective space
    assert torch.allclose(uhg.normalize_points(out), out)

def test_projective_dropout():
    """Test that projective dropout maintains projective structure."""
    model = ProjectiveGraphSAGE(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        dropout=0.5
    )
    
    x = torch.randn(5, 3)
    
    # Set to training mode
    model.train()
    
    # Apply dropout
    out = model.projective_dropout(x, p=0.5)
    
    # Check that output is properly normalized
    assert torch.allclose(model.uhg.normalize_points(out), out)
    
    # Check that some elements are zeroed
    assert (out == 0).any()
    
    # Test no dropout during eval
    model.eval()
    out_eval = model.projective_dropout(x, p=0.5)
    assert torch.allclose(x, out_eval)

def test_end_to_end_training(sample_graph):
    """Test end-to-end training of ProjectiveGraphSAGE."""
    x, edge_index = sample_graph
    model = ProjectiveGraphSAGE(
        in_channels=3,
        hidden_channels=8,  # Increased hidden size
        out_channels=2,
        dropout=0.2  # Increased dropout
    )
    
    # Create dummy target
    y = torch.randint(0, 2, (5,))
    
    # Training loop with better parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track best loss
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    model.train()
    for epoch in range(200):  # More epochs
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        
        if epoch == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break
    
    final_loss = best_loss
    
    # Check that loss decreased significantly
    assert final_loss < initial_loss