"""
Tests for hierarchical GNN implementation using Universal Hyperbolic Geometry.

These tests verify that the hierarchical GNN implementation strictly follows
UHG principles and maintains projective invariants throughout all operations.
"""

import torch
import pytest
from uhg.nn.layers.hierarchical import ProjectiveHierarchicalLayer
from uhg.nn.models.hierarchical import ProjectiveHierarchicalGNN
from uhg.projective import ProjectiveUHG
from uhg.utils.cross_ratio import compute_cross_ratio, verify_cross_ratio_preservation

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_data(device):
    # Create sample hierarchical graph data
    num_nodes = 12
    in_features = 8
    
    # Create node features
    x = torch.randn(num_nodes, in_features, device=device)
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True)  # Normalize features
    x = torch.cat([x, torch.ones(num_nodes, 1, device=device)], dim=1)
    
    # Create hierarchical structure:
    # Level 0: nodes 0-3 (root level)
    # Level 1: nodes 4-7 (middle level)
    # Level 2: nodes 8-11 (leaf level)
    node_levels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], device=device)
    
    # Create edges between:
    # 1. Nodes in same level
    # 2. Parent-child connections
    edge_index = torch.tensor([
        # Same level connections (level 0)
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 2, 0, 3, 0, 3, 1, 2],
        
        # Same level connections (level 1)
        [4, 4, 5, 5, 6, 6, 7, 7],
        [5, 6, 4, 7, 4, 7, 5, 6],
        
        # Same level connections (level 2)
        [8, 8, 9, 9, 10, 10, 11, 11],
        [9, 10, 8, 11, 8, 11, 9, 10],
        
        # Parent-child connections (0->1)
        [0, 1, 2, 3, 0, 1, 2, 3],
        [4, 5, 6, 7, 5, 6, 7, 4],
        
        # Parent-child connections (1->2)
        [4, 5, 6, 7, 4, 5, 6, 7],
        [8, 9, 10, 11, 9, 10, 11, 8]
    ], device=device)
    
    # Reshape to [2, num_edges] format
    edge_index = torch.cat([edge_index[0::2], edge_index[1::2]], dim=1)
    edge_index = edge_index.reshape(2, -1)
    
    return x, edge_index, node_levels

def test_hierarchical_layer_init():
    """Test initialization of hierarchical layer."""
    layer = ProjectiveHierarchicalLayer(
        in_features=8,
        out_features=16,
        num_levels=3,
        level_dim=8
    )
    
    assert layer.in_features == 8
    assert layer.out_features == 16
    assert layer.num_levels == 3
    assert layer.level_dim == 8
    assert isinstance(layer.uhg, ProjectiveUHG)
    
def test_hierarchical_layer_projective_transform(device, sample_data):
    """Test that projective transformations preserve cross-ratios."""
    x, _, _ = sample_data
    layer = ProjectiveHierarchicalLayer(
        in_features=8,
        out_features=16,
        num_levels=3
    ).to(device)
    
    # Apply projective transformation
    transformed = layer.projective_transform(x, layer.W_self)
    transformed = layer._normalize_with_homogeneous(transformed)
    
    # Verify cross-ratio preservation
    assert verify_cross_ratio_preservation(x, transformed, rtol=1e-4)
    
def test_hierarchical_layer_cross_ratio_weights(device, sample_data):
    """Test that cross-ratio based weights respect level differences."""
    x, _, _ = sample_data
    layer = ProjectiveHierarchicalLayer(
        in_features=8,
        out_features=16,
        num_levels=3
    ).to(device)
    
    # Compute weights for same level and different levels
    weight_same = layer.compute_cross_ratio_weight(x[0], x[1], level_diff=0)
    weight_diff = layer.compute_cross_ratio_weight(x[0], x[4], level_diff=1)
    weight_far = layer.compute_cross_ratio_weight(x[0], x[8], level_diff=2)
    
    # Same level weight should be higher than different level
    assert weight_same > weight_diff > weight_far
    assert 0.05 <= weight_same <= 0.95
    assert 0.05 <= weight_diff <= 0.95
    assert 0.05 <= weight_far <= 0.95
    
def test_dynamic_weights(device, sample_data):
    """Test dynamic weight computation based on level structure."""
    _, _, node_levels = sample_data
    layer = ProjectiveHierarchicalLayer(
        in_features=8,
        out_features=16,
        num_levels=3
    ).to(device)
    
    N = len(node_levels)
    weights = layer._compute_dynamic_weights(node_levels, N, device)
    
    # Check weight properties
    assert weights.shape == (N, 5)  # 5 weight types
    assert torch.allclose(weights.sum(1), torch.ones(N, device=device))
    assert torch.all(weights >= 0)
    
def test_hierarchical_layer_forward(device, sample_data):
    """Test forward pass of hierarchical layer."""
    x, edge_index, node_levels = sample_data
    layer = ProjectiveHierarchicalLayer(
        in_features=8,
        out_features=16,
        num_levels=3
    ).to(device)
    
    # Forward pass
    out = layer(x, edge_index, node_levels)
    
    # Check output shape and normalization
    assert out.shape == (x.size(0), layer.out_features)
    assert torch.allclose(torch.norm(out, p=2, dim=-1), torch.ones_like(out[:, 0]), rtol=1e-4)
    
    # Check that output preserves projective structure
    out_with_homogeneous = torch.cat([out, torch.ones_like(out[:, :1])], dim=1)
    assert verify_cross_ratio_preservation(x, out_with_homogeneous, rtol=1e-4)
    
    # Check level-based feature similarity
    level_0_features = out[node_levels == 0]
    level_1_features = out[node_levels == 1]
    level_2_features = out[node_levels == 2]
    
    # Within-level similarity should be higher than between-level
    sim_within_0 = torch.mean(torch.pdist(level_0_features))
    sim_within_1 = torch.mean(torch.pdist(level_1_features))
    sim_within_2 = torch.mean(torch.pdist(level_2_features))
    
    sim_between_01 = torch.mean(torch.cdist(level_0_features, level_1_features))
    sim_between_12 = torch.mean(torch.cdist(level_1_features, level_2_features))
    
    assert sim_within_0 < sim_between_01
    assert sim_within_1 < sim_between_01 and sim_within_1 < sim_between_12
    assert sim_within_2 < sim_between_12
    
def test_hierarchical_gnn_init():
    """Test initialization of hierarchical GNN model."""
    model = ProjectiveHierarchicalGNN(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        num_layers=3,
        num_levels=3
    )
    
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], ProjectiveHierarchicalLayer)
    assert isinstance(model.uhg, ProjectiveUHG)
    
def test_hierarchical_gnn_projective_dropout(device):
    """Test that dropout preserves projective structure."""
    model = ProjectiveHierarchicalGNN(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        dropout=0.5
    ).to(device)
    
    # Create normalized input
    x = torch.randn(10, 8, device=device)
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
    x = torch.cat([x, torch.ones(10, 1, device=device)], dim=1)
    
    # Apply dropout
    model.train()
    dropped = model.projective_dropout(x, p=0.5)
    
    # Check that homogeneous coordinate is preserved
    assert torch.allclose(dropped[:, -1], x[:, -1])
    
    # Check that features are normalized
    norms = torch.norm(dropped[:, :-1], p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-4)
    
    # Check that cross-ratio is preserved
    assert verify_cross_ratio_preservation(x, dropped, rtol=1e-4)
    
def test_hierarchical_gnn_forward(device, sample_data):
    """Test forward pass of hierarchical GNN model."""
    x, edge_index, node_levels = sample_data
    model = ProjectiveHierarchicalGNN(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        num_layers=3,
        num_levels=3
    ).to(device)
    
    # Forward pass
    out = model(x, edge_index, node_levels)
    
    # Check output shape and normalization
    assert out.shape == (x.size(0), 4)  # out_channels = 4
    assert torch.allclose(torch.norm(out, p=2, dim=-1), torch.ones_like(out[:, 0]), rtol=1e-4)
    
    # Check that hierarchical structure is preserved
    level_0_features = out[node_levels == 0]
    level_1_features = out[node_levels == 1]
    level_2_features = out[node_levels == 2]
    
    # Within-level similarity should be higher than between-level
    sim_within_0 = torch.mean(torch.pdist(level_0_features))
    sim_within_1 = torch.mean(torch.pdist(level_1_features))
    sim_within_2 = torch.mean(torch.pdist(level_2_features))
    
    sim_between_01 = torch.mean(torch.cdist(level_0_features, level_1_features))
    sim_between_12 = torch.mean(torch.cdist(level_1_features, level_2_features))
    
    assert sim_within_0 < sim_between_01
    assert sim_within_1 < sim_between_01 and sim_within_1 < sim_between_12
    assert sim_within_2 < sim_between_12
    
def test_hierarchical_gnn_cross_ratio_preservation(device, sample_data):
    """Test that the full model preserves cross-ratios through all layers."""
    x, edge_index, node_levels = sample_data
    model = ProjectiveHierarchicalGNN(
        in_channels=8,
        hidden_channels=16,
        out_channels=4,
        num_layers=3,
        num_levels=3,
        dropout=0.2  # Test with dropout
    ).to(device)
    
    # Set to training mode to test dropout
    model.train()
    
    # Forward pass
    out = model(x, edge_index, node_levels)
    
    # Add homogeneous coordinate for cross-ratio check
    out_with_homogeneous = torch.cat([out, torch.ones_like(out[:, :1])], dim=1)
    
    # Track cross-ratios through layers
    crs_preserved = []
    for i in range(0, x.size(0)-3, 2):
        cr_before = compute_cross_ratio(
            x[i], x[i+1], x[i+2], x[i+3]
        )
        cr_after = compute_cross_ratio(
            out_with_homogeneous[i],
            out_with_homogeneous[i+1],
            out_with_homogeneous[i+2],
            out_with_homogeneous[i+3]
        )
        if not torch.isnan(cr_before) and not torch.isnan(cr_after):
            log_cr_before = torch.log(cr_before + 1e-8)
            log_cr_after = torch.log(cr_after + 1e-8)
            crs_preserved.append(
                torch.allclose(log_cr_before, log_cr_after, rtol=1e-4)
            )
    
    # Verify that most cross-ratios are preserved
    assert sum(crs_preserved) / len(crs_preserved) > 0.8