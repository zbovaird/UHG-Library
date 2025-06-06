import torch
import pytest
import numpy as np
from uhg.nn.models import HGCN, HGAT
from uhg.manifolds import HyperbolicManifold
from uhg.metrics import hyperbolic_distance

def create_test_graph(batch_size=2, num_nodes=10, num_edges=20, in_channels=16, hidden_channels=32, out_channels=8, edge_dim=None):
    """Create a test graph with random features and connectivity."""
    # Create random node features
    x = torch.randn(batch_size * num_nodes, in_channels)
    
    # Create random edge indices
    edge_index = torch.randint(0, batch_size * num_nodes, (2, num_edges))
    
    # Create random edge features
    if edge_dim is not None:
        edge_attr = torch.randn(num_edges, edge_dim)
    else:
        edge_attr = None
    
    # Create batch vector
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    return x, edge_index, edge_attr, batch

@pytest.fixture
def manifold():
    """Create a hyperbolic manifold instance for testing."""
    return HyperbolicManifold()

@pytest.fixture
def hgcn(manifold):
    """Create an HGCN model instance for testing."""
    return HGCN(
        manifold=manifold,
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        num_layers=2,
        dropout=0.1,
        bias=True
    )

@pytest.fixture
def hgat(manifold):
    """Create an HGAT model instance for testing."""
    return HGAT(
        manifold=manifold,
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        bias=True
    )

def test_hgcn_initialization(hgcn):
    """Test HGCN model initialization."""
    assert hgcn.in_channels == 16
    assert hgcn.hidden_channels == 32
    assert hgcn.out_channels == 8
    assert hgcn.num_layers == 2
    assert hgcn.dropout == 0.1
    assert hgcn.bias is True
    assert len(hgcn.layers) == 2

def test_hgat_initialization(hgat):
    """Test HGAT model initialization."""
    assert hgat.in_channels == 16
    assert hgat.hidden_channels == 32
    assert hgat.out_channels == 8
    assert hgat.num_layers == 2
    assert hgat.num_heads == 4
    assert hgat.dropout == 0.1
    assert hgat.bias is True
    assert len(hgat.layers) == 2
    assert len(hgat.attention_layers) == 2

def test_hgcn_forward(hgcn, manifold):
    """Test HGCN forward pass."""
    x, edge_index, edge_attr, batch = create_test_graph()
    
    # Forward pass
    out = hgcn(x, edge_index, edge_attr, batch)
    
    # Check output shape
    assert out.shape == (20, 8)  # 2 batches * 10 nodes, 8 output channels
    
    # Check that output lies on the hyperbolic manifold
    assert torch.allclose(manifold.inner_product(out, out), torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgat_forward(hgat, manifold):
    """Test HGAT forward pass."""
    x, edge_index, edge_attr, batch = create_test_graph()
    
    # Forward pass
    out = hgat(x, edge_index, edge_attr, batch)
    
    # Check output shape
    assert out.shape == (20, 8)  # 2 batches * 10 nodes, 8 output channels
    
    # Check that output lies on the hyperbolic manifold
    assert torch.allclose(manifold.inner_product(out, out), torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgcn_message_passing(hgcn, manifold):
    """Test HGCN message passing."""
    x, edge_index, edge_attr, _ = create_test_graph(batch_size=1, num_nodes=5, num_edges=10)
    
    # Get initial features
    initial_features = hgcn.layers[0](x)
    
    # Perform message passing
    updated_features = hgcn._message_passing(initial_features, edge_index, edge_attr)
    
    # Check that features are updated
    assert not torch.allclose(initial_features, updated_features)
    
    # Check that updated features lie on the hyperbolic manifold
    assert torch.allclose(manifold.inner_product(updated_features, updated_features), 
                         torch.ones_like(updated_features[:, 0]), atol=1e-6)

def test_hgat_attention(hgat, manifold):
    """Test HGAT attention mechanism."""
    x, edge_index, edge_attr, _ = create_test_graph(batch_size=1, num_nodes=5, num_edges=10)
    
    # Get initial features
    initial_features = hgat.layers[0](x)
    
    # Perform attention-based message passing
    updated_features = hgat._message_passing(initial_features, edge_index, edge_attr)
    
    # Check that features are updated
    assert not torch.allclose(initial_features, updated_features)
    
    # Check that updated features lie on the hyperbolic manifold
    assert torch.allclose(manifold.inner_product(updated_features, updated_features), 
                         torch.ones_like(updated_features[:, 0]), atol=1e-6)

def test_hgcn_gradient_flow(hgcn):
    """Test HGCN gradient flow."""
    x, edge_index, edge_attr, batch = create_test_graph()
    
    # Enable gradient computation
    x.requires_grad_(True)
    
    # Forward pass
    out = hgcn(x, edge_index, edge_attr, batch)
    
    # Compute loss
    loss = out.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_hgat_gradient_flow(hgat):
    """Test HGAT gradient flow."""
    x, edge_index, edge_attr, batch = create_test_graph()
    
    # Enable gradient computation
    x.requires_grad_(True)
    
    # Forward pass
    out = hgat(x, edge_index, edge_attr, batch)
    
    # Compute loss
    loss = out.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_hgcn_hyperbolic_invariants(hgcn, manifold):
    """Test that HGCN preserves hyperbolic invariants."""
    x, edge_index, edge_attr, batch = create_test_graph()
    out = hgcn(x, edge_index, edge_attr, batch)
    for i in range(out.shape[0]):
        for j in range(i + 1, out.shape[0]):
            dist = hyperbolic_distance(out[i], out[j])
            assert dist >= 0
            assert not torch.isnan(dist)
            assert not torch.isinf(dist)

def test_hgat_hyperbolic_invariants(hgat, manifold):
    """Test that HGAT preserves hyperbolic invariants."""
    x, edge_index, edge_attr, batch = create_test_graph()
    out = hgat(x, edge_index, edge_attr, batch)
    for i in range(out.shape[0]):
        for j in range(i + 1, out.shape[0]):
            dist = hyperbolic_distance(out[i], out[j])
            assert dist >= 0
            assert not torch.isnan(dist)
            assert not torch.isinf(dist)

def test_hgcn_batch_processing(hgcn, manifold):
    """Test HGCN batch processing."""
    batch_sizes = [1, 2, 4]
    num_nodes = 10
    
    for batch_size in batch_sizes:
        x, edge_index, edge_attr, batch = create_test_graph(
            batch_size=batch_size,
            num_nodes=num_nodes,
            num_edges=20
        )
        
        # Forward pass
        out = hgcn(x, edge_index, edge_attr, batch)
        
        # Check output shape
        assert out.shape == (batch_size * num_nodes, 8)
        
        # Check that output lies on the hyperbolic manifold
        assert torch.allclose(manifold.inner_product(out, out), 
                            torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgat_batch_processing(hgat, manifold):
    """Test HGAT batch processing."""
    batch_sizes = [1, 2, 4]
    num_nodes = 10
    
    for batch_size in batch_sizes:
        x, edge_index, edge_attr, batch = create_test_graph(
            batch_size=batch_size,
            num_nodes=num_nodes,
            num_edges=20
        )
        
        # Forward pass
        out = hgat(x, edge_index, edge_attr, batch)
        
        # Check output shape
        assert out.shape == (batch_size * num_nodes, 8)
        
        # Check that output lies on the hyperbolic manifold
        assert torch.allclose(manifold.inner_product(out, out), 
                            torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgcn_forward_no_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with no edge features."""
    x, edge_index, _, batch = create_test_graph(edge_dim=None)
    out = hgcn(x, edge_index, None, batch)
    assert out.shape == (20, 8)

def test_hgcn_forward_matching_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with matching-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=32)  # hidden_channels
    out = hgcn(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 8)

def test_hgcn_forward_mismatched_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with mismatched-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=5)
    out = hgcn(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 8)

def test_hgat_forward_no_edge_attr(hgat, manifold):
    """Test HGAT forward pass with no edge features."""
    x, edge_index, _, batch = create_test_graph(edge_dim=None)
    out = hgat(x, edge_index, None, batch)
    assert out.shape == (20, 8)

def test_hgat_forward_matching_edge_attr(hgat, manifold):
    """Test HGAT forward pass with matching-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=16)  # in_channels
    out = hgat(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 8)

def test_hgat_forward_mismatched_edge_attr(hgat, manifold):
    """Test HGAT forward pass with mismatched-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=7)
    out = hgat(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 8) 