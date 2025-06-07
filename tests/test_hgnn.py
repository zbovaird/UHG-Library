import torch
import pytest
import numpy as np
from uhg.nn.models import HGCN, HGAT
from uhg.manifolds import HyperbolicManifold
from uhg.metrics import hyperbolic_distance
from uhg.utils.cross_ratio import compute_cross_ratio

def create_test_graph(batch_size=2, num_nodes=10, num_edges=20, in_channels=16, hidden_channels=32, out_channels=8, edge_dim=None):
    """Create a test graph with random features and connectivity, including homogeneous coordinate."""
    # Create random node features and add homogeneous coordinate
    x = torch.randn(batch_size * num_nodes, in_channels)
    ones = torch.ones(batch_size * num_nodes, 1)
    x = torch.cat([x, ones], dim=1)  # [N, in_channels+1]
    
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
        in_channels=17,
        hidden_channels=33,
        out_channels=9,
        num_layers=2,
        dropout=0.1,
        bias=True
    )

@pytest.fixture
def hgat(manifold):
    """Create an HGAT model instance for testing."""
    return HGAT(
        manifold=manifold,
        in_channels=17,
        hidden_channels=33,
        out_channels=9,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        bias=True
    )

def test_hgcn_initialization(hgcn):
    """Test HGCN model initialization."""
    assert hgcn.in_channels == 17
    assert hgcn.hidden_channels == 33
    assert hgcn.out_channels == 9
    assert hgcn.num_layers == 2
    assert hgcn.dropout == 0.1
    assert hgcn.bias is True
    assert len(hgcn.layers) == 2

def test_hgat_initialization(hgat):
    """Test HGAT model initialization."""
    assert hgat.in_channels == 17
    assert hgat.hidden_channels == 33
    assert hgat.out_channels == 9
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
    assert out.shape == (20, 9)  # 2 batches * 10 nodes, 9 output channels
    
    # Check that output lies on the hyperbolic manifold
    assert torch.allclose(manifold.inner_product(out, out), torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgat_forward(hgat, manifold):
    """Test HGAT forward pass."""
    x, edge_index, edge_attr, batch = create_test_graph()
    
    # Forward pass
    out = hgat(x, edge_index, edge_attr, batch)
    
    # Check output shape
    assert out.shape == (20, 9)  # 2 batches * 10 nodes, 9 output channels
    
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
        assert out.shape == (batch_size * num_nodes, 9)
        
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
        assert out.shape == (batch_size * num_nodes, 9)
        
        # Check that output lies on the hyperbolic manifold
        assert torch.allclose(manifold.inner_product(out, out), 
                            torch.ones_like(out[:, 0]), atol=1e-6)

def test_hgcn_forward_no_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with no edge features."""
    x, edge_index, _, batch = create_test_graph(edge_dim=None)
    out = hgcn(x, edge_index, None, batch)
    assert out.shape == (20, 9)

def test_hgcn_forward_matching_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with matching-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=32)  # hidden_channels
    out = hgcn(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 9)

def test_hgcn_forward_mismatched_edge_attr(hgcn, manifold):
    """Test HGCN forward pass with mismatched-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=5)
    out = hgcn(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 9)

def test_hgat_forward_no_edge_attr(hgat, manifold):
    """Test HGAT forward pass with no edge features."""
    x, edge_index, _, batch = create_test_graph(edge_dim=None)
    out = hgat(x, edge_index, None, batch)
    assert out.shape == (20, 9)

def test_hgat_forward_matching_edge_attr(hgat, manifold):
    """Test HGAT forward pass with matching-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=16)  # in_channels
    out = hgat(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 9)

def test_hgat_forward_mismatched_edge_attr(hgat, manifold):
    """Test HGAT forward pass with mismatched-dimension edge features."""
    x, edge_index, edge_attr, batch = create_test_graph(edge_dim=7)
    out = hgat(x, edge_index, edge_attr, batch)
    assert out.shape == (20, 9)

def test_hgcn_basic():
    """Test basic functionality of HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    
    # Create model (input features include homogeneous coordinate)
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Create test data
    x, edge_index, edge_attr, batch = create_test_graph(
        in_channels=in_channels,
        edge_dim=4
    )
    
    # Forward pass
    out = model(x, edge_index, edge_attr)
    
    # Check output shape
    assert out.shape == (x.size(0), out_channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_hgcn_geometric_invariants():
    """Test preservation of geometric invariants in HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    
    # Create model
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Create test data with at least 4 nodes
    x = torch.randn(4, in_channels+1)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    
    # Compute initial cross-ratio
    cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
    
    # Forward pass
    out = model(x, edge_index)
    
    # Compute final cross-ratio
    cr_final = compute_cross_ratio(out[0], out[1], out[2], out[3])
    
    # Check cross-ratio preservation
    assert torch.allclose(cr_initial, cr_final, rtol=1e-3, atol=1e-3)

def test_hgcn_attention():
    """Test HGCN with attention mechanism."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    num_heads = 4
    
    # Create model with attention
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        use_attn=True,
        attn_heads=num_heads
    )
    
    # Create test data
    x, edge_index, edge_attr, batch = create_test_graph(
        in_channels=in_channels,
        edge_dim=4
    )
    
    # Forward pass
    out = model(x, edge_index, edge_attr)
    
    # Check output shape
    assert out.shape == (x.size(0), out_channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_hgcn_message_passing():
    """Test message passing in HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    
    # Create model
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Create test data
    x, edge_index, edge_attr, batch = create_test_graph(
        in_channels=in_channels,
        edge_dim=4
    )
    
    # Forward pass
    out = model(x, edge_index, edge_attr)
    
    # Check that messages are properly aggregated
    assert out.shape == (x.size(0), out_channels)
    
    # Check that output features are different from input
    assert not torch.allclose(out, x[:, :out_channels])

def test_hgcn_dropout():
    """Test dropout in HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    dropout = 0.5
    
    # Create model with dropout
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create test data
    x, edge_index, edge_attr, batch = create_test_graph(
        in_channels=in_channels,
        edge_dim=4
    )
    
    # Forward pass in training mode
    model.train()
    out_train = model(x, edge_index, edge_attr)
    
    # Forward pass in eval mode
    model.eval()
    out_eval = model(x, edge_index, edge_attr)
    
    # Check that outputs are different in train and eval modes
    assert not torch.allclose(out_train, out_eval)

def test_hgcn_batch_processing():
    """Test batch processing in HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    batch_size = 2
    
    # Create model
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Create test data with batch
    x, edge_index, edge_attr, batch = create_test_graph(
        batch_size=batch_size,
        in_channels=in_channels,
        edge_dim=4
    )
    
    # Forward pass
    out = model(x, edge_index, edge_attr, batch)
    
    # Check output shape
    assert out.shape == (x.size(0), out_channels)
    
    # Check that batch processing works
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_hgcn_edge_cases():
    """Test edge cases in HGCN."""
    manifold = HyperbolicManifold()
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_layers = 2
    
    # Create model
    model = HGCN(
        manifold=manifold,
        in_channels=in_channels+1,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Test with empty graph
    x = torch.randn(0, in_channels+1)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (0, out_channels)
    
    # Test with single node
    x = torch.randn(1, in_channels+1)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (1, out_channels)
    
    # Test with disconnected graph
    x = torch.randn(4, in_channels+1)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (4, out_channels)

if __name__ == "__main__":
    # Run all tests
    test_hgcn_basic()
    test_hgcn_geometric_invariants()
    test_hgcn_attention()
    test_hgcn_message_passing()
    test_hgcn_dropout()
    test_hgcn_batch_processing()
    test_hgcn_edge_cases()
    print("All HGCN tests passed!") 