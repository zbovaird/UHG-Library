"""
Unit tests for the Hyperbolic Graph Transformer (HGT) model.
"""

import torch
import pytest
from uhg.nn.models.hgt import HGT, HyperbolicPositionalEncoding, HyperbolicTransformerLayer
from uhg.projective import ProjectiveUHG
from uhg.utils.cross_ratio import compute_cross_ratio

def test_hyperbolic_positional_encoding():
    """Test HyperbolicPositionalEncoding module."""
    # Initialize module
    d_model = 64
    max_len = 100
    dropout = 0.1
    pos_encoder = HyperbolicPositionalEncoding(d_model, max_len, dropout)
    
    # Create input
    batch_size = 32
    x = torch.randn(batch_size, d_model)
    
    # Test forward pass
    out = pos_encoder(x)
    assert out.shape == (batch_size, d_model + 1)  # +1 for homogeneous coordinate
    
    # Test dropout
    pos_encoder.train()
    out_train = pos_encoder(x)
    pos_encoder.eval()
    out_eval = pos_encoder(x)
    assert not torch.allclose(out_train, out_eval)  # Should be different due to dropout
    
    # Test cross-ratio preservation
    if batch_size >= 4:
        cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
        cr_after = compute_cross_ratio(out[0], out[1], out[2], out[3])
        assert torch.allclose(cr_before, cr_after, atol=1e-6)

def test_hyperbolic_transformer_layer():
    """Test HyperbolicTransformerLayer module."""
    # Initialize module
    d_model = 64
    nhead = 4
    dim_feedforward = 256
    dropout = 0.1
    layer = HyperbolicTransformerLayer(d_model, nhead, dim_feedforward, dropout)
    
    # Create input
    num_nodes = 32
    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, d_model)
    
    # Test forward pass
    out = layer(x, edge_index, edge_attr)
    assert out.shape == (num_nodes, d_model)
    
    # Test cross-ratio preservation
    if num_nodes >= 4:
        cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
        cr_after = compute_cross_ratio(out[0], out[1], out[2], out[3])
        assert torch.allclose(cr_before, cr_after, atol=1e-6)
    
    # Test attention mask
    mask = torch.ones(num_nodes, num_nodes)
    out_masked = layer(x, edge_index, edge_attr, mask)
    assert out_masked.shape == (num_nodes, d_model)

def test_hgt_model():
    """Test HGT model."""
    # Initialize model
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    model = HGT(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # Create input
    num_nodes = 32
    x = torch.randn(num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, d_model)
    
    # Test forward pass
    out = model(x, edge_index, edge_attr)
    assert out.shape == (num_nodes, d_model)
    
    # Test cross-ratio preservation
    if num_nodes >= 4:
        cr_before = compute_cross_ratio(x[0], x[1], x[2], x[3])
        cr_after = compute_cross_ratio(out[0], out[1], out[2], out[3])
        assert torch.allclose(cr_before, cr_after, atol=1e-6)
    
    # Test attention mask
    mask = torch.ones(num_nodes, num_nodes)
    out_masked = model(x, edge_index, edge_attr, mask)
    assert out_masked.shape == (num_nodes, d_model)

def test_hgt_gradient_flow():
    """Test gradient flow through HGT model."""
    # Initialize model
    d_model = 64
    model = HGT(d_model=d_model, num_layers=2)
    
    # Create input
    num_nodes = 32
    x = torch.randn(num_nodes, d_model, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, d_model)
    
    # Forward pass
    out = model(x, edge_index, edge_attr)
    
    # Compute loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_hgt_batch_processing():
    """Test HGT model with batched inputs."""
    # Initialize model
    d_model = 64
    model = HGT(d_model=d_model, num_layers=2)
    
    # Create batched input
    batch_size = 4
    num_nodes = 32
    x = torch.randn(batch_size, num_nodes, d_model)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, d_model)
    
    # Process each batch
    outputs = []
    for i in range(batch_size):
        out = model(x[i], edge_index, edge_attr)
        outputs.append(out)
    
    # Stack outputs
    stacked_outputs = torch.stack(outputs)
    assert stacked_outputs.shape == (batch_size, num_nodes, d_model)

def test_hgt_edge_cases():
    """Test HGT model with edge cases."""
    # Initialize model
    d_model = 64
    model = HGT(d_model=d_model, num_layers=2)
    
    # Test with empty graph
    x = torch.randn(0, d_model)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, d_model))
    out = model(x, edge_index, edge_attr)
    assert out.shape == (0, d_model)
    
    # Test with single node
    x = torch.randn(1, d_model)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, d_model))
    out = model(x, edge_index, edge_attr)
    assert out.shape == (1, d_model)
    
    # Test with disconnected graph
    num_nodes = 32
    x = torch.randn(num_nodes, d_model)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, d_model))
    out = model(x, edge_index, edge_attr)
    assert out.shape == (num_nodes, d_model)

def test_hgt_numerical_stability():
    """Test numerical stability of HGT model."""
    # Initialize model
    d_model = 64
    model = HGT(d_model=d_model, num_layers=2)
    
    # Create input with extreme values
    num_nodes = 32
    x = torch.randn(num_nodes, d_model) * 1e6
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, d_model) * 1e6
    
    # Test forward pass
    out = model(x, edge_index, edge_attr)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    # Test with very small values
    x = torch.randn(num_nodes, d_model) * 1e-6
    edge_attr = torch.randn(100, d_model) * 1e-6
    out = model(x, edge_index, edge_attr)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any() 