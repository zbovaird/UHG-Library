"""
Tests for UHG neural layers.
"""

import pytest
import torch
import math
import logging
from uhg.layers import (
    UHGLinear, UHGConv, UHGAttention,
    UHGTransformer, UHGMultiheadAttention, UHGLayerNorm
)

@pytest.fixture
def batch_size():
    """Return batch size for testing."""
    return 4

@pytest.fixture
def in_features():
    """Return input feature dimension for testing."""
    return 3

@pytest.fixture
def out_features():
    """Return output feature dimension for testing."""
    return 2

@pytest.fixture
def num_nodes():
    """Return number of nodes for graph testing."""
    return 5

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def d_model():
    return 64

def test_uhg_linear(batch_size, in_features, out_features):
    """Test UHG linear layer."""
    # Create layer
    layer = UHGLinear(in_features, out_features)
    
    # Create input
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    out = layer(x)
    
    # Check output shape
    assert out.shape == (batch_size, out_features)
    
    # Check that output lies on unit sphere
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Test with zero input
    x_zero = torch.zeros(batch_size, in_features)
    out_zero = layer(x_zero)
    assert torch.allclose(torch.norm(out_zero, dim=-1), 
                         torch.ones_like(out_zero[:, 0]), atol=1e-6)

def test_uhg_conv(batch_size, in_features, out_features, num_nodes):
    """Test UHG graph convolution layer."""
    # Create layer
    layer = UHGConv(in_features, out_features)
    
    # Create input features and edge index
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Simple path graph
    
    # Forward pass
    out = layer(x, edge_index)
    
    # Check output shape
    assert out.shape == (num_nodes, out_features)
    
    # Check that output lies on unit sphere
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Test with empty graph
    x_empty = torch.randn(num_nodes, in_features)
    edge_index_empty = torch.zeros((2, 0), dtype=torch.long)
    out_empty = layer(x_empty, edge_index_empty)
    assert out_empty.shape == (num_nodes, out_features)

def test_uhg_attention(batch_size, in_features):
    """Test UHG attention layer."""
    # Create layer
    layer = UHGAttention(in_features, heads=2)
    
    # Create input
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    out = layer(x)
    
    # Check output shape
    assert out.shape == (batch_size, in_features)
    
    # Check that output lies on unit sphere
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Test with attention mask
    mask = torch.ones(batch_size, batch_size)
    mask[0, 1:] = 0  # Mask out some attention
    out_masked = layer(x, mask)
    assert out_masked.shape == (batch_size, in_features)
    
    # Test with zero input
    x_zero = torch.zeros(batch_size, in_features)
    out_zero = layer(x_zero)
    assert torch.allclose(torch.norm(out_zero, dim=-1), 
                         torch.ones_like(out_zero[:, 0]), atol=1e-6)

def test_uhg_linear_gradients(batch_size, in_features, out_features):
    """Test gradient flow through UHG linear layer."""
    layer = UHGLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    
    # Forward pass
    out = layer(x)
    
    # Compute loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None

def test_uhg_conv_gradients(batch_size, in_features, out_features, num_nodes):
    """Test gradient flow through UHG graph convolution layer."""
    layer = UHGConv(in_features, out_features)
    x = torch.randn(num_nodes, in_features, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    
    # Forward pass
    out = layer(x, edge_index)
    
    # Compute loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert layer.weight.grad is not None
    if layer.bias is not None:
        assert layer.bias.grad is not None

def test_uhg_attention_gradients(batch_size, in_features):
    """Test gradient flow through UHG attention layer."""
    layer = UHGAttention(in_features, heads=2)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    
    # Forward pass
    out = layer(x)
    
    # Compute loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert layer.query.grad is not None
    assert layer.key.grad is not None
    assert layer.value.grad is not None

def test_uhg_transformer(seq_len, batch_size, d_model):
    """Test UHG transformer layer."""
    # Create layer
    layer = UHGTransformer(d_model, nhead=8)
    
    # Create input
    x = torch.randn(seq_len, batch_size, d_model)
    
    # Forward pass
    out = layer(x)
    
    # Check output shape
    assert out.shape == (seq_len, batch_size, d_model)
    
    # Check that output lies on unit sphere
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Test with attention mask
    mask = torch.ones(seq_len, seq_len)
    mask[0, 1:] = 0  # Mask out some attention
    out_masked = layer(x, src_mask=mask)
    assert out_masked.shape == (seq_len, batch_size, d_model)
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -2:] = True  # Mask last two positions
    out_padded = layer(x, src_key_padding_mask=padding_mask)
    assert out_padded.shape == (seq_len, batch_size, d_model)

def test_uhg_multihead_attention(seq_len, batch_size, d_model):
    """Test UHG multi-head attention."""
    # Create layer
    layer = UHGMultiheadAttention(d_model, num_heads=8)
    
    # Create input
    query = torch.randn(seq_len, batch_size, d_model)
    key = torch.randn(seq_len, batch_size, d_model)
    value = torch.randn(seq_len, batch_size, d_model)
    
    # Forward pass
    out, attn_weights = layer(query, key, value)
    
    # Check output shape
    assert out.shape == (seq_len, batch_size, d_model)
    assert attn_weights.shape == (seq_len, 8, seq_len, seq_len)
    
    # Check that output lies on unit sphere
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Test with attention mask
    mask = torch.ones(seq_len, seq_len)
    mask[0, 1:] = 0  # Mask out some attention
    out_masked, attn_weights_masked = layer(query, key, value, attn_mask=mask)
    assert out_masked.shape == (seq_len, batch_size, d_model)
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -2:] = True  # Mask last two positions
    out_padded, attn_weights_padded = layer(query, key, value, key_padding_mask=padding_mask)
    assert out_padded.shape == (seq_len, batch_size, d_model)

def test_uhg_layer_norm(seq_len, batch_size, d_model):
    """Test UHG layer normalization."""
    # Create layer
    layer = UHGLayerNorm(d_model)
    
    # Create input
    x = torch.randn(seq_len, batch_size, d_model)
    
    # Forward pass
    out = layer(x)
    
    # Check output shape
    assert out.shape == (seq_len, batch_size, d_model)
    
    # Check that output is normalized
    mean = torch.mean(out, dim=-1)
    var = torch.var(out, dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)  # Relaxed tolerance for variance
    
    # Test with zero input
    x_zero = torch.zeros(seq_len, batch_size, d_model)
    out_zero = layer(x_zero)
    assert out_zero.shape == (seq_len, batch_size, d_model)
    
    # Test gradient flow
    x = torch.randn(seq_len, batch_size, d_model, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None 