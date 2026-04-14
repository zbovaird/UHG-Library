import torch
import pytest
import numpy as np
from uhg.utils.metrics import (
    minkowski_inner_product,
    projective_normalize,
    uhg_inner_product,
    uhg_norm,
    uhg_quadrance,
    uhg_quadrance_vectorized,
    uhg_spread,
    verify_uhg_constraints,
)

# Constants for testing
BATCH_SIZES = [1, 10, 100, 1000]
FEATURE_DIMS = [3, 8, 16, 32]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("feature_dim", FEATURE_DIMS)
def test_uhg_inner_product_vectorized(batch_size, feature_dim):
    """Test that uhg_inner_product handles vectorization correctly."""
    # Create random tensors with batch dimension
    a = torch.randn(batch_size, feature_dim, device=DEVICE)
    b = torch.randn(batch_size, feature_dim, device=DEVICE)
    
    # Add homogeneous coordinate
    a = torch.cat([a, torch.ones(batch_size, 1, device=DEVICE)], dim=1)
    b = torch.cat([b, torch.ones(batch_size, 1, device=DEVICE)], dim=1)
    
    # Compute vectorized inner product
    result = uhg_inner_product(a, b)
    
    # Check shape
    assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    # Check against loop implementation for correctness
    loop_result = torch.zeros(batch_size, device=DEVICE)
    for i in range(batch_size):
        loop_result[i] = uhg_inner_product(a[i:i+1], b[i:i+1]).squeeze()
    
    assert torch.allclose(result, loop_result, rtol=1e-5, atol=1e-5), \
        f"Vectorized result doesn't match loop implementation"
    
    # Test broadcasting
    c = torch.randn(1, feature_dim + 1, device=DEVICE)  # Single vector with homogeneous coordinate
    broadcast_result = uhg_inner_product(a, c)
    
    assert broadcast_result.shape == (batch_size,), \
        f"Expected broadcasting to shape ({batch_size},), got {broadcast_result.shape}"

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("feature_dim", FEATURE_DIMS)
def test_uhg_norm_vectorized(batch_size, feature_dim):
    """Test that uhg_norm handles vectorization correctly."""
    # Create random tensors with batch dimension
    a = torch.randn(batch_size, feature_dim, device=DEVICE)
    
    # Add homogeneous coordinate
    a = torch.cat([a, torch.ones(batch_size, 1, device=DEVICE)], dim=1)
    
    # Compute vectorized norm
    result = uhg_norm(a)
    
    # Check shape
    assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    # Check against loop implementation for correctness
    loop_result = torch.zeros(batch_size, device=DEVICE)
    for i in range(batch_size):
        loop_result[i] = uhg_norm(a[i:i+1]).squeeze()
    
    assert torch.allclose(result, loop_result, rtol=1e-5, atol=1e-5), \
        f"Vectorized result doesn't match loop implementation"

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("feature_dim", FEATURE_DIMS)
def test_uhg_quadrance_vectorized(batch_size, feature_dim):
    """Test that uhg_quadrance handles vectorization correctly."""
    # Create random tensors with batch dimension
    a = torch.randn(batch_size, feature_dim, device=DEVICE)
    b = torch.randn(batch_size, feature_dim, device=DEVICE)
    
    # Add homogeneous coordinate
    a = torch.cat([a, torch.ones(batch_size, 1, device=DEVICE)], dim=1)
    b = torch.cat([b, torch.ones(batch_size, 1, device=DEVICE)], dim=1)
    
    # Compute vectorized quadrance
    result = uhg_quadrance_vectorized(a, b)
    
    # Check shape
    assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    # Check against loop implementation for correctness
    loop_result = torch.zeros(batch_size, device=DEVICE)
    for i in range(batch_size):
        loop_result[i] = uhg_quadrance_vectorized(a[i:i+1], b[i:i+1]).squeeze()
    
    assert torch.allclose(result, loop_result, rtol=1e-5, atol=1e-5), \
        f"Vectorized result doesn't match loop implementation"
    
    # Test broadcasting
    c = torch.randn(1, feature_dim + 1, device=DEVICE)  # Single vector with homogeneous coordinate
    broadcast_result = uhg_quadrance_vectorized(a, c)
    
    assert broadcast_result.shape == (batch_size,), \
        f"Expected broadcasting to shape ({batch_size},), got {broadcast_result.shape}"

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("feature_dim", FEATURE_DIMS)
def test_uhg_spread_vectorized(batch_size, feature_dim):
    """Test that uhg_spread handles vectorization correctly."""
    # Create random tensors with batch dimension (representing lines)
    L = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
    M = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
    
    # Compute vectorized spread
    result = uhg_spread(L, M)
    
    # Check shape
    assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    # Check against loop implementation for correctness
    loop_result = torch.zeros(batch_size, device=DEVICE)
    for i in range(batch_size):
        loop_result[i] = uhg_spread(L[i:i+1], M[i:i+1]).squeeze()
    
    assert torch.allclose(result, loop_result, rtol=1e-5, atol=1e-5), \
        f"Vectorized result doesn't match loop implementation"
    
    # Test broadcasting
    N = torch.randn(1, feature_dim + 1, device=DEVICE)  # Single line
    broadcast_result = uhg_spread(L, N)
    
    assert broadcast_result.shape == (batch_size,), \
        f"Expected broadcasting to shape ({batch_size},), got {broadcast_result.shape}"

@pytest.mark.parametrize("batch_size", [1000, 10000])
def test_performance_improvement(batch_size):
    """Test that vectorized operations are faster than loop implementations."""
    import time
    
    feature_dim = 16
    a = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
    b = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
    
    # Time vectorized implementation
    start = time.time()
    _ = uhg_quadrance(a, b)
    vectorized_time = time.time() - start
    
    # Time loop implementation
    start = time.time()
    loop_result = torch.zeros(batch_size, device=DEVICE)
    for i in range(batch_size):
        loop_result[i] = uhg_quadrance(a[i:i+1], b[i:i+1]).squeeze()
    loop_time = time.time() - start
    
    # Vectorized should be significantly faster
    speedup = loop_time / vectorized_time
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Should be at least 5x faster for large batches
    assert speedup > 5, f"Expected at least 5x speedup, got {speedup:.2f}x"


def test_projective_normalize_places_points_on_hyperboloid():
    x = torch.tensor([[1.0, 2.0, 1.0], [-0.5, 0.25, 1.0]], device=DEVICE)
    x_norm = projective_normalize(x)
    norms = minkowski_inner_product(x_norm, x_norm)
    assert torch.allclose(norms, -torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_verify_uhg_constraints_reports_small_violation_for_normalized_points():
    x = torch.randn(16, 5, device=DEVICE)
    x = torch.cat([x, torch.ones(16, 1, device=DEVICE)], dim=1)
    x_norm = projective_normalize(x)
    stats = verify_uhg_constraints(x_norm)
    assert stats["max_violation"] < 1e-5
    assert stats["mean_violation"] < 1e-6

if __name__ == "__main__":
    # Run tests
    test_uhg_inner_product_vectorized(100, 16)
    test_uhg_norm_vectorized(100, 16)
    test_uhg_quadrance_vectorized(100, 16)
    test_uhg_spread_vectorized(100, 16)
    test_performance_improvement(1000)
    print("All tests passed!") 