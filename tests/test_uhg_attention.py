"""Tests for UHG attention mechanism."""

import pytest
import torch
from uhg.attention import UHGAttentionConfig, UHGMultiHeadAttention
from uhg.projective import ProjectiveUHG

def test_attention_config():
    """Test UHG attention configuration."""
    print("\n=== Testing Attention Config ===")
    
    # Test valid config
    config = UHGAttentionConfig(
        feature_dim=32,
        num_heads=4,
        dropout=0.1
    )
    assert config.head_dim == 8
    print("Valid config passed")
    
    # Test invalid config
    with pytest.raises(AssertionError):
        config = UHGAttentionConfig(
            feature_dim=30,  # Not divisible by num_heads
            num_heads=4
        )
    print("Invalid config caught")

def test_projective_properties():
    """Test that attention preserves projective properties."""
    print("\n=== Testing Projective Properties ===")
    
    # Initialize attention
    config = UHGAttentionConfig(feature_dim=4, num_heads=2)
    attention = UHGMultiHeadAttention(config)
    
    # Create test points in projective space
    points = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 1.0],  # Standard basis point 1
        [0.0, 1.0, 0.0, 0.0, 1.0],  # Standard basis point 2
        [0.0, 0.0, 1.0, 0.0, 1.0],  # Standard basis point 3
        [1.0, 1.0, 1.0, 1.0, 1.0],  # Regular point
    ], dtype=torch.float32)
    
    # Add batch dimension and normalize
    x = points.unsqueeze(0).repeat(2, 1, 1)  # [2, 4, 5]
    x = attention._normalize_projective(x)
    
    # Test projective transformation
    transformed = attention._projective_transform(
        x,
        attention.W_q[0]  # Use first head's transform
    )
    
    # Check homogeneous coordinate normalization
    assert torch.allclose(
        transformed[..., -1],
        torch.ones_like(transformed[..., -1]),
        rtol=1e-5
    )
    print("Homogeneous coordinate normalization passed")
    
    # Check cross-ratio preservation
    cr_before = attention.uhg.cross_ratio(
        x[0, 0],
        x[0, 1],
        x[0, 2],
        x[0, 3]
    )
    cr_after = attention.uhg.cross_ratio(
        transformed[0, 0],
        transformed[0, 1],
        transformed[0, 2],
        transformed[0, 3]
    )
    print(f"Cross-ratio before: {cr_before}")
    print(f"Cross-ratio after: {cr_after}")
    assert torch.allclose(cr_before, cr_after, rtol=1e-3, atol=1e-3)
    print("Cross-ratio preservation passed")

def test_ideal_points():
    """Test ideal points computation."""
    print("\n=== Testing Ideal Points ===")
    
    config = UHGAttentionConfig(feature_dim=4, num_heads=2)
    attention = UHGMultiHeadAttention(config)
    
    # Create a test line
    line = torch.tensor([1.0, 0.0, 0.0, 1.0])
    
    # Get ideal points
    i1, i2 = attention._get_ideal_points(line)
    
    # Check points are on the absolute (last coordinate = 0)
    assert torch.allclose(i1[-1], torch.tensor(0.0))
    assert torch.allclose(i2[-1], torch.tensor(0.0))
    print("Ideal points on absolute")
    
    # Check orthogonality
    assert torch.allclose(
        torch.sum(i1 * i2),
        torch.tensor(0.0),
        atol=1e-6
    )
    print("Ideal points orthogonal")
    
    # Check normalization
    assert torch.allclose(torch.norm(i1), torch.tensor(1.0))
    assert torch.allclose(torch.norm(i2), torch.tensor(1.0))
    print("Ideal points normalized")

def test_attention_scores():
    """Test attention score computation."""
    print("\n=== Testing Attention Scores ===")
    
    config = UHGAttentionConfig(feature_dim=4, num_heads=2)
    attention = UHGMultiHeadAttention(config)
    
    # Create test queries and keys
    batch_size = 2
    num_queries = 3
    num_keys = 4
    
    q = torch.randn(batch_size, num_queries, config.feature_dim + 1)
    k = torch.randn(batch_size, num_keys, config.feature_dim + 1)
    
    # Test cross-ratio based attention
    scores_cr = attention._compute_attention_scores(q, k)
    assert scores_cr.shape == (batch_size, num_queries, num_keys)
    assert not torch.any(torch.isnan(scores_cr))
    print("Cross-ratio attention scores passed")
    
    # Test distance based attention
    config.use_cross_ratio = False
    attention = UHGMultiHeadAttention(config)
    scores_dist = attention._compute_attention_scores(q, k)
    assert scores_dist.shape == (batch_size, num_queries, num_keys)
    assert not torch.any(torch.isnan(scores_dist))
    print("Distance-based attention scores passed")

def test_full_attention():
    """Test full attention mechanism."""
    print("\n=== Testing Full Attention ===")
    
    config = UHGAttentionConfig(
        feature_dim=8,
        num_heads=2,
        dropout=0.0  # Disable dropout for testing
    )
    attention = UHGMultiHeadAttention(config)
    
    # Create test inputs
    batch_size = 2
    num_queries = 3
    num_keys = 4
    
    query = torch.randn(batch_size, num_queries, config.feature_dim)
    key = torch.randn(batch_size, num_keys, config.feature_dim)
    value = torch.randn(batch_size, num_keys, config.feature_dim)
    
    # Test forward pass
    output, weights = attention(query, key, value)
    
    # Check shapes
    assert output.shape == (batch_size, num_queries, config.feature_dim)
    assert weights.shape == (batch_size, config.num_heads, num_queries, num_keys)
    print("Output shapes correct")
    
    # Check attention weights sum to 1
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(batch_size, config.num_heads, num_queries),
        rtol=1e-5
    )
    print("Attention weights normalized")
    
    # Test with mask
    mask = torch.ones(batch_size, num_queries, num_keys)
    mask[:, :, -1] = 0  # Mask out last key
    output_masked, weights_masked = attention(query, key, value, mask)
    
    # Check masked weights are zero
    assert torch.allclose(
        weights_masked[:, :, :, -1],
        torch.zeros_like(weights_masked[:, :, :, -1])
    )
    print("Attention masking passed")

def test_geometric_properties():
    """Test geometric properties of attention mechanism."""
    print("\n=== Testing Geometric Properties ===")
    
    config = UHGAttentionConfig(feature_dim=4, num_heads=2)
    attention = UHGMultiHeadAttention(config)
    
    # Create test points along a geodesic
    t = torch.linspace(-0.8, 0.8, 5)
    x = 0.5 * t  # Points in Klein model
    y = torch.zeros_like(t)
    points = torch.stack([x, y, torch.ones_like(x)], dim=1)
    
    # Add batch dimension
    points = points.unsqueeze(0)  # [1, 5, 3]
    
    # Apply attention
    output, _ = attention(points, points, points)
    
    # Check output preserves geometric structure
    for i in range(len(points)-2):
        p1 = output[0, i]
        p2 = output[0, i+1]
        p3 = output[0, i+2]
        
        # Check points still form geodesic (approximately)
        d12 = attention.uhg.distance(p1, p2)
        d23 = attention.uhg.distance(p2, p3)
        d13 = attention.uhg.distance(p1, p3)
        
        # Points on geodesic should approximately satisfy additivity
        rel_error = torch.abs(d13 - (d12 + d23)) / (d13 + attention.config.eps)
        assert rel_error < 0.1, "Points too far from geodesic"
    
    print("Geometric structure preserved")

if __name__ == "__main__":
    print("Testing UHG Attention Mechanisms\n")
    
    try:
        test_attention_config()
        test_projective_properties()
        test_ideal_points()
        test_attention_scores()
        test_full_attention()
        test_geometric_properties()
        print("\n✅ All attention tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}") 