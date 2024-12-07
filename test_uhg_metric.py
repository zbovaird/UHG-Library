"""Tests for UHG metric learning module."""

import torch
import pytest
import os

from uhg_metric import UHGMetricConfig, UHGMetricLearner

def test_metric_initialization():
    """Test initialization of metric learner."""
    config = UHGMetricConfig(feature_dim=10)
    model = UHGMetricLearner(config)
    
    assert model.config.feature_dim == 10
    assert model.config.embedding_dim == 32
    assert isinstance(model.proj_matrix, torch.nn.Parameter)
    assert model.proj_matrix.shape == (33, 33)  # embedding_dim + 1 for homogeneous coord
    assert isinstance(model.ref_points, torch.nn.Parameter)
    assert model.ref_points.shape == (3, 33)  # 3 reference points

def test_projective_normalization():
    """Test projective normalization."""
    config = UHGMetricConfig(feature_dim=5)
    model = UHGMetricLearner(config)
    
    # Test points with different scales
    x = torch.randn(10, 6)  # 5 + 1 for homogeneous coordinate
    normalized = model._normalize_projective(x)
    
    # Check that last coordinate is 1
    assert torch.allclose(normalized[..., -1], torch.ones(10))
    
    # Check that projective equivalence is preserved
    scale = torch.randn(10, 1)
    scaled_x = x * scale
    scaled_normalized = model._normalize_projective(scaled_x)
    assert torch.allclose(normalized, scaled_normalized, atol=1e-5)

def test_cross_ratio():
    """Test cross-ratio computation."""
    config = UHGMetricConfig(feature_dim=3)
    model = UHGMetricLearner(config)
    
    # Create projective points
    a = torch.tensor([1.0, 0.0, 0.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    c = torch.tensor([0.0, 0.0, 1.0, 1.0])
    d = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    # Test cross-ratio
    cr = model._cross_ratio(a, b, c, d)
    
    # Cross-ratio should be invariant under projective transformations
    scale = torch.randn(4)
    scaled_a = a * scale[0]
    scaled_b = b * scale[1]
    scaled_c = c * scale[2]
    scaled_d = d * scale[3]
    scaled_cr = model._cross_ratio(scaled_a, scaled_b, scaled_c, scaled_d)
    assert torch.allclose(cr, scaled_cr, atol=1e-5)

def test_projective_distance():
    """Test projective distance computation."""
    config = UHGMetricConfig(feature_dim=4)
    model = UHGMetricLearner(config)
    
    # Create points
    x = torch.tensor([0.1, 0.0, 0.0, 1.0])
    y = torch.tensor([0.2, 0.0, 0.0, 1.0])
    
    # Compute distance
    dist = model._projective_distance(x, y)
    
    # Distance should be positive
    assert dist > 0
    
    # Distance should be symmetric
    dist_reverse = model._projective_distance(y, x)
    assert torch.allclose(dist, dist_reverse, atol=1e-5)
    
    # Distance should be projectively invariant
    scale = torch.randn(2)
    scaled_x = x * scale[0]
    scaled_y = y * scale[1]
    scaled_dist = model._projective_distance(scaled_x, scaled_y)
    assert torch.allclose(dist, scaled_dist, atol=1e-5)

def test_forward_pass():
    """Test forward pass through the model."""
    config = UHGMetricConfig(feature_dim=5)
    model = UHGMetricLearner(config)
    
    # Create batch of inputs
    x = torch.randn(32, 5)
    
    # Forward pass
    output = model(x)
    
    # Check output shape (includes homogeneous coordinate)
    assert output.shape == (32, config.embedding_dim + 1)
    
    # Check projective normalization
    assert torch.allclose(output[..., -1], torch.ones(32))
    
    # Check projective transformation is applied correctly
    transformed = torch.matmul(
        torch.cat([x, torch.ones(32, 1)], dim=-1),
        model.proj_matrix
    )
    normalized = transformed / transformed[..., -1:]
    assert torch.allclose(output, normalized, atol=1e-5)

def test_distance_computation():
    """Test distance computation between points."""
    config = UHGMetricConfig(feature_dim=4)
    model = UHGMetricLearner(config)
    
    # Create points
    x = torch.randn(16, 4)
    y = torch.randn(16, 4)
    
    # Compute distances
    distances = model.compute_distance(x, y)
    
    # Check properties
    assert distances.shape == (16,)
    assert torch.all(distances >= 0)
    assert not torch.any(torch.isnan(distances))
    
    # Check projective invariance
    scale = torch.randn(16, 1)
    scaled_x = x * scale
    scaled_y = y * scale
    scaled_distances = model.compute_distance(scaled_x, scaled_y)
    assert torch.allclose(distances, scaled_distances, atol=1e-5)

def test_training_step():
    """Test single training step."""
    config = UHGMetricConfig(feature_dim=6)
    model = UHGMetricLearner(config)
    
    # Create triplet batch
    anchor = torch.randn(8, 6)
    positive = anchor + 0.1 * torch.randn(8, 6)  # Similar to anchor
    negative = torch.randn(8, 6)  # Different from anchor
    
    # Get initial parameters
    init_proj = model.proj_matrix.clone()
    init_ref = model.ref_points.clone()
    
    # Perform training step
    metrics = model.train_step(anchor, positive, negative)
    
    # Check metrics
    assert 'total_loss' in metrics
    assert 'triplet_loss' in metrics
    assert 'cr_loss' in metrics
    assert metrics['pos_dist'] < metrics['neg_dist']
    
    # Check parameters were updated
    assert not torch.allclose(init_proj, model.proj_matrix)
    assert not torch.allclose(init_ref, model.ref_points)

def test_save_load():
    """Test model state saving and loading."""
    config = UHGMetricConfig(feature_dim=4)
    model = UHGMetricLearner(config)
    
    # Create some data
    x = torch.randn(10, 4)
    y = torch.randn(10, 4)
    
    # Get initial distances
    initial_distances = model.compute_distance(x, y)
    
    # Save model
    model.save_state('test_model.pt')
    
    # Load model
    loaded_model = UHGMetricLearner.load_state('test_model.pt')
    
    # Compare distances
    loaded_distances = loaded_model.compute_distance(x, y)
    assert torch.allclose(initial_distances, loaded_distances)
    
    # Clean up
    os.remove('test_model.pt')

if __name__ == '__main__':
    pytest.main([__file__]) 