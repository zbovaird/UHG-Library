"""
Tests for UHG Anomaly Detection module.
"""

import torch
import numpy as np
import pytest
from uhg_anomaly import UHGAnomalyConfig, UHGAnomalyScorer, create_anomaly_scorer

def test_anomaly_scorer_initialization():
    """Test proper initialization of anomaly scorer."""
    config = UHGAnomalyConfig(feature_dim=10)
    scorer = UHGAnomalyScorer(config)
    
    assert scorer.config.feature_dim == 10
    assert scorer.config.window_size == 100
    assert scorer.config.use_cross_ratio == True
    assert not scorer.is_baseline_ready

def test_baseline_statistics():
    """Test computation of baseline statistics."""
    feature_dim = 5
    window_size = 10
    scorer = create_anomaly_scorer(
        feature_dim=feature_dim,
        window_size=window_size
    )
    
    # Create normal baseline data
    baseline_data = torch.randn(window_size, feature_dim)
    
    # Update baseline
    for features in baseline_data:
        scorer._update_baseline(features)
    
    assert scorer.is_baseline_ready
    assert scorer.mean_baseline is not None
    assert scorer.std_baseline_score is not None
    assert scorer.threshold is not None

def test_cross_ratio_preservation():
    """Test if cross-ratio is preserved in anomaly scoring."""
    feature_dim = 3
    scorer = create_anomaly_scorer(feature_dim=feature_dim)
    
    # Create baseline with known geometric structure
    baseline = torch.tensor([
        [1.0, 0.0, 0.0],  # Point on x-axis
        [0.0, 1.0, 0.0],  # Point on y-axis
        [0.0, 0.0, 1.0],  # Point on z-axis
        [1.0, 1.0, 1.0]   # Point on diagonal
    ])
    
    # Update baseline
    for features in baseline:
        scorer._update_baseline(features)
    
    # Test point
    x = torch.tensor([0.5, 0.5, 0.5])
    
    # Compute cross-ratio score
    cr_score = scorer._compute_cross_ratio_score(x)
    
    # Cross-ratio should be finite and non-negative
    assert torch.isfinite(cr_score)
    assert cr_score >= 0

def test_anomaly_detection():
    """Test anomaly detection on synthetic data."""
    feature_dim = 5
    window_size = 20
    scorer = create_anomaly_scorer(
        feature_dim=feature_dim,
        window_size=window_size
    )
    
    # Create normal baseline data
    baseline_data = torch.randn(window_size, feature_dim)
    
    # Update baseline
    for features in baseline_data:
        scorer._update_baseline(features)
    
    # Test normal point (should not be anomalous)
    normal_point = torch.randn(feature_dim)
    normal_result = scorer.compute_score(normal_point)
    
    # Test anomalous point (significantly different from baseline)
    anomalous_point = 10.0 * torch.ones(feature_dim)
    anomalous_result = scorer.compute_score(anomalous_point)
    
    # Normal point should have lower score than anomalous point
    assert normal_result['score'] < anomalous_result['score']
    assert not normal_result['is_anomaly']
    assert anomalous_result['is_anomaly']

def test_geometric_deviation():
    """Test geometric deviation computation."""
    feature_dim = 3
    scorer = create_anomaly_scorer(feature_dim=feature_dim)
    
    # Create points with known geometric relationship
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    
    # Points should have non-zero deviation
    dev = scorer._compute_deviation(
        torch.cat([p1, torch.ones(1)]),
        torch.cat([p2, torch.ones(1)])
    )
    
    assert dev > 0
    assert torch.isfinite(dev)

def test_uhg_mean():
    """Test UHG mean computation."""
    feature_dim = 3
    scorer = create_anomaly_scorer(feature_dim=feature_dim)
    
    # Create points in general position
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Add homogeneous coordinates
    points_h = torch.cat([
        points,
        torch.ones(3, 1)
    ], dim=1)
    
    # Compute UHG mean
    mean = scorer._uhg_mean(points_h)
    
    # Mean should be normalized and finite
    assert torch.allclose(torch.norm(mean), torch.tensor(1.0), rtol=1e-5)
    assert torch.all(torch.isfinite(mean))

def test_real_traffic_patterns():
    """Test anomaly detection with realistic traffic patterns."""
    feature_dim = 10
    window_size = 50
    scorer = create_anomaly_scorer(
        feature_dim=feature_dim,
        window_size=window_size
    )
    
    # Create baseline of normal traffic
    baseline = torch.rand(window_size, feature_dim) * 0.5  # Normal range
    
    # Update baseline
    for features in baseline:
        scorer._update_baseline(features)
    
    # Test cases
    test_cases = [
        # Normal traffic (should not be anomalous)
        (torch.rand(feature_dim) * 0.5, False),
        
        # Volume-based anomaly (high values)
        (torch.ones(feature_dim) * 2.0, True),
        
        # Pattern-based anomaly (unusual distribution)
        (torch.cat([torch.zeros(5), torch.ones(5) * 2.0]), True),
        
        # Subtle anomaly (slightly outside normal range)
        (torch.rand(feature_dim) * 0.8, False)
    ]
    
    for features, expected_anomaly in test_cases:
        result = scorer.compute_score(features)
        assert result['is_anomaly'] == expected_anomaly

if __name__ == "__main__":
    pytest.main([__file__]) 