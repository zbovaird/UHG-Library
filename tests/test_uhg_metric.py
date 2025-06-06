"""
Tests for UHG metric operations.
"""

import pytest
import torch
from uhg.metric import UHGMetric

@pytest.fixture
def metric():
    """Create a UHGMetric instance for testing."""
    return UHGMetric(eps=1e-10)

def test_quadrance(metric):
    """Test quadrance calculation between points."""
    # Test points in projective space
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    
    # Calculate quadrance
    quad = metric.quadrance(p1, p2)
    
    # Points are orthogonal, so quadrance should be 1
    assert torch.isclose(quad, torch.tensor(1.0))
    
    # Test with same point
    p3 = torch.tensor([1.0, 1.0, 1.0])
    quad_same = metric.quadrance(p3, p3)
    
    # Quadrance between same point should be close to 0
    assert torch.isclose(quad_same, torch.tensor(0.0), atol=1e-6)

def test_spread(metric):
    """Test spread calculation between three points."""
    # Test points forming a right angle
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 0.0, 1.0])
    p3 = torch.tensor([0.0, 1.0, 0.0])
    
    # Calculate spread
    spread = metric.spread(p1, p2, p3)
    
    # For right angle, spread should be 1
    assert torch.isclose(spread, torch.tensor(1.0))
    
    # Test with collinear points
    p4 = torch.tensor([2.0, 0.0, 0.0])
    spread_collinear = metric.spread(p1, p2, p4)
    
    # Spread should be 0 for collinear points
    assert torch.isclose(spread_collinear, torch.tensor(0.0))

def test_distance(metric):
    """Test distance calculation between points."""
    # Test points in projective space
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    
    # Calculate distance
    dist = metric.distance(p1, p2)
    
    # Distance should be sqrt of quadrance
    expected_dist = torch.sqrt(torch.tensor(1.0))
    assert torch.isclose(dist, expected_dist)

def test_cross_ratio(metric):
    """Test cross-ratio calculation of four points."""
    # Test points in projective space
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    p3 = torch.tensor([0.0, 0.0, 1.0])
    p4 = torch.tensor([1.0, 1.0, 1.0])
    
    # Calculate cross-ratio
    cr = metric.cross_ratio(p1, p2, p3, p4)
    
    # Cross-ratio should be invariant under projective transformations
    assert cr > 0.0

def test_is_collinear(metric):
    """Test collinearity check for three points."""
    # Test collinear points
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([2.0, 0.0, 0.0])
    p3 = torch.tensor([3.0, 0.0, 0.0])
    
    # Check collinearity
    is_collinear = metric.is_collinear(p1, p2, p3)
    assert is_collinear
    
    # Test non-collinear points
    p4 = torch.tensor([0.0, 1.0, 0.0])
    is_not_collinear = metric.is_collinear(p1, p2, p4)
    assert not is_not_collinear 