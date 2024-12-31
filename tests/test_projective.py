import pytest
import torch
import math
from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-10)

def test_quadrance_basic(uhg):
    # Test points with negative norm (proper hyperbolic points)
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    q = uhg.quadrance(a, b)
    # Expected = 1 - (a·b)² / ((a·a)(b·b))
    # a·b = 1*2 + 0*0 - 2*3 = 2 - 6 = -4
    # a·a = 1*1 + 0*0 - 2*2 = 1 - 4 = -3
    # b·b = 2*2 + 0*0 - 3*3 = 4 - 9 = -5
    expected = 1 - (-4)**2 / ((-3)*(-5))
    assert torch.abs(q - expected) < 1e-6, f"Expected {expected}, got {q}"

    # Test orthogonal points (should give 1)
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    q = uhg.quadrance(a, b)
    # a·b = 1*0 + 0*1 - 2*2 = -4
    # a·a = 1*1 + 0*0 - 2*2 = -3
    # b·b = 0*0 + 1*1 - 2*2 = -3
    expected = 1 - (-4)**2 / ((-3)*(-3))
    assert torch.abs(q - expected) < 1e-6, f"Expected {expected}, got {q}"

def test_quadrance_batch(uhg):
    # Test batch computation with proper hyperbolic points
    a = torch.tensor([[1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])  # norms = -3, -2
    b = torch.tensor([[2.0, 0.0, 3.0], [0.0, 1.0, 2.0]])  # norms = -5, -3
    q = uhg.quadrance(a, b)
    assert q.shape == torch.Size([2]), f"Expected shape [2], got {q.shape}"
    
    # Calculate expected values for first pair
    # a·b = 1*2 + 0*0 - 2*3 = 2 - 6 = -4
    # a·a = 1*1 + 0*0 - 2*2 = 1 - 4 = -3
    # b·b = 2*2 + 0*0 - 3*3 = 4 - 9 = -5
    expected0 = 1 - (-4)**2 / ((-3)*(-5))
    
    # Calculate expected values for second pair
    # a·b = 1*0 + 1*1 - 2*2 = 1 - 4 = -3
    # a·a = 1*1 + 1*1 - 2*2 = 2 - 4 = -2
    # b·b = 0*0 + 1*1 - 2*2 = 1 - 4 = -3
    expected1 = 1 - (-3)**2 / ((-2)*(-3))
    
    assert torch.abs(q[0] - expected0) < 1e-6, f"First pair should have quadrance {expected0}"
    assert torch.abs(q[1] - expected1) < 1e-6, f"Second pair should have quadrance {expected1}"

def test_quadrance_normalization(uhg):
    # Test that scaling points doesn't affect quadrance
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    q1 = uhg.quadrance(a, b)
    
    a_scaled = a * 2.0
    b_scaled = b * 3.0
    q2 = uhg.quadrance(a_scaled, b_scaled)
    
    assert torch.abs(q1 - q2) < 1e-6, f"Quadrance should be scale-invariant"

def test_cross_ratio_basic(uhg):
    # Test cross ratio with proper hyperbolic points
    # Using points with negative norms to ensure they're in hyperbolic space
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    c = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    d = torch.tensor([1.0, 1.0, 3.0])  # norm = -7
    
    cr = uhg.cross_ratio(a, b, c, d)
    
    # Cross ratio should be real and finite
    assert not torch.isnan(cr), "Cross ratio should not be NaN"
    assert torch.isreal(cr), "Cross ratio should be real"
    assert torch.isfinite(cr), "Cross ratio should be finite"
    
    # Test scale invariance (essential for ML)
    cr_scaled = uhg.cross_ratio(2*a, 3*b, 4*c, 5*d)
    assert torch.abs(cr - cr_scaled) < 1e-6, "Cross ratio should be scale-invariant"

def test_cross_ratio_batch(uhg):
    # Test batch computation of cross ratios
    a = torch.tensor([[1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])  # norms = -3, -2
    b = torch.tensor([[2.0, 0.0, 3.0], [0.0, 1.0, 2.0]])  # norms = -5, -3
    c = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 2.0]])  # norms = -3, -3
    d = torch.tensor([[1.0, 1.0, 3.0], [-2.0, 0.0, 3.0]])  # norms = -7, -5
    
    cr = uhg.cross_ratio(a, b, c, d)
    assert cr.shape == torch.Size([2]), f"Expected shape [2], got {cr.shape}"
    assert not torch.any(torch.isnan(cr)), "Cross ratios should not be NaN"
    assert torch.all(torch.isreal(cr)), "Cross ratios should be real"

def test_cross_ratio_normalization(uhg):
    # Test that scaling points doesn't affect cross ratio
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    c = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    d = torch.tensor([1.0, 1.0, 3.0])  # norm = -7
    
    cr1 = uhg.cross_ratio(a, b, c, d)
    
    # Scale points
    a_scaled = a * 2.0
    b_scaled = b * 3.0
    c_scaled = c * 4.0
    d_scaled = d * 5.0
    
    cr2 = uhg.cross_ratio(a_scaled, b_scaled, c_scaled, d_scaled)
    assert torch.abs(cr1 - cr2) < 1e-6, "Cross ratio should be scale-invariant" 