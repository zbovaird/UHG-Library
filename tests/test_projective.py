import pytest
import torch
import math
from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-10)

def test_quadrance_basic(uhg):
    # Test points on the same line (should give 0)
    a = torch.tensor([1.0, 0.0, 1.0])
    b = torch.tensor([2.0, 0.0, 2.0])
    q = uhg.quadrance(a, b)
    assert torch.abs(q) < 1e-6, f"Expected 0, got {q}"

    # Test orthogonal points (should give 1)
    a = torch.tensor([1.0, 0.0, 1.0])
    b = torch.tensor([0.0, 1.0, 1.0])
    q = uhg.quadrance(a, b)
    assert torch.abs(q - 1.0) < 1e-6, f"Expected 1, got {q}"

def test_quadrance_null_points(uhg):
    # Test with null points (should give 0)
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 1.0])
    q = uhg.quadrance(a, b)
    assert torch.abs(q) < 1e-6, f"Expected 0, got {q}"

def test_quadrance_batch(uhg):
    # Test batch computation
    a = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    q = uhg.quadrance(a, b)
    assert q.shape == torch.Size([2]), f"Expected shape [2], got {q.shape}"
    assert torch.abs(q[0]) < 1e-6, f"First pair should have quadrance 0"
    assert torch.abs(q[1] - 0.5) < 1e-6, f"Second pair should have quadrance 0.5"

def test_quadrance_normalization(uhg):
    # Test that scaling points doesn't affect quadrance
    a = torch.tensor([1.0, 0.0, 1.0])
    b = torch.tensor([0.0, 1.0, 1.0])
    q1 = uhg.quadrance(a, b)
    
    a_scaled = a * 2.0
    b_scaled = b * 3.0
    q2 = uhg.quadrance(a_scaled, b_scaled)
    
    assert torch.abs(q1 - q2) < 1e-6, f"Quadrance should be scale-invariant" 