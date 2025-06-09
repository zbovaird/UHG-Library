"""
Tests for UHG metric operations.

These tests verify that all metric operations strictly follow UHG principles:
- Using projective geometry
- Working with cross-ratios
- No differential geometry
- No curvature parameters
- Pure projective operations

References:
    - UHG.pdf Chapter 3: Projective Geometry
    - UHG.pdf Chapter 4: Cross-ratios and Invariants
    - UHG.pdf Chapter 5: The Fundamental Operations
"""

import pytest
import torch
import numpy as np
from uhg.metrics import UHGMetric as HyperbolicMetric
from uhg.projective import ProjectiveUHG

@pytest.fixture
def metric():
    """Create a HyperbolicMetric instance with small epsilon for numerical stability."""
    return HyperbolicMetric(epsilon=1e-10)

@pytest.fixture
def projective():
    """Create a ProjectiveUHG instance with small epsilon for numerical stability."""
    return ProjectiveUHG(epsilon=1e-10)

def test_quadrance_basic(metric, projective):
    """Test basic quadrance calculations."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    q = metric.quadrance(a, b)
    assert isinstance(q, torch.Tensor)
    assert q.shape == ()
    assert 0 < q < 1

def test_quadrance_symmetry(metric, projective):
    """Test that quadrance is symmetric."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    q1 = metric.quadrance(a, b)
    q2 = metric.quadrance(b, a)
    assert torch.allclose(q1, q2)

def test_quadrance_triangle_inequality(metric, projective):
    """Test triangle inequality for quadrance."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    q1 = metric.quadrance(a, b)
    q2 = metric.quadrance(b, c)
    q3 = metric.quadrance(a, c)
    assert q3 <= q1 + q2 - q1*q2

def test_spread_basic(metric, projective):
    """Test basic spread calculations."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    S = metric.spread(a, b, c)
    assert isinstance(S, torch.Tensor)
    assert S.shape == ()
    assert 0 <= S <= 1

def test_spread_symmetry(metric, projective):
    """Test that spread is symmetric."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    S1 = metric.spread(a, b, c)
    S2 = metric.spread(a, c, b)
    assert torch.allclose(S1, S2)

def test_cross_ratio_invariance(metric, projective):
    """Test that cross-ratio is preserved under projective transformations."""
    v1 = torch.tensor([0.3, 0.4, 1.0])
    v2 = torch.tensor([-0.2, 0.5, 1.0])
    u1 = torch.tensor([0.1, -0.3, 1.0])
    u2 = torch.tensor([-0.4, -0.2, 1.0])
    cr1 = metric.cross_ratio(v1, v2, u1, u2)
    matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])
    v1_t = metric.transform(v1, matrix)
    v2_t = metric.transform(v2, matrix)
    u1_t = metric.transform(u1, matrix)
    u2_t = metric.transform(u2, matrix)
    cr2 = metric.cross_ratio(v1_t, v2_t, u1_t, u2_t)
    assert torch.allclose(cr1, cr2)

def test_pythagoras(metric, projective):
    """Test Pythagoras' theorem in UHG."""
    # Create a right triangle
    q1 = torch.tensor(0.5)  # First leg
    q2 = torch.tensor(0.3)  # Second leg
    
    # Calculate expected hypotenuse
    expected_q3 = metric.pythagoras(q1, q2)
    
    # Verify the theorem
    assert metric.pythagoras(q1, q2, expected_q3)

def test_dual_pythagoras(metric, projective):
    """Test dual Pythagoras' theorem in UHG."""
    # Create a right triangle
    S1 = torch.tensor(0.5)  # First leg spread
    S2 = torch.tensor(0.3)  # Second leg spread
    
    # Calculate expected hypotenuse spread
    expected_S3 = metric.dual_pythagoras(S1, S2)
    
    # Verify the theorem
    assert metric.dual_pythagoras(S1, S2, expected_S3)

def test_triple_quad_formula(metric, projective):
    """Test triple quad formula."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    q1 = metric.quadrance(a, b)
    q2 = metric.quadrance(b, c)
    q3 = metric.quadrance(a, c)
    assert metric.triple_quad_formula(q1, q2, q3)

def test_triple_spread_formula(metric, projective):
    """Test triple spread formula."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    S1 = metric.spread(a, b, c)
    S2 = metric.spread(b, c, a)
    S3 = metric.spread(c, a, b)
    assert metric.triple_spread_formula(S1, S2, S3)

def test_cross_law(metric, projective):
    """Test cross law."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    q1 = metric.quadrance(a, b)
    q2 = metric.quadrance(b, c)
    q3 = metric.quadrance(a, c)
    S1 = metric.spread(a, b, c)
    S2 = metric.spread(b, c, a)
    S3 = metric.spread(c, a, b)
    assert metric.cross_law(q1, q2, q3, S1, S2, S3)

def test_cross_dual_law(metric, projective):
    """Test dual cross law."""
    a = torch.tensor([0.3, 0.4, 1.0])
    b = torch.tensor([-0.2, 0.5, 1.0])
    c = torch.tensor([0.1, -0.3, 1.0])
    q1 = metric.quadrance(a, b)
    S1 = metric.spread(a, b, c)
    S2 = metric.spread(b, c, a)
    S3 = metric.spread(c, a, b)
    assert metric.cross_dual_law(S1, S2, S3, q1)

def test_batch_operations(metric, projective):
    """Test operations on batches of points."""
    a = torch.tensor([
        [0.3, 0.4, 1.0],
        [-0.2, 0.5, 1.0]
    ])
    b = torch.tensor([
        [-0.2, 0.5, 1.0],
        [0.1, -0.3, 1.0]
    ])
    q = metric.quadrance(a, b)
    assert isinstance(q, torch.Tensor)
    assert q.shape == (2,)
    assert torch.all((q > 0) & (q < 1))

def test_numerical_stability(metric, projective):
    """Test numerical stability with extreme values."""
    # Create points with very small and large values (non-null points)
    a = torch.tensor([1e-10, 1e-10, 1.0])  # x² + y² < z²
    b = torch.tensor([1e10, 1e10, 2e10])   # x² + y² < z²
    
    # Calculate quadrance
    q = metric.quadrance(a, b)
    
    # Verify result is finite and reasonable
    assert torch.isfinite(q)
    assert q >= 0
    assert q <= 1

def test_gradient_flow(metric, projective):
    """Test that gradients flow correctly through metric calculations."""
    # Create points that require gradients (non-null points)
    a = torch.tensor([1.0, 1.0, 2.0], requires_grad=True)  # x² + y² < z²
    b = torch.tensor([2.0, 2.0, 3.0], requires_grad=True)  # x² + y² < z²
    
    # Calculate quadrance
    q = metric.quadrance(a, b)
    
    # Compute gradient
    q.backward()
    
    # Verify gradients exist
    assert a.grad is not None
    assert b.grad is not None
    assert torch.all(torch.isfinite(a.grad))
    assert torch.all(torch.isfinite(b.grad))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_consistency(metric, projective):
    """Test operations on different devices."""
    # Create points on CPU (non-null points)
    a_cpu = torch.tensor([1.0, 1.0, 2.0])  # x² + y² < z²
    b_cpu = torch.tensor([2.0, 2.0, 3.0])  # x² + y² < z²
    
    # Move to GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    
    # Calculate quadrance on both devices
    q_cpu = metric.quadrance(a_cpu, b_cpu)
    q_gpu = metric.quadrance(a_gpu, b_gpu)
    
    # Move GPU result back to CPU for comparison
    q_gpu_cpu = q_gpu.cpu()
    
    # Verify results match
    assert torch.allclose(q_cpu, q_gpu_cpu) 