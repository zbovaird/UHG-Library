"""
Test fundamental UHG measurements: quadrance and spread.
All tests verify that measurements follow UHG principles.
"""

import pytest
import torch
import math
from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG()

@pytest.fixture
def sample_points():
    """Generate test points with known properties."""
    return {
        'origin': torch.tensor([0.0, 0.0, 1.0]),
        'unit_x': torch.tensor([1.0, 0.0, 2.0]),
        'unit_y': torch.tensor([0.0, 1.0, 2.0]),
        'perp_points': (
            torch.tensor([1.0, 0.0, 2.0]),
            torch.tensor([0.0, 1.0, 2.0])
        ),
        'same_points': (
            torch.tensor([1.0, 1.0, 2.0]),
            torch.tensor([1.0, 1.0, 2.0])
        ),
        'null_point': torch.tensor([1.0, 1.0, math.sqrt(2.0)]),
    }

@pytest.fixture
def sample_lines():
    """Generate test lines with known properties."""
    return {
        'x_axis': torch.tensor([0.0, 1.0, 0.0]),
        'y_axis': torch.tensor([1.0, 0.0, 0.0]),
        'diagonal': torch.tensor([1.0, -1.0, 0.0]),
        'perp_lines': (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0])
        ),
    }

def test_quadrance_properties(uhg, sample_points):
    """Test fundamental properties of quadrance."""
    p1, p2 = sample_points['same_points']
    assert torch.isclose(uhg.quadrance(p1, p2), torch.tensor(0.0), atol=1e-5)

    p1, p2 = sample_points['perp_points']
    q12 = uhg.quadrance(p1, p2)
    q21 = uhg.quadrance(p2, p1)
    assert torch.isclose(q12, q21, rtol=1e-5)

    with pytest.raises(ValueError):
        uhg.quadrance(sample_points['null_point'], sample_points['unit_x'])

def test_quadrance_cross_ratio(uhg, sample_points):
    """Test that quadrance equals the cross-ratio with opposite points."""
    p1, p2 = sample_points['perp_points']

    aa = uhg.hyperbolic_dot(p1, p1)
    bb = uhg.hyperbolic_dot(p2, p2)
    ab = uhg.hyperbolic_dot(p1, p2)

    o1 = ab * p1 - aa * p2
    o2 = bb * p1 - ab * p2

    q_direct = uhg.quadrance(p1, p2)
    q_cross = uhg.cross_ratio(p1, o2, p2, o1)
    assert torch.isclose(q_direct, q_cross, rtol=1e-4)

def test_spread_properties(uhg, sample_lines):
    """Test fundamental properties of spread."""
    l1, l2 = sample_lines['perp_lines']
    assert torch.isclose(uhg.spread(l1, l2).squeeze(), torch.tensor(1.0), rtol=1e-5)

    s12 = uhg.spread(l1, l2)
    s21 = uhg.spread(l2, l1)
    assert torch.isclose(s12.squeeze(), s21.squeeze(), rtol=1e-5)

@pytest.mark.skip(reason="Projective matrix generation may not preserve quadrance in PyPI v0.3.7")
def test_projective_invariance(uhg, sample_points):
    """Test that quadrance is invariant under projective transformations."""
    p1, p2 = sample_points['perp_points']

    matrix = uhg.get_projective_matrix(2)

    p1_trans = uhg.transform(p1, matrix)
    p2_trans = uhg.transform(p2, matrix)

    q_orig = uhg.quadrance(p1, p2)
    q_trans = uhg.quadrance(p1_trans, p2_trans)
    assert torch.isclose(q_orig, q_trans, rtol=1e-4)

def test_null_point_properties(uhg, sample_points):
    """Test properties of null points."""
    null_point = sample_points['null_point']
    assert uhg.is_null_point(null_point)

    t = torch.tensor(1.0)
    u = torch.tensor(0.5)
    generated_null = uhg.null_point(t, u)
    assert uhg.is_null_point(generated_null)

    non_null = torch.tensor([1.0, 0.0, 2.0])
    assert not uhg.is_null_point(non_null)

def test_join_null_points(uhg):
    """Test joining of null points."""
    t1, u1 = torch.tensor(1.0), torch.tensor(0.0)
    t2, u2 = torch.tensor(0.0), torch.tensor(1.0)

    p1 = uhg.null_point(t1, u1)
    p2 = uhg.null_point(t2, u2)

    assert uhg.is_null_point(p1)
    assert uhg.is_null_point(p2)

    join = uhg.join_null_points(t1, u1, t2, u2)
    assert join.shape == (3,)

def test_quadrance_with_null_points(uhg, sample_points):
    """Test quadrance behavior with null points."""
    null_point = sample_points['null_point']
    non_null = torch.tensor([1.0, 0.0, 2.0])

    with pytest.raises(ValueError):
        uhg.quadrance(null_point, non_null)

    with pytest.raises(ValueError):
        uhg.quadrance(non_null, null_point)

    with pytest.raises(ValueError):
        uhg.quadrance(null_point, null_point)
