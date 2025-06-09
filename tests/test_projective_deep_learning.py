import torch
from torch.testing import assert_close
import math
import pytest
from uhg.projective import ProjectiveUHG

def test_quadrance():
    """
    Test quadrance calculation according to UHG principles:
    q(A,B) = 1 - <A,B>²/(<A,A><B,B>)
    """
    uhg = ProjectiveUHG()
    
    # Test case 1: Points at known quadrance
    a = torch.tensor([0.0, 0.0, 1.0])  # Origin
    b = torch.tensor([1.0, 0.0, math.sqrt(2.0)])  # Point with known quadrance
    q = uhg.quadrance(a, b)
    assert_close(q, torch.tensor(1.0), rtol=1e-4, atol=1e-4)
    
    # Test case 2: Same point should have zero quadrance
    q = uhg.quadrance(a, a)
    assert_close(q, torch.tensor(0.0), rtol=1e-4, atol=1e-4)
    
    # Test case 3: Quadrance should be symmetric
    q1 = uhg.quadrance(a, b)
    q2 = uhg.quadrance(b, a)
    assert_close(q1, q2, rtol=1e-4, atol=1e-4)
    
    # Test case 4: Null points should raise error
    null_point = torch.tensor([1.0, 0.0, 1.0])  # x² + y² = z²
    with pytest.raises(ValueError):
        uhg.quadrance(a, null_point)

def test_midpoints():
    """
    Test midpoint calculation in hyperbolic space.
    The midpoints should:
    1. Be equidistant from both input points
    2. Be perpendicular to each other
    3. Exist only when p(a₁,a₂) = 1 - q(a₁,a₂) is a square
    4. Handle null points and null lines correctly
    """
    uhg = ProjectiveUHG()
    
    # Test case 1: Points with known midpoints
    # Use points that satisfy x² + y² < z² (inside hyperbolic disk)
    # and are not perpendicular to each other
    a = torch.tensor([0.3, 0.0, 1.0])  # Point on x-axis
    b = torch.tensor([0.4, 0.2, 1.0])  # Point at angle to x-axis
    m1, m2 = uhg.midpoints(a, b)
    
    # Verify midpoints exist and are valid
    assert m1 is not None and m2 is not None
    assert uhg.verify_midpoints(a, b, m1, m2)
    
    # Test case 2: Midpoints should be invariant under swapping points
    m1_swap, m2_swap = uhg.midpoints(b, a)
    assert_close(m1, m1_swap, rtol=1e-4, atol=1e-4)
    assert_close(m2, m2_swap, rtol=1e-4, atol=1e-4)
    
    # Test case 3: Point with itself should return (point, None)
    m1, m2 = uhg.midpoints(a, a)
    assert_close(m1, a, rtol=1e-4, atol=1e-4)
    assert m2 is None
    
    # Test case 4: Points too far apart should return no midpoints
    far_point = torch.tensor([2.0, 0.0, 1.0])  # Point outside unit disk
    m1, m2 = uhg.midpoints(a, far_point)
    assert m1 is None and m2 is None
    
    # Test case 5: Null point should be its own midpoint
    null_point = torch.tensor([1.0, 0.0, 1.0])  # x² + y² = z²
    m1, m2 = uhg.midpoints(null_point, b)
    assert_close(m1, null_point, rtol=1e-4, atol=1e-4)
    assert m2 is None
    
    # Test case 6: Points on null line should have no midpoints
    t1, u1 = torch.tensor(1.0), torch.tensor(0.0)
    t2, u2 = torch.tensor(2.0), torch.tensor(0.0)
    p1 = uhg.null_point(t1, u1)
    p2 = uhg.null_point(t2, u2)
    m1, m2 = uhg.midpoints(p1, p2)
    assert m1 is None and m2 is None

def test_midpoint_mathematics():
    """
    Test mathematical properties of midpoints in hyperbolic space.
    The midpoints should:
    1. Cross-ratio relationships
    2. Preservation of hyperbolic distance
    3. Perpendicularity properties
    4. Invariance under projective transformations
    """
    uhg = ProjectiveUHG()
    eps = 1e-4
    
    # Use points that satisfy x² + y² < z² (inside hyperbolic disk)
    # and are not perpendicular to each other
    a = torch.tensor([0.3, 0.0, 1.0])  # Point on x-axis
    b = torch.tensor([0.4, 0.2, 1.0])  # Point at angle to x-axis
    m1, m2 = uhg.midpoints(a, b)
    
    # 1. Cross-ratio relationships
    # For midpoints m₁,m₂ of side AB:
    # (A,B:m₁,m₂) = -1
    cr = uhg.cross_ratio(a, b, m1, m2)
    assert_close(cr, torch.tensor(-1.0), rtol=eps, atol=eps)
    
    # 2. Distance relationships
    # Distance from each midpoint to endpoints should be equal
    d1a = uhg.distance(m1, a)
    d1b = uhg.distance(m1, b)
    assert_close(d1a, d1b, rtol=eps, atol=eps)
    
    d2a = uhg.distance(m2, a)
    d2b = uhg.distance(m2, b)
    assert_close(d2a, d2b, rtol=eps, atol=eps)
    
    # 3. Perpendicularity
    # m₁⊥m₂ means their hyperbolic dot product should be 0
    dot = uhg.hyperbolic_dot(m1, m2)
    assert_close(dot, torch.tensor(0.0), rtol=eps, atol=eps)
    
    # 4. Projective invariance
    # Apply random projective transformation
    matrix = uhg.get_projective_matrix(2)  # 2D projective matrix
    
    # Transform points
    a_trans = uhg.transform(a, matrix)
    b_trans = uhg.transform(b, matrix)
    
    # Get midpoints of transformed points
    m1_trans, m2_trans = uhg.midpoints(a_trans, b_trans)
    
    # Transform original midpoints
    m1_expected = uhg.transform(m1, matrix)
    m2_expected = uhg.transform(m2, matrix)
    
    # Should match up to sign (projective equivalence)
    assert (torch.allclose(m1_trans, m1_expected, rtol=eps, atol=eps) or
            torch.allclose(m1_trans, -m1_expected, rtol=eps, atol=eps))
    assert (torch.allclose(m2_trans, m2_expected, rtol=eps, atol=eps) or
            torch.allclose(m2_trans, -m2_expected, rtol=eps, atol=eps))
    
    # Additional properties for null points
    null_point = torch.tensor([1.0, 0.0, 1.0])  # x² + y² = z²
    m1, m2 = uhg.midpoints(null_point, b)
    
    # A null point should be its own midpoint
    assert_close(m1, null_point, rtol=eps, atol=eps)
    assert m2 is None
    
    # The quadrance between a null point and its midpoint should be 0
    q = uhg.quadrance(null_point, m1)
    assert_close(q, torch.tensor(0.0), rtol=eps, atol=eps)

def test_null_line():
    """Test detection of null lines"""
    uhg = ProjectiveUHG()
    
    # Test case 1: Line through two null points should be null
    t1, u1 = torch.tensor(1.0), torch.tensor(0.0)
    t2, u2 = torch.tensor(2.0), torch.tensor(0.0)
    p1 = uhg.null_point(t1, u1)
    p2 = uhg.null_point(t2, u2)
    line = uhg.join_points(p1, p2)
    assert uhg.is_null_line(line)
    
    # Test case 2: Line through origin and non-null point should not be null
    a = torch.tensor([0.0, 0.0, 1.0])  # Origin
    b = torch.tensor([1.0, 0.0, math.sqrt(2.0)])  # Non-null point
    line = uhg.join_points(a, b)
    assert not uhg.is_null_line(line)
