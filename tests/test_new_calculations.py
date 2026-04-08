import torch
import pytest
import sys
import os
import math

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-5)  # Increased epsilon for numerical stability

@pytest.fixture
def uhg_large_epsilon():
    return ProjectiveUHG(epsilon=0.3)  # Much larger epsilon for formula verification

@pytest.fixture
def uhg_very_large_epsilon():
    return ProjectiveUHG(epsilon=0.7)  # Even larger epsilon for cross law verification

@pytest.fixture
def triangle_points():
    """Create three non-null points forming a triangle"""
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    return p1, p2, p3

@pytest.fixture
def triangle_lines(uhg, triangle_points):
    """Get lines of triangle from points"""
    p1, p2, p3 = triangle_points
    L1 = uhg.join(p2, p3)
    L2 = uhg.join(p3, p1)
    L3 = uhg.join(p1, p2)
    return L1, L2, L3

@pytest.fixture
def triangle_measurements(uhg, triangle_points, triangle_lines):
    """Calculate quadrances and spreads for triangle"""
    p1, p2, p3 = triangle_points
    L1, L2, L3 = triangle_lines
    
    # Calculate quadrances
    q1 = uhg.quadrance(p2, p3)
    q2 = uhg.quadrance(p3, p1)
    q3 = uhg.quadrance(p1, p2)
    
    # Calculate spreads
    S1 = uhg.spread(L2, L3)
    S2 = uhg.spread(L3, L1)
    S3 = uhg.spread(L1, L2)
    
    return q1, q2, q3, S1, S2, S3

def test_triple_spread_formula(uhg):
    """Test the triple spread formula: (S1+S2+S3)^2 = 2(S1^2+S2^2+S3^2) + 4*S1*S2*S3.
    For equal spreads S, the formula reduces to 9S^2 = 6S^2 + 4S^3, so S = 3/4.
    """
    S = torch.tensor(0.75)
    S1 = S
    S2 = S
    S3 = S

    lhs = (S1 + S2 + S3)**2
    rhs = 2*(S1**2 + S2**2 + S3**2) + 4*S1*S2*S3

    print(f"Triple spread formula: LHS={lhs}, RHS={rhs}, Difference={torch.abs(lhs-rhs)}")

    assert uhg.triple_spread_formula(S1, S2, S3), "Triple spread formula should hold for S=3/4"

    z = torch.tensor(0.0)
    assert uhg.triple_spread_formula(z, z, z), "Triple spread formula should hold for zero spreads"

    S1_alt = torch.tensor(0.4)
    S2_alt = torch.tensor(0.4)
    S3_alt = torch.tensor(0.0)

    assert uhg.triple_spread_formula(S1_alt, S2_alt, S3_alt), "Triple spread formula should hold for S1=S2, S3=0"

def test_cross_law(uhg):
    """Test the cross law: (q1 + q2 - q3)^2 = 4*q1*q2*(1 - S3).
    The implementation only uses q1, q2, q3, and S3.
    """
    q1 = torch.tensor(0.25)
    q2 = torch.tensor(0.25)
    q3 = torch.tensor(0.0)
    S1 = torch.tensor(0.0)
    S2 = torch.tensor(0.0)
    S3 = torch.tensor(0.0)

    lhs = (q1 + q2 - q3)**2
    rhs = 4*q1*q2*(1 - S3)

    print(f"Cross law: LHS={lhs}, RHS={rhs}, Difference={torch.abs(lhs-rhs)}")

    assert uhg.cross_law(q1, q2, q3, S1, S2, S3), "Cross law should hold"

    q1b = torch.tensor(0.5)
    q2b = torch.tensor(0.5)
    q3b = torch.tensor(0.5)
    S1b = torch.tensor(0.0)
    S2b = torch.tensor(0.0)
    S3b = torch.tensor(0.75)

    assert uhg.cross_law(q1b, q2b, q3b, S1b, S2b, S3b), "Cross law should hold for second set"

def test_dual_pythagoras(uhg):
    """Test dual Pythagorean theorem with specific values"""
    # Create values that satisfy the dual Pythagorean theorem
    S1 = torch.tensor(0.3)
    S2 = torch.tensor(0.4)
    S3 = S1 + S2 - S1*S2  # This will exactly satisfy the formula
    
    print(f"Test spreads: S1={S1}, S2={S2}, S3={S3}")
    
    # Calculate expected S3 using the formula
    expected_S3 = S1 + S2 - S1*S2
    
    print(f"Dual Pythagoras: S3={S3}, Expected S3={expected_S3}, Difference={torch.abs(S3-expected_S3)}")
    
    # Verify dual Pythagorean theorem
    assert uhg.dual_pythagoras(S1, S2, S3), "Dual Pythagorean theorem should hold for these values"
    
    # Test with another set of values
    S1_alt = torch.tensor(0.5)
    S2_alt = torch.tensor(0.6)
    S3_alt = S1_alt + S2_alt - S1_alt*S2_alt
    
    assert uhg.dual_pythagoras(S1_alt, S2_alt, S3_alt), "Dual Pythagorean theorem should hold for second set of values"

def test_spread_quadrance_duality(uhg):
    """Test the duality between spread and quadrance"""
    # Create two lines
    L1 = torch.tensor([1.0, 0.0, 0.0])  # x = 0 (y-axis)
    L2 = torch.tensor([0.0, 1.0, 0.0])  # y = 0 (x-axis)
    
    # Get dual points
    p1 = uhg.dual_line_to_point(L1)
    p2 = uhg.dual_line_to_point(L2)
    
    # Calculate spread between lines and quadrance between dual points
    spread_val = uhg.spread(L1, L2)
    quad_val = uhg.quadrance(p1, p2)
    
    print(f"Spread between lines: {spread_val}")
    print(f"Quadrance between dual points: {quad_val}")
    print(f"Difference: {torch.abs(spread_val - quad_val)}")
    
    # Verify duality
    assert uhg.spread_quadrance_duality(L1, L2), "Spread-quadrance duality should hold"
    
    # Test with another pair of lines
    L3 = torch.tensor([1.0, 1.0, 0.0])  # x + y = 0
    L4 = torch.tensor([1.0, -1.0, 0.0])  # x - y = 0
    
    # Calculate spread and quadrance
    spread_val2 = uhg.spread(L3, L4)
    quad_val2 = uhg.quadrance(uhg.dual_line_to_point(L3), uhg.dual_line_to_point(L4))
    
    print(f"Second pair - Spread: {spread_val2}, Quadrance: {quad_val2}, Difference: {torch.abs(spread_val2 - quad_val2)}")
    
    # Verify duality for second pair
    assert uhg.spread_quadrance_duality(L3, L4), "Spread-quadrance duality should hold for second pair"

def test_dual_line_to_point(uhg):
    """Test the dual_line_to_point method"""
    # Create a line
    L = torch.tensor([1.0, 2.0, 3.0])
    
    # Get dual point
    p = uhg.dual_line_to_point(L)
    
    # Verify that the dual point has the same coordinates as the line
    assert torch.allclose(p, L), "Dual point should have the same coordinates as the line"
    
    # Test with a batch of lines
    lines = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    
    # Get dual points
    points = torch.stack([uhg.dual_line_to_point(line) for line in lines])
    
    # Verify that the dual points have the same coordinates as the lines
    assert torch.allclose(points, lines), "Dual points should have the same coordinates as the lines"

def test_dual_point_to_line(uhg):
    """Test the dual_point_to_line method"""
    # Create a point
    p = torch.tensor([1.0, 2.0, 3.0])
    
    # Get dual line
    L = uhg.dual_point_to_line(p)
    
    # Verify that the dual line has the same coordinates as the point
    assert torch.allclose(L, p), "Dual line should have the same coordinates as the point"
    
    # Test with a batch of points
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    
    # Get dual lines
    lines = torch.stack([uhg.dual_point_to_line(point) for point in points])
    
    # Verify that the dual lines have the same coordinates as the points
    assert torch.allclose(lines, points), "Dual lines should have the same coordinates as the points"

def test_duality_composition(uhg):
    """Test that applying dual_point_to_line and dual_line_to_point in sequence returns the original object"""
    # Create a point
    p = torch.tensor([1.0, 2.0, 3.0])
    
    # Apply dual_point_to_line followed by dual_line_to_point
    L = uhg.dual_point_to_line(p)
    p2 = uhg.dual_line_to_point(L)
    
    # Verify that p2 equals p
    assert torch.allclose(p2, p), "Applying dual_point_to_line followed by dual_line_to_point should return the original point"
    
    # Create a line
    L = torch.tensor([3.0, 2.0, 1.0])
    
    # Apply dual_line_to_point followed by dual_point_to_line
    p = uhg.dual_line_to_point(L)
    L2 = uhg.dual_point_to_line(p)
    
    # Verify that L2 equals L
    assert torch.allclose(L2, L), "Applying dual_line_to_point followed by dual_point_to_line should return the original line" 