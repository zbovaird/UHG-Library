import torch
import pytest
import sys
import os

# Add the docs directory to the Python path
docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
sys.path.append(docs_dir)

from projective_cursor import ProjectiveUHG

def test_triple_quad_formula():
    """Test the triple quad formula with non-null points forming a triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)  # Increased epsilon for numerical stability
    
    # Create three non-null points forming a triangle
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    
    # Calculate quadrances
    q1 = uhg.quadrance(p1, p2)
    q2 = uhg.quadrance(p2, p3)
    q3 = uhg.quadrance(p3, p1)
    
    print(f"Triangle quadrances: q1={q1}, q2={q2}, q3={q3}")
    
    # Verify triple quad formula
    assert uhg.triple_quad_formula(q1, q2, q3), "Triple quad formula should hold for triangle"

def test_pythagoras():
    """Test Pythagorean theorem with right-angled triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three non-null points forming a right triangle
    # Using points that form a right triangle
    p1 = torch.tensor([1.0, 0.0, 1.0])  # Base point
    p2 = torch.tensor([2.0, 0.0, 1.0])  # Point on x-axis
    p3 = torch.tensor([2.0, 1.0, 1.0])  # Point making right angle
    
    # Calculate quadrances
    q1 = uhg.quadrance(p1, p2)  # base
    q2 = uhg.quadrance(p2, p3)  # height
    q3 = uhg.quadrance(p1, p3)  # hypotenuse
    
    print(f"Right triangle quadrances: q1={q1}, q2={q2}, q3={q3}")
    
    # Verify Pythagorean theorem
    assert uhg.pythagoras(q1, q2, q3), "Pythagorean theorem should hold for right triangle"

def test_spread_law():
    """Test spread law with non-null triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three non-null points forming a triangle
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    
    # Get lines of triangle
    L1 = uhg.join(p2, p3)
    L2 = uhg.join(p3, p1)
    L3 = uhg.join(p1, p2)
    
    # Calculate spreads and quadrances
    S1 = uhg.spread(L2, L3)
    S2 = uhg.spread(L3, L1)
    S3 = uhg.spread(L1, L2)
    
    q1 = uhg.quadrance(p2, p3)
    q2 = uhg.quadrance(p3, p1)
    q3 = uhg.quadrance(p1, p2)
    
    print(f"Spread law ratios: S1/q1={S1/q1}, S2/q2={S2/q2}, S3/q3={S3/q3}")
    
    # Verify spread law
    assert uhg.spread_law(S1, S2, S3, q1, q2, q3), "Spread law should hold for triangle"

def test_cross_dual_law():
    """Test cross dual law with non-null triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three non-null points forming a triangle
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    
    # Get lines of triangle
    L1 = uhg.join(p2, p3)
    L2 = uhg.join(p3, p1)
    L3 = uhg.join(p1, p2)
    
    # Calculate spreads and quadrance
    S1 = uhg.spread(L2, L3)
    S2 = uhg.spread(L3, L1)
    S3 = uhg.spread(L1, L2)
    q1 = uhg.quadrance(p2, p3)
    
    # Verify cross dual law
    assert uhg.cross_dual_law(S1, S2, S3, q1), "Cross dual law should hold for triangle"

def test_triple_spread_formula():
    """Test the triple spread formula with non-null lines forming a triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)  # Increased epsilon for numerical stability
    
    # Create three non-null points forming a triangle
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    
    # Get lines of triangle
    L1 = uhg.join(p2, p3)
    L2 = uhg.join(p3, p1)
    L3 = uhg.join(p1, p2)
    
    # Calculate spreads
    S1 = uhg.spread(L2, L3)
    S2 = uhg.spread(L3, L1)
    S3 = uhg.spread(L1, L2)
    
    print(f"Triangle spreads: S1={S1}, S2={S2}, S3={S3}")
    
    # Verify triple spread formula
    assert uhg.triple_spread_formula(S1, S2, S3), "Triple spread formula should hold for triangle"

def test_cross_law():
    """Test the cross law with non-null triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three non-null points forming a triangle
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 2.0, 1.0])  # [0:2:1]
    p3 = torch.tensor([2.0, 2.0, 2.0])  # [1:1:1]
    
    # Get lines of triangle
    L1 = uhg.join(p2, p3)
    L2 = uhg.join(p3, p1)
    L3 = uhg.join(p1, p2)
    
    # Calculate quadrances and spreads
    q1 = uhg.quadrance(p2, p3)
    q2 = uhg.quadrance(p3, p1)
    q3 = uhg.quadrance(p1, p2)
    
    S1 = uhg.spread(L2, L3)
    S2 = uhg.spread(L3, L1)
    S3 = uhg.spread(L1, L2)
    
    print(f"Triangle quadrances: q1={q1}, q2={q2}, q3={q3}")
    print(f"Triangle spreads: S1={S1}, S2={S2}, S3={S3}")
    
    # Verify cross law
    assert uhg.cross_law(q1, q2, q3, S1, S2, S3), "Cross law should hold for triangle"

def test_spread_quadrance_duality():
    """Test the duality between spread and quadrance"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
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
    
    # Verify duality
    assert uhg.spread_quadrance_duality(L1, L2), "Spread-quadrance duality should hold"
    
    # Test with another pair of lines
    L3 = torch.tensor([1.0, 1.0, 0.0])  # x + y = 0
    L4 = torch.tensor([1.0, -1.0, 0.0])  # x - y = 0
    
    # Verify duality for second pair
    assert uhg.spread_quadrance_duality(L3, L4), "Spread-quadrance duality should hold for second pair"

def test_dual_pythagoras():
    """Test dual Pythagorean theorem with right-angled triangle"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three lines forming a right triangle
    # Using lines that form a right triangle
    L1 = torch.tensor([1.0, 0.0, 0.0])  # x = 0 (y-axis)
    L2 = torch.tensor([0.0, 1.0, 0.0])  # y = 0 (x-axis)
    L3 = torch.tensor([1.0, 1.0, 1.0])  # x + y = z
    
    # Calculate spreads
    S1 = uhg.spread(L1, L3)  # spread between y-axis and x+y=z
    S2 = uhg.spread(L2, L3)  # spread between x-axis and x+y=z
    S3 = uhg.spread(L1, L2)  # spread between x-axis and y-axis (should be 1 for perpendicular lines)
    
    print(f"Right triangle spreads: S1={S1}, S2={S2}, S3={S3}")
    
    # Verify dual Pythagorean theorem
    assert uhg.dual_pythagoras(S1, S2, S3), "Dual Pythagorean theorem should hold for right triangle"

def test_point_lies_on_line():
    """Test point-line incidence with non-null point and line"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create a non-null line and a point on it
    L = torch.tensor([1.0, 1.0, 1.0])  # x + y = z
    p = torch.tensor([0.5, 0.5, 1.0])  # Point satisfying x + y = z
    
    # Verify point lies on line
    assert uhg.point_lies_on_line(p, L), "Point should lie on line"
    
    # Test with point not on line
    p2 = torch.tensor([2.0, 0.0, 1.0])  # Point not satisfying x + y = z
    assert not uhg.point_lies_on_line(p2, L), "Point should not lie on line"

def test_perpendicular_relations():
    """Test perpendicular relations between points and lines"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create perpendicular points
    # For points to be perpendicular: x₁x₂ + y₁y₂ - z₁z₂ = 0
    p1 = torch.tensor([1.0, 0.0, 0.0])  # [1:0:0]
    p2 = torch.tensor([0.0, 1.0, 0.0])  # [0:1:0]
    
    # Create perpendicular lines
    L1 = torch.tensor([1.0, 0.0, 0.0])  # x = 0
    L2 = torch.tensor([0.0, 1.0, 0.0])  # y = 0
    
    # Test perpendicular points
    assert uhg.points_perpendicular(p1, p2), "Points should be perpendicular"
    
    # Test perpendicular lines
    assert uhg.lines_perpendicular(L1, L2), "Lines should be perpendicular"

def test_collinearity_and_concurrency():
    """Test collinearity of points and concurrency of lines"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create three collinear points on the line x = y
    p1 = torch.tensor([1.0, 1.0, 1.0])  # [1:1:1]
    p2 = torch.tensor([2.0, 2.0, 1.0])  # [2:2:1]
    p3 = torch.tensor([3.0, 3.0, 1.0])  # [3:3:1]
    
    # Create three concurrent lines through origin
    L1 = torch.tensor([1.0, 0.0, 0.0])  # x = 0
    L2 = torch.tensor([0.0, 1.0, 0.0])  # y = 0
    L3 = torch.tensor([1.0, 1.0, 0.0])  # x + y = 0
    
    # Test collinearity
    assert uhg.are_collinear(p1, p2, p3), "Points should be collinear"
    
    # Test concurrency
    assert uhg.are_concurrent(L1, L2, L3), "Lines should be concurrent"

def test_geometric_constructions():
    """Test geometric constructions with non-null points and lines"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create a point and a line
    a = torch.tensor([1.0, 1.0, 1.0])  # Point [1:1:1] not on y = 0
    L = torch.tensor([0.0, 1.0, 0.0])  # Line y = 0
    
    # Test altitude line
    alt_line = uhg.altitude_line(a, L)
    assert uhg.lines_perpendicular(alt_line, L), "Altitude line should be perpendicular to original line"
    
    # Test altitude point
    alt_point = uhg.altitude_point(a, L)
    assert uhg.point_lies_on_line(alt_point, L), "Altitude point should lie on original line"
    
    # Test parallel line
    par_line = uhg.parallel_line(a, L)
    assert not uhg.point_lies_on_line(a, L), "Point should not lie on original line"
    assert not uhg.point_lies_on_line(a, par_line), "Point should not lie on parallel line"
    
    # Test parallel point
    par_point = uhg.parallel_point(a, L)
    assert uhg.point_lies_on_line(par_point, L), "Parallel point should lie on original line"

def test_line_parametrization():
    """Test parametrization of points on a line"""
    uhg = ProjectiveUHG(epsilon=1e-5)
    
    # Create a line
    L = torch.tensor([1.0, 1.0, 1.0])  # Line x + y = z
    
    # Create parameters
    p = torch.tensor(1.0)
    r = torch.tensor(0.0)
    s = torch.tensor(1.0)
    
    # Get point on line
    point = uhg.parametrize_line_point(L, p, r, s)
    
    # Verify point lies on line
    assert uhg.point_lies_on_line(point, L), "Parametrized point should lie on line"
