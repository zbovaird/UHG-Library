import torch
import pytest
import math
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uhg.projective import ProjectiveUHG

@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-6)

@pytest.fixture
def uhg_relaxed():
    return ProjectiveUHG(epsilon=1e-4)

@pytest.fixture
def triangle_points(uhg):
    """Create three non-null points forming a triangle from UHG.pdf Example 15"""
    a1 = torch.tensor([math.sqrt(2), 0.0, math.sqrt(5)])
    a2 = torch.tensor([-1.0, math.sqrt(3), math.sqrt(10)])
    a3 = torch.tensor([-1.0, -math.sqrt(3), math.sqrt(10)])
    
    # Normalize points to hyperboloid
    a1 = uhg.normalize_points(a1)
    a2 = uhg.normalize_points(a2)
    a3 = uhg.normalize_points(a3)
    return a1, a2, a3

@pytest.fixture
def triangle_lines(uhg, triangle_points):
    """Get lines of triangle from points"""
    a1, a2, a3 = triangle_points
    L1 = uhg.join(a2, a3)  # Line through a2 and a3
    L2 = uhg.join(a3, a1)  # Line through a3 and a1
    L3 = uhg.join(a1, a2)  # Line through a1 and a2
    return L1, L2, L3

@pytest.fixture
def triangle_measurements(uhg, triangle_points, triangle_lines):
    """Calculate quadrances and spreads for triangle"""
    a1, a2, a3 = triangle_points
    L1, L2, L3 = triangle_lines
    
    # Calculate quadrances
    q1 = uhg.quadrance(a2, a3)  # Between a2 and a3
    q2 = uhg.quadrance(a3, a1)  # Between a3 and a1
    q3 = uhg.quadrance(a1, a2)  # Between a1 and a2
    
    # Calculate spreads using reference lines
    S1 = uhg.spread(L2, L3, L1)  # Spread at vertex a1
    S2 = uhg.spread(L3, L1, L2)  # Spread at vertex a2
    S3 = uhg.spread(L1, L2, L3)  # Spread at vertex a3
    
    return q1, q2, q3, S1, S2, S3

def test_quadrance_basic(uhg):
    """Test basic quadrance properties according to UHG.pdf"""
    # Test points with negative norm (proper hyperbolic points)
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    
    # Normalize points to hyperboloid
    a = uhg.normalize_points(a)
    b = uhg.normalize_points(b)
    
    q = uhg.quadrance(a, b)
    
    # Expected = 1 - (a·b)² since points are normalized
    a_dot_b = uhg.inner_product(a, b)
    expected = 1 - (a_dot_b * a_dot_b)
    assert torch.abs(q - expected) < 1e-6, f"Expected {expected}, got {q}"
    
    # Test perpendicular points (should give 1)
    p1 = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    p2 = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    
    # Normalize points to hyperboloid
    p1 = uhg.normalize_points(p1)
    p2 = uhg.normalize_points(p2)
    
    dot = uhg.inner_product(p1, p2)
    assert torch.abs(dot) < 1e-6, f"Points should be perpendicular (dot product = 0), got {dot}"
    
    q_perp = uhg.quadrance(p1, p2)
    assert torch.abs(q_perp - 1.0) < 1e-6, f"Expected quadrance to be 1 for perpendicular points, got {q_perp}"
    
    # Test points on same line (should give 0)
    c = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    d = torch.tensor([2.0, 0.0, 4.0])  # norm = -12
    
    # Normalize points to hyperboloid
    c = uhg.normalize_points(c)
    d = uhg.normalize_points(d)
    
    q_line = uhg.quadrance(c, d)
    assert torch.abs(q_line) < 1e-6, f"Expected quadrance to be 0 for points on same line, got {q_line}"

def test_quadrance_from_cross_ratio(uhg):
    """Test quadrance calculation using cross ratio method"""
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    
    q_direct = uhg.quadrance(a, b)
    q_cr = uhg.quadrance_from_cross_ratio(a, b)
    
    assert torch.abs(q_direct - q_cr) < 1e-6, f"Quadrance calculations should match: direct={q_direct}, cross_ratio={q_cr}"

def test_quadrance_example15(uhg, triangle_points):
    """Test quadrance with Example 15 from UHG.pdf"""
    a1, a2, a3 = triangle_points
    
    q1 = uhg.quadrance(a2, a3)
    q2 = uhg.quadrance(a3, a1)
    q3 = uhg.quadrance(a1, a2)
    
    # According to Example 15, all quadrances should be -3
    assert torch.abs(q1 - (-3.0)) < 1e-5, f"Expected q1 to be -3, got {q1}"
    assert torch.abs(q2 - (-3.0)) < 1e-5, f"Expected q2 to be -3, got {q2}"
    assert torch.abs(q3 - (-3.0)) < 1e-5, f"Expected q3 to be -3, got {q3}"

def test_spread_basic(uhg):
    """Test basic spread properties according to UHG.pdf"""
    # Create non-null lines
    L1 = torch.tensor([1.0, 0.0, 1.0])  # x + z = 0
    L2 = torch.tensor([1.0, 0.0, 2.0])  # x + 2z = 0
    L3 = torch.tensor([0.0, 1.0, 1.0])  # y + z = 0
    
    # Normalize lines
    L1 = L1 / torch.sqrt(torch.abs(uhg.inner_product(L1, L1)))
    L2 = L2 / torch.sqrt(torch.abs(uhg.inner_product(L2, L2)))
    L3 = L3 / torch.sqrt(torch.abs(uhg.inner_product(L3, L3)))
    
    # Test parallel lines (should give 0)
    s = uhg.spread(L1, L2, L3)
    assert torch.abs(s) < 1e-6, f"Parallel lines should have spread 0, got {s}"
    
    # Test perpendicular lines (should give 1)
    L4 = torch.tensor([1.0, 0.0, 1.0])  # x + z = 0
    L5 = torch.tensor([0.0, 1.0, 1.0])  # y + z = 0
    
    # Normalize lines
    L4 = L4 / torch.sqrt(torch.abs(uhg.inner_product(L4, L4)))
    L5 = L5 / torch.sqrt(torch.abs(uhg.inner_product(L5, L5)))
    
    s_perp = uhg.spread(L4, L5, L3)
    assert torch.abs(s_perp - 1.0) < 1e-6, f"Expected spread to be 1 for perpendicular lines, got {s_perp}"
    
    # Test lines at 45 degrees (should give 0.5)
    L6 = torch.tensor([1.0, 0.0, 1.0])  # x + z = 0
    L7 = torch.tensor([1.0, 1.0, 1.0])  # x + y + z = 0
    
    # Normalize lines
    L6 = L6 / torch.sqrt(torch.abs(uhg.inner_product(L6, L6)))
    L7 = L7 / torch.sqrt(torch.abs(uhg.inner_product(L7, L7)))
    
    s_45 = uhg.spread(L6, L7, L3)
    assert torch.abs(s_45 - 0.5) < 1e-6, f"Expected spread to be 0.5 for 45-degree lines, got {s_45}"

def test_cross_ratio_basic(uhg):
    """Test basic cross ratio properties"""
    # Create four points
    a = torch.tensor([1.0, 0.0, 2.0])
    b = torch.tensor([2.0, 0.0, 3.0])
    c = torch.tensor([0.0, 1.0, 2.0])
    d = torch.tensor([1.0, 1.0, 3.0])
    
    # Cross ratio should be real and finite
    cr = uhg.cross_ratio(a, b, c, d)
    assert not torch.isnan(cr), "Cross ratio should not be NaN"
    assert torch.isfinite(cr), "Cross ratio should be finite"
    
    # Test identity: CR(A,B;C,D) * CR(A,B;D,C) = 1
    cr1 = uhg.cross_ratio(a, b, c, d)
    cr2 = uhg.cross_ratio(a, b, d, c)
    product = cr1 * cr2
    assert torch.abs(product - 1.0) < 1e-5, f"Cross ratio identity should hold: {cr1} * {cr2} = {product}"
    
    # Test scale invariance
    cr_scaled = uhg.cross_ratio(2*a, 3*b, 4*c, 5*d)
    assert torch.abs(cr - cr_scaled) < 1e-6, f"Cross ratio should be scale-invariant: {cr} vs {cr_scaled}"

def test_triple_quad_formula(uhg_relaxed, triangle_measurements):
    """Test the triple quad formula with triangle from Example 15"""
    q1, q2, q3, _, _, _ = triangle_measurements
    
    # Verify triple quad formula: (q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4q₁q₂q₃
    lhs = (q1 + q2 + q3)**2
    rhs = 2*(q1**2 + q2**2 + q3**2) + 4*q1*q2*q3
    
    assert torch.abs(lhs - rhs) < 1e-4, f"Triple quad formula not satisfied: {lhs} vs {rhs}"
    assert uhg_relaxed.triple_quad_formula(q1, q2, q3), "Triple quad formula should hold for triangle"

def test_triple_spread_formula(uhg_relaxed, triangle_measurements):
    """Test the triple spread formula with triangle from Example 15"""
    _, _, _, S1, S2, S3 = triangle_measurements
    
    # Verify triple spread formula: (S₁ + S₂ + S₃)² = 2(S₁² + S₂² + S₃²) + 4S₁S₂S₃
    lhs = (S1 + S2 + S3)**2
    rhs = 2*(S1**2 + S2**2 + S3**2) + 4*S1*S2*S3
    
    assert torch.abs(lhs - rhs) < 1e-4, f"Triple spread formula not satisfied: {lhs} vs {rhs}"
    assert uhg_relaxed.triple_spread_formula(S1, S2, S3), "Triple spread formula should hold for triangle"

def test_pythagoras(uhg):
    """Test Pythagorean theorem with right-angled triangle"""
    # Create three points forming a right triangle
    p1 = torch.tensor([1.0, 0.0, 1.0])  # Base point
    p2 = torch.tensor([2.0, 0.0, 1.0])  # Point on x-axis
    p3 = torch.tensor([2.0, 1.0, 1.0])  # Point making right angle
    
    # Calculate quadrances
    q1 = uhg.quadrance(p1, p2)  # base
    q2 = uhg.quadrance(p2, p3)  # height
    q3 = uhg.quadrance(p1, p3)  # hypotenuse
    
    # Verify Pythagorean theorem: q₃ = q₁ + q₂ - q₁q₂
    expected = q1 + q2 - q1*q2
    assert torch.abs(q3 - expected) < 1e-6, f"Pythagorean theorem not satisfied: {q3} vs {expected}"
    assert uhg.pythagoras(q1, q2, q3), "Pythagorean theorem should hold for right triangle"

def test_dual_pythagoras(uhg, triangle_measurements):
    """Test dual Pythagoras theorem for spreads"""
    q1, q2, q3, S1, S2, S3 = triangle_measurements
    
    # S1 + S2 + S3 = 1 + 2*S1*S2*S3
    left = S1 + S2 + S3
    right = 1 + 2 * S1 * S2 * S3
    assert torch.abs(left - right) < 1e-6, f"Expected {right}, got {left}"

def test_cross_law(uhg_relaxed, triangle_measurements):
    """Test the cross law with triangle from Example 15"""
    q1, q2, q3, S1, S2, S3 = triangle_measurements
    
    # Verify cross law: q₁q₂q₃S₁S₂S₃ = (q₁q₂S₃ + q₂q₃S₁ + q₃q₁S₂ - q₁ - q₂ - q₃ - S₁ - S₂ - S₃ + 2)²
    lhs = q1*q2*q3*S1*S2*S3
    inside_term = q1*q2*S3 + q2*q3*S1 + q3*q1*S2 - q1 - q2 - q3 - S1 - S2 - S3 + 2
    rhs = inside_term**2
    
    assert torch.abs(lhs - rhs) < 1e-4, f"Cross law not satisfied: {lhs} vs {rhs}"
    assert uhg_relaxed.cross_law(q1, q2, q3, S1, S2, S3), "Cross law should hold for triangle"

def test_spread_quadrance_duality(uhg, triangle_measurements):
    """Test duality between spread and quadrance"""
    q1, q2, q3, S1, S2, S3 = triangle_measurements
    
    # q1*S1 = q2*S2 = q3*S3
    assert torch.abs(q1*S1 - q2*S2) < 1e-6, f"Expected q1*S1 = q2*S2, got {q1*S1} and {q2*S2}"
    assert torch.abs(q2*S2 - q3*S3) < 1e-6, f"Expected q2*S2 = q3*S3, got {q2*S2} and {q3*S3}"

def test_null_point_detection(uhg):
    """Test detection of null points"""
    # Create a null point [1:0:1] (lies on the null cone where x² + y² = z²)
    null_point = torch.tensor([1.0, 0.0, 1.0])
    regular_point = torch.tensor([2.0, 0.0, 1.0])
    
    # Verify the point is indeed null
    assert uhg.is_null_point(null_point), "Point [1:0:1] should be null"
    assert not uhg.is_null_point(regular_point), "Point [2:0:1] should not be null"
    
    # Attempt to calculate quadrance with null point - should raise ValueError
    with pytest.raises(ValueError, match="Quadrance is undefined for null points"):
        uhg.quadrance(null_point, regular_point)
    
    with pytest.raises(ValueError, match="Quadrance is undefined for null points"):
        uhg.quadrance(regular_point, null_point)

def test_null_line_detection(uhg):
    """Test detection of null lines"""
    # Create a null line (Minkowski norm = 0)
    null_line = torch.tensor([1.0, 1.0, math.sqrt(2)])  # x + y + √2z = 0
    norm = uhg.inner_product(null_line, null_line)
    assert torch.abs(norm) < 1e-6, f"Expected null line to have norm 0, got {norm}"
    
    # Create non-null lines for spread calculation
    L1 = torch.tensor([1.0, 0.0, 1.0])  # x + z = 0
    L2 = torch.tensor([0.0, 1.0, 1.0])  # y + z = 0
    
    # Normalize lines
    L1 = L1 / torch.sqrt(torch.abs(uhg.inner_product(L1, L1)))
    L2 = L2 / torch.sqrt(torch.abs(uhg.inner_product(L2, L2)))
    
    # Attempt to use null line in spread calculation
    with pytest.raises(ValueError, match="Spread is undefined for null lines"):
        uhg.spread(null_line, L1, L2)

def test_midpoints(uhg):
    """Test midpoint calculation"""
    # Create two points
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    
    # Normalize points to hyperboloid
    a = uhg.normalize_points(a)
    b = uhg.normalize_points(b)
    
    # Calculate midpoint
    m = uhg.midpoint(a, b)
    
    # Verify midpoint is normalized
    norm = uhg.inner_product(m, m)
    assert torch.abs(norm + 1.0) < 1e-6, f"Expected midpoint to have norm -1, got {norm}"
    
    # Verify midpoint is equidistant
    q1 = uhg.quadrance(a, m)
    q2 = uhg.quadrance(m, b)
    assert torch.abs(q1 - q2) < 1e-6, f"Expected equal quadrances, got {q1} and {q2}"

def test_projective_transformation(uhg):
    """Test projective transformation properties"""
    # Create points
    a = torch.tensor([1.0, 0.0, 2.0])  # norm = -3
    b = torch.tensor([2.0, 0.0, 3.0])  # norm = -5
    c = torch.tensor([0.0, 1.0, 2.0])  # norm = -3
    
    # Normalize points to hyperboloid
    a = uhg.normalize_points(a)
    b = uhg.normalize_points(b)
    c = uhg.normalize_points(c)
    
    # Create transformation matrix
    T = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Transform points
    a_transformed = uhg.transform(a, T)
    b_transformed = uhg.transform(b, T)
    c_transformed = uhg.transform(c, T)
    
    # Verify transformed points are normalized
    for p in [a_transformed, b_transformed, c_transformed]:
        norm = uhg.inner_product(p, p)
        assert torch.abs(norm + 1.0) < 1e-6, f"Expected transformed point to have norm -1, got {norm}"
    
    # Verify cross ratio is preserved
    cr_before = uhg.cross_ratio(a, b, c)
    cr_after = uhg.cross_ratio(a_transformed, b_transformed, c_transformed)
    assert torch.abs(cr_before - cr_after) < 1e-6, f"Expected cross ratio to be preserved, got {cr_before} and {cr_after}"

if __name__ == "__main__":
    # Create UHG instance
    uhg_instance = ProjectiveUHG()
    
    # Run tests
    test_quadrance_basic(uhg_instance)
    test_spread_basic(uhg_instance)
    test_cross_ratio_basic(uhg_instance)
    test_null_point_detection(uhg_instance)
    test_null_line_detection(uhg_instance)
    test_midpoints(uhg_instance)
    test_projective_transformation(uhg_instance)
    
    print("All tests passed!") 