"""
Test UHG calculations with synthetic data to verify mathematical correctness.
Focuses on quadrance, spread, and cross law calculations.
"""

import torch
import numpy as np
from uhg.projective import ProjectiveUHG
import math

def test_quadrance():
    """Test quadrance calculation with known values."""
    print("\nTesting Quadrance Calculation")
    print("-" * 50)
    
    uhg = ProjectiveUHG()
    
    # Test case 1: Points on same line (should give 0)
    # Using hyperbolic points (z-coordinate larger than x,y)
    a1 = torch.tensor([1.0, 0.0, 2.0])  # Non-null hyperbolic point
    b1 = torch.tensor([2.0, 0.0, 4.0])  # Same direction, different scale
    q1 = uhg.quadrance(a1, b1)
    q1_cr = uhg.quadrance_from_cross_ratio(a1, b1)
    print(f"Test 1 - Points on same line:")
    print(f"a1: {a1}")
    print(f"b1: {b1}")
    print(f"Quadrance: {q1:.6f} (Expected: 0)")
    print(f"Quadrance from cross ratio: {q1_cr:.6f}")
    assert torch.abs(q1) < 1e-6, "Quadrance should be 0 for points on same line"
    assert torch.abs(q1 - q1_cr) < 1e-6, "Quadrance calculations should match"
    
    # Test case 2: Using points from UHG.pdf Example 15 (second set)
    # These points form a hyperbolic triangle with equal quadrances q1 = q2 = q3 = -3
    a2 = torch.tensor([math.sqrt(2), 0.0, math.sqrt(5)])  # a1 from second example
    b2 = torch.tensor([-1.0, math.sqrt(3), math.sqrt(10)])  # a2 from second example
    q2 = uhg.quadrance(a2, b2)
    
    # Debug prints
    print(f"\nTest 2 - Points from UHG.pdf Example 15 (second set):")
    print(f"a2: {a2}")
    print(f"b2: {b2}")
    print(f"inner_prod: {a2[0]*b2[0] + a2[1]*b2[1] - a2[2]*b2[2]}")
    print(f"norm_a: {a2[0]*a2[0] + a2[1]*a2[1] - a2[2]*a2[2]}")
    print(f"norm_b: {b2[0]*b2[0] + b2[1]*b2[1] - b2[2]*b2[2]}")
    print(f"Quadrance: {q2:.6f} (Expected: -3)")
    assert torch.abs(q2 - (-3.0)) < 1e-6, "Quadrance should be -3 for these points from Example 15"
    
    # Test case 3: Null points should raise ValueError
    a3 = torch.tensor([1.0, 0.0, 1.0])  # Null point (norm = 0)
    b3 = torch.tensor([2.0, 0.0, 2.0])  # Null point (norm = 0)
    try:
        q3 = uhg.quadrance(a3, b3)
        q3_cr = uhg.quadrance_from_cross_ratio(a3, b3)
        assert False, "Should have raised ValueError for null points"
    except ValueError as e:
        print("\nTest 3 - Null points correctly raise ValueError")
        assert str(e) == "Quadrance is undefined for null points"

def test_spread():
    """Test spread calculation with known values."""
    print("\nTesting Spread Calculation")
    print("-" * 50)
    
    uhg = ProjectiveUHG()
    
    # Test case 1: Parallel lines (should give 0)
    L1 = torch.tensor([1.0, 0.0, 0.0])
    L2 = torch.tensor([1.0, 0.0, 1.0])
    s1 = uhg.spread(L1, L2)
    print(f"Test 1 - Parallel lines:")
    print(f"L1: {L1}")
    print(f"L2: {L2}")
    print(f"Spread: {s1:.6f} (Expected: 0)")
    assert torch.abs(s1) < 1e-6, "Spread should be 0 for parallel lines"
    
    # Test case 2: Perpendicular lines (should give 1)
    L3 = torch.tensor([1.0, 0.0, 0.0])
    L4 = torch.tensor([0.0, 1.0, 0.0])
    s2 = uhg.spread(L3, L4)
    print(f"\nTest 2 - Perpendicular lines:")
    print(f"L3: {L3}")
    print(f"L4: {L4}")
    print(f"Spread: {s2:.6f} (Expected: 1)")
    assert torch.abs(s2 - 1.0) < 1e-6, "Spread should be 1 for perpendicular lines"
    
    # Test case 3: Lines at 45 degrees (should give 0.5)
    L5 = torch.tensor([1.0, 0.0, 0.0])
    L6 = torch.tensor([1.0, 1.0, 0.0])
    s3 = uhg.spread(L5, L6)
    print(f"\nTest 3 - Lines at 45 degrees:")
    print(f"L5: {L5}")
    print(f"L6: {L6}")
    print(f"Spread: {s3:.6f} (Expected: 0.5)")
    assert torch.abs(s3 - 0.5) < 1e-6, "Spread should be 0.5 for 45-degree lines"
    
    # Test case 4: Scale invariance
    s4_orig = uhg.spread(L3, L4)
    s4_scaled = uhg.spread(2*L3, 3*L4)
    print(f"\nTest 4 - Scale invariance:")
    print(f"Original spread: {s4_orig:.6f}")
    print(f"Scaled spread: {s4_scaled:.6f}")
    assert torch.abs(s4_orig - s4_scaled) < 1e-6, "Spread should be scale invariant"

def test_cross_law():
    """Test cross law with known values."""
    print("\nTesting Cross Law")
    print("-" * 50)
    
    uhg = ProjectiveUHG()
    
    # Test case 1: Using points from Example 15 in UHG.pdf
    # Triangle with equal quadrances q1 = q2 = q3 = -3
    a1 = torch.tensor([math.sqrt(2), 0.0, math.sqrt(5)])
    a2 = torch.tensor([-1.0, math.sqrt(3), math.sqrt(10)])
    a3 = torch.tensor([-1.0, -math.sqrt(3), math.sqrt(10)])
    
    # Compute quadrances
    q1 = uhg.quadrance(a2, a3)  # Between a2 and a3
    q2 = uhg.quadrance(a1, a3)  # Between a1 and a3
    q3 = uhg.quadrance(a1, a2)  # Between a1 and a2
    
    # Get lines for spread calculation
    L12 = uhg.join(a1, a2)  # Line through a1 and a2
    L13 = uhg.join(a1, a3)  # Line through a1 and a3
    S1 = uhg.spread(L12, L13)  # Spread at a1
    
    print(f"\nTest 1 - Example 15 triangle:")
    print(f"Point a1: {a1}")
    print(f"Point a2: {a2}")
    print(f"Point a3: {a3}")
    print(f"q1 (a2a3): {q1:.6f}")
    print(f"q2 (a1a3): {q2:.6f}")
    print(f"q3 (a1a2): {q3:.6f}")
    print(f"S1: {S1:.6f}")
    
    # Verify cross law: (q2q3S1 - q1 - q2 - q3 + 2)² = 4(1 - q1)(1 - q2)(1 - q3)
    lhs = (q2*q3*S1 - q1 - q2 - q3 + 2)**2
    rhs = 4*(1 - q1)*(1 - q2)*(1 - q3)
    
    print(f"\nCross Law verification:")
    print(f"LHS = (q2q3S1 - q1 - q2 - q3 + 2)² = {lhs:.6f}")
    print(f"RHS = 4(1 - q1)(1 - q2)(1 - q3) = {rhs:.6f}")
    print(f"Difference: {torch.abs(lhs - rhs):.6f}")
    
    # Use a relative tolerance for the comparison
    relative_error = torch.abs(lhs - rhs) / (torch.abs(rhs) + 1e-6)
    assert relative_error < 1e-5, f"Cross law not satisfied, relative error: {relative_error:.6f}"
    
    # Test case 2: Generate random non-null points
    print("\nTest 2 - Random triangles:")
    for i in range(5):  # Test 5 different triangles
        # Generate random points and ensure they're non-null
        while True:
            # Generate points with z-coordinate larger than x,y to ensure negative norms
            z1 = 2.0 + abs(torch.randn(1).item())
            z2 = 2.0 + abs(torch.randn(1).item())
            z3 = 2.0 + abs(torch.randn(1).item())
            
            A = torch.tensor([torch.randn(1).item(), torch.randn(1).item(), z1])
            B = torch.tensor([torch.randn(1).item(), torch.randn(1).item(), z2])
            C = torch.tensor([torch.randn(1).item(), torch.randn(1).item(), z3])
            
            # Compute norms
            norm_a = A[0]**2 + A[1]**2 - A[2]**2
            norm_b = B[0]**2 + B[1]**2 - B[2]**2
            norm_c = C[0]**2 + C[1]**2 - C[2]**2
            
            # Check if points are non-null and have negative norms (hyperbolic condition)
            if (norm_a < -1e-6 and norm_b < -1e-6 and norm_c < -1e-6):
                break
        
        # Compute quadrances
        q1 = uhg.quadrance(B, C)
        q2 = uhg.quadrance(A, C)
        q3 = uhg.quadrance(A, B)
        
        # Get lines for spread calculation
        L1 = uhg.join(A, B)  # Line through A and B
        L2 = uhg.join(A, C)  # Line through A and C
        S1 = uhg.spread(L1, L2)  # Spread at A
        
        print(f"\nRandom triangle {i+1}:")
        print(f"Point A: {A}")
        print(f"Point B: {B}")
        print(f"Point C: {C}")
        print(f"q1 (BC): {q1:.6f}")
        print(f"q2 (AC): {q2:.6f}")
        print(f"q3 (AB): {q3:.6f}")
        print(f"S1: {S1:.6f}")
        
        # Verify cross law: (q2q3S1 - q1 - q2 - q3 + 2)² = 4(1 - q1)(1 - q2)(1 - q3)
        lhs = (q2*q3*S1 - q1 - q2 - q3 + 2)**2
        rhs = 4*(1 - q1)*(1 - q2)*(1 - q3)
        
        print(f"\nCross Law verification:")
        print(f"LHS = (q2q3S1 - q1 - q2 - q3 + 2)² = {lhs:.6f}")
        print(f"RHS = 4(1 - q1)(1 - q2)(1 - q3) = {rhs:.6f}")
        print(f"Difference: {torch.abs(lhs - rhs):.6f}")
        
        # Use a relative tolerance for the comparison
        relative_error = torch.abs(lhs - rhs) / (torch.abs(rhs) + 1e-6)
        assert relative_error < 1e-5, f"Cross law not satisfied, relative error: {relative_error:.6f}"
    
    print("\nCross law verified for Example 15 and random triangles")

def test_cross_ratio():
    """Test cross ratio calculation and its properties."""
    print("\nTesting Cross Ratio Calculation")
    print("-" * 50)
    
    uhg = ProjectiveUHG()
    
    # Test case 1: Using points from UHG.pdf example 15
    a = torch.tensor([math.sqrt(2), 0.0, math.sqrt(5)])
    b = torch.tensor([-1.0, math.sqrt(3), math.sqrt(10)])
    c = torch.tensor([-1.0, -math.sqrt(3), math.sqrt(10)])
    d = torch.tensor([0.0, 0.0, 1.0])  # Reference point
    
    # Compute cross ratio
    cr1 = uhg.cross_ratio(a, b, c, d)
    print(f"\nTest 1 - Cross ratio with example points:")
    print(f"CR(A,B;C,D) = {cr1:.6f}")
    
    # Test case 2: Verify CR(A,B;C,D) * CR(A,B;D,C) = 1
    cr2 = uhg.cross_ratio(a, b, d, c)
    product = cr1 * cr2
    print(f"\nTest 2 - Cross ratio identity:")
    print(f"CR(A,B;C,D) = {cr1:.6f}")
    print(f"CR(A,B;D,C) = {cr2:.6f}")
    print(f"Product = {product:.6f} (Expected: 1.0)")
    assert torch.abs(product - 1.0) < 1e-5, "Cross ratio identity should hold"
    
    # Test case 3: Projective invariance
    # Use a proper hyperbolic transformation matrix
    # This matrix preserves the hyperbolic form x² + y² - z² = 0
    matrix = torch.tensor([
        [math.cosh(0.5), 0.0, math.sinh(0.5)],
        [0.0, 1.0, 0.0],
        [math.sinh(0.5), 0.0, math.cosh(0.5)]
    ])
    
    # Transform and normalize points
    def hyperbolic_normalize(p):
        norm = torch.sqrt(torch.abs(p[0]**2 + p[1]**2 - p[2]**2))
        return p / norm
    
    a_transformed = hyperbolic_normalize(matrix @ a)
    b_transformed = hyperbolic_normalize(matrix @ b)
    c_transformed = hyperbolic_normalize(matrix @ c)
    d_transformed = hyperbolic_normalize(matrix @ d)
    
    cr_transformed = uhg.cross_ratio(a_transformed, b_transformed, c_transformed, d_transformed)
    print(f"\nTest 3 - Projective invariance:")
    print(f"Original CR = {cr1:.6f}")
    print(f"Transformed CR = {cr_transformed:.6f}")
    assert torch.abs(cr1 - cr_transformed) < 1e-5, "Cross ratio should be projectively invariant"
    
    # Test case 4: Numerical stability with scaled points
    scale = torch.tensor([2.0, 3.0, 4.0, 5.0])
    cr_scaled = uhg.cross_ratio(
        a * scale[0], 
        b * scale[1], 
        c * scale[2], 
        d * scale[3]
    )
    print(f"\nTest 4 - Scale invariance:")
    print(f"Original CR = {cr1:.6f}")
    print(f"Scaled CR = {cr_scaled:.6f}")
    assert torch.abs(cr1 - cr_scaled) < 1e-5, "Cross ratio should be scale invariant"
    
    # Test case 5: Collinear points
    print("\nTest 5 - Collinear points:")
    # Points on the same line in hyperbolic space
    col_a = torch.tensor([0.0, 0.0, 1.0])
    col_b = torch.tensor([1.0, 0.0, 2.0])
    col_c = torch.tensor([2.0, 0.0, 3.0])
    col_d = torch.tensor([3.0, 0.0, 4.0])
    
    cr_col = uhg.cross_ratio(col_a, col_b, col_c, col_d)
    print(f"CR for collinear points = {cr_col:.6f}")
    
    # Test case 6: Points approaching null points
    print("\nTest 6 - Near null points:")
    # Points very close to the null cone
    epsilon = 1e-6
    near_null_a = torch.tensor([1.0, 0.0, 1.0 + epsilon])
    near_null_b = torch.tensor([1.0, epsilon, 1.0 + epsilon])
    near_null_c = torch.tensor([1.0, -epsilon, 1.0 + epsilon])
    near_null_d = torch.tensor([1.0 + epsilon, 0.0, 1.0 + epsilon])
    
    try:
        cr_near_null = uhg.cross_ratio(near_null_a, near_null_b, near_null_c, near_null_d)
        print(f"CR for near-null points = {cr_near_null:.6f}")
    except ValueError as e:
        print(f"Expected error for near-null points: {str(e)}")
    
    # Test case 7: Points at infinity
    print("\nTest 7 - Points at infinity:")
    # Points with very large z-coordinate
    inf_scale = 1e6
    inf_a = torch.tensor([1.0, 0.0, inf_scale])
    inf_b = torch.tensor([2.0, 0.0, inf_scale])
    inf_c = torch.tensor([3.0, 0.0, inf_scale])
    inf_d = torch.tensor([4.0, 0.0, inf_scale])
    
    cr_inf = uhg.cross_ratio(inf_a, inf_b, inf_c, inf_d)
    print(f"CR for points at infinity = {cr_inf:.6f}")
    
    # Test case 8: Degenerate cases
    print("\nTest 8 - Degenerate cases:")
    # Test when three points coincide
    degen_a = torch.tensor([1.0, 0.0, 2.0])
    degen_b = torch.tensor([1.0, 0.0, 2.0])  # Same as a
    degen_c = torch.tensor([1.0, 0.0, 2.0])  # Same as a
    degen_d = torch.tensor([2.0, 0.0, 3.0])  # Different point
    
    try:
        cr_degen = uhg.cross_ratio(degen_a, degen_b, degen_c, degen_d)
        print(f"CR for degenerate case = {cr_degen:.6f}")
    except ValueError as e:
        print(f"Expected error for degenerate case: {str(e)}")

def main():
    """Run all tests."""
    print("\nTesting UHG Calculations")
    print("=" * 50)
    
    test_quadrance()
    test_spread()
    test_cross_law()
    test_cross_ratio()

if __name__ == "__main__":
    main() 