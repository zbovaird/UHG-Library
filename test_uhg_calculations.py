"""
Test UHG calculations with synthetic data to verify mathematical correctness.
Focuses on quadrance, spread, and cross law calculations.
"""

import torch
import numpy as np
from uhg.projective import ProjectiveUHG

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
    print(f"Test 1 - Points on same line:")
    print(f"a1: {a1}")
    print(f"b1: {b1}")
    print(f"Quadrance: {q1:.6f} (Expected: 0)")
    assert torch.abs(q1) < 1e-6, "Quadrance should be 0 for points on same line"
    
    # Test case 2: Perpendicular points (should give 1)
    # Using hyperbolic points with zero inner product
    a2 = torch.tensor([1.0, 0.0, 2.0])  # Non-null hyperbolic point
    b2 = torch.tensor([0.0, 1.0, 2.0])  # Non-null hyperbolic point
    q2 = uhg.quadrance(a2, b2)
    
    # Debug prints
    print(f"\nTest 2 - Perpendicular points (Debug):")
    print(f"a2: {a2}")
    print(f"b2: {b2}")
    print(f"inner_prod: {a2[0]*b2[0] + a2[1]*b2[1] - a2[2]*b2[2]}")
    print(f"norm_a: {a2[0]*a2[0] + a2[1]*a2[1] - a2[2]*a2[2]}")
    print(f"norm_b: {b2[0]*b2[0] + b2[1]*b2[1] - b2[2]*b2[2]}")
    print(f"Quadrance: {q2:.6f} (Expected: 1)")
    assert torch.abs(q2 - 1.0) < 1e-6, "Quadrance should be 1 for perpendicular points"
    
    # Test case 3: Null points should raise ValueError
    a3 = torch.tensor([1.0, 0.0, 1.0])  # Null point (norm = 0)
    b3 = torch.tensor([2.0, 0.0, 2.0])  # Null point (norm = 0)
    try:
        q3 = uhg.quadrance(a3, b3)
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
    
    # Test case 1: Generate random non-null points
    # We'll test multiple sets to ensure the law holds generally
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
        
        print(f"\nTest triangle {i+1}:")
        print(f"Point A: {A}")
        print(f"Point B: {B}")
        print(f"Point C: {C}")
        print(f"q1 (BC): {q1:.6f}")
        print(f"q2 (AC): {q2:.6f}")
        print(f"q3 (AB): {q3:.6f}")
        
        # Debug prints for understanding the values
        print("\nDebug information:")
        print(f"A norm: {norm_a}")
        print(f"B norm: {norm_b}")
        print(f"C norm: {norm_c}")
        
        # Verify cross law: (q1 + q2 + q3)^2 = 2(q1^2 + q2^2 + q3^2)
        lhs = (q1 + q2 + q3)**2
        rhs = 2 * (q1**2 + q2**2 + q3**2)
        print(f"\nCross Law verification:")
        print(f"LHS: {lhs:.6f}")
        print(f"RHS: {rhs:.6f}")
        print(f"Difference: {torch.abs(lhs - rhs):.6f}")
        
        # Use a relative tolerance for the comparison
        relative_error = torch.abs(lhs - rhs) / (torch.abs(rhs) + 1e-6)
        assert relative_error < 1e-5, f"Cross law not satisfied, relative error: {relative_error:.6f}"
        
    print("\nCross law verified for multiple random triangles")

def main():
    """Run all tests."""
    print("\nTesting UHG Calculations")
    print("=" * 50)
    
    test_quadrance()
    test_spread()
    test_cross_law()

if __name__ == "__main__":
    main() 