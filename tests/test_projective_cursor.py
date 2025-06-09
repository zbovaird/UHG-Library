import pytest
import torch
from torch.testing import assert_close
import sys
import os
import math

# Add the docs directory to the Python path
docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
sys.path.append(docs_dir)

from projective_cursor import ProjectiveUHG

def test_join_null_points():
    """Test joining two null points"""
    uhg = ProjectiveUHG(epsilon=1e-4)
    
    # Test case 1: Distinct null points
    t1, u1 = torch.tensor(1.0), torch.tensor(0.0)  # [1:0:1]
    t2, u2 = torch.tensor(1.0), torch.tensor(1.0)  # [0:2:2]
    
    line = uhg.join_null_points(t1, u1, t2, u2)
    assert line is not None, "Line should be created for distinct null points"
    
    # Test case 2: Same null points (should raise error)
    t1, u1 = torch.tensor(2.0), torch.tensor(1.0)
    t2, u2 = torch.tensor(4.0), torch.tensor(2.0)  # Same proportion as t1:u1
    
    with pytest.raises(ValueError, match="Null points must be distinct"):
        uhg.join_null_points(t1, u1, t2, u2)
        
    # Test case 3: Check if resulting line is normalized
    t1, u1 = torch.tensor(1.0), torch.tensor(0.0)
    t2, u2 = torch.tensor(0.0), torch.tensor(1.0)
    
    line = uhg.join_null_points(t1, u1, t2, u2)
    norm = torch.norm(line)
    # Use larger relative tolerance for floating point comparison
    assert torch.allclose(norm, torch.tensor(1.0), rtol=1e-3), "Line should be normalized"

def test_pythagoras():
    """
    Test Pythagoras' theorem in hyperbolic space
    According to UHG.pdf Theorem 42:
    For a right triangle (S₃ = 1), q₃ = q₁ + q₂ - q₁q₂
    """
    uhg = ProjectiveUHG()
    
    # Create points for a right triangle using the Poincaré disk model
    # In this model, perpendicular lines to a diameter are arcs of circles
    # orthogonal to the unit circle
    r = 0.5  # Distance from origin
    theta = math.pi/4  # 45 degree angle
    
    # Convert from Poincaré disk to hyperboloid model
    # x = (2x')/(1 + |x'|²), y = (2y')/(1 + |x'|²), z = (1 - |x'|²)/(1 + |x'|²)
    # where x' and y' are coordinates in the Poincaré disk
    
    # Point at origin
    a1 = torch.tensor([0.0, 0.0, 1.0])
    
    # Point along x-axis at distance r
    x2 = r
    y2 = 0
    denom2 = 1 + x2*x2 + y2*y2
    a2 = torch.tensor([2*x2/denom2, 2*y2/denom2, (1 - (x2*x2 + y2*y2))/denom2])
    
    # Point at angle theta and distance r
    x3 = r * math.cos(theta)
    y3 = r * math.sin(theta)
    denom3 = 1 + x3*x3 + y3*y3
    a3 = torch.tensor([2*x3/denom3, 2*y3/denom3, (1 - (x3*x3 + y3*y3))/denom3])
    
    # Calculate lines
    L1 = uhg.join_points(a2, a3)  # Line from a2 to a3
    L2 = uhg.join_points(a1, a2)  # Line from a1 to a2
    
    # Calculate spread (should be 1 for right angle)
    S3 = uhg.spread(L1, L2)
    print(f"\nSpread S₃ = {S3:.6f} (should be 1 for right angle)")
    
    # Calculate quadrances
    q1 = uhg.quadrance(a1, a2)  # First leg
    q2 = uhg.quadrance(a2, a3)  # Second leg
    q3 = uhg.quadrance(a1, a3)  # Hypotenuse
    
    print(f"q₁ = {q1:.6f}")
    print(f"q₂ = {q2:.6f}")
    print(f"q₃ = {q3:.6f}")
    
    # Calculate expected q3 using Pythagoras
    expected_q3 = q1 + q2 - q1*q2
    print(f"Expected q₃ = {expected_q3:.6f}")
    
    # Test assertions with appropriate tolerances
    assert_close(S3, torch.tensor(1.0), rtol=1e-4, atol=1e-4)
    assert_close(q3, expected_q3, rtol=1e-4, atol=1e-4)
