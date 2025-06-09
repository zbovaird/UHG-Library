import torch
import uhg
import numpy as np

def test_cross_ratio_invariance():
    """Test that cross-ratio is preserved under projective transformations."""
    print("\n=== Testing Cross-Ratio Invariance ===")
    
    # Create four collinear points
    t = torch.linspace(0, 1, 4).reshape(-1, 1)
    points = torch.cat([t, t**2, torch.ones_like(t)], dim=1)
    
    uhg_proj = uhg.ProjectiveUHG()
    
    # Compute initial cross-ratio
    cr_before = uhg_proj.cross_ratio(points[0], points[1], points[2], points[3])
    print("Cross-ratio before transformation:", cr_before)
    
    # Apply random projective transformation
    matrix = uhg_proj.get_projective_matrix(dim=2)
    transformed_points = uhg_proj.transform(points, matrix)
    
    # Compute cross-ratio after transformation
    cr_after = uhg_proj.cross_ratio(
        transformed_points[0], 
        transformed_points[1], 
        transformed_points[2], 
        transformed_points[3]
    )
    print("Cross-ratio after transformation:", cr_after)
    
    # Check if cross-ratio is preserved
    is_preserved = torch.allclose(cr_before, cr_after, rtol=1e-5)
    print("Cross-ratio preserved:", is_preserved)
    assert is_preserved, "Cross-ratio not preserved under projective transformation!"

def test_projective_operations():
    """Test basic projective geometry operations."""
    print("\n=== Testing Projective Operations ===")
    
    uhg_proj = uhg.ProjectiveUHG()
    
    # Test join operation (line through two points)
    p1 = torch.tensor([1.0, 0.0, 1.0])
    p2 = torch.tensor([0.0, 1.0, 1.0])
    line = uhg_proj.join(p1, p2)
    print("Line through points:", line)
    
    # Test meet operation (intersection of two lines)
    l1 = torch.tensor([1.0, 0.0, 0.0])  # x = 0 line
    l2 = torch.tensor([0.0, 1.0, 0.0])  # y = 0 line
    intersection = uhg_proj.meet(l1, l2)
    print("Intersection point:", intersection)
    
    # Verify that points lie on the line
    dot1 = torch.dot(line, p1)
    dot2 = torch.dot(line, p2)
    print("Points lie on line:", torch.allclose(dot1, torch.tensor(0.0)) and 
                                torch.allclose(dot2, torch.tensor(0.0)))

def test_hyperbolic_properties():
    """Test basic hyperbolic geometric properties."""
    print("\n=== Testing Hyperbolic Properties ===")
    
    # Create points in the Klein model
    # Use smaller radius to avoid boundary issues
    radius = 0.5  # Points well inside unit disk
    n_points = 10
    
    # Create points along a hyperbolic geodesic (straight line in Klein model)
    t = torch.linspace(-0.8, 0.8, n_points)
    x = radius * t
    y = torch.zeros_like(t)
    points = torch.stack([x, y, torch.ones_like(x)], dim=1)
    
    uhg_proj = uhg.ProjectiveUHG()
    
    # Apply projective transformation
    matrix = uhg_proj.get_projective_matrix(dim=2)
    transformed_points = uhg_proj.transform(points, matrix)
    
    # Compute distances between consecutive points
    distances = []
    for i in range(len(points)-1):
        dist = uhg_proj.distance(transformed_points[i], transformed_points[i+1])
        distances.append(dist)
    distances = torch.stack(distances)
    
    print("Mean distance:", distances.mean().item())
    print("Distance std:", distances.std().item())
    
    # Verify distances are positive
    print("All distances positive:", torch.all(distances > 0))
    assert torch.all(distances > 0), "Found non-positive distances"
    
    # Verify triangle inequality for points along geodesic
    for i in range(len(points)-2):
        d12 = uhg_proj.distance(transformed_points[i], transformed_points[i+1])
        d23 = uhg_proj.distance(transformed_points[i+1], transformed_points[i+2])
        d13 = uhg_proj.distance(transformed_points[i], transformed_points[i+2])
        
        # Triangle inequality
        assert d13 <= d12 + d23 + uhg_proj.eps, "Triangle inequality violated"
        
        # Points on geodesic should approximately satisfy additivity
        # Allow for numerical error and projective distortion
        rel_error = torch.abs(d13 - (d12 + d23)) / (d13 + uhg_proj.eps)
        assert rel_error < 0.1, "Points too far from geodesic"
    
    print("Triangle inequality satisfied")
    print("Geodesic property verified")

if __name__ == "__main__":
    print(f"Testing UHG version: {uhg.__version__}\n")
    
    try:
        test_cross_ratio_invariance()
        test_projective_operations()
        test_hyperbolic_properties()
        print("\n✅ All core UHG tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}") 