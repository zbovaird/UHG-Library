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
    
    # Create points in the unit disk model
    theta = torch.linspace(0, 2*np.pi, 100)
    radius = 0.9  # Points inside unit disk
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y, torch.ones_like(x)], dim=1)
    
    uhg_proj = uhg.ProjectiveUHG()
    
    # Apply projective transformation
    matrix = uhg_proj.get_projective_matrix(dim=2)
    transformed_points = uhg_proj.transform(points, matrix)
    
    # Compute distances between consecutive points
    distances = []
    for i in range(len(points)-1):
        dist = uhg_proj.proj_dist(transformed_points[i], transformed_points[i+1])
        distances.append(dist)
    distances = torch.stack(distances)
    
    print("Mean distance:", distances.mean().item())
    print("Distance std:", distances.std().item())
    
    # Verify distances are positive
    print("All distances positive:", torch.all(distances > 0))

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