import torch
import uhg

def test_basic_operations():
    # Create points in projective space
    x = torch.randn(10, 3)
    y = torch.randn(10, 3)

    # Initialize UHG
    uhg_proj = uhg.ProjectiveUHG()

    # Get a projective transformation matrix
    matrix = uhg_proj.get_projective_matrix(dim=2)  # for 3D points (2D + 1 for homogeneous coordinates)

    # Transform points
    x_proj = uhg_proj.transform(x, matrix)
    y_proj = uhg_proj.transform(y, matrix)

    # Compute projective distance
    dist = uhg_proj.proj_dist(x_proj, y_proj)
    print("Projective distances:", dist)

    # Compute cross-ratio
    cr = uhg_proj.cross_ratio(x_proj[0], x_proj[1], x_proj[2], x_proj[3])
    print("Cross-ratio:", cr)

if __name__ == "__main__":
    print("Testing UHG version:", uhg.__version__)
    test_basic_operations() 