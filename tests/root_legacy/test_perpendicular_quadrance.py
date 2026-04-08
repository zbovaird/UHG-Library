import torch
from uhg.projective import ProjectiveUHG

def test_perpendicular_quadrance():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create two perpendicular points
    # In hyperbolic geometry, points are perpendicular when their hyperbolic dot product is 0
    # Points [2:0:1] and [0:1:0] are perpendicular
    p1 = torch.tensor([2.0, 0.0, 1.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])

    # First verify these points are perpendicular via hyperbolic dot product
    dot = uhg.hyperbolic_dot(p1, p2)
    print(f'Points:')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')
    print(f'\nHyperbolic dot product: {dot}')
    assert torch.abs(dot) < 1e-6, f"Points should be perpendicular (dot product = 0), got {dot}"

    # Calculate quadrance
    q = uhg.quadrance(p1, p2)
    print(f'Quadrance: {q}')
    print(f'Test passed: {torch.abs(q - 1.0) < 1e-6}')

    # Assert that quadrance is 1 (within numerical tolerance)
    assert torch.abs(q - 1.0) < 1e-6, f"Expected quadrance to be 1, got {q}"

    # Also verify that these points are not null
    assert not uhg.is_null_point(p1), "p1 should not be a null point"
    assert not uhg.is_null_point(p2), "p2 should not be a null point"

def test_null_point_quadrance():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create a null point [1:0:1] (lies on the null cone where x² + y² = z²)
    null_point = torch.tensor([1.0, 0.0, 1.0])
    regular_point = torch.tensor([2.0, 0.0, 1.0])

    # Verify the point is indeed null
    assert uhg.is_null_point(null_point), "Point [1:0:1] should be null"
    assert not uhg.is_null_point(regular_point), "Point [2:0:1] should not be null"

    # Attempt to calculate quadrance with null point - should raise ValueError
    try:
        q = uhg.quadrance(null_point, regular_point)
        assert False, "Expected ValueError when calculating quadrance with null point"
    except ValueError as e:
        print(f"\nExpected error raised: {e}")
        assert str(e) == "Quadrance is undefined for null points"

    try:
        q = uhg.quadrance(regular_point, null_point)
        assert False, "Expected ValueError when calculating quadrance with null point"
    except ValueError as e:
        print(f"Expected error raised: {e}")
        assert str(e) == "Quadrance is undefined for null points"

if __name__ == "__main__":
    test_perpendicular_quadrance()
    print("\nTesting null point behavior:")
    test_null_point_quadrance() 