import torch
from uhg.projective import ProjectiveUHG

def test_non_null_points():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create two non-null points
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 1.0, 0.0])  # [0:1:0]

    # Print points
    print(f'Points:')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')

    # First verify these points are not null
    print(f'\nChecking if points are null:')
    print(f'p1 is null: {uhg.is_null_point(p1)}')
    print(f'p2 is null: {uhg.is_null_point(p2)}')
    assert not uhg.is_null_point(p1), "p1 should not be a null point"
    assert not uhg.is_null_point(p2), "p2 should not be a null point"

    # Calculate hyperbolic dot product
    dot = uhg.hyperbolic_dot(p1, p2)
    print(f'\nHyperbolic dot product: {dot}')

    # Calculate quadrance
    q = uhg.quadrance(p1, p2)
    print(f'\nQuadrance: {q}')

    # Verify properties
    print(f'\nVerifying properties:')
    print(f'1. Quadrance is non-negative: {q >= 0}')
    assert q >= 0, f"Quadrance should be non-negative, got {q}"

    print(f'2. Quadrance is symmetric:')
    q_reverse = uhg.quadrance(p2, p1)
    print(f'   q(p1,p2) = {q}')
    print(f'   q(p2,p1) = {q_reverse}')
    assert torch.isclose(q, q_reverse, rtol=1e-5), f"Quadrance should be symmetric"

    print(f'3. If points are perpendicular (dot product = 0), quadrance should be 1')
    if torch.abs(dot) < 1e-6:
        assert torch.abs(q - 1.0) < 1e-6, f"Expected quadrance to be 1 for perpendicular points, got {q}"
        print(f'   Points are perpendicular and quadrance = {q} ≈ 1')
    else:
        print(f'   Points are not perpendicular (dot = {dot})')

def test_perpendicular_points():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create two non-null perpendicular points
    p1 = torch.tensor([2.0, 0.0, 1.0])  # [2:0:1]
    p2 = torch.tensor([0.0, 1.0, 0.0])  # [0:1:0]

    # Print points
    print(f'\nTest 1: Perpendicular Points')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')

    # First verify these points are not null
    print(f'\nChecking if points are null:')
    print(f'p1 is null: {uhg.is_null_point(p1)}')
    print(f'p2 is null: {uhg.is_null_point(p2)}')
    assert not uhg.is_null_point(p1), "p1 should not be a null point"
    assert not uhg.is_null_point(p2), "p2 should not be a null point"

    # Calculate hyperbolic dot product
    dot = uhg.hyperbolic_dot(p1, p2)
    print(f'\nHyperbolic dot product: {dot}')

    # Calculate quadrance
    q = uhg.quadrance(p1, p2)
    print(f'\nQuadrance: {q}')

    # Verify properties
    print(f'\nVerifying properties:')
    print(f'1. Quadrance is non-negative: {q >= 0}')
    assert q >= 0, f"Quadrance should be non-negative, got {q}"

    print(f'2. Quadrance is symmetric:')
    q_reverse = uhg.quadrance(p2, p1)
    print(f'   q(p1,p2) = {q}')
    print(f'   q(p2,p1) = {q_reverse}')
    assert torch.isclose(q, q_reverse, rtol=1e-5), f"Quadrance should be symmetric"

    print(f'3. If points are perpendicular (dot product = 0), quadrance should be 1')
    if torch.abs(dot) < 1e-6:
        assert torch.abs(q - 1.0) < 1e-6, f"Expected quadrance to be 1 for perpendicular points, got {q}"
        print(f'   Points are perpendicular and quadrance = {q} ≈ 1')
    else:
        print(f'   Points are not perpendicular (dot = {dot})')

def test_same_direction_points():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create two non-null points in the same direction
    p1 = torch.tensor([1.0, 0.0, 0.0])  # [1:0:0]
    p2 = torch.tensor([2.0, 0.0, 0.0])  # [2:0:0]

    # Print points
    print(f'\nTest 2: Points in Same Direction')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')

    # First verify these points are not null
    print(f'\nChecking if points are null:')
    print(f'p1 is null: {uhg.is_null_point(p1)}')
    print(f'p2 is null: {uhg.is_null_point(p2)}')
    assert not uhg.is_null_point(p1), "p1 should not be a null point"
    assert not uhg.is_null_point(p2), "p2 should not be a null point"

    # Calculate hyperbolic dot product
    dot = uhg.hyperbolic_dot(p1, p2)
    print(f'\nHyperbolic dot product: {dot}')

    # Calculate quadrance
    q = uhg.quadrance(p1, p2)
    print(f'\nQuadrance: {q}')

    # Verify properties
    print(f'\nVerifying properties:')
    print(f'1. Quadrance is non-negative: {q >= 0}')
    assert q >= 0, f"Quadrance should be non-negative, got {q}"

    print(f'2. Quadrance is symmetric:')
    q_reverse = uhg.quadrance(p2, p1)
    print(f'   q(p1,p2) = {q}')
    print(f'   q(p2,p1) = {q_reverse}')
    assert torch.isclose(q, q_reverse, rtol=1e-5), f"Quadrance should be symmetric"

    print(f'3. Points in same direction should have quadrance 0')
    assert torch.abs(q) < 1e-6, f"Expected quadrance to be 0 for same direction points, got {q}"
    print(f'   Points are in same direction and quadrance = {q} ≈ 0')

def test_another_pair_of_non_null_points():
    # Initialize UHG
    uhg = ProjectiveUHG()

    # Create two non-null points
    p1 = torch.tensor([1.0, 1.0, 1.0])  # [1:1:1]
    p2 = torch.tensor([1.0, 2.0, 1.0])  # [1:2:1]

    # Print points
    print(f'\nTest 3: Another Pair of Non-Null Points')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')

    # First verify these points are not null
    print(f'\nChecking if points are null:')
    print(f'p1 is null: {uhg.is_null_point(p1)}')
    print(f'p2 is null: {uhg.is_null_point(p2)}')
    assert not uhg.is_null_point(p1), "p1 should not be a null point"
    assert not uhg.is_null_point(p2), "p2 should not be a null point"

    # Calculate hyperbolic dot product
    dot = uhg.hyperbolic_dot(p1, p2)
    print(f'\nHyperbolic dot product: {dot}')

    # Calculate quadrance
    q = uhg.quadrance(p1, p2)
    print(f'\nQuadrance: {q}')

    # Verify properties
    print(f'\nVerifying properties:')
    print(f'1. Quadrance is non-negative: {q >= 0}')
    assert q >= 0, f"Quadrance should be non-negative, got {q}"

    print(f'2. Quadrance is symmetric:')
    q_reverse = uhg.quadrance(p2, p1)
    print(f'   q(p1,p2) = {q}')
    print(f'   q(p2,p1) = {q_reverse}')
    assert torch.isclose(q, q_reverse, rtol=1e-5), f"Quadrance should be symmetric"

    print(f'3. If points are perpendicular (dot product = 0), quadrance should be 1')
    if torch.abs(dot) < 1e-6:
        assert torch.abs(q - 1.0) < 1e-6, f"Expected quadrance to be 1 for perpendicular points, got {q}"
        print(f'   Points are perpendicular and quadrance = {q} ≈ 1')
    else:
        print(f'   Points are not perpendicular (dot = {dot})')

if __name__ == "__main__":
    test_non_null_points()
    test_perpendicular_points()
    test_same_direction_points()
    test_another_pair_of_non_null_points()
