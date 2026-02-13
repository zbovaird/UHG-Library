import torch
import pytest

from uhg.projective import ProjectiveUHG
from uhg.utils.metrics import uhg_inner_product, uhg_quadrance, compute_cross_ratio


def minkowski_norm(x: torch.Tensor) -> torch.Tensor:
    return uhg_inner_product(x, x)


@pytest.mark.parametrize("points", [
    torch.tensor([0.3, 0.4, 1.0]),
    torch.tensor([0.8, 0.1, 1.5]),
])
def test_normalize_points_hyperboloid(points):
    uhg = ProjectiveUHG()
    x = uhg.normalize_points(points)
    assert x.shape[-1] == 3
    # Minkowski norm -1 within tolerance
    assert torch.allclose(minkowski_norm(x), torch.tensor(-1.0), atol=1e-6)
    # Time-like component positive (choose upper sheet)
    assert (x[..., -1] > 0).all()


def test_normalize_points_projects_null_to_hyperboloid():
    """Null points are projected onto the hyperboloid by recomputing time-like.

    This is the numerically robust behavior needed for ML training where
    float32 rounding can push points onto the null cone.
    """
    uhg = ProjectiveUHG()
    # Null point satisfies x^2 + y^2 - z^2 = 0
    p_null = torch.tensor([1.0, 0.0, 1.0])
    result = uhg.normalize_points(p_null)
    # Should now be on the hyperboloid (norm = -1)
    norm = uhg.inner_product(result, result)
    assert torch.abs(norm + 1.0) < 1e-6, f"Expected norm -1.0, got {norm}"
    # Time-like component positive
    assert result[-1] > 0


def test_distance_properties():
    uhg = ProjectiveUHG()
    a = torch.tensor([0.3, 0.1, 1.0])
    b = torch.tensor([0.2, 0.2, 1.0])
    a_n = uhg.normalize_points(a)
    b_n = uhg.normalize_points(b)
    d_ab = uhg.distance(a_n, b_n)
    d_aa = uhg.distance(a_n, a_n)
    assert d_ab >= 0
    assert torch.allclose(d_aa, torch.tensor(0.0), atol=1e-3)


def test_cross_ratio_invariance_under_projective_transform_collinear():
    uhg = ProjectiveUHG()
    # Build four collinear points on the line y = 0.5 x + 0.1, z=1
    xs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ys = 0.5 * xs + 0.1
    P = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    P = uhg.normalize_points(P)
    cr_before = compute_cross_ratio(P[0], P[1], P[2], P[3])

    # Apply an invertible projective transform
    T = torch.tensor([[1.2, 0.1, 0.0],
                      [0.0, 0.9, 0.1],
                      [0.1, 0.2, 1.1]])
    Q = uhg.transform(P, T)
    Q = uhg.normalize_points(Q)

    cr_after = compute_cross_ratio(Q[0], Q[1], Q[2], Q[3])
    assert torch.allclose(cr_before, cr_after, atol=1e-3, rtol=1e-3)


def test_join_incidence():
    uhg = ProjectiveUHG()
    p1 = uhg.normalize_points(torch.tensor([0.2, 0.0, 1.0]))
    p2 = uhg.normalize_points(torch.tensor([0.0, 0.3, 1.0]))
    line = uhg.join(p1, p2)
    # Incidence: a point lies on a line if dot(line, point)=0
    lhs1 = torch.sum(line * p1)
    lhs2 = torch.sum(line * p2)
    assert torch.allclose(lhs1, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(lhs2, torch.tensor(0.0), atol=1e-6)


def test_projective_aggregate_weighted_average_invariant_norm():
    uhg = ProjectiveUHG()
    points = uhg.normalize_points(torch.tensor([
        [0.2, 0.0, 1.0],
        [0.0, 0.3, 1.0],
        [0.1, 0.1, 1.0],
    ]))
    weights = torch.tensor([0.2, 0.5, 0.3])
    agg = uhg.aggregate(points.unsqueeze(0), weights.unsqueeze(0))
    # Aggregated point should be normalizable with Minkowski norm -1
    agg_n = uhg.normalize_points(agg.squeeze(0))
    assert torch.allclose(minkowski_norm(agg_n), torch.tensor(-1.0), atol=1e-6) 