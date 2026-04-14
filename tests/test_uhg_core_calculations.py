import math

import pytest
import torch

from uhg.projective import ProjectiveUHG


@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-6)


def test_quadrance_from_cross_ratio_matches_quadrance(uhg):
    a = torch.tensor([1.0, 0.0, 2.0])
    b = torch.tensor([2.0, 0.0, 3.0])
    q_direct = uhg.quadrance(a, b)
    q_cr = uhg.quadrance_from_cross_ratio(a, b)
    assert torch.allclose(q_direct, q_cr, atol=1e-6, rtol=1e-6)


def test_transform_preserves_cross_ratio_for_projective_scaling(uhg):
    a = uhg.normalize_points(torch.tensor([1.0, 0.0, 2.0]))
    b = uhg.normalize_points(torch.tensor([2.0, 0.0, 3.0]))
    c = uhg.normalize_points(torch.tensor([0.0, 1.0, 2.0]))
    d = uhg.normalize_points(torch.tensor([1.0, 1.0, 3.0]))
    t = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    cr_before = uhg.cross_ratio(a, b, c, d)
    a_t = uhg.transform(a, t)
    b_t = uhg.transform(b, t)
    c_t = uhg.transform(c, t)
    d_t = uhg.transform(d, t)
    cr_after = uhg.cross_ratio(a_t, b_t, c_t, d_t)

    for p in (a_t, b_t, c_t, d_t):
        norm = uhg.inner_product(p, p)
        assert torch.allclose(norm, torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
    assert torch.allclose(cr_before, cr_after, atol=1e-5, rtol=1e-5)


def test_midpoint_is_on_hyperboloid_and_equidistant(uhg):
    a = uhg.normalize_points(torch.tensor([1.0, 0.0, 2.0]))
    b = uhg.normalize_points(torch.tensor([2.0, 0.0, 3.0]))
    m = uhg.midpoint(a, b)

    norm = uhg.inner_product(m, m)
    assert torch.allclose(norm, torch.tensor(-1.0), atol=1e-5, rtol=1e-5)

    q1 = uhg.quadrance(a, m)
    q2 = uhg.quadrance(m, b)
    assert torch.allclose(q1, q2, atol=1e-5, rtol=1e-5)


def test_null_point_detection_and_quadrance_error(uhg):
    null_point = torch.tensor([1.0, 0.0, 1.0])
    regular_point = torch.tensor([2.0, 0.0, 1.0])

    assert uhg.is_null_point(null_point)
    assert not uhg.is_null_point(regular_point)

    with pytest.raises(ValueError, match="Quadrance is undefined for null points"):
        uhg.quadrance(null_point, regular_point)


def test_null_line_detection_raises_in_spread(uhg):
    null_line = torch.tensor([1.0, 1.0, math.sqrt(2.0)])
    non_null = torch.tensor([2.0, 0.0, 3.0])
    assert torch.abs(uhg.inner_product(null_line, null_line)) < 1e-6

    with pytest.raises(ValueError, match="Spread is undefined for null lines"):
        uhg.spread(null_line, non_null, non_null)


def test_triple_quad_formula_with_known_solution(uhg):
    q = torch.tensor(0.75)
    lhs = (q + q + q) ** 2
    rhs = 2 * (q**2 + q**2 + q**2) + 4 * q * q * q
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)
    assert uhg.triple_quad_formula(q, q, q)


def test_triple_spread_formula_with_known_solution(uhg):
    s = torch.tensor(0.75)
    lhs = (s + s + s) ** 2
    rhs = 2 * (s**2 + s**2 + s**2) + 4 * s * s * s
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)
    assert uhg.triple_spread_formula(s, s, s)


def test_pythagoras_and_dual_pythagoras_helpers(uhg):
    q1 = torch.tensor(0.2)
    q2 = torch.tensor(0.3)
    q3 = q1 + q2 - q1 * q2
    assert uhg.pythagoras(q1, q2, q3)

    s1 = torch.tensor(0.2)
    s2 = torch.tensor(0.3)
    s3 = s1 + s2 - s1 * s2
    assert uhg.dual_pythagoras(s1, s2, s3)


def test_cross_law_helper_accepts_consistent_inputs(uhg):
    q1 = torch.tensor(0.5)
    q2 = torch.tensor(0.5)
    q3 = torch.tensor(0.75)
    s3 = 1.0 - ((q1 + q2 - q3) ** 2) / (4.0 * q1 * q2)
    assert uhg.cross_law(q1, q2, q3, torch.tensor(0.1), torch.tensor(0.2), s3)