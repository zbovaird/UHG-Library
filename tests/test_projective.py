import math

import numpy as np
import pytest
import torch

from uhg.projective import ProjectiveUHG
from uhg.utils.metrics import projective_normalize, uhg_quadrance_vectorized


@pytest.fixture
def uhg():
    return ProjectiveUHG(epsilon=1e-8)


def test_quadrance_matches_vectorized_helper(uhg):
    a = torch.tensor([1.0, 0.0, 2.0])
    b = torch.tensor([2.0, 0.0, 3.0])
    a_norm = uhg.normalize_points(a)
    b_norm = uhg.normalize_points(b)

    q = uhg.quadrance(a, b)
    expected = uhg_quadrance_vectorized(a_norm.unsqueeze(0), b_norm.unsqueeze(0))[0]
    assert torch.allclose(q, expected, atol=1e-6, rtol=1e-6)


def test_quadrance_is_scale_invariant_for_homogeneous_points(uhg):
    a = torch.tensor([1.0, 0.0, 2.0])
    b = torch.tensor([0.0, 1.0, 2.0])
    q1 = uhg.quadrance(a, b)
    q2 = uhg.quadrance(2.0 * a, 3.0 * b)
    assert torch.allclose(q1, q2, atol=1e-6, rtol=1e-6)


def test_cross_ratio_is_scale_invariant_and_involutive(uhg):
    a = torch.tensor([1.0, 0.0, 2.0])
    b = torch.tensor([2.0, 0.0, 3.0])
    c = torch.tensor([0.0, 1.0, 2.0])
    d = torch.tensor([1.0, 1.0, 3.0])

    cr = uhg.cross_ratio(a, b, c, d)
    cr_swapped = uhg.cross_ratio(a, b, d, c)
    cr_scaled = uhg.cross_ratio(2 * a, 3 * b, 4 * c, 5 * d)

    assert torch.isfinite(cr)
    assert torch.allclose(cr * cr_swapped, torch.ones_like(cr), atol=1e-6, rtol=1e-6)
    assert torch.allclose(cr, cr_scaled, atol=1e-6, rtol=1e-6)


def test_cross_ratio_batch_shape_and_finiteness(uhg):
    a = torch.tensor([[1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    b = torch.tensor([[2.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
    c = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 2.0]])
    d = torch.tensor([[1.0, 1.0, 3.0], [-2.0, 0.0, 3.0]])
    cr = uhg.cross_ratio(a, b, c, d)
    assert cr.shape == (2,)
    assert torch.isfinite(cr).all()


def test_spread_uses_pairwise_line_formula(uhg):
    l1 = torch.tensor([1.0, 0.0, 0.0])
    l2 = torch.tensor([2.0, 0.0, 0.0])
    l3 = torch.tensor([0.0, 1.0, 0.0])
    ref = torch.tensor([1.0, 1.0, 3.0])

    s_parallel = uhg.spread(l1, l2, ref)
    s_perp = uhg.spread(l1, l3, ref)

    assert torch.allclose(
        s_parallel, torch.zeros_like(s_parallel), atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(s_perp, torch.ones_like(s_perp), atol=1e-6, rtol=1e-6)


def test_normalize_points_requires_hyperbolic_input(uhg):
    p = torch.tensor([1.0, 2.0, 3.0])
    p_norm = uhg.normalize_points(p)
    norm = uhg.inner_product(p_norm, p_norm)
    assert torch.allclose(norm, torch.tensor(-1.0), atol=1e-5, rtol=1e-5)

    p_null = torch.tensor([1.0, 1.0, math.sqrt(2)])
    with pytest.raises(ValueError):
        uhg.normalize_points(p_null)

    p_eucl = torch.tensor([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        uhg.normalize_points(p_eucl)


def test_projective_normalize_lifts_euclidean_features(uhg):
    p = torch.tensor([[1.0, 1.0, 1.0], [0.5, -0.5, 1.0]])
    p_norm = uhg.projective_normalize(p)
    norms = uhg.inner_product(p_norm, p_norm)
    assert torch.allclose(norms, -torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_numerical_stability_on_small_and_large_hyperbolic_inputs(uhg):
    a = projective_normalize(torch.tensor([[1e-10, 0.0, 1.0]], dtype=torch.float32))[0]
    b = projective_normalize(torch.tensor([[0.0, 1e-10, 1.0]], dtype=torch.float32))[0]
    q_small = uhg.quadrance(a, b)
    assert torch.isfinite(q_small)

    a_big = projective_normalize(torch.tensor([[1e6, 0.0, 1.0]], dtype=torch.float64))[
        0
    ]
    b_big = projective_normalize(torch.tensor([[0.0, 1e6, 1.0]], dtype=torch.float64))[
        0
    ]
    q_big = uhg.quadrance(a_big, b_big)
    assert torch.isfinite(q_big)


def test_distance_requires_torch_tensor_inputs(uhg):
    a = np.array([1.0, 0.0, 2.0])
    b = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="torch.Tensor"):
        uhg.distance(a, b)


def test_distance_validates_homogeneous_shape_compatibility(uhg):
    a = torch.tensor([1.0, 0.0, 2.0])
    wrong_dim = torch.tensor([0.0, 1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="same homogeneous dimension"):
        uhg.distance(a, wrong_dim)

    batch = torch.ones(2, 3, 3)
    incompatible = torch.ones(4, 3)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        uhg.distance(batch, incompatible)
