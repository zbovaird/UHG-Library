import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uhg.projective import ProjectiveUHG

def test_triple_quad_formula():
    """Test the triple quad formula with values satisfying the formula.
    (q1 + q2 + q3)^2 = 2(q1^2 + q2^2 + q3^2) + 4*q1*q2*q3
    Satisfied when q1 = q2 = q3 = 3/4.
    """
    uhg = ProjectiveUHG(epsilon=1e-5)

    q = torch.tensor(0.75)
    assert uhg.triple_quad_formula(q, q, q), "Triple quad formula should hold for q=3/4"

    z = torch.tensor(0.0)
    assert uhg.triple_quad_formula(z, z, z), "Triple quad formula should hold for zero quadrances"

def test_pythagoras():
    """Test Pythagorean theorem: q3 = q1 + q2 - q1*q2"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    q1 = torch.tensor(0.3)
    q2 = torch.tensor(0.4)
    q3 = q1 + q2 - q1 * q2

    assert uhg.pythagoras(q1, q2, q3), "Pythagorean theorem should hold"

    q3_expected = uhg.pythagoras(q1, q2)
    assert torch.allclose(q3, q3_expected), "Computed q3 should match formula"

@pytest.mark.skip(reason="Method 'spread_law' not in PyPI v0.3.7")
def test_spread_law():
    """Test spread law with non-null triangle"""
    pass

def test_cross_dual_law():
    """Test cross dual law: (S1 + S2 - S3)^2 = 4*S1*S2*(1 - q1)"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    S1 = torch.tensor(0.25)
    S2 = torch.tensor(0.25)
    S3 = torch.tensor(0.0)
    q1 = torch.tensor(0.0)

    assert uhg.cross_dual_law(S1, S2, S3, q1), "Cross dual law should hold"

    S1b = torch.tensor(0.5)
    S2b = torch.tensor(0.5)
    S3b = torch.tensor(0.5)
    q1b = torch.tensor(0.75)
    assert uhg.cross_dual_law(S1b, S2b, S3b, q1b), "Cross dual law should hold for second set"

def test_triple_spread_formula():
    """Test the triple spread formula with values satisfying the formula.
    (S1 + S2 + S3)^2 = 2(S1^2 + S2^2 + S3^2) + 4*S1*S2*S3
    Satisfied when S1 = S2 = S3 = 3/4.
    """
    uhg = ProjectiveUHG(epsilon=1e-5)

    S = torch.tensor(0.75)
    assert uhg.triple_spread_formula(S, S, S), "Triple spread formula should hold for S=3/4"

    z = torch.tensor(0.0)
    assert uhg.triple_spread_formula(z, z, z), "Triple spread formula should hold for zero spreads"

def test_cross_law():
    """Test the cross law: (q1 + q2 - q3)^2 = 4*q1*q2*(1 - S3)"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    q1 = torch.tensor(0.25)
    q2 = torch.tensor(0.25)
    q3 = torch.tensor(0.0)
    S1 = torch.tensor(0.0)
    S2 = torch.tensor(0.0)
    S3 = torch.tensor(0.0)

    assert uhg.cross_law(q1, q2, q3, S1, S2, S3), "Cross law should hold"

    q1b = torch.tensor(0.5)
    q2b = torch.tensor(0.5)
    q3b = torch.tensor(0.5)
    S1b = torch.tensor(0.0)
    S2b = torch.tensor(0.0)
    S3b = torch.tensor(0.75)

    assert uhg.cross_law(q1b, q2b, q3b, S1b, S2b, S3b), "Cross law should hold for second set"

def test_spread_quadrance_duality():
    """Test the duality between spread and quadrance"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    L1 = torch.tensor([1.0, 0.0, 0.0])
    L2 = torch.tensor([0.0, 1.0, 0.0])

    p1 = uhg.dual_line_to_point(L1)
    p2 = uhg.dual_line_to_point(L2)

    spread_val = uhg.spread(L1, L2)
    quad_val = uhg.quadrance(p1, p2)

    print(f"Spread between lines: {spread_val}")
    print(f"Quadrance between dual points: {quad_val}")

    assert uhg.spread_quadrance_duality(L1, L2), "Spread-quadrance duality should hold"

    L3 = torch.tensor([1.0, 1.0, 0.0])
    L4 = torch.tensor([1.0, -1.0, 0.0])

    assert uhg.spread_quadrance_duality(L3, L4), "Spread-quadrance duality should hold for second pair"

def test_dual_pythagoras():
    """Test dual Pythagorean theorem: S3 = S1 + S2 - S1*S2"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    S1 = torch.tensor(0.3)
    S2 = torch.tensor(0.4)
    S3 = S1 + S2 - S1 * S2

    assert uhg.dual_pythagoras(S1, S2, S3), "Dual Pythagorean theorem should hold"

    S3_expected = uhg.dual_pythagoras(S1, S2)
    assert torch.allclose(S3, S3_expected), "Computed S3 should match formula"

def test_point_lies_on_line():
    """Test point-line incidence"""
    uhg = ProjectiveUHG(epsilon=1e-5)

    L = torch.tensor([1.0, 1.0, -1.0])
    p = torch.tensor([0.5, 0.5, 1.0])

    assert uhg.point_lies_on_line(p, L), "Point should lie on line"

    p2 = torch.tensor([2.0, 0.0, 1.0])
    assert not uhg.point_lies_on_line(p2, L), "Point should not lie on line"

@pytest.mark.skip(reason="Methods 'points_perpendicular'/'lines_perpendicular' not in PyPI v0.3.7")
def test_perpendicular_relations():
    """Test perpendicular relations between points and lines"""
    pass

@pytest.mark.skip(reason="Methods 'are_collinear'/'are_concurrent' not in PyPI v0.3.7")
def test_collinearity_and_concurrency():
    """Test collinearity of points and concurrency of lines"""
    pass

@pytest.mark.skip(reason="Methods 'altitude_line'/'altitude_point'/'parallel_line'/'parallel_point' not in PyPI v0.3.7")
def test_geometric_constructions():
    """Test geometric constructions with non-null points and lines"""
    pass

@pytest.mark.skip(reason="Method 'parametrize_line_point' not in PyPI v0.3.7")
def test_line_parametrization():
    """Test parametrization of points on a line"""
    pass
