import pytest
import torch
from uhg.points import UHGPoint

def test_point_initialization():
    """Test initialization of a UHG point."""
    coords = torch.tensor([1.0, 2.0, 3.0])
    point = UHGPoint(coords)
    assert torch.allclose(point.coords, coords)

def test_point_normalization():
    """Test normalizing a UHG point."""
    coords = torch.tensor([3.0, 4.0, 0.0])
    point = UHGPoint(coords)
    normalized_point = point.normalize()
    expected_coords = coords / torch.norm(coords)
    assert torch.allclose(normalized_point.coords, expected_coords)

def test_point_is_null():
    """Test checking if a UHG point is null."""
    null_point = UHGPoint(torch.tensor([1.0, 0.0, 1.0]))
    non_null_point = UHGPoint(torch.tensor([1.0, 0.0, 0.0]))
    assert null_point.is_null()
    assert not non_null_point.is_null()

def test_dual_line():
    """Test computing the dual line of a UHG point."""
    point = UHGPoint(torch.tensor([1.0, 2.0, 3.0]))
    dual_line = point.dual_line()
    assert torch.allclose(dual_line, point.coords) 