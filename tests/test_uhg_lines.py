import pytest
import torch
from uhg.lines import UHGLine

def test_line_initialization():
    """Test initialization of a UHG line."""
    coords = torch.tensor([1.0, 2.0, 3.0])
    line = UHGLine(coords)
    assert torch.allclose(line.coords, coords)

def test_line_from_points():
    """Test constructing a UHG line from two points."""
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    line = UHGLine.from_points(p1, p2)
    expected_coords = torch.cross(p1, p2)
    assert torch.allclose(line.coords, expected_coords)

def test_point_lies_on_line():
    """Test checking if a point lies on a UHG line."""
    line = UHGLine(torch.tensor([1.0, 1.0, 0.0]))
    point_on_line = torch.tensor([1.0, -1.0, 0.0])
    point_not_on_line = torch.tensor([1.0, 1.0, 1.0])
    assert line.point_lies_on_line(point_on_line)
    assert not line.point_lies_on_line(point_not_on_line)

def test_dual_point():
    """Test computing the dual point of a UHG line."""
    line = UHGLine(torch.tensor([1.0, 2.0, 3.0]))
    dual_point = line.dual_point()
    assert torch.allclose(dual_point, line.coords) 