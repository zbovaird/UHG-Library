import pytest
import torch
from uhg.transform import UHGTransform

def test_transform_initialization():
    """Test initialization of a UHG transformation."""
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform = UHGTransform(matrix)
    assert torch.allclose(transform.matrix, matrix)

def test_apply_to_point():
    """Test applying a UHG transformation to a point."""
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform = UHGTransform(matrix)
    point = torch.tensor([1.0, 2.0, 3.0])
    transformed_point = transform.apply_to_point(point)
    assert torch.allclose(transformed_point, point)

def test_apply_to_line():
    """Test applying a UHG transformation to a line."""
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform = UHGTransform(matrix)
    line = torch.tensor([1.0, 2.0, 3.0])
    transformed_line = transform.apply_to_line(line)
    assert torch.allclose(transformed_line, line) 