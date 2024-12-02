"""Utility functions for UHG library."""

import torch
from typing import Tuple, Union, List

def size2shape(*size: Union[torch.Size, Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
    """Convert size to shape.
    
    Args:
        *size: Size to convert
        
    Returns:
        Shape tuple
    """
    if len(size) == 1:
        size = size[0]
        if isinstance(size, (list, tuple)):
            return tuple(size)
        if isinstance(size, torch.Size):
            return tuple(size)
    return size

def broadcast_shapes(*shapes: Union[torch.Size, Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
    """Broadcast shapes to compatible size.
    
    Args:
        *shapes: Shapes to broadcast
        
    Returns:
        Broadcasted shape
    """
    shapes = [size2shape(s) for s in shapes]
    max_dim = max(len(s) for s in shapes)
    shapes = [(1,) * (max_dim - len(s)) + tuple(s) for s in shapes]
    result = []
    for dims in zip(*shapes):
        non_ones = [d for d in dims if d != 1]
        if not non_ones:
            result.append(1)
        elif all(d == non_ones[0] for d in non_ones):
            result.append(max(dims))
        else:
            raise ValueError(f"Shapes {shapes} cannot be broadcast together")
    return tuple(result) 