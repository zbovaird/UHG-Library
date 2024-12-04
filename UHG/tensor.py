import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
from .projective import ProjectiveUHG

class UHGTensor(torch.Tensor):
    """
    A tensor class for UHG operations that uses projective geometry.
    All operations are performed using pure projective geometry principles.
    """
    
    @staticmethod
    def __new__(cls, data, requires_grad=False):
        """Create a new UHG tensor."""
        return torch.as_tensor(data, dtype=torch.float32).requires_grad_(requires_grad)
    
    def proj_transform(self, matrix=None):
        """Apply projective transformation."""
        uhg = ProjectiveUHG()
        if matrix is None:
            matrix = uhg.get_projective_matrix(self.size(-1))
        return uhg.transform(self, matrix)
    
    def cross_ratio(self, a, b, c):
        """Compute cross-ratio with three other points."""
        uhg = ProjectiveUHG()
        return uhg.cross_ratio(self, a, b, c)
    
    def proj_dist(self, other):
        """Compute projective distance to another point."""
        uhg = ProjectiveUHG()
        return uhg.proj_dist(self, other)

class UHGParameter(nn.Parameter):
    """Parameter class for UHG operations."""
    
    @staticmethod
    def __new__(cls, data=None, requires_grad=True):
        """Create a new UHG parameter."""
        if data is None:
            data = torch.Tensor()
        return nn.Parameter(data, requires_grad=requires_grad)
    
    def proj_transform(self, matrix=None):
        """Apply projective transformation."""
        uhg = ProjectiveUHG()
        if matrix is None:
            matrix = uhg.get_projective_matrix(self.size(-1))
        return uhg.transform(self, matrix)
    
    def cross_ratio(self, a, b, c):
        """Compute cross-ratio with three other points."""
        uhg = ProjectiveUHG()
        return uhg.cross_ratio(self, a, b, c)
    
    def proj_dist(self, other):
        """Compute projective distance to another point."""
        uhg = ProjectiveUHG()
        return uhg.proj_dist(self, other)
