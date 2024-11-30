import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
from .manifolds.base import Manifold

class HyperbolicTensor(torch.Tensor):
    """Tensor class for Universal Hyperbolic Geometry.
    
    All operations are performed directly in hyperbolic space
    using UHG principles, without tangent space mappings.
    
    Args:
        data: Input tensor data
        manifold: The hyperbolic manifold to operate on
        requires_grad: Whether to track gradients
    """
    def __new__(
        cls,
        data: Union[torch.Tensor, List[float]],
        manifold: Optional[Manifold] = None,
        requires_grad: bool = False,
        **kwargs
    ):
        if isinstance(data, torch.Tensor):
            tensor = data.clone().detach()
        else:
            tensor = torch.tensor(data, **kwargs)
            
        if manifold is not None:
            # Ensure tensor lies in hyperbolic space
            tensor = manifold.normalize(tensor)
            
        instance = torch.Tensor._make_subclass(cls, tensor, requires_grad)
        instance.manifold = manifold
        return instance
        
    def mobius_add(self, other: torch.Tensor) -> torch.Tensor:
        """Addition in hyperbolic space using Möbius operations."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.mobius_add(self, other)
        
    def mobius_mul(self, scalar: Union[float, torch.Tensor]) -> torch.Tensor:
        """Scalar multiplication in hyperbolic space."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.mobius_scalar_mul(scalar, self)
        
    def distance(self, other: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance to another point."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.dist(self, other)
        
    def project(self) -> torch.Tensor:
        """Project tensor onto hyperbolic manifold."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.project(self)
        
    def normalize(self) -> torch.Tensor:
        """Normalize tensor to satisfy hyperbolic constraints."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.normalize(self)
        
    def mobius_matvec(self, matrix: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication in hyperbolic space."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.mobius_matvec(matrix, self)
        
    def reflect(self, line: torch.Tensor) -> torch.Tensor:
        """Reflect tensor in a line using UHG operations."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.reflect(self, line)
        
    def rotate(self, center: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rotate tensor around a center point."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.rotate(self, center, angle)
        
    def translate(self, direction: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Translate tensor along a direction vector."""
        if self.manifold is None:
            raise ValueError("Tensor must have an associated manifold")
        return self.manifold.translate(self, direction, distance)
        
    def __repr__(self) -> str:
        manifold_str = f" on {self.manifold}" if self.manifold else ""
        return f"HyperbolicTensor{manifold_str} containing:\n{super().__repr__()}"


class HyperbolicParameter(HyperbolicTensor, nn.Parameter):
    """Parameter class for Universal Hyperbolic Geometry.
    
    Extends HyperbolicTensor to be recognized as a module parameter.
    All operations follow UHG principles without tangent space mappings.
    
    Args:
        data: Input tensor data
        manifold: The hyperbolic manifold to operate on
        requires_grad: Whether to track gradients (default: True)
    """
    def __new__(
        cls,
        data: Union[torch.Tensor, List[float]],
        manifold: Optional[Manifold] = None,
        requires_grad: bool = True
    ):
        if isinstance(data, HyperbolicTensor):
            if manifold is not None and data.manifold != manifold:
                raise ValueError(
                    f"Manifold mismatch: {data.manifold} != {manifold}"
                )
            manifold = data.manifold
            
        instance = HyperbolicTensor.__new__(
            cls, data, manifold=manifold, requires_grad=requires_grad
        )
        instance._is_param = True
        return instance
        
    def __repr__(self) -> str:
        manifold_str = f" on {self.manifold}" if self.manifold else ""
        return f"HyperbolicParameter{manifold_str} containing:\n{super(HyperbolicTensor, self).__repr__()}"
