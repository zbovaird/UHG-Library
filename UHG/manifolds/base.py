import torch
from typing import Union, Tuple, Optional, NamedTuple, Callable, Any
from abc import ABC, abstractmethod
import functools

__all__ = ["Manifold", "ScalingInfo"]

class ScalingInfo(NamedTuple):
    """Information about scaling operations in projective space."""
    scale: float
    factor: float
    bias: float

def scaling(info: ScalingInfo, method_name: str) -> Callable:
    """Decorator for scaling operations in projective space.
    
    Args:
        info: Scaling information
        method_name: Name of the method being decorated
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: Any, *args, **kwargs) -> Any:
            # Apply scaling before operation
            scaled_args = [arg * info.scale + info.bias if isinstance(arg, torch.Tensor) else arg 
                         for arg in args]
            scaled_kwargs = {k: v * info.scale + info.bias if isinstance(v, torch.Tensor) else v 
                           for k, v in kwargs.items()}
            
            # Call original function
            result = func(self, *scaled_args, **scaled_kwargs)
            
            # Apply inverse scaling after operation
            if isinstance(result, torch.Tensor):
                result = (result - info.bias) / info.scale
            
            return result
        return wrapper
    return decorator

class Manifold(ABC):
    """Base class for projective geometry in Universal Hyperbolic Geometry.
    
    This class defines the interface for projective operations.
    All implementations must strictly follow the principles outlined in UHG.pdf,
    particularly the projective geometry foundations in Chapter 3 and the
    cross-ratio invariants in Chapter 4.
    """
    
    @abstractmethod
    def cross_ratio(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Compute the cross-ratio of four points.
        
        Args:
            a, b, c, d: Points in projective space
            
        Returns:
            Cross-ratio value
        """
        pass
        
    @abstractmethod
    def join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the join of two points.
        
        Args:
            a, b: Points to join
            
        Returns:
            Join line in projective coordinates
        """
        pass
        
    @abstractmethod
    def meet(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """Compute the meet of two lines.
        
        Args:
            l1, l2: Lines to intersect
            
        Returns:
            Meet point in projective coordinates
        """
        pass
        
    @abstractmethod
    def transform(self, points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Apply a projective transformation.
        
        Args:
            points: Points to transform
            matrix: Projective transformation matrix
            
        Returns:
            Transformed points
        """
        pass