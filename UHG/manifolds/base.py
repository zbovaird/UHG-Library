import torch
from typing import Union, Tuple, Optional, NamedTuple
from abc import ABC, abstractmethod

class ScalingInfo(NamedTuple):
    """Information about scaling operations in hyperbolic space."""
    scale: float
    factor: float
    bias: float

class Manifold(ABC):
    """Base class for all manifolds in Universal Hyperbolic Geometry.
    
    This class defines the interface that all manifolds must implement.
    Each manifold represents a geometric space with its own metric and operations.
    All implementations must strictly follow the principles outlined in UHG.pdf,
    particularly the foundational axioms in Chapter 1 and the metric properties
    in Chapter 3.
    
    References:
        - Chapter 1: Foundational Principles of UHG
        - Chapter 3: Metric Properties and Distance Functions
        - Chapter 4: Transformations and Invariants
    """
    
    def __init__(self):
        """Initialize the manifold with numerical stability parameters."""
        super().__init__()
        self.eps = 1e-8  # Numerical stability threshold
        self.max_norm = 1e8  # Maximum allowed norm for numerical stability
    
    @abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """Compute the hyperbolic distance between points x and y on the manifold.
        
        As per UHG.pdf Chapter 3, all distance calculations must preserve
        hyperbolic invariants and use proper hyperbolic metrics.
        
        Args:
            x: Point on the manifold
            y: Point on the manifold
            keepdim: Whether to keep the dimension of the output
            
        Returns:
            Hyperbolic distance between x and y
        """
        raise NotImplementedError
    
    @abstractmethod
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at point x with tangent vector v.
        
        Implements the exponential map as defined in UHG.pdf Chapter 4,
        preserving hyperbolic invariants during the transformation.
        
        Args:
            x: Point on the manifold
            v: Tangent vector at x
            
        Returns:
            Point on the manifold reached by following v from x
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at point x for target point y.
        
        Implements the inverse of the exponential map, following
        UHG.pdf Chapter 4's principles on hyperbolic transformations.
        
        Args:
            x: Point on the manifold
            y: Target point on the manifold
            
        Returns:
            Tangent vector at x pointing toward y
        """
        raise NotImplementedError
    
    @abstractmethod
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport of tangent vector v from x to y.
        
        Implements parallel transport as defined in UHG.pdf Chapter 5,
        preserving the hyperbolic inner product during transport.
        
        Args:
            x: Source point on the manifold
            y: Target point on the manifold
            v: Tangent vector at x
            
        Returns:
            Transported vector at y
        """
        raise NotImplementedError
    
    @abstractmethod
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto the tangent space at point x.
        
        Following UHG.pdf Chapter 4's principles on tangent spaces
        and their relationship to the manifold structure.
        
        Args:
            x: Point on the manifold
            v: Vector to project
            
        Returns:
            Projected vector in the tangent space at x
        """
        raise NotImplementedError
    
    @abstractmethod
    def proj_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project point x onto the manifold.
        
        Ensures points satisfy the manifold constraints as per
        UHG.pdf Chapter 2's geometric constraints.
        
        Args:
            x: Point to project
            
        Returns:
            Projected point on the manifold
        """
        raise NotImplementedError
    
    @abstractmethod
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, 
             *, keepdim=False) -> torch.Tensor:
        """Compute the hyperbolic inner product of tangent vectors u and v at point x.
        
        Implements the hyperbolic inner product as defined in UHG.pdf Chapter 3,
        ensuring all metric properties are preserved.
        
        Args:
            x: Point on the manifold
            u: First tangent vector
            v: Second tangent vector (optional, defaults to u)
            keepdim: Whether to keep the dimension of the output
            
        Returns:
            Inner product value
        """
        raise NotImplementedError
    
    def check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
                              ) -> Union[Tuple[bool, Optional[str]], bool]:
        """Check if point x lies on the manifold.
        
        Verifies that points satisfy the geometric constraints
        defined in UHG.pdf Chapter 2.
        """
        if torch.isnan(x).any():
            return False, "NaN values detected in point coordinates"
        if torch.isinf(x).any():
            return False, "Infinite values detected in point coordinates"
        if x.abs().max() > self.max_norm:
            return False, f"Point coordinates exceed maximum allowed norm: {self.max_norm}"
        return self._check_point_on_manifold(x, atol=atol, rtol=rtol)
    
    @abstractmethod
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
                               ) -> Union[Tuple[bool, Optional[str]], bool]:
        """Implementation of point checking logic."""
        raise NotImplementedError
    
    def check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
                               ) -> Union[Tuple[bool, Optional[str]], bool]:
        """Check if vector u lies in the tangent space of x.
        
        Verifies that vectors satisfy the tangent space constraints
        defined in UHG.pdf Chapter 4.
        """
        if torch.isnan(u).any():
            return False, "NaN values detected in vector coordinates"
        if torch.isinf(u).any():
            return False, "Infinite values detected in vector coordinates"
        if u.abs().max() > self.max_norm:
            return False, f"Vector coordinates exceed maximum allowed norm: {self.max_norm}"
        return self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
    
    @abstractmethod
    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
                                ) -> Union[Tuple[bool, Optional[str]], bool]:
        """Implementation of vector checking logic."""
        raise NotImplementedError
    
    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute points on the geodesic between x and y at times t.
        
        Implements geodesic curves as defined in UHG.pdf Chapter 3,
        ensuring they satisfy the hyperbolic axioms and preserve
        the cross-ratio.
        
        Args:
            x: Starting point on the manifold
            y: Ending point on the manifold
            t: Times at which to evaluate the geodesic
            
        Returns:
            Points on the geodesic at times t
        """
        # Ensure inputs are valid
        self.check_point_on_manifold(x)
        self.check_point_on_manifold(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("Invalid time parameters for geodesic")
            
        v = self.log_map(x, y)
        return self.exp_map(x, t.view(-1, 1, 1) * v)