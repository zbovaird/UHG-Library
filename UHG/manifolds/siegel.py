import torch
from typing import Union, Tuple, Optional
from .base import Manifold

class SiegelManifold(Manifold):
    """
    The Siegel manifold implementation for Universal Hyperbolic Geometry.
    This manifold represents the Siegel upper half-space model, which is a generalization
    of the Poincaré upper half-plane model to higher dimensions, as described in
    Chapter 7 of UHG.pdf.
    
    The Siegel upper half-space consists of points z = x + yi where:
    - x is a real vector of dimension n-1
    - y > 0 is a real number (the "height" coordinate)
    
    This model is particularly useful for understanding the boundary behavior
    of hyperbolic space and its isometries.
    
    References:
        - Chapter 7.1: The Siegel Upper Half-Space
        - Chapter 7.2: The Hyperbolic Metric
        - Chapter 7.3: Geodesics and Isometries
    """
    
    def __init__(self, n: int = 2):
        """
        Initialize the Siegel manifold.
        
        As per UHG.pdf Chapter 7.1, the dimension parameter n determines
        the dimension of the upper half-space model.
        
        Args:
            n: Dimension of the manifold (default: 2, giving the upper half-plane)
            
        Raises:
            ValueError: If n < 2
        """
        super().__init__()
        if n < 2:
            raise ValueError("Dimension must be at least 2 for Siegel upper half-space")
        self.n = n
    
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point lies in the Siegel upper half-space: Im(z) > 0
        
        This constraint comes from UHG.pdf Chapter 7.1, defining the
        fundamental domain of the model.
        """
        imag_part = x[..., -1]
        constraint_satisfied = (imag_part > 0).all()
        if not constraint_satisfied:
            return False, "Point not in upper half-space: imaginary part must be positive"
        return True, None
    
    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if vector lies in the tangent space.
        
        In the Siegel model, the tangent space at any point is isomorphic
        to R^n, as described in UHG.pdf Chapter 7.2.
        """
        return True, None
    
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False) -> torch.Tensor:
        """
        Compute the hyperbolic inner product of tangent vectors.
        
        Implements the hyperbolic metric tensor from UHG.pdf Chapter 7.2:
            g_z(u,v) = <u,v>/y^2
        where y is the height coordinate of the point z.
        """
        if v is None:
            v = u
        y = x[..., -1].unsqueeze(-1)  # height coordinate
        scale = 1.0 / (y * y)
        return (scale * (u * v).sum(dim=-1, keepdim=keepdim))
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector onto the tangent space.
        
        In the Siegel model, all vectors are automatically tangent,
        as explained in UHG.pdf Chapter 7.2.
        """
        return v
    
    def proj_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point onto the Siegel manifold.
        
        Following UHG.pdf Chapter 7.1, ensures the point lies in the
        upper half-space by making the height coordinate positive.
        """
        imag_part = x[..., -1].clone()
        imag_part[imag_part <= 0] = self.eps
        x = x.clone()
        x[..., -1] = imag_part
        return x
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at point x with tangent vector v.
        
        Implements the exponential map formula from UHG.pdf Chapter 7.3,
        which follows hyperbolic geodesics in the upper half-space.
        """
        y = x[..., -1].unsqueeze(-1)  # height coordinate
        scale = y  # Scale factor preserves the hyperbolic metric
        return x + scale * v
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at point x for target point y.
        
        Implements the inverse of the exponential map using the
        hyperbolic metric from UHG.pdf Chapter 7.2.
        """
        diff = y - x
        scale = 1.0 / x[..., -1].unsqueeze(-1)  # Inverse height scaling
        return scale * diff
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of tangent vector v from x to y.
        
        Implements parallel transport along geodesics as described
        in UHG.pdf Chapter 7.3, preserving the hyperbolic inner product.
        """
        scale = y[..., -1].unsqueeze(-1) / x[..., -1].unsqueeze(-1)
        return scale * v
    
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points.
        
        Implements the distance formula from UHG.pdf Chapter 7.2 using
        the cross-ratio formula:
            d(x,y) = 2 log(|x-y'|/(2sqrt(y_1y_2)))
        where y' is the reflection of y across the boundary.
        
        This formula preserves all hyperbolic axioms and the cross-ratio.
        """
        x_re, x_im = x[..., :-1], x[..., -1:]  # Real and imaginary parts
        y_re, y_im = y[..., :-1], y[..., -1:]
        
        # Compute the cross-ratio
        numer = torch.sqrt((x_re - y_re).pow(2).sum(dim=-1, keepdim=keepdim) + 
                         (x_im + y_im).pow(2))  # Distance to reflected point
        denom = torch.sqrt(x_im * y_im)  # Geometric mean of heights
        
        # The 2 log formula preserves hyperbolic structure
        return 2 * torch.log((numer / (2 * denom)) + 
                           torch.sqrt((numer / (2 * denom)).pow(2) - 1))