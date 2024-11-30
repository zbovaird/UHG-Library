import torch
from typing import Union, Tuple, Optional
from .base import Manifold

class LorentzManifold(Manifold):
    """
    The Lorentz manifold implementation for Universal Hyperbolic Geometry.
    This manifold represents the hyperboloid model of hyperbolic space,
    as described in Chapter 6 of UHG.pdf.
    
    The Lorentz model embeds hyperbolic space in Minkowski space,
    where points satisfy the constraint:
        -x[0]^2 + sum(x[1:]^2) = -1/k
    
    This model has the advantage of making the hyperbolic inner product
    explicit through the Minkowski metric.
    
    References:
        - Chapter 6.2: The Lorentz Model
        - Chapter 6.3: Geodesics in the Lorentz Model
        - Chapter 6.4: Isometries and the Cross-Ratio
    """
    
    def __init__(self, k: float = 1.0):
        """
        Initialize the Lorentz manifold.
        
        The curvature k determines the scale of the hyperbolic space.
        As per UHG.pdf Chapter 6.2, k > 0 gives the standard hyperbolic space,
        while k < 0 would give spherical space (not implemented).
        
        Args:
            k: Curvature of the manifold (default: 1.0)
        
        Raises:
            ValueError: If k <= 0
        """
        super().__init__()
        if k <= 0:
            raise ValueError("Curvature must be positive for hyperbolic space")
        self.k = k
    
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point satisfies the Lorentz constraint: -x[0]^2 + sum(x[1:]^2) = -1/k
        
        This constraint comes from UHG.pdf Chapter 6.2, equation (6.2.1).
        Points must lie on the upper sheet of the hyperboloid.
        """
        if x[..., 0].lt(0).any():
            return False, "Points must lie on the upper sheet of the hyperboloid"
            
        lorentz_norm = -x[..., 0].pow(2) + x[..., 1:].pow(2).sum(dim=-1)
        constraint_satisfied = torch.allclose(lorentz_norm, torch.tensor(-1.0/self.k), atol=atol, rtol=rtol)
        if not constraint_satisfied:
            return False, f"Point does not satisfy Lorentz constraint: {lorentz_norm}"
        return True, None
    
    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if vector satisfies the Lorentz tangent space constraint: <x,u>_L = 0
        
        This constraint ensures the vector lies in the tangent space of the hyperboloid,
        as derived in UHG.pdf Chapter 6.3.
        """
        inner_prod = self.lorentz_inner(x, u)
        tangent_constraint = torch.allclose(inner_prod, torch.tensor(0.0), atol=atol, rtol=rtol)
        if not tangent_constraint:
            return False, f"Vector not in tangent space: {inner_prod}"
        return True, None
    
    def lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lorentz inner product.
        
        Implements the Minkowski metric as defined in UHG.pdf Chapter 6.2:
            <x,y>_L = -x[0]y[0] + sum(x[i]y[i]) for i > 0
        """
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False) -> torch.Tensor:
        """
        Compute the Riemannian metric at point x.
        
        The Riemannian metric in the Lorentz model is inherited from
        the ambient Minkowski space, as shown in UHG.pdf Chapter 6.2.
        """
        if v is None:
            v = u
        return self.lorentz_inner(u, v)
    
    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto the tangent space at x.
        
        The projection formula follows from the orthogonality condition
        in UHG.pdf Chapter 6.3, ensuring the result is tangent to the hyperboloid.
        """
        inner_prod = self.lorentz_inner(x, v)
        return v + inner_prod.unsqueeze(-1) * x * self.k
    
    def proj_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point x onto the Lorentz manifold.
        
        Projects onto the upper sheet of the hyperboloid while preserving
        the direction of x in the ambient space, as per UHG.pdf Chapter 6.2.
        """
        norm = torch.sqrt(torch.abs(self.lorentz_inner(x, x)))
        result = x / (norm.unsqueeze(-1) * torch.sqrt(torch.abs(self.k)))
        # Ensure points lie on the upper sheet
        result[..., 0] = torch.abs(result[..., 0])
        return result
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at point x with tangent vector v.
        
        Implements the exponential map formula from UHG.pdf Chapter 6.3,
        which follows geodesics in the hyperboloid model.
        """
        v_norm = torch.sqrt(torch.abs(self.lorentz_inner(v, v)))
        v_norm = torch.clamp(v_norm, min=self.eps)
        
        return torch.cosh(v_norm.unsqueeze(-1)) * x + \
               torch.sinh(v_norm.unsqueeze(-1)) * v / v_norm.unsqueeze(-1)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at point x for target point y.
        
        Implements the inverse of the exponential map using the
        hyperbolic distance formula from UHG.pdf Chapter 6.3.
        """
        inner = -self.lorentz_inner(x, y)
        inner = torch.clamp(inner, min=1.0 + self.eps)
        
        factor = torch.acosh(inner)
        diff = y - inner.unsqueeze(-1) * x
        return factor.unsqueeze(-1) * diff / torch.sqrt(inner.pow(2) - 1).unsqueeze(-1)
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of tangent vector v from x to y.
        
        Implements parallel transport along geodesics as described
        in UHG.pdf Chapter 6.4, preserving the hyperbolic inner product.
        """
        inner = -self.lorentz_inner(x, y)
        inner = torch.clamp(inner, min=1.0 + self.eps)
        
        return v + self.lorentz_inner(y - inner.unsqueeze(-1) * x, v).unsqueeze(-1) * \
               (x + y) / (inner + 1).unsqueeze(-1)
    
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points x and y.
        
        Implements the distance formula from UHG.pdf Chapter 6.3:
            d(x,y) = acosh(-<x,y>_L) / sqrt(k)
        
        This formula preserves all hyperbolic axioms and the cross-ratio.
        """
        inner = -self.lorentz_inner(x, y)
        inner = torch.clamp(inner, min=1.0 + self.eps)
        return torch.acosh(inner) / torch.sqrt(self.k)
    
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """Compute the distance between points x and y on the manifold."""
        inner = -self.lorentz_inner(x, y)
        inner = torch.clamp(inner, min=1.0 + self.eps)
        return torch.acosh(inner) / torch.sqrt(self.k) 