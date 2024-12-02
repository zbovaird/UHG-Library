import torch
from typing import Union, Tuple, Optional
from ..base import Manifold, ScalingInfo
from ...utils import size2shape, broadcast_shapes
from .math import lorentz_inner, lorentz_norm, lorentz_distance
import geoopt


__all__ = ["LorentzManifold"]


class LorentzManifold(Manifold):
    """
    Lorentz manifold implementation following UHG principles.
    
    This manifold represents hyperbolic space using the Lorentz model,
    which preserves hyperbolic invariants and follows the geometric
    constraints defined in UHG.pdf.
    """

    _scaling = Manifold._scaling
    name = "Lorentz"
    ndim = 0
    reversible = True

    def __init__(self, ndim=0):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        inner = lorentz_inner(x, x)
        valid = torch.abs(inner + 1) <= atol + rtol * torch.abs(inner)
        if not valid.all():
            return False, "Points do not lie on the Lorentz manifold"
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        inner = lorentz_inner(x, u)
        valid = torch.abs(inner) <= atol + rtol * torch.abs(inner)
        if not valid.all():
            return False, "Vectors do not lie in the tangent space"
        return True, None

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return lorentz_inner(u, v, keepdim=keepdim)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        return lorentz_norm(u, keepdim=keepdim)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inner = lorentz_inner(x, u, keepdim=True)
        return u + inner * x

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        norm = lorentz_norm(x, keepdim=True)
        return x / norm

    def exp_map(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = lorentz_norm(u, keepdim=True)
        exp = x * torch.cosh(norm_u) + u * torch.sinh(norm_u) / norm_u
        return self.projx(exp)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner = -lorentz_inner(x, y, keepdim=True)
        norm_diff = torch.sqrt(inner * inner - 1)
        return norm_diff * (y + inner * x) / inner

    def parallel_transport(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        inner = -lorentz_inner(x, y, keepdim=True)
        return v + lorentz_inner(y - x * inner, v, keepdim=True) * (x + y) / (inner + 1)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        return lorentz_distance(x, y, keepdim=keepdim)

    @_scaling(ScalingInfo(scale=-1, factor=1.0, bias=0.0), "random")
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=device, dtype=dtype)
        tens = std.new_empty(*size).normal_() * std + mean
        return geoopt.ManifoldTensor(tens, manifold=self)

    random = random_normal

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), "x")
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )

    def extra_repr(self):
        return "ndim={}".format(self.ndim)