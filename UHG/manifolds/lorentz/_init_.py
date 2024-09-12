import torch as th
import torch.nn
import numpy as np
from typing import Tuple, Optional
from . import math
import geoopt
from ..base import Manifold, ScalingInfo
from ...utils import size2shape, broadcast_shapes

__all__ = ["UHGLorentz"]

_uhg_lorentz_doc = r"""
    Universal Hyperbolic Geometry Lorentz model

    Notes
    -----
    This implementation uses Universal Hyperbolic Geometry principles.
"""

class UHGLorentz(Manifold):
    __doc__ = r"""{}
    """.format(_uhg_lorentz_doc)

    ndim = 1
    reversible = False
    name = "UHGLorentz"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self):
        super().__init__()

    def _check_point_on_manifold(
        self, x: th.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        ok = not math.is_null_point(x, dim=dim)
        if not ok:
            reason = "'x' is a null point"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: th.Tensor, u: th.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        inner_ = math._inner(u, x, dim=dim)
        ok = th.allclose(inner_, th.zeros(1), atol=atol, rtol=rtol)
        if not ok:
            reason = "Inner product is not equal to zero"
        else:
            reason = None
        return ok, reason

    def quadrance(
        self, x: th.Tensor, y: th.Tensor, *, keepdim=False, dim=-1
    ) -> th.Tensor:
        return math.quadrance(x, y, keepdim=keepdim, dim=dim)

    def spread(
        self, L1: th.Tensor, L2: th.Tensor, *, keepdim=False, dim=-1
    ) -> th.Tensor:
        return math.spread(L1, L2, keepdim=keepdim, dim=dim)

    def norm(self, u: th.Tensor, *, keepdim=False, dim=-1) -> th.Tensor:
        return math._norm(u, keepdim=keepdim, dim=dim)

    def projx(self, x: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.project(x, dim=dim)

    def proju(self, x: th.Tensor, v: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.project_tangent(x, v, dim=dim)

    def expmap(
        self, x: th.Tensor, u: th.Tensor, *, project=True, dim=-1
    ) -> th.Tensor:
        res = math.expmap(x, u, dim=dim)
        if project:
            return math.project(res, dim=dim)
        else:
            return res

    def logmap(self, x: th.Tensor, y: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.logmap(x, y, dim=dim)

    def gyration(self, a: th.Tensor, b: th.Tensor, x: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.gyration(a, b, x, dim=dim)

    def mobius_add(self, x: th.Tensor, y: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.mobius_add(x, y, dim=dim)

    def inner(
        self,
        x: th.Tensor,
        u: th.Tensor,
        v: th.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> th.Tensor:
        if v is None:
            v = u
        return math._inner(u, v, dim=dim, keepdim=keepdim)

    def transp(
        self, x: th.Tensor, y: th.Tensor, v: th.Tensor, *, dim=-1
    ) -> th.Tensor:
        return math.parallel_transport(x, y, v, dim=dim)

    def join(self, a: th.Tensor, b: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.join(a, b, dim=dim)

    def meet(self, L1: th.Tensor, L2: th.Tensor, *, dim=-1) -> th.Tensor:
        return math.meet(L1, L2, dim=dim)

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` argument to None"
            )
        tens = th.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap(self.origin(*size), tens), manifold=self)

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        if dtype is None:
            dtype = th.get_default_dtype()
        if device is None:
            device = th.device('cpu')

        zero_point = th.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = 1
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    retr = expmap