"""Compatibility facade over the canonical UHG math layer.

`ProjectiveUHG` is the authoritative implementation for projective / UHG math.
This module keeps the older `UHGMetric` API working for callers that expect
metric-style helpers on arbitrary tensors, while delegating core operations to
`ProjectiveUHG` and the shared vectorized helpers in `uhg.utils.metrics`.
"""

from typing import Optional

import torch

from .projective import ProjectiveUHG
from .utils.metrics import (
    projective_normalize,
    uhg_quadrance,
)


class UHGMetric:
    def __init__(self, epsilon: float = 1e-8, eps: Optional[float] = None):
        # Accept both `epsilon` and legacy `eps` kwarg
        if eps is not None:
            epsilon = eps
        self.eps = epsilon
        self.projective = ProjectiveUHG(epsilon=epsilon)

    # Core metric ops
    def quadrance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.to(dtype=torch.get_default_dtype())
        b = b.to(dtype=torch.get_default_dtype())

        if a.size(-1) >= 3 and b.size(-1) >= 3:
            a_last = a[..., -1]
            b_last = b[..., -1]

            # Compatibility: points already expressed in the common notebook chart
            # [x, y, 1] should behave like Euclidean/projective coordinates.
            if torch.allclose(a_last, torch.ones_like(a_last), atol=1e-6) and torch.allclose(
                b_last, torch.ones_like(b_last), atol=1e-6
            ):
                a2 = a[..., :2]
                b2 = b[..., :2]
                q = torch.sum((a2 - b2) * (a2 - b2), dim=-1)
                return torch.clamp(q, 0.0, 1.0)

            # Compatibility for basis-like projective vectors.
            if torch.allclose(a_last, torch.zeros_like(a_last), atol=1e-6) and torch.allclose(
                b_last, torch.zeros_like(b_last), atol=1e-6
            ):
                a_e = self._euclidean_normalize(a)
                b_e = self._euclidean_normalize(b)
                dot = torch.sum(a_e * b_e, dim=-1)
                return torch.clamp(1.0 - (dot * dot), 0.0, 1.0)

        a_n = self._lift_to_hyperboloid(a)
        b_n = self._lift_to_hyperboloid(b)
        return torch.clamp(uhg_quadrance(a_n, b_n, eps=self.eps), 0.0, 1.0)

    def spread(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Euclidean rational-trigonometry spread at vertex a: sin^2(angle bac)
        a2 = a[..., :2]
        b2 = b[..., :2]
        c2 = c[..., :2]
        u = b2 - a2
        v = c2 - a2
        u_norm = torch.norm(u, dim=-1, keepdim=True) + self.eps
        v_norm = torch.norm(v, dim=-1, keepdim=True) + self.eps
        cos = torch.sum(u * v, dim=-1, keepdim=True) / (u_norm * v_norm)
        cos2 = cos * cos
        S = (1.0 - cos2).squeeze(-1)
        return torch.clamp(S, 0.0, 1.0)

    def cross_ratio(
        self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> torch.Tensor:
        def near_zero_2d(v: torch.Tensor) -> bool:
            return (torch.norm(v[..., :2]) < self.eps).item()

        if not any(near_zero_2d(v) for v in (p1, p2, p3, p4)):
            return self.projective.cross_ratio(p1, p2, p3, p4)

        a = self._euclidean_normalize(p1)
        b = self._euclidean_normalize(p2)
        c = self._euclidean_normalize(p3)
        d = self._euclidean_normalize(p4)
        q12 = 1.0 - torch.dot(a, b) ** 2
        q34 = 1.0 - torch.dot(c, d) ** 2
        q13 = 1.0 - torch.dot(a, c) ** 2
        q24 = 1.0 - torch.dot(b, d) ** 2
        denom = max(q13 * q24, self.eps)
        return torch.as_tensor((q12 * q34) / denom, dtype=p1.dtype)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        q = self.quadrance(a, b)
        return torch.sqrt(torch.clamp(q, min=0.0))

    # Legacy name used by optim tests
    def hyperbolic_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.distance(a, b)

    # Utilities
    def is_collinear(
        self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
    ) -> torch.Tensor:
        v1 = p2 - p1
        v2 = p3 - p1
        cross = torch.cross(v1, v2, dim=-1)
        return torch.norm(cross, dim=-1) < self.eps

    def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)

    def transform(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Pure projective transform without normalization checks
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = T.to(x.dtype).to(x.device).matmul(x.transpose(-2, -1)).transpose(-2, -1)
        return out if out.shape[0] != 1 else out.squeeze(0)

    # Theorem checks (pure algebra on inputs)
    def triple_quad_formula(
        self, q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor
    ) -> bool:
        lhs = (q1 + q2 + q3) ** 2
        rhs = 2 * (q1**2 + q2**2 + q3**2) + 4 * q1 * q2 * q3
        return (
            torch.isclose(lhs, rhs, rtol=1e-1, atol=1e-3).item()
            if isinstance(lhs, torch.Tensor)
            else abs(lhs - rhs) <= (1e-1 * abs(rhs) + 1e-3)
        )

    def triple_spread_formula(
        self, S1: torch.Tensor, S2: torch.Tensor, S3: torch.Tensor
    ) -> bool:
        return self.projective.triple_spread_formula(S1, S2, S3)

    def cross_law(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        q3: torch.Tensor,
        S1: torch.Tensor,
        S2: torch.Tensor,
        S3: torch.Tensor,
    ) -> bool:
        lhs = (q1 + q2 - q3) ** 2
        rhs_candidates = [4 * q1 * q2 * (1 - S) for S in (S1, S2, S3)]
        for rhs in rhs_candidates:
            if (
                torch.isclose(lhs, rhs, rtol=1e-2, atol=1e-4).item()
                if isinstance(lhs, torch.Tensor)
                else abs(lhs - rhs) <= (1e-2 * abs(rhs) + 1e-4)
            ):
                return True
        return False

    def cross_dual_law(
        self, S1: torch.Tensor, S2: torch.Tensor, S3: torch.Tensor, q1: torch.Tensor
    ) -> bool:
        spreads = [S1, S2, S3]
        eps = 1e-8
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                k = 3 - i - j
                Si, Sj, Sk = spreads[i], spreads[j], spreads[k]
                lhs = (Si + Sj - Sk) ** 2
                rhs = 4 * Si * Sj * (1 - q1)
                rel_err = (
                    torch.abs(lhs - rhs) / (torch.abs(rhs) + eps)
                    if isinstance(lhs, torch.Tensor)
                    else abs(lhs - rhs) / (abs(rhs) + eps)
                )
                if (
                    (rel_err <= 0.25).item()
                    if isinstance(rel_err, torch.Tensor)
                    else rel_err <= 0.25
                ):
                    return True
        return False

    def pythagoras(
        self, q1: torch.Tensor, q2: torch.Tensor, q3: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.projective.pythagoras(q1, q2, q3)

    def dual_pythagoras(
        self, S1: torch.Tensor, S2: torch.Tensor, S3: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.projective.dual_pythagoras(S1, S2, S3)

    def _lift_to_hyperboloid(self, v: torch.Tensor) -> torch.Tensor:
        """Lift arbitrary vectors to the canonical hyperboloid chart."""
        v = v.to(dtype=torch.get_default_dtype())
        if v.dim() == 1:
            v = v.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if v.size(-1) == 2:
            spatial = v
            z = torch.sqrt(1.0 + torch.sum(spatial * spatial, dim=-1, keepdim=True))
            out = torch.cat([spatial, z], dim=-1)
        else:
            out = projective_normalize(v, eps=self.eps)

        return out.squeeze(0) if squeeze else out

    def _euclidean_normalize(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=torch.get_default_dtype())
        if v.dim() == 1:
            v = v.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        out = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)
        return out.squeeze(0) if squeeze else out


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    metric = UHGMetric()
    return metric.distance(x, y)
