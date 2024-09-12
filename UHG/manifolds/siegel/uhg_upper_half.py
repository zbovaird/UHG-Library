import torch
from typing import Optional, Tuple, Union
from ..base import Manifold
from . import uhg_math

__all__ = ["UHGUpperHalf"]

class UHGUpperHalf(Manifold):
    """
    Universal Hyperbolic Geometry adaptation of the Upper Half Space model.

    In UHG, we represent points in this space using 2x2 real matrices of the form:
    [a b]
    [b d]
    where d > 0 and ad - b^2 > 0.
    """

    name = "UHG Upper Half Space"

    def __init__(self):
        super().__init__()

    def quadrance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quadrance between two points in the UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            First point, shape (..., 2, 2)
        y : torch.Tensor
            Second point, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Quadrance between x and y
        """
        diff = x - y
        return 1 - (torch.det(diff) / (torch.det(x) * torch.det(y)))**2

    def spread(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """
        Compute the spread between two lines in the UHG Upper Half Space.

        Parameters
        ----------
        l1 : torch.Tensor
            First line, represented as a point on the boundary, shape (..., 2)
        l2 : torch.Tensor
            Second line, represented as a point on the boundary, shape (..., 2)

        Returns
        -------
        torch.Tensor
            Spread between l1 and l2
        """
        diff = l1 - l2
        return torch.sum(diff**2, dim=-1) / (4 * l1[..., 1] * l2[..., 1])

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            Point to project, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Projected point
        """
        a, b, d = x[..., 0, 0], x[..., 0, 1], x[..., 1, 1]
        d = torch.clamp(d, min=1e-6)  # Ensure d > 0
        det = a * d - b**2
        det = torch.clamp(det, min=1e-6)  # Ensure ad - b^2 > 0
        a = torch.where(det <= 0, d + 1e-6, a)  # Adjust a if necessary
        return torch.stack([
            torch.stack([a, b], dim=-1),
            torch.stack([b, d], dim=-1)
        ], dim=-2)

    def random(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Generate random points in the UHG Upper Half Space.

        Parameters
        ----------
        size : tuple
            Shape of the tensor to generate
        dtype : torch.dtype, optional
            Data type of the tensor
        device : torch.device, optional
            Device on which to generate the tensor

        Returns
        -------
        torch.Tensor
            Random points in the UHG Upper Half Space
        """
        shape = size + (2, 2)
        x = torch.randn(shape, dtype=dtype, device=device)
        return self.projx(x)

    def origin(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Generate the origin point(s) in the UHG Upper Half Space.

        Parameters
        ----------
        size : tuple
            Shape of the tensor to generate
        dtype : torch.dtype, optional
            Data type of the tensor
        device : torch.device, optional
            Device on which to generate the tensor

        Returns
        -------
        torch.Tensor
            Origin point(s) in the UHG Upper Half Space
        """
        shape = size + (2, 2)
        origin = torch.eye(2, dtype=dtype, device=device).expand(shape)
        return origin

    def gyration(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Perform gyration operation in UHG Upper Half Space.

        Parameters
        ----------
        a : torch.Tensor
            First point, shape (..., 2, 2)
        b : torch.Tensor
            Second point, shape (..., 2, 2)
        x : torch.Tensor
            Point to gyrate, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Gyrated point
        """
        m = torch.matmul(torch.inverse(a), b)
        return torch.matmul(torch.matmul(m, x), torch.inverse(m))

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Mobius addition in UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            First point, shape (..., 2, 2)
        y : torch.Tensor
            Second point, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Result of Mobius addition
        """
        i = torch.eye(2, dtype=x.dtype, device=x.device).expand_as(x)
        return torch.matmul(x, torch.matmul(i + y, torch.inverse(i + torch.matmul(torch.inverse(x), y))))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance between two points in the UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            First point, shape (..., 2, 2)
        y : torch.Tensor
            Second point, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Distance between x and y
        """
        return 2 * torch.arcsinh(torch.sqrt(self.quadrance(x, y) / 2))

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Exponential map in UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            Base point, shape (..., 2, 2)
        u : torch.Tensor
            Tangent vector, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Point reached by the exponential map
        """
        norm_u = torch.norm(u, dim=(-2, -1), keepdim=True)
        return self.mobius_add(x, torch.tanh(norm_u / 2) * u / norm_u)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map in UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            Base point, shape (..., 2, 2)
        y : torch.Tensor
            Target point, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Tangent vector
        """
        diff = self.mobius_add(-x, y)
        norm_diff = torch.norm(diff, dim=(-2, -1), keepdim=True)
        return 2 * torch.arctanh(norm_diff) * diff / norm_diff

    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport in UHG Upper Half Space.

        Parameters
        ----------
        x : torch.Tensor
            Start point, shape (..., 2, 2)
        y : torch.Tensor
            End point, shape (..., 2, 2)
        u : torch.Tensor
            Vector to transport, shape (..., 2, 2)

        Returns
        -------
        torch.Tensor
            Transported vector
        """
        lambda_x = 2 / (1 - torch.det(x))
        lambda_y = 2 / (1 - torch.det(y))
        return torch.sqrt(lambda_x / lambda_y).unsqueeze(-1).unsqueeze(-1) * u