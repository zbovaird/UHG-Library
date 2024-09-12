import torch.jit
import torch

# Helper functions for UHG operations
@torch.jit.script
def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )

@torch.jit.script
def _norm(u, keepdim: bool = False, dim: int = -1):
    return torch.sqrt(torch.clamp_min(_inner(u, u, keepdim=keepdim), 1e-8))

# Quadrance between points
@torch.jit.script
def quadrance(x, y, keepdim: bool = False, dim: int = -1):
    inner_prod = _inner(x, y, dim=dim, keepdim=keepdim)
    x_norm = _inner(x, x, dim=dim, keepdim=keepdim)
    y_norm = _inner(y, y, dim=dim, keepdim=keepdim)
    return 1 - (inner_prod**2) / (x_norm * y_norm)

# Quadrance from a point to the origin
@torch.jit.script
def quadrance0(x, keepdim: bool = False, dim: int = -1):
    x_norm = _inner(x, x, dim=dim, keepdim=keepdim)
    return 1 - x.narrow(dim, 0, 1)**2 / x_norm

# Spread between lines
@torch.jit.script
def spread(L1, L2, keepdim: bool = False, dim: int = -1):
    inner_prod = _inner(L1, L2, dim=dim, keepdim=keepdim)
    L1_norm = _inner(L1, L1, dim=dim, keepdim=keepdim)
    L2_norm = _inner(L2, L2, dim=dim, keepdim=keepdim)
    return 1 - (inner_prod**2) / (L1_norm * L2_norm)

# Project a point onto the UHG space
@torch.jit.script
def project(x, dim: int = -1):
    dn = x.size(dim) - 1
    left_ = torch.sqrt(
        1 + torch.norm(x.narrow(dim, 1, dn), p=2, dim=dim) ** 2
    ).unsqueeze(dim)
    right_ = x.narrow(dim, 1, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj

# Project a vector onto the tangent space at x
@torch.jit.script
def project_tangent(x, v, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x)

# Exponential map in UHG
@torch.jit.script
def expmap(x, u, dim: int = -1):
    u_norm = _norm(u, keepdim=True, dim=dim)
    return x * torch.cosh(u_norm) + u * torch.sinh(u_norm) / u_norm

# Logarithmic map in UHG
@torch.jit.script
def logmap(x, y, dim: int = -1):
    dist_ = quadrance(x, y, dim=dim, keepdim=True)
    diff = y - x * _inner(x, y, keepdim=True)
    return dist_.sqrt() * diff / _norm(diff, keepdim=True)

# Parallel transport in UHG
@torch.jit.script
def parallel_transport(x, y, v, dim: int = -1):
    logxy = logmap(x, y, dim=dim)
    dist = quadrance(x, y, dim=dim, keepdim=True).sqrt()
    return v - _inner(logxy, v, keepdim=True) / dist**2 * (logxy + logmap(y, x, dim=dim))

# Gyration operation (specific to UHG)
@torch.jit.script
def gyration(a, b, x, dim: int = -1):
    a_inner_x = _inner(a, x, keepdim=True, dim=dim)
    b_inner_x = _inner(b, x, keepdim=True, dim=dim)
    a_inner_b = _inner(a, b, keepdim=True, dim=dim)
    a_norm = _inner(a, a, keepdim=True, dim=dim)
    b_norm = _inner(b, b, keepdim=True, dim=dim)
    
    coeff = 2 * a_inner_x * b_inner_x / (1 + a_inner_b)
    return x + coeff * (a / a_norm + b / b_norm)

# Möbius addition in UHG
@torch.jit.script
def mobius_add(x, y, dim: int = -1):
    x_inner_y = _inner(x, y, keepdim=True, dim=dim)
    x_norm = _inner(x, x, keepdim=True, dim=dim)
    y_norm = _inner(y, y, keepdim=True, dim=dim)
    
    numerator = (1 + 2*x_inner_y + y_norm) * x + (1 - x_norm) * y
    denominator = 1 + 2*x_inner_y + x_norm * y_norm
    return numerator / denominator

# Triple spread function
@torch.jit.script
def triple_spread(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (a + b + c)**2 - 2*(a**2 + b**2 + c**2) - 4*a*b*c

# Cross law
@torch.jit.script
def cross_law(q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor, S1: torch.Tensor) -> torch.Tensor:
    return (q2*q3*S1 - q1 - q2 - q3 + 2)**2 - 4*(1 - q1)*(1 - q2)*(1 - q3)

# Spread law
@torch.jit.script
def spread_law(q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor, S3: torch.Tensor) -> bool:
    return torch.allclose(S1/q1, S2/q2) and torch.allclose(S2/q2, S3/q3)

# Check if a point is null
@torch.jit.script
def is_null_point(x, dim: int = -1):
    return torch.allclose(_inner(x, x, dim=dim), torch.tensor(0.0))

# Check if a line is null
@torch.jit.script
def is_null_line(L, dim: int = -1):
    return torch.allclose(_inner(L, L, dim=dim), torch.tensor(0.0))

# Join of two points (returns a line)
@torch.jit.script
def join(a, b, dim: int = -1):
    return torch.cross(a, b, dim=dim)

# Meet of two lines (returns a point)
@torch.jit.script
def meet(L1, L2, dim: int = -1):
    return torch.cross(L1, L2, dim=dim)

# Compute the perpendicular point to a side
@torch.jit.script
def perpendicular_point(a1, a2, dim: int = -1):
    L = join(a1, a2, dim=dim)
    return meet(L, torch.cross(a1, a2, dim=dim), dim=dim)

# Compute the perpendicular line to a vertex
@torch.jit.script
def perpendicular_line(L1, L2, dim: int = -1):
    a = meet(L1, L2, dim=dim)
    return join(a, torch.cross(L1, L2, dim=dim), dim=dim)

# Helper function for spread polynomials
@torch.jit.script
def spread_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return x
    else:
        S_n_minus_1 = x
        S_n_minus_2 = torch.zeros_like(x)
        for _ in range(2, n+1):
            S_n = 2*(1 - 2*x)*S_n_minus_1 - S_n_minus_2 + 2*x
            S_n_minus_2 = S_n_minus_1
            S_n_minus_1 = S_n
        return S_n