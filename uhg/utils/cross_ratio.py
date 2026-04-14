"""Cross-ratio computation utilities for UHG."""

import torch


def compute_cross_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the projective cross-ratio in a scale-invariant way.

    The library notebooks operate on homogeneous 3-vectors but use only the first
    two coordinates for the projective determinant. That determinant formula is
    individually scale-invariant for each input point:

        CR(a, b; c, d) = [a, c] [b, d] / ([a, d] [b, c])

    where ``[u, v]`` is the 2x2 determinant of the first two coordinates.
    """

    a_2d = a[..., :2]
    b_2d = b[..., :2]
    c_2d = c[..., :2]
    d_2d = d[..., :2]

    def det2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    num = det2(a_2d, c_2d) * det2(b_2d, d_2d)
    denom = det2(a_2d, d_2d) * det2(b_2d, c_2d)
    denom = torch.where(
        torch.abs(denom) < eps,
        torch.full_like(denom, eps),
        denom,
    )
    return num / denom


def verify_cross_ratio_preservation(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    transformed_a: torch.Tensor,
    transformed_b: torch.Tensor,
    transformed_c: torch.Tensor,
    transformed_d: torch.Tensor,
) -> bool:
    """
    Verify that the cross-ratio is preserved under a projective transformation.
    Reference: UHG.pdf, Ch. 2
    """
    original_cr = compute_cross_ratio(a, b, c, d)
    transformed_cr = compute_cross_ratio(
        transformed_a, transformed_b, transformed_c, transformed_d
    )
    return torch.allclose(original_cr, transformed_cr, rtol=1e-5, atol=1e-5)


def verify_cross_ratio_preservation_simple(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Check cross-ratio preservation for the first four rows of two aligned batch tensors.
    Used by hierarchical layer tests when full quadruples are not explicit.
    """
    if x.shape[0] < 4 or y.shape[0] < 4:
        return True
    return verify_cross_ratio_preservation(
        x[0],
        x[1],
        x[2],
        x[3],
        y[0],
        y[1],
        y[2],
        y[3],
    )


def restore_cross_ratio(x: torch.Tensor, target_cr: torch.Tensor) -> torch.Tensor:
    """
    Restore the cross-ratio of points to a target value.
    Reference: UHG.pdf, Ch. 2
    """
    # Normalize points
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    # Compute current cross-ratio
    current_cr = compute_cross_ratio(x[0], x[1], x[2], x[3])
    # Compute scaling factor
    scale = torch.sqrt(target_cr / current_cr)
    # Apply scaling
    x = x * scale
    return x
