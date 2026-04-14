"""UHG utilities."""

from .cross_ratio import (
    compute_cross_ratio,
    restore_cross_ratio,
    verify_cross_ratio_preservation,
)
from .metrics import (
    minkowski_inner_product,
    projective_normalize,
    uhg_inner_product,
    uhg_norm,
    uhg_quadrance,
    uhg_quadrance_vectorized,
    uhg_spread,
    verify_uhg_constraints,
)

__all__ = [
    "compute_cross_ratio",
    "verify_cross_ratio_preservation",
    "restore_cross_ratio",
    "uhg_inner_product",
    "minkowski_inner_product",
    "projective_normalize",
    "uhg_norm",
    "uhg_quadrance",
    "uhg_quadrance_vectorized",
    "uhg_spread",
    "verify_uhg_constraints",
]
