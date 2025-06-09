# Universal Hyperbolic Geometry (UHG) Module Template

## 1. Module Structure
- All operations must be defined in terms of projective/hyperbolic geometry (no tangent space, no Riemannian exp/log maps).
- Use Minkowski (Lorentzian) inner product and projective normalization.
- Preserve hyperbolic invariants (cross-ratio, Minkowski norm, etc.).
- Reference specific sections of UHG.pdf in docstrings.

## 2. Example Class Skeleton
```python
import torch
from .base import Manifold

class UHGModuleName(Manifold):
    """
    [Short description of the module's purpose.]

    UHG-compliant: All operations are performed directly in projective/hyperbolic space.
    No tangent space, exponential map, or logarithmic map operations are present.
    Reference: UHG.pdf, Ch. [X]
    """
    def __init__(self, ...):
        # Initialization code
        super().__init__()

    def normalize_points(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize points to lie on the hyperbolic manifold (Minkowski norm -1).
        Reference: UHG.pdf, Ch. 3
        """
        # ...

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points onto the hyperbolic manifold (Minkowski norm -1).
        Reference: UHG.pdf, Ch. 3
        """
        # ...

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance using the Minkowski inner product.
        Reference: UHG.pdf, Ch. 4
        """
        # ...

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Minkowski inner product between points.
        Reference: UHG.pdf, Ch. 3
        """
        # ...

    # Add other projective/hyperbolic operations as needed
```

## 3. Docstring Requirements
- Every method must reference the relevant section of UHG.pdf.
- Explain the mathematical principle and how it is implemented in code.
- State explicitly that no tangent space or Riemannian operations are used.

## 4. Test Guidelines
- All tests must check preservation of hyperbolic/projective invariants:
    - Minkowski norm (e.g., \(x_1^2 + x_2^2 - x_3^2 = -1\))
    - Time component sign (last coordinate positive)
    - Cross-ratio or other UHG invariants if relevant
- Use robust tolerances for floating-point comparisons (e.g., `atol=1e-3` for near-zero checks).
- No tests for expmap/logmap or tangent-space operations.

## 5. Example Test Skeleton
```python
import torch
import pytest
from uhg.manifolds import UHGModuleName

def minkowski_norm(x):
    spatial = x[..., :-1]
    time = x[..., -1]
    return torch.sum(spatial ** 2, dim=-1) - time ** 2

def test_normalize_points():
    module = UHGModuleName(...)
    x = torch.tensor([...])
    x_norm = module.normalize_points(x)
    assert torch.allclose(minkowski_norm(x_norm), torch.tensor(-1.0), atol=1e-6)
    assert x_norm[..., -1] > 0
```

---

**Cursor Rule Addition:**
> For all future UHG modules, strictly follow the template in `docs/UHG_module_template.md`. All code, docstrings, and tests must adhere to UHG principles: no tangent space, no exp/log maps, only projective/hyperbolic operations, and explicit reference to UHG.pdf in documentation. 