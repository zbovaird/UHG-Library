### UHG-Library codebase review (v0.1.2)

#### Scope
- Reviewed repository structure, core geometry (`uhg/projective.py`, `uhg/utils/metrics.py`), models and layers (`uhg/nn/*`), optimizers (`uhg/optim/*`), datasets/samplers, and tests.
- Focused on UHG.pdf compliance: projective-only methods, cross-ratio invariants, no tangent-space ops, hyperbolic distance via Minkowski form.

### Pros
- **Projective core present**: `ProjectiveUHG` implements normalization to Minkowski norm −1, inner product, distance, weighted aggregation, and several UHG primitives.
- **Hyperbolic distance**: Uses hyperboloid model formula d(x,y)=arccosh(−⟨x,y⟩) on normalized points.
- **Vectorized utilities**: `uhg/utils/metrics.py` provides vectorized inner product, norm, quadrance, spread with broadcasting.
- **HGNN coverage**: Includes transformer-style (`HGT`), GAT-like (`HGAT`), and hierarchical layers with intent to preserve cross-ratios.
- **Documentation**: Multiple files reference UHG principles and highlight projective constraints; extensive tests exist for geometry and models.

### Cons and inconsistencies
- **Missing manifold implementation files**
  - `uhg/manifolds/__init__.py` imports `base` and `hyperbolic`, but these files are not present (recently deleted), causing import errors across the codebase and tests.

```1:11:uhg/manifolds/__init__.py
from .base import Manifold, ScalingInfo
from .hyperbolic import HyperbolicManifold
```

- **Optimizer API divergence and UHG violations**
  - Multiple optimizer stacks coexist with conflicting philosophies:
    - `uhg/optim/base.py` explicitly projects to tangent space and uses an exponential map, violating “no tangent space” and “no exp/log” requirements. It also calls `self.manifold.add`/`normalize_points` even though no `self.manifold` is defined in the class.

```112:166:uhg/optim/base.py
def _project_to_tangent_space(self, vector: torch.Tensor, point: torch.Tensor, group: dict) -> torch.Tensor:
    ...

def _exponential_map(self, point: torch.Tensor, tangent_vector: torch.Tensor) -> torch.Tensor:
    ...
    new_point = cosh_v * point + (sinh_v / v_norm) * tangent_vector
```

  - `uhg/optim/adam.py` and `uhg/optim/sgd.py` rely on `mobius_*` operations that do not exist in `ProjectiveUHG`.

```91:136:uhg/optim/adam.py
p_new = self.uhg.mobius_add(
    p,
    self.uhg.mobius_scalar_mul(
        -step_size,
        exp_avg_new / denom
    )
)
```

```66:102:uhg/optim/sgd.py
grad = self.uhg.mobius_add(
    grad,
    self.uhg.mobius_scalar_mul(
        group['weight_decay'],
        p
    )
)
```

- **Model/attention signature mismatches**
  - `HGT` and `HGNN` call `HyperbolicAttention` with parameters that don’t match its implemented signature (`in_features`, `out_features`, ...). The models pass `manifold=` and `in_channels=`, which will raise runtime errors.

```120:128:uhg/nn/models/hgt.py
self.self_attn = HyperbolicAttention(
    manifold=self.uhg.manifold,
    in_channels=d_model,
    num_heads=nhead,
    dropout=dropout
)
```

```11:22:uhg/nn/attention.py
class HyperbolicAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, dropout: float = 0.0, concat: bool = True):
```

- **Dual library trees**
  - Both `uhg/` and `UHG/` directories exist with overlapping functionality (e.g., `projective.py`). This duplication risks divergence and confusion about the “canonical” implementation.

- **Homogeneous vs. feature space usage confusion**
  - Some modules treat inputs as Euclidean features and later normalize with Minkowski form without first appending/maintaining a homogeneous coordinate. In `uhg/nn/attention.py`, `normalize_points` is applied to `x` without an explicit homogeneous coordinate addition within the module; upstream callers sometimes add one, but the API expectations are inconsistent.

```55:81:uhg/nn/attention.py
# Store initial cross-ratio if possible
...
# Projective normalization (homogeneous coordinates)
x_proj = self.uhg.normalize_points(x)
```

- **Tests reference missing manifold class**
  - `tests/test_manifolds.py` imports `HyperbolicManifold` which is absent, leading to immediate failures.

```1:12:tests/test_manifolds.py
from uhg.manifolds import HyperbolicManifold
...
manifold = HyperbolicManifold(curvature=-1.0)
```

### Mathematical/UHG compliance notes
- **Hyperbolic distance**
  - `uhg/projective.py` implements d(x,y)=arccosh(−⟨x,y⟩) on points normalized to Minkowski norm −1, consistent with the hyperboloid model (UHG.pdf Ch. 3–4).

```436:450:uhg/projective.py
def distance(self, x, y):
    x = self.normalize_points(x)
    y = self.normalize_points(y)
    spatial_dot = torch.sum(x[..., :-1] * y[..., :-1], dim=-1)
    time_dot = x[..., -1] * y[..., -1]
    inner_prod = spatial_dot - time_dot
    inner_prod = torch.clamp(inner_prod, max=-1.0 - self.eps)
    d = torch.acosh(-inner_prod)
```

- **Inner product and normalization**
  - Minkowski form ⟨a,b⟩ = a₁b₁ + a₂b₂ − a₃b₃ is used consistently in `projective.py` and `uhg/utils/metrics.py`. Normalization to Minkowski norm −1 is enforced; null points are rejected.

- **Cross-ratio**
  - Implementations exist via determinant-based forms and via quadrance-based expressions in `uhg/utils/metrics.py`. The 2D determinant approach drops the homogeneous coordinate; correctness depends on consistent planar embeddings and may be fragile numerically near degenerate configurations.

- **Violations of UHG principles**
  - Tangent-space constructs and exponential maps in `uhg/optim/base.py` and derived optimizers directly contradict the repo’s UHG rules (no tangent space; no exp/log maps). Even where “projective-only” optimizers exist, they reference `mobius_*` operations.

### Performance and numerical notes
- Vectorized operations are common and GPU-friendly. Stability clamps are present (e.g., acosh domain, eps in denominators). Some printing in critical paths (`join`) could hinder performance.
- Cross-ratio and aggregation normalize weights and results, which is good; edge-case handling around null/degenerate cases exists in places and is missing in others.

### Packaging/structure observations
- PyPI scaffolding present (`pyproject.toml`, `setup.py`, `MANIFEST.in`), docs and CI config exist. The dual trees (`uhg/` vs `UHG/`) and missing `uhg/manifolds/*` will block installation and imports.

### Testing landscape
- Many tests target geometry, metrics, layers, models, and CUDA ops. Given the inconsistencies above, several test suites will likely fail immediately due to missing manifold modules and incompatible APIs.

### High-level takeaways
- The project has a solid projective core and good coverage of UHG concepts, with extensive intent to preserve invariants.
- Critical repository breakages and conceptual drift in optimizers (tangent-space vs. projective-only) currently prevent consistent usage and testing.
- Model/attention API mismatches and missing manifold files are immediate blockers. 