import torch
import pytest
from uhg.manifolds import HyperbolicManifold

@pytest.fixture
def manifold():
    """Create a HyperbolicManifold instance for testing."""
    return HyperbolicManifold(curvature=-1.0)

@pytest.fixture
def points():
    """Create test points in hyperbolic space."""
    # Create points in hyperbolic space (last coordinate is time-like)
    p1 = torch.tensor([0.5, 0.5, 1.0])
    p2 = torch.tensor([0.3, 0.7, 1.0])
    p3 = torch.tensor([0.8, 0.2, 1.0])
    return p1, p2, p3

def minkowski_norm(x):
    spatial = x[..., :-1]
    time = x[..., -1]
    return torch.sum(spatial ** 2, dim=-1) - time ** 2

def test_initialization():
    """Test manifold initialization."""
    manifold = HyperbolicManifold(curvature=-1.0)
    assert manifold.curvature == -1.0
    with pytest.raises(ValueError):
        HyperbolicManifold(curvature=1.0)

def test_normalize_points(manifold, points):
    """Test point normalization (Minkowski norm)."""
    p1, p2, p3 = points
    normalized = manifold.normalize_points(p1)
    assert torch.allclose(minkowski_norm(normalized), torch.tensor(-1.0), atol=1e-6)
    batch = torch.stack([p1, p2, p3])
    normalized_batch = manifold.normalize_points(batch)
    assert torch.allclose(minkowski_norm(normalized_batch), -torch.ones(3), atol=1e-6)
    # Time component positive
    assert (normalized_batch[..., -1] > 0).all()

def test_project(manifold, points):
    """Test projection to manifold (Minkowski norm)."""
    p1, p2, p3 = points
    projected = manifold.project(p1)
    assert torch.allclose(minkowski_norm(projected), torch.tensor(-1.0), atol=1e-6)
    batch = torch.stack([p1, p2, p3])
    projected_batch = manifold.project(batch)
    assert torch.allclose(minkowski_norm(projected_batch), -torch.ones(3), atol=1e-6)
    assert (projected_batch[..., -1] > 0).all()

def test_distance(manifold, points):
    """Test hyperbolic distance calculation."""
    p1, p2, p3 = points
    d = manifold.distance(p1, p2)
    assert d >= 0
    assert not torch.isnan(d)
    batch = torch.stack([p1, p2, p3])
    d_batch = manifold.distance(batch[0], batch[1:])
    assert (d_batch >= 0).all()
    assert not torch.isnan(d_batch).any()
    d_self = manifold.distance(p1, p1)
    assert torch.allclose(d_self, torch.tensor(0.0), atol=1e-3)

def test_inner_product(manifold, points):
    """Test Minkowski inner product."""
    p1, p2, p3 = points
    ip = manifold.inner_product(p1, p2)
    assert not torch.isnan(ip)
    batch = torch.stack([p1, p2, p3])
    ip_batch = manifold.inner_product(batch[0], batch[1:])
    assert not torch.isnan(ip_batch).any()
    ip_self = manifold.inner_product(p1, p1)
    assert not torch.isnan(ip_self)

def test_uhg_invariants(manifold, points):
    """Test preservation of UHG invariants (Minkowski norm, time sign)."""
    p1, p2, p3 = points
    projected_p1 = manifold.project(p1)
    projected_p2 = manifold.project(p2)
    projected_p3 = manifold.project(p3)
    # Minkowski norm preserved
    assert torch.allclose(minkowski_norm(projected_p1), torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(minkowski_norm(projected_p2), torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(minkowski_norm(projected_p3), torch.tensor(-1.0), atol=1e-6)
    # Time component positive
    assert projected_p1[-1] > 0
    assert projected_p2[-1] > 0
    assert projected_p3[-1] > 0 