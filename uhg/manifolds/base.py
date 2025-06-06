class Manifold:
    """Stub for Manifold base class."""
    pass

class ScalingInfo:
    """Stub for ScalingInfo class."""
    pass

class HyperbolicManifold(Manifold):
    """Stub for HyperbolicManifold class for testing purposes."""
    def logmap0(self, x):
        return x
    def expmap0(self, x):
        return x
    def inner_product(self, x, y):
        # Return ones for shape compatibility in tests
        return (x * y).sum(dim=-1, keepdim=True) 