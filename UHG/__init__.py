"""
Universal Hyperbolic Geometry (UHG) Library

A PyTorch library for hyperbolic deep learning using UHG principles.
All operations are performed directly in hyperbolic space without tangent space mappings.
"""

from .tensor import HyperbolicTensor, HyperbolicParameter
from . import manifolds
from . import nn
from . import optim
from . import samplers
from . import datasets
from . import core

__version__ = "0.1.0"

# Core functionality
from .core import (
    join,
    meet,
    cross_ratio,
    perpendicular,
    reflect,
    rotate,
    translate,
    distance
)

# Manifolds
from .manifolds import (
    Manifold,
    LorentzManifold,
    SiegelManifold
)

# Neural network components
from .nn.layers import (
    HyperbolicLayer,
    HyperbolicGraphConv,
    HyperbolicAttention
)

from .nn.models import (
    HyperbolicGraphSAGE
)

# Optimizers
from .optim import (
    HyperbolicOptimizer,
    HyperbolicAdam,
    HyperbolicSGD
)

# Samplers
from .samplers import (
    HyperbolicSampler,
    HyperbolicHMC,
    HyperbolicLangevin
)

# Datasets
from .datasets import (
    HyperbolicDataset,
    HyperbolicGraphDataset
)

__all__ = [
    # Version
    "__version__",
    
    # Tensors
    "HyperbolicTensor",
    "HyperbolicParameter",
    
    # Core operations
    "join",
    "meet", 
    "cross_ratio",
    "perpendicular",
    "reflect",
    "rotate",
    "translate",
    "distance",
    
    # Manifolds
    "Manifold",
    "LorentzManifold",
    "SiegelManifold",
    
    # Neural networks
    "HyperbolicLayer",
    "HyperbolicGraphConv",
    "HyperbolicAttention",
    "HyperbolicGraphSAGE",
    
    # Optimizers
    "HyperbolicOptimizer",
    "HyperbolicAdam", 
    "HyperbolicSGD",
    
    # Samplers
    "HyperbolicSampler",
    "HyperbolicHMC",
    "HyperbolicLangevin",
    
    # Datasets
    "HyperbolicDataset",
    "HyperbolicGraphDataset",
    
    # Submodules
    "manifolds",
    "nn",
    "optim",
    "samplers",
    "datasets",
    "core"
] 