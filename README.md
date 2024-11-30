# UHG (Universal Hyperbolic Geometry)

A PyTorch library for Universal Hyperbolic Geometry, providing tools and implementations for working with hyperbolic spaces and their applications.

## Features

- Manifold-aware tensor operations
- Support for various hyperbolic spaces
- Efficient implementations of geometric operations
- PyTorch integration for deep learning applications

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from uhg import ManifoldTensor
from uhg.manifolds import Hyperbolic

# Create a tensor on a hyperbolic manifold
x = ManifoldTensor(torch.randn(3, 2), manifold=Hyperbolic())

# Perform manifold-aware operations
y = x.exp_map(torch.randn_like(x))
```

## Documentation

For detailed documentation and mathematical background, please refer to the included PDF documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
