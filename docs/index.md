# Universal Hyperbolic Geometry Library

A PyTorch library for Universal Hyperbolic Geometry (UHG) and Hyperbolic Graph Neural Networks.

## Overview

The UHG Library provides a comprehensive implementation of Universal Hyperbolic Geometry principles for deep learning applications, with a focus on graph neural networks. It offers:

- Pure hyperbolic operations without tangent space approximations
- Multiple hyperbolic manifolds (Lorentz, Siegel)
- Hyperbolic graph neural network layers
- Hyperbolic attention mechanisms
- Specialized optimizers for hyperbolic space

## Quick Start

Install the package:
```bash
pip install uhg
```

Basic usage:
```python
import torch
import uhg

# Create a hyperbolic manifold
manifold = uhg.manifolds.LorentzManifold()

# Create hyperbolic points
x = manifold.random_points(10)
y = manifold.random_points(10)

# Compute hyperbolic distance
dist = manifold.dist(x, y)
```

## Features

- **Pure Hyperbolic Operations**: All operations are performed directly in hyperbolic space without tangent space mappings
- **Multiple Manifolds**: Support for different hyperbolic geometries
- **Graph Neural Networks**: Specialized layers for graph learning in hyperbolic space
- **Optimizers**: Riemannian optimization methods
- **GPU Support**: Full acceleration for all operations
- **Batched Operations**: Efficient processing of batched data

## Documentation

For detailed documentation, please visit:

- [Installation Guide](installation.md)
- [User Guide](guide/getting-started.md)
- [API Reference](api/core.md)
- [Examples](examples/basic.md) 