# Universal Hyperbolic Geometry Library

[![PyPI version](https://badge.fury.io/py/uhg.svg)](https://badge.fury.io/py/uhg)
[![License](https://img.shields.io/github/license/zachbovaird/UHG-Library.svg)](https://github.com/zachbovaird/UHG-Library/blob/main/LICENSE)
[![Build Status](https://github.com/zachbovaird/UHG-Library/workflows/CI/badge.svg)](https://github.com/zachbovaird/UHG-Library/actions)
[![Code Coverage](https://codecov.io/gh/zachbovaird/UHG-Library/branch/main/graph/badge.svg)](https://codecov.io/gh/zachbovaird/UHG-Library)

A PyTorch library for Universal Hyperbolic Geometry (UHG) and Hyperbolic Graph Neural Networks. All operations are performed directly in hyperbolic space without tangent space mappings.

## Installation

### Basic Installation

```bash
pip install uhg
```

### With GPU Support

```bash
pip install uhg[gpu]
```

### CPU-Only Version

```bash
pip install uhg[cpu]
```

### Development Version

```bash
pip install uhg[dev]
```

### Documentation Tools

```bash
pip install uhg[docs]
```

## Quick Start

```python
import uhg
import torch

# Create hyperbolic tensors
manifold = uhg.LorentzManifold()
x = uhg.HyperbolicTensor([1.0, 0.0, 0.0], manifold=manifold)
y = uhg.HyperbolicTensor([0.0, 1.0, 0.0], manifold=manifold)

# Compute hyperbolic distance
dist = uhg.distance(x, y)

# Create a hyperbolic neural network
model = uhg.nn.layers.HyperbolicGraphConv(
    manifold=manifold,
    in_features=10,
    out_features=5
)

# Use hyperbolic optimizer
optimizer = uhg.optim.HyperbolicAdam(
    model.parameters(),
    manifold=manifold,
    lr=0.01
)
```

## Features

- Pure UHG implementation without tangent space operations
- Hyperbolic neural network layers and models
- Hyperbolic optimizers (Adam, SGD)
- Hyperbolic samplers (HMC, Langevin)
- Graph neural networks in hyperbolic space
- Comprehensive documentation and examples

## Platform Support

- Linux (all major distributions)
- macOS (including Apple Silicon)
- Windows
- Docker containers
- Splunk environments

## Documentation

Full documentation is available in the [docs](docs/) directory and in the [GitHub repository](https://github.com/zachbovaird/UHG-Library).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use UHG in your research, please cite:

```bibtex
@software{uhg2023,
  title = {UHG: Universal Hyperbolic Geometry Library},
  author = {Bovaird, Zach},
  year = {2023},
  url = {https://github.com/zachbovaird/UHG-Library}
}
```
