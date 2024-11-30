Universal Hyperbolic Geometry (UHG) Library
======================================

.. image:: https://img.shields.io/pypi/v/uhg.svg
   :target: https://pypi.org/project/uhg/
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/zachbovaird/UHG-Library.svg
   :target: https://github.com/zachbovaird/UHG-Library/blob/main/LICENSE
   :alt: License

.. image:: https://github.com/zachbovaird/UHG-Library/workflows/CI/badge.svg
   :target: https://github.com/zachbovaird/UHG-Library/actions
   :alt: Build Status

.. image:: https://codecov.io/gh/zachbovaird/UHG-Library/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/zachbovaird/UHG-Library
   :alt: Code Coverage

A PyTorch library for hyperbolic deep learning using Universal Hyperbolic Geometry principles.
All operations are performed directly in hyperbolic space without tangent space mappings.

Features
--------

- Pure UHG implementation without tangent space operations
- Hyperbolic neural network layers and models
- Hyperbolic optimizers (Adam, SGD)
- Hyperbolic samplers (HMC, Langevin)
- Graph neural networks in hyperbolic space
- Comprehensive documentation and examples

Installation
-----------

.. code-block:: bash

   pip install uhg

For development installation:

.. code-block:: bash

   git clone https://github.com/zachbovaird/UHG-Library.git
   cd UHG-Library
   pip install -e .[dev]

Quick Start
----------

.. code-block:: python

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

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorials/index
   api/index
   examples/index
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 