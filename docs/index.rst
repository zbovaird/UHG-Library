Universal Hyperbolic Geometry (UHG)
================================

A PyTorch library for hyperbolic deep learning using pure UHG principles.

Features
--------

- Pure projective geometry implementation
- No differential geometry or manifold concepts
- Cross-ratio preservation
- Projective transformations
- Graph neural networks
- Optimizers and samplers

Quick Start
----------

.. code-block:: python

    import torch
    import uhg

    # Create points in projective space
    x = torch.randn(10, 3)
    y = torch.randn(10, 3)

    # Initialize UHG
    uhg_proj = uhg.ProjectiveUHG()

    # Transform points
    x_proj = uhg_proj.transform(x)
    y_proj = uhg_proj.transform(y)

    # Compute projective distance
    dist = uhg_proj.proj_dist(x_proj, y_proj)

    # Compute cross-ratio
    cr = uhg_proj.cross_ratio(x_proj[0], x_proj[1], x_proj[2], x_proj[3])

Installation
-----------

.. code-block:: bash

    pip install uhg

Documentation
------------

For detailed documentation, visit `uhg.readthedocs.io <https://uhg.readthedocs.io>`_. 