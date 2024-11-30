Installation Guide
==================

Basic Installation
-----------------

UHG can be installed using pip:

.. code-block:: bash

   pip install uhg

This will install the CPU version with basic dependencies.

Baremetal Linux Installation
-------------------------

For baremetal Linux systems, use the provided installation script:

.. code-block:: bash

   # Basic installation
   ./install_baremetal.sh

   # With CUDA support
   ./install_baremetal.sh --cuda

   # With virtual environment
   ./install_baremetal.sh --venv

   # With custom virtual environment path
   ./install_baremetal.sh --venv-path /path/to/venv

The script handles:
- Distribution detection (Debian/Ubuntu, Fedora/RHEL, Arch, openSUSE)
- System dependency installation
- CUDA detection and setup
- Python environment configuration
- UHG installation

Supported Linux distributions:
- Debian/Ubuntu
- Fedora/RHEL
- Arch Linux
- openSUSE

Required system dependencies will be installed automatically:
- Python development files
- Build tools (gcc, g++, cmake)
- OpenBLAS
- Git
- Additional distribution-specific dependencies

GPU Support
----------

For GPU support, install with the gpu extra:

.. code-block:: bash

   pip install uhg[gpu]

This includes CUDA-enabled versions of PyTorch and PyTorch Geometric.

CPU-Only Version
---------------

For a CPU-only version (smaller install size):

.. code-block:: bash

   pip install uhg[cpu]

Virtual Environment Installation
------------------------------

UHG provides a script to set up virtual environments:

.. code-block:: bash

   # Using venv (default)
   python setup_venv.py

   # Using conda
   python setup_venv.py --type conda

   # Using pipenv
   python setup_venv.py --type pipenv

   # With CUDA support
   python setup_venv.py --cuda

Conda Installation
----------------

Install using the provided environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate uhg

Splunk Installation
-----------------

UHG can be installed as a Splunk app:

.. code-block:: bash

   # Basic installation
   python install_splunk.py

   # With custom app name
   python install_splunk.py --app-name my_uhg_app

   # With CUDA support
   python install_splunk.py --cuda

   # With custom Splunk apps directory
   python install_splunk.py --app-dir /path/to/splunk/apps

After installation, UHG can be used in Splunk Python scripts:

.. code-block:: python

   from uhg_wrapper import setup_uhg
   manifold = setup_uhg()

Development Installation
----------------------

For development, clone the repository and install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/zachbovaird/UHG-Library.git
   cd UHG-Library
   pip install -e .[dev,docs]

Or use make commands:

.. code-block:: bash

   make install-dev  # CPU version with dev tools
   make install-gpu  # GPU version with dev tools

Docker Installation
-----------------

UHG provides Docker support for containerized development:

.. code-block:: bash

   # Build and start all services
   docker-compose up -d

   # Or use make commands
   make docker-build
   make docker-up

Available containers:
- uhg: Main development environment
- docs: Documentation server (http://localhost:8000)
- jupyter: JupyterLab server (http://localhost:8888)

Dependencies
-----------

Core Dependencies:
- Python >= 3.7
- PyTorch >= 1.7.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0

Optional Dependencies:
- GPU support: CUDA toolkit >= 11.0
- Development: pytest, black, isort, flake8, mypy
- Documentation: sphinx, sphinx-rtd-theme, nbsphinx

System Dependencies (Linux):
- Python development files (python3-dev)
- Build tools (gcc, g++, cmake)
- OpenBLAS
- Git

Environment Support
-----------------

UHG supports installation in various environments:

- Standard Python environments
- Virtual environments (venv)
- Conda environments
- Pipenv environments
- Docker containers
- Splunk environments
- Jupyter environments
- Baremetal Linux systems

Troubleshooting
--------------

CUDA Version Mismatch
^^^^^^^^^^^^^^^^^^^^
If you encounter CUDA version mismatches, install specific versions:

.. code-block:: bash

   pip install torch==1.9.0+cu111 torch-geometric==2.0.0+cu111

Memory Issues
^^^^^^^^^^^^
For systems with limited memory, install the CPU-only version:

.. code-block:: bash

   pip install uhg[cpu]

Splunk Issues
^^^^^^^^^^^^
If you encounter issues with Splunk installation:

1. Ensure Splunk's Python is being used:

   .. code-block:: bash

      python install_splunk.py --app-dir "$(splunk cmd python -c 'import os; print(os.path.join(os.environ["SPLUNK_HOME"], "etc", "apps"))')"

2. Install in Splunk's Python environment directly:

   .. code-block:: bash

      $SPLUNK_HOME/bin/python -m pip install uhg[cpu]

Linux System Issues
^^^^^^^^^^^^^^^^^
If you encounter issues on Linux:

1. Missing system dependencies:

   .. code-block:: bash

      # Debian/Ubuntu
      sudo apt-get install python3-dev build-essential cmake libopenblas-dev

      # Fedora/RHEL
      sudo dnf install python3-devel gcc gcc-c++ cmake openblas-devel

      # Arch Linux
      sudo pacman -S python base-devel cmake openblas

      # openSUSE
      sudo zypper install python3-devel gcc gcc-c++ cmake openblas-devel

2. Permission issues:

   .. code-block:: bash

      # Use virtual environment instead of system Python
      ./install_baremetal.sh --venv

3. CUDA detection:

   .. code-block:: bash

      # Check CUDA availability
      nvidia-smi
      # If available, install with CUDA support
      ./install_baremetal.sh --cuda

Build Issues
^^^^^^^^^^^
If you encounter build issues:

1. Ensure you have the latest pip:

   .. code-block:: bash

      python -m pip install --upgrade pip

2. Install build dependencies:

   .. code-block:: bash

      pip install wheel setuptools

3. Clear pip cache:

   .. code-block:: bash

      pip cache purge 