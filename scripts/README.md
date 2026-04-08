# Maintainer and helper scripts

Scripts in this directory are **not** part of the published `uhg` package. They exist for local development, publishing, and experiments.

| Script | Purpose |
|--------|---------|
| `color_test.py` | Development utility |
| `examine_uhg.py` | Inspect / explore UHG objects locally |
| `install_baremetal.sh` | Environment setup (reference only) |
| `install_mac_silicon.sh` | macOS Silicon setup (reference only) |
| `install_splunk.py` | Legacy integration helper |
| `progress_test.py` | Progress / timing checks |
| `publish.py` | Build and publish to PyPI (`python -m build`, `twine`) |
| `setup_venv.py` | Virtualenv helper |
| `terminal_network_colab.py` | Notebook-oriented network demo |
| `terminal_network_uhg.py` | Terminal network demo |

Install the library itself with `pip install -e ".[dev]"` from the repository root (see [README.md](../README.md)).
