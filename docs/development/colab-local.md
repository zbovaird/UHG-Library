# Colab and local dev environments

## Google Colab (recommended for GPU training)

For full CIC-scale runs, use **Google Colab** with a **GPU** runtime:

1. Open the notebook from the repo, e.g.  
   [colab/uhg_cic_intrusion_detection.ipynb](https://colab.research.google.com/github/zbovaird/UHG-Library/blob/main/colab/uhg_cic_intrusion_detection.ipynb)
2. **Runtime → Change runtime type →** choose **T4 / L4** (or another GPU) and **Python 3**.
3. Install the library (from PyPI or `main`):

   ```bash
   pip install uhg
   # or latest main:
   # pip install "git+https://github.com/zbovaird/UHG-Library.git"
   ```

4. Mount Drive if your dataset lives there (`CIC_data.csv`).
5. Optional extras matched to the **[colab]** optional dependency group (see `pyproject.toml`): faster IO and neighbor search for IDS-style notebooks.

**torch-scatter:** PyTorch Geometric stacks often need `torch-scatter` built for your exact `torch` and CUDA/CPU build. On Colab, after `import torch`, install from the PyG wheel index, e.g.:

```text
https://data.pyg.org/whl/torch-{TORCH}+{cu124}.html
```

(use `+cpu` on CPU-only machines). Same pattern as `tests/UHG_IDS_4_9.ipynb`.

## Local “Colab-like” venv (Cursor / laptop)

Use this when you want to edit `uhg` and run **small** tests without Colab:

- **Python 3.11 or 3.12**
- Create a venv, then:

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e ".[dev,colab]"
  ```

- Install **torch-scatter** from the PyG index that matches your `torch` version (see above).
- Optional: `faiss-cpu` is included in `uhg[colab]` where supported; on platforms without a wheel, skip it and rely on scikit-learn kNN in `build_knn_graph`.

This stack is enough for **`pytest`** and **smoke** training on **downsampled** CSV rows. Full CIC training belongs on **Colab GPU** or another machine with enough RAM/GPU.

## Environment variables (notebooks)

- **`CIC_CSV`**: Path to `CIC_data.csv` (defaults in notebooks often assume `/content/drive/MyDrive/...` on Colab).
