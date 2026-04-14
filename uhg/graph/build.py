"""KNN graph building with caching support."""

import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import coalesce as pyg_coalesce


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert to numpy, handling tensor."""
    if isinstance(X, torch.Tensor):
        # Avoid torch->numpy bridge in environments where torch was built
        # against an incompatible NumPy ABI.
        return np.asarray(X.detach().cpu().tolist(), dtype=np.float32)
    return np.asarray(X)


def build_knn_graph(
    X: Union[torch.Tensor, np.ndarray],
    k: int,
    metric: str = "euclidean",
    undirected: bool = False,
    cache_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Build k-nearest neighbors graph. Returns edge_index [2, E] in COO format.

    Uses Euclidean distance by default (NearestNeighbors). By default edges are
    directed: (i, j) means j is a k-nearest neighbor of i. Set ``undirected=True``
    to add reverse edges and merge duplicates (symmetric kNN).

    Args:
        X: Feature matrix [N, D].
        k: Number of neighbors per node.
        metric: Distance metric; only "euclidean" supported for now.
        undirected: If True, graph is symmetrized via union of (i,j) and (j,i).
        cache_key: If provided, save/load graph from cache. Auto-generated from
            (X.shape, k, metric, undirected) if None and caching is used.
        cache_dir: Directory for cache files. Default: current dir.

    Returns:
        edge_index: Tensor [2, E], dtype long.
    """
    X_np = _to_numpy(X)
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_np.shape}")
    n = X_np.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n_samples ({n}), got k={k}")

    if cache_key is not None or cache_dir is not None:
        key = cache_key or _cache_key(X_np.shape, k, metric, undirected)
        cdir = Path(cache_dir) if cache_dir else Path(".")
        cache_path = cdir / f"{key}.edge_index.pt"
        if cache_path.exists():
            return load_edge_index(str(cache_path))

    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(X_np)
    indices = nn.kneighbors(return_distance=False)
    row = np.repeat(np.arange(n), k)
    col = indices.ravel()
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
    if undirected:
        combined = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        coalesced = pyg_coalesce(combined, num_nodes=n)
        edge_index = coalesced[0] if isinstance(coalesced, tuple) else coalesced
    if isinstance(X, torch.Tensor) and X.is_cuda:
        edge_index = edge_index.to(X.device)

    if cache_key is not None or cache_dir is not None:
        key = cache_key or _cache_key(X_np.shape, k, metric, undirected)
        cdir = Path(cache_dir) if cache_dir else Path(".")
        cache_path = cdir / f"{key}.edge_index.pt"
        save_edge_index(str(cache_path), edge_index.cpu())

    return edge_index


def _cache_key(shape: tuple, k: int, metric: str, undirected: bool = False) -> str:
    h = hashlib.sha256(f"{shape}_{k}_{metric}_{undirected}".encode()).hexdigest()[:16]
    return f"knn_{shape[0]}_{shape[1]}_{k}_{h}"


def save_edge_index(path: str, edge_index: torch.Tensor) -> None:
    """Save edge_index to file.

    Args:
        path: File path (.pt recommended).
        edge_index: Tensor [2, E].
    """
    torch.save(edge_index.cpu(), path)


def load_edge_index(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load edge_index from file.

    Args:
        path: File path.
        device: Optional device to move tensor to.

    Returns:
        edge_index: Tensor [2, E].
    """
    ei = torch.load(path, weights_only=True)
    if device is not None:
        ei = ei.to(device)
    return ei


def build_maxk_then_slice(
    X: Union[torch.Tensor, np.ndarray],
    max_k: int,
    k: int,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Build kNN graph at max_k, then slice to first k neighbors per node.

    Useful when experimenting with different k without recomputing.

    Args:
        X: Feature matrix [N, D].
        max_k: Build graph with this many neighbors.
        k: Return only first k neighbors per node.
        metric: Distance metric.

    Returns:
        edge_index: Tensor [2, E] with at most k edges per source node.
    """
    if k > max_k:
        raise ValueError(f"k ({k}) must be <= max_k ({max_k})")
    X_np = _to_numpy(X)
    n = X_np.shape[0]
    if max_k >= n:
        raise ValueError(f"max_k must be < n_samples ({n})")

    nn = NearestNeighbors(n_neighbors=max_k, metric=metric, algorithm="auto")
    nn.fit(X_np)
    indices = nn.kneighbors(return_distance=False)
    indices_slice = indices[:, :k]
    row = np.repeat(np.arange(n), k)
    col = indices_slice.ravel()
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
    if isinstance(X, torch.Tensor) and X.is_cuda:
        edge_index = edge_index.to(X.device)
    return edge_index
