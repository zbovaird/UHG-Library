"""KNN graph building with caching support."""

import hashlib
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from torch_geometric.utils import coalesce as pyg_coalesce

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from pynndescent import NNDescent  # type: ignore

    _PYNNDESCENT_AVAILABLE = True
except ImportError:
    _PYNNDESCENT_AVAILABLE = False

KnnBackend = Literal["auto", "faiss", "pynndescent", "sklearn"]


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert to numpy, handling tensor."""
    if isinstance(X, torch.Tensor):
        # Avoid torch->numpy bridge in environments where torch was built
        # against an incompatible NumPy ABI.
        return np.asarray(X.detach().cpu().tolist(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _maybe_pca(
    X: np.ndarray, pca_components: Optional[int]
) -> tuple[np.ndarray, Optional[object]]:
    """Reduce dimensionality for KNN search only (matches IDS v4.9 notebook)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if pca_components is None or X.shape[1] <= int(pca_components):
        return X, None
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_components))
        reduced = pca.fit_transform(X)
        return np.ascontiguousarray(reduced.astype(np.float32)), pca
    except Exception:
        x_t = torch.tensor(X.tolist(), dtype=torch.float32)
        x_centered = x_t - x_t.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(x_centered, q=int(pca_components))
        reduced = x_centered @ v[:, : int(pca_components)]
        return np.ascontiguousarray(reduced.numpy().astype(np.float32)), None


def _knn_indices_faiss(X: np.ndarray, k: int) -> np.ndarray:
    """kNN indices [N, k] using FAISS CPU L2 (self neighbor excluded)."""
    if not _FAISS_AVAILABLE:
        raise RuntimeError("faiss is not installed")
    n, d = X.shape
    x = np.ascontiguousarray(X, dtype=np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(x)
    _, indices = index.search(x, k + 1)
    return indices[:, 1:].copy()


def _knn_indices_pynndescent(X: np.ndarray, k: int) -> np.ndarray:
    """Approximate kNN using PyNNDescent."""
    if not _PYNNDESCENT_AVAILABLE:
        raise RuntimeError("pynndescent is not installed")
    index = NNDescent(
        X,
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
        verbose=False,
    )
    indices, _ = index.neighbor_graph
    return np.ascontiguousarray(indices[:, 1:], dtype=np.int64)


def _knn_indices_sklearn(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
        nn.fit(X)
        return nn.kneighbors(return_distance=False)
    except Exception:
        return _knn_indices_bruteforce(X, k, metric)


def _knn_indices_bruteforce(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Dependency-free exact kNN fallback for small/medium arrays."""
    x = torch.tensor(
        np.ascontiguousarray(X, dtype=np.float32).tolist(), dtype=torch.float32
    )
    if metric in ("euclidean", "l2"):
        distances = torch.cdist(x, x, p=2)
    elif metric == "cosine":
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        distances = 1.0 - x_norm @ x_norm.t()
    else:
        raise RuntimeError(
            f"metric={metric!r} requires scikit-learn; install a working "
            "scikit-learn/SciPy stack or use metric='euclidean'/'cosine'."
        )
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=k, largest=False).indices.cpu().numpy()


def _select_knn_indices(
    X_knn: np.ndarray,
    k: int,
    metric: str,
    knn_backend: KnnBackend,
) -> tuple[np.ndarray, str]:
    """Return neighbor indices and the backend actually used."""
    if knn_backend == "faiss":
        if not _FAISS_AVAILABLE:
            raise RuntimeError("knn_backend='faiss' but faiss is not installed")
        return _knn_indices_faiss(X_knn, k), "faiss"

    if knn_backend == "pynndescent":
        if not _PYNNDESCENT_AVAILABLE:
            raise RuntimeError(
                "knn_backend='pynndescent' but pynndescent is not installed"
            )
        return _knn_indices_pynndescent(X_knn, k), "pynndescent"

    if knn_backend == "sklearn":
        return _knn_indices_sklearn(X_knn, k, metric), "sklearn"

    # auto: faiss -> pynndescent -> sklearn
    if _FAISS_AVAILABLE:
        try:
            return _knn_indices_faiss(X_knn, k), "faiss"
        except Exception:
            pass
    if _PYNNDESCENT_AVAILABLE:
        try:
            return _knn_indices_pynndescent(X_knn, k), "pynndescent"
        except Exception:
            pass
    return _knn_indices_sklearn(X_knn, k, metric), "sklearn"


def build_knn_graph(
    X: Union[torch.Tensor, np.ndarray],
    k: int,
    metric: str = "euclidean",
    undirected: bool = False,
    cache_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    pca_components: Optional[int] = None,
    knn_backend: KnnBackend = "auto",
) -> torch.Tensor:
    """Build k-nearest neighbors graph. Returns edge_index [2, E] in COO format.

    Neighbors are computed in Euclidean space. By default edges are directed:
    ``(i, j)`` means ``j`` is a k-nearest neighbor of ``i``. Set ``undirected=True``
    to add reverse edges and merge duplicates (symmetric kNN).

    For large ``N`` and high ``D``, use ``pca_components`` to run KNN on a PCA
    projection **only for neighbor search** (node features ``X`` passed to the
    model are unchanged). Prefer ``knn_backend='faiss'`` or ``'auto'`` (tries
    FAISS CPU, then PyNNDescent, then scikit-learn) to reduce RAM vs pure sklearn.

    Args:
        X: Feature matrix [N, D].
        k: Number of neighbors per node.
        metric: Distance metric for sklearn fallback; FAISS path uses L2.
        undirected: If True, graph is symmetrized via union of (i,j) and (j,i).
        cache_key: If provided, save/load graph from cache.
        cache_dir: Directory for cache files. Default: current dir.
        pca_components: If set and D > this value, PCA-reduce features for KNN only.
        knn_backend: ``auto`` | ``faiss`` | ``pynndescent`` | ``sklearn``.

    Returns:
        edge_index: Tensor [2, E], dtype long.
    """
    X_np = _to_numpy(X)
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_np.shape}")
    n = X_np.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n_samples ({n}), got k={k}")

    X_knn, _pca = _maybe_pca(X_np, pca_components)

    cache_suffix = f"{pca_components}_{knn_backend}"
    if cache_key is not None or cache_dir is not None:
        key = cache_key or _cache_key(
            X_np.shape, k, metric, undirected, pca_components, knn_backend
        )
        cdir = Path(cache_dir) if cache_dir else Path(".")
        cache_path = cdir / f"{key}.edge_index.pt"
        if cache_path.exists():
            return load_edge_index(str(cache_path))

    indices, _ = _select_knn_indices(X_knn, k, metric, knn_backend)
    del X_knn

    row = np.repeat(np.arange(n, dtype=np.int64), k).tolist()
    col = indices.ravel().astype(np.int64).tolist()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    if undirected:
        combined = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        coalesced = pyg_coalesce(combined, num_nodes=n)
        edge_index = coalesced[0] if isinstance(coalesced, tuple) else coalesced
    if isinstance(X, torch.Tensor) and X.is_cuda:
        edge_index = edge_index.to(X.device)

    if cache_key is not None or cache_dir is not None:
        key = cache_key or _cache_key(
            X_np.shape, k, metric, undirected, pca_components, knn_backend
        )
        cdir = Path(cache_dir) if cache_dir else Path(".")
        cache_path = cdir / f"{key}.edge_index.pt"
        save_edge_index(str(cache_path), edge_index.cpu())

    return edge_index


def _cache_key(
    shape: tuple,
    k: int,
    metric: str,
    undirected: bool,
    pca_components: Optional[int],
    knn_backend: str,
) -> str:
    h = hashlib.sha256(
        f"{shape}_{k}_{metric}_{undirected}_{pca_components}_{knn_backend}".encode()
    ).hexdigest()[:16]
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
        k: Return only first k neighbors per source node.
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

    indices = _knn_indices_sklearn(X_np, max_k, metric)
    indices_slice = indices[:, :k]
    row = np.repeat(np.arange(n, dtype=np.int64), k).tolist()
    col = indices_slice.ravel().astype(np.int64).tolist()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    if isinstance(X, torch.Tensor) and X.is_cuda:
        edge_index = edge_index.to(X.device)
    return edge_index
