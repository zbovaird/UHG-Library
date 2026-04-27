"""DBSCAN clustering with grid search and auto-eps utilities."""

from typing import List, Union

import numpy as np
import torch

from .metrics import calinski_harabasz, davies_bouldin, silhouette


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return np.asarray(X.detach().cpu().tolist(), dtype=np.float64)
    return np.asarray(X)


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _dbscan_fallback(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small dependency-free DBSCAN fallback used when sklearn is unavailable."""
    n = X.shape[0]
    distances = _pairwise_distances(X)
    neighborhoods = [np.flatnonzero(distances[i] <= eps) for i in range(n)]
    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = neighborhoods[i]
        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seeds = list(neighbors[neighbors != i])
        while seeds:
            j = seeds.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = neighborhoods[j]
                if len(j_neighbors) >= min_samples:
                    for candidate in j_neighbors:
                        if candidate not in seeds:
                            seeds.append(int(candidate))
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def run_dbscan(
    emb: Union[np.ndarray, torch.Tensor],
    eps: float,
    min_samples: int,
) -> dict:
    """Run DBSCAN clustering.

    Args:
        emb: Embeddings [N, D].
        eps: Maximum distance for neighborhood.
        min_samples: Min points to form a core point.

    Returns:
        Dict with keys: labels (np.ndarray), core_mask (bool array, True for non-noise).
    """
    X = _to_numpy(emb)
    try:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
    except Exception:
        labels = _dbscan_fallback(X, eps, min_samples)
    core_mask = labels >= 0
    return {"labels": labels, "core_mask": core_mask}


def eps_grid_search(
    emb: Union[np.ndarray, torch.Tensor],
    eps_list: List[float],
    min_samples_list: List[int],
    score: str = "db",
) -> dict:
    """Search for best eps and min_samples by clustering metric.

    Args:
        emb: Embeddings [N, D].
        eps_list: Candidate eps values.
        min_samples_list: Candidate min_samples values.
        score: Metric to optimize - "db" (lower better), "silhouette" (higher), "ch" (higher).

    Returns:
        Dict with keys: labels, params (best eps, min_samples), metrics (dict of best run).
    """
    X = _to_numpy(emb)
    best = None
    best_val = float("-inf") if score in ("silhouette", "ch") else float("inf")
    best_labels = None
    best_params = None

    for eps in eps_list:
        for ms in min_samples_list:
            out = run_dbscan(X, eps, ms)
            labels = out["labels"]
            n_clusters = len(set(labels) - {-1})
            if n_clusters < 2:
                continue
            try:
                db_val = davies_bouldin(X, labels)
                sil_val = silhouette(X, labels)
                ch_val = calinski_harabasz(X, labels)
            except Exception:
                continue
            metrics = {
                "davies_bouldin": db_val,
                "silhouette": sil_val,
                "calinski_harabasz": ch_val,
            }
            if score == "db":
                val = db_val
                better = val < best_val
            elif score == "silhouette":
                val = sil_val
                better = val > best_val
            else:
                val = ch_val
                better = val > best_val
            if better:
                best_val = val
                best_labels = labels
                best_params = {"eps": eps, "min_samples": ms}
                best = {
                    "labels": best_labels,
                    "params": best_params,
                    "metrics": metrics,
                }

    if best is None:
        return {"labels": np.full(X.shape[0], -1), "params": {}, "metrics": None}
    return best


def auto_eps_kdist(
    emb: Union[np.ndarray, torch.Tensor],
    k: int = 4,
) -> float:
    """Heuristic eps from k-distance curve elbow.

    Computes k-th neighbor distance for each point, sorts, and picks a heuristic
    elbow (e.g. max curvature or 90th percentile of sorted distances).

    Args:
        emb: Embeddings [N, D].
        k: k for k-distance.

    Returns:
        Suggested eps value.
    """
    X = _to_numpy(emb)
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
    except Exception:
        distances = _pairwise_distances(X)
        distances.sort(axis=1)
        dists = distances[:, : k + 1]
    k_dist = np.sort(dists[:, k])
    n = len(k_dist)
    idx_90 = min(int(0.9 * n), n - 1)
    return float(k_dist[idx_90])
