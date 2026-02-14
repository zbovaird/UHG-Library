"""DBSCAN clustering with grid search and auto-eps utilities."""

from typing import List, Union

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .metrics import calinski_harabasz, davies_bouldin, silhouette


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.cpu().numpy()
    return np.asarray(X)


def _sanitize_for_sklearn(X: np.ndarray) -> np.ndarray:
    """Make embeddings safe for sklearn (finite + float32-range bounded)."""
    X = np.asarray(X, dtype=np.float64)

    # Replace NaN/Inf first.
    if not np.all(np.isfinite(X)):
        import warnings

        n_bad = int(np.sum(~np.isfinite(X)))
        warnings.warn(
            f"dbscan: {n_bad} non-finite values in embeddings; replacing with 0"
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Clamp huge finite magnitudes that overflow sklearn float32 validation.
    # Use a conservative safe bound well below float32 max.
    safe_abs_max = 1.0e30
    max_abs = float(np.max(np.abs(X))) if X.size else 0.0
    if max_abs > safe_abs_max:
        import warnings

        warnings.warn(
            f"dbscan: large embedding magnitude detected (max_abs={max_abs:.3e}); "
            f"clipping to +/-{safe_abs_max:.1e}"
        )
        X = np.clip(X, -safe_abs_max, safe_abs_max)

    # Keep dtype explicit so downstream sklearn checks don't cast unexpectedly.
    X = X.astype(np.float32, copy=False)

    # Final defensive check.
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


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
    X = _sanitize_for_sklearn(_to_numpy(emb))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
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
            metrics = {"davies_bouldin": db_val, "silhouette": sil_val, "calinski_harabasz": ch_val}
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
                best = {"labels": best_labels, "params": best_params, "metrics": metrics}

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
    X = _sanitize_for_sklearn(_to_numpy(emb))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    k_dist = np.sort(dists[:, k])
    n = len(k_dist)
    idx_90 = min(int(0.9 * n), n - 1)
    return float(k_dist[idx_90])
