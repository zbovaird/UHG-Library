"""Clustering quality metrics."""

from typing import Union

import numpy as np
import torch


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return np.asarray(X.detach().cpu().tolist())
    return np.asarray(X)


def _valid_cluster_data(
    emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _to_numpy(emb).astype(np.float64)
    y = _to_numpy(labels).astype(np.int32)
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError("At least two clusters are required")
    return X, y, unique


def _cluster_centroids(X: np.ndarray, y: np.ndarray, unique: np.ndarray) -> np.ndarray:
    return np.stack([X[y == label].mean(axis=0) for label in unique])


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def davies_bouldin(
    emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]
) -> float:
    """Davies-Bouldin index (lower is better).

    Wraps sklearn.metrics.davies_bouldin_score.
    Requires at least 2 clusters (labels with distinct values).

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N], -1 for noise in DBSCAN.

    Returns:
        Davies-Bouldin score.
    """
    X, y, unique = _valid_cluster_data(emb, labels)
    try:
        from sklearn.metrics import davies_bouldin_score

        return float(davies_bouldin_score(X, y))
    except Exception:
        centroids = _cluster_centroids(X, y, unique)
        scatter = np.array(
            [
                np.mean(np.linalg.norm(X[y == label] - centroids[i], axis=1))
                for i, label in enumerate(unique)
            ]
        )
        centroid_dist = _pairwise_distances(centroids, centroids)
        np.fill_diagonal(centroid_dist, np.inf)
        ratios = (scatter[:, None] + scatter[None, :]) / centroid_dist
        return float(np.mean(np.max(ratios, axis=1)))


def silhouette(
    emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]
) -> float:
    """Silhouette score (higher is better, range [-1, 1]).

    Wraps sklearn.metrics.silhouette_score.
    Noise points (label -1) are excluded by sklearn.

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N].

    Returns:
        Silhouette score.
    """
    X, y, unique = _valid_cluster_data(emb, labels)
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(X, y))
    except Exception:
        distances = _pairwise_distances(X, X)
        scores = []
        for i in range(X.shape[0]):
            same = y == y[i]
            same[i] = False
            a = float(np.mean(distances[i, same])) if np.any(same) else 0.0
            b = min(
                float(np.mean(distances[i, y == label]))
                for label in unique
                if label != y[i]
            )
            denom = max(a, b)
            scores.append(0.0 if denom == 0.0 else (b - a) / denom)
        return float(np.mean(scores))


def calinski_harabasz(
    emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]
) -> float:
    """Calinski-Harabasz index (higher is better).

    Wraps sklearn.metrics.calinski_harabasz_score.
    Requires at least 2 clusters.

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N].

    Returns:
        Calinski-Harabasz score.
    """
    X, y, unique = _valid_cluster_data(emb, labels)
    try:
        from sklearn.metrics import calinski_harabasz_score

        return float(calinski_harabasz_score(X, y))
    except Exception:
        n_samples = X.shape[0]
        n_clusters = unique.size
        overall_mean = X.mean(axis=0)
        centroids = _cluster_centroids(X, y, unique)
        counts = np.array([np.sum(y == label) for label in unique], dtype=np.float64)
        between = np.sum(counts * np.sum((centroids - overall_mean) ** 2, axis=1))
        within = sum(
            np.sum((X[y == label] - centroids[i]) ** 2)
            for i, label in enumerate(unique)
        )
        if within == 0.0 or n_samples == n_clusters:
            return float("inf")
        return float((between / (n_clusters - 1)) / (within / (n_samples - n_clusters)))
