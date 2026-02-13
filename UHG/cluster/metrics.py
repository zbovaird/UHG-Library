"""Clustering quality metrics."""

from typing import Union

import numpy as np
import torch
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _to_numpy(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.cpu().numpy()
    return np.asarray(X)


def davies_bouldin(emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
    """Davies-Bouldin index (lower is better).

    Wraps sklearn.metrics.davies_bouldin_score.
    Requires at least 2 clusters (labels with distinct values).

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N], -1 for noise in DBSCAN.

    Returns:
        Davies-Bouldin score.
    """
    X = _to_numpy(emb)
    y = _to_numpy(labels).astype(np.int32)
    return float(davies_bouldin_score(X, y))


def silhouette(emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
    """Silhouette score (higher is better, range [-1, 1]).

    Wraps sklearn.metrics.silhouette_score.
    Noise points (label -1) are excluded by sklearn.

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N].

    Returns:
        Silhouette score.
    """
    X = _to_numpy(emb)
    y = _to_numpy(labels).astype(np.int32)
    return float(silhouette_score(X, y))


def calinski_harabasz(emb: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
    """Calinski-Harabasz index (higher is better).

    Wraps sklearn.metrics.calinski_harabasz_score.
    Requires at least 2 clusters.

    Args:
        emb: Embeddings [N, D].
        labels: Cluster labels [N].

    Returns:
        Calinski-Harabasz score.
    """
    X = _to_numpy(emb)
    y = _to_numpy(labels).astype(np.int32)
    return float(calinski_harabasz_score(X, y))
