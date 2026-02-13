"""Anomaly scoring functions using UHG quadrance."""

from typing import Dict, Optional, Union

import numpy as np
import torch

from uhg.graph.build import build_knn_graph
from uhg.utils.metrics import uhg_quadrance


def _to_tensor(X, device=None):
    if isinstance(X, np.ndarray):
        t = torch.from_numpy(X).float()
    else:
        t = X.float()
    if device is not None:
        t = t.to(device)
    return t


def _ensure_homogeneous(emb: torch.Tensor) -> torch.Tensor:
    """Ensure embeddings have homogeneous coordinate for UHG (Minkowski -1).
    spatial = emb[..., :-1] or emb, time = sqrt(1 + |spatial|^2).
    """
    if emb.size(-1) >= 2:
        spatial = emb[..., :-1]
    else:
        spatial = emb
    ip = (spatial * spatial).sum(dim=-1, keepdim=True)
    z = torch.sqrt(torch.clamp(1.0 + ip, min=1e-9))
    return torch.cat([spatial, z], dim=-1)


def _normalize_to_uhg(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Project point to UHG (Minkowski norm -1): spatial part, time = sqrt(1 + |spatial|^2)."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    spatial = a[..., :-1]
    ip = (spatial * spatial).sum(dim=-1, keepdim=True)
    time = torch.sqrt(torch.clamp(1.0 + ip, min=eps))
    return torch.cat([spatial, time], dim=-1)


def centroid_quadrance(emb: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """Anomaly score: quadrance from each point to centroid.

    Centroid is mean of embeddings, renormalized to UHG manifold.
    Higher quadrance = more anomalous.

    Args:
        emb: Embeddings [N, D] or [N, D+1] in homogeneous coords.
        eps: Numerical stability.

    Returns:
        Scores [N], higher = more anomalous.
    """
    emb = _to_tensor(emb)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    emb = _ensure_homogeneous(emb)
    centroid = emb.mean(dim=0)
    centroid = _normalize_to_uhg(centroid.unsqueeze(0), eps).squeeze(0)
    q = uhg_quadrance(emb, centroid.unsqueeze(0).expand(emb.size(0), -1), eps=eps)
    return q.squeeze(-1) if q.dim() > 1 else q


def neighbor_quadrance(
    emb: torch.Tensor,
    k: int = 5,
    *,
    edge_index: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Anomaly score: mean quadrance to k nearest neighbors in embedding space.

    Higher mean quadrance = more anomalous (further from neighbors).

    Args:
        emb: Embeddings [N, D+1].
        k: Number of neighbors.
        edge_index: Optional precomputed kNN edges [2, E]. If None, computed from emb.
        eps: Numerical stability.

    Returns:
        Scores [N].
    """
    emb = _to_tensor(emb)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    emb = _ensure_homogeneous(emb)
    n = emb.size(0)
    if edge_index is None:
        edge_index = build_knn_graph(emb.cpu().numpy(), k=k)
        if emb.is_cuda:
            edge_index = edge_index.to(emb.device)
    row, col = edge_index[0], edge_index[1]
    q = uhg_quadrance(emb[row], emb[col], eps=eps)
    if q.dim() > 1:
        q = q.squeeze(-1)
    out = torch.zeros(n, device=emb.device, dtype=emb.dtype)
    cnt = torch.zeros(n, device=emb.device, dtype=torch.long)
    out.scatter_add_(0, row, q)
    cnt.scatter_add_(0, row, torch.ones_like(row, dtype=torch.long))
    cnt = cnt.clamp(min=1)
    return out / cnt.float()


def boundary_score(
    labels: np.ndarray,
    core_mask: np.ndarray,
    emb: torch.Tensor,
    *,
    k: int = 3,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Anomaly score: higher for non-core (noise) points; core points get mean quadrance to core neighbors.

    Non-core (noise) points receive score 1.0 (max anomalous).
    Core points receive mean quadrance to their k nearest core neighbors.

    Args:
        labels: Cluster labels from DBSCAN.
        core_mask: True for core points.
        emb: Embeddings [N, D+1].
        k: Neighbors for core points.
        eps: Numerical stability.

    Returns:
        Scores [N].
    """
    emb = _to_tensor(emb)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    emb = _ensure_homogeneous(emb)
    n = emb.size(0)
    core_mask = np.asarray(core_mask)
    scores = torch.ones(n, device=emb.device, dtype=emb.dtype)
    core_idx = np.where(core_mask)[0]
    if len(core_idx) < 2:
        return scores
    emb_core = emb[core_idx]
    ei = build_knn_graph(emb_core.cpu().numpy(), k=min(k, len(core_idx) - 1))
    if emb.is_cuda:
        ei = ei.to(emb.device)
    row_c, col_c = ei[0], ei[1]
    q = uhg_quadrance(emb_core[row_c], emb_core[col_c], eps=eps)
    out = torch.zeros(len(core_idx), device=emb.device, dtype=emb.dtype)
    cnt = torch.zeros(len(core_idx), device=emb.device, dtype=torch.long)
    out.scatter_add_(0, row_c, q)
    cnt.scatter_add_(0, row_c, torch.ones_like(row_c, dtype=torch.long))
    cnt = cnt.clamp(min=1)
    core_scores = out / cnt.float()
    for i, idx in enumerate(core_idx):
        scores[idx] = core_scores[i].item()
    return scores


def composite_score(scores: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """Weighted sum of named score tensors.

    Args:
        scores: Dict of name -> tensor [N].
        weights: Dict of name -> weight.

    Returns:
        Combined score [N].
    """
    out = None
    for name, w in weights.items():
        if name not in scores:
            continue
        s = scores[name] * w
        out = s if out is None else out + s
    if out is None:
        raise ValueError("No matching scores for given weights")
    return out
