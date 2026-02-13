"""Reporting utilities for anomaly detection."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch


def rank_topk(
    scores: Union[torch.Tensor, np.ndarray],
    k: int,
    ids: Optional[np.ndarray] = None,
) -> List[dict]:
    """Rank top-k highest-scoring entities.

    Args:
        scores: Anomaly scores [N].
        k: Number of top results.
        ids: Optional entity IDs [N]. If None, use indices.

    Returns:
        List of dicts with keys: rank, id (or index), score.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores)
    order = np.argsort(-scores)[:k]
    out = []
    for r, idx in enumerate(order):
        rec = {"rank": r + 1, "score": float(scores[idx])}
        if ids is not None:
            rec["id"] = ids[idx]
        else:
            rec["index"] = int(idx)
        out.append(rec)
    return out


def aggregate_by_entity(
    scores: Union[torch.Tensor, np.ndarray],
    entity_ids: np.ndarray,
    stats: Tuple[str, ...] = ("mean", "p95", "count"),
) -> dict:
    """Aggregate scores by entity.

    Args:
        scores: Per-record scores [N].
        entity_ids: Entity index per record [N] (from build_entity_index).
        stats: Which stats to compute.

    Returns:
        Dict mapping entity_idx -> {mean, p95, count, ...}.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores).ravel()
    entity_ids = np.asarray(entity_ids).ravel()
    valid = entity_ids >= 0
    scores = scores[valid]
    entity_ids = entity_ids[valid]
    uniq = np.unique(entity_ids)
    out = {}
    for eid in uniq:
        mask = entity_ids == eid
        s = scores[mask]
        rec = {}
        if "mean" in stats:
            rec["mean"] = float(np.mean(s))
        if "p95" in stats:
            rec["p95"] = float(np.percentile(s, 95))
        if "count" in stats:
            rec["count"] = int(len(s))
        out[int(eid)] = rec
    return out


def summary_to_json(summary: dict, path: str) -> None:
    """Write summary dict to JSON file."""
    import json

    def _enc(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): _enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_enc(x) for x in o]
        return o

    with open(path, "w") as f:
        json.dump(_enc(summary), f, indent=2)


def display_summary(summary: dict) -> None:
    """Print summary to stdout (N, E, k, timings, DB score, cluster counts, top entities)."""
    lines = []
    if "n_nodes" in summary:
        lines.append(f"N: {summary['n_nodes']}")
    if "n_edges" in summary:
        lines.append(f"E: {summary['n_edges']}")
    if "k" in summary:
        lines.append(f"k: {summary['k']}")
    if "timings" in summary:
        for k, v in summary["timings"].items():
            lines.append(f"  {k}: {v:.3f}s")
    if "cluster_metrics" in summary:
        for k, v in summary["cluster_metrics"].items():
            lines.append(f"  {k}: {v}")
    if "top_entities" in summary:
        lines.append("Top entities by score:")
        for r in summary["top_entities"][:10]:
            lines.append(f"  {r}")
    print("\n".join(lines))
