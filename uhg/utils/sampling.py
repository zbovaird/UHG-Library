"""Stratified subsampling helpers (e.g. CIC-style imbalanced traffic data)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def stratified_subsample_indices(
    y: np.ndarray,
    n_total: int,
    min_per_class: int = 1,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Pick ``n_total`` row indices with a per-class floor when possible.

    Round 1: up to ``min_per_class`` samples from each class (without replacement).
    Round 2: fill remaining slots uniformly at random from unused rows.

    Args:
        y: Label vector of length N (any comparable dtype).
        n_total: Target number of indices (must be <= N).
        min_per_class: Minimum samples to draw per class in round 1, capped by
            class count. If ``min_per_class * n_classes > n_total``, ``ValueError``.
        random_state: RNG seed for reproducibility (numpy Generator).

    Returns:
        Integer array of shape ``(n_total,)`` — indices into the original ``y``.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    n = y.shape[0]
    if n_total > n:
        raise ValueError(f"n_total ({n_total}) cannot exceed len(y) ({n})")
    if n_total < 1:
        raise ValueError("n_total must be >= 1")

    classes = np.unique(y)
    n_classes = len(classes)
    if min_per_class < 0:
        raise ValueError("min_per_class must be >= 0")

    floor = max(0, min_per_class)
    need = floor * n_classes
    if need > n_total:
        raise ValueError(
            f"min_per_class * n_classes ({need}) > n_total ({n_total}); "
            "lower min_per_class or increase n_total."
        )

    picked: list[int] = []
    for c in classes:
        pool = np.where(y == c)[0]
        take = min(floor, len(pool))
        if take > 0:
            picked.extend(rng.choice(pool, size=take, replace=False).tolist())

    picked_arr = np.array(picked, dtype=np.int64)
    if picked_arr.size > n_total:
        return rng.choice(picked_arr, size=n_total, replace=False)

    used = np.zeros(n, dtype=bool)
    used[picked_arr] = True
    pool_rest = np.where(~used)[0]
    need_more = n_total - int(picked_arr.size)
    if need_more > 0:
        if need_more > pool_rest.size:
            raise ValueError("Internal error: not enough remaining indices.")
        extra = rng.choice(pool_rest, size=need_more, replace=False)
        picked_arr = np.concatenate([picked_arr, extra])
    rng.shuffle(picked_arr)
    return picked_arr
