"""Schema and data utilities for detecting columns and preparing DataFrames."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


# Common column name variants for label and entity detection
LABEL_VARIANTS = frozenset({
    "label", "labels", "Label", "Labels", "target", "Target",
    "y", "class", "Class", "outcome", "Outcome"
})
ENTITY_VARIANTS = frozenset({
    "entity", "Entity", "entity_id", "entity_id", "id", "ID",
    "entityid", "entityId", "entity_id", "customer_id", "user_id"
})


def detect_label_column(df: "pd.DataFrame") -> Optional[str]:
    """Detect column that likely contains labels/targets.

    Checks common variants (label, target, y, class, etc.) case-insensitively.

    Args:
        df: DataFrame to inspect.

    Returns:
        Column name if found, else None.
    """
    if not HAS_PANDAS or df is None:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for v in LABEL_VARIANTS:
        if v.lower() in cols_lower:
            return cols_lower[v.lower()]
    return None


def detect_entity_column(df: "pd.DataFrame") -> Optional[str]:
    """Detect column that likely contains entity IDs.

    Checks common variants (entity_id, id, customer_id, etc.).

    Args:
        df: DataFrame to inspect.

    Returns:
        Column name if found, else None.
    """
    if not HAS_PANDAS or df is None:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for v in ENTITY_VARIANTS:
        if v.lower() in cols_lower:
            return cols_lower[v.lower()]
    return None


def enforce_numeric(
    df: "pd.DataFrame",
    fill: str = "mean",
    replace_inf: bool = True,
) -> "pd.DataFrame":
    """Ensure all columns are numeric. Fill NaN and optionally replace inf.

    Args:
        df: Input DataFrame.
        fill: How to fill NaN - "mean" (column mean), "median", "zero", or "drop" (drop rows).
        replace_inf: If True, replace inf/-inf with NaN then apply fill.

    Returns:
        DataFrame with numeric columns only.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for enforce_numeric")
    out = df.apply(pd.to_numeric, errors="coerce")
    if replace_inf:
        out = out.replace([np.inf, -np.inf], np.nan)
    if fill == "mean":
        out = out.fillna(out.mean())
    elif fill == "median":
        out = out.fillna(out.median())
    elif fill == "zero":
        out = out.fillna(0)
    elif fill == "drop":
        out = out.dropna()
    else:
        out = out.fillna(out.mean())
    return out


def build_entity_index(series: "pd.Series") -> Tuple[np.ndarray, dict, dict]:
    """Build integer index for entity IDs and mapping dicts.

    Args:
        series: Series of entity IDs (e.g. from df["entity_id"]).

    Returns:
        Tuple of (index_array, id_to_idx, idx_to_id).
        index_array: Integer array mapping row -> entity index.
        id_to_idx: dict mapping entity_id -> integer index.
        idx_to_id: dict mapping integer index -> entity_id.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for build_entity_index")
    uniq = series.dropna().unique()
    id_to_idx = {v: i for i, v in enumerate(uniq)}
    idx_to_id = {i: v for v, i in id_to_idx.items()}
    index_array = series.map(id_to_idx).fillna(-1).astype(np.int64).values
    return index_array, id_to_idx, idx_to_id
