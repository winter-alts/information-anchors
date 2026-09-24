"""Sidecar score alignment for the TS-RAG ARM attention prior."""

from __future__ import annotations

import numpy as np


def attention_prior_source_key(prior_method: str) -> str:
    if prior_method == "high_mi":
        return "high_mi_distances"
    if prior_method == "low_mi":
        return "low_mi_distances"
    if prior_method == "random":
        return "random_distances"
    if prior_method == "uniform":
        return "uniform_distances"
    if prior_method == "mi_prior":
        return "selection_scores_mi_prior"
    if prior_method == "null_mi_prior":
        return "selection_scores_null_mi_prior"
    raise ValueError(f"unknown ARM attention prior method: {prior_method}")


def attention_prior_scores(
    sidecar: dict[str, np.ndarray],
    *,
    method: str,
    prior_method: str,
    selected: np.ndarray,
    local_offset: int,
    batch_size: int,
) -> np.ndarray:
    """Return high-is-better scores aligned with the selected ARM tokens.

    Distance and selector-score sidecars are lower-is-better.  The model-side
    attention helper standardises high-is-better scores, so negate them here.
    Selection-score arrays are already aligned to their method's selected
    candidates and cannot be indexed by top-20 ranks.
    """
    if method.startswith("mi_global_"):
        raise ValueError("ARM attention prior currently requires a bounded top-20 sidecar")
    key = attention_prior_source_key(prior_method)
    if key not in sidecar:
        raise ValueError(f"artifact is missing {key} for ARM attention prior")
    values = np.asarray(sidecar[key])[local_offset:local_offset + batch_size]
    selected = np.asarray(selected, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != batch_size:
        raise ValueError(f"{key} has incompatible batch shape {values.shape}")
    if key.startswith("selection_scores_") and method != prior_method:
        raise ValueError(
            f"{key} is aligned to method={prior_method}, not method={method}; "
            "use a distance prior for cross-method attention controls"
        )
    if key.startswith("selection_scores_"):
        if values.shape[1] != selected.shape[1]:
            raise ValueError(
                f"{key} width {values.shape[1]} does not match selected width "
                f"{selected.shape[1]}"
            )
        aligned = values
    else:
        if selected.size and int(selected.max()) >= values.shape[1]:
            raise ValueError(
                f"cannot align {key} width {values.shape[1]} with selected "
                f"top-{selected.shape[1]} ranks"
            )
        aligned = np.take_along_axis(values, selected, axis=1)
    return -np.asarray(aligned, dtype=np.float32)

