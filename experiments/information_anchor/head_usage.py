"""Head-aware selection utilities for held-out activation interventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.information_anchor.interventions.common import UnitSelection


@dataclass(frozen=True)
class HeadAwareSelection:
    """一个 MI/head 二维象限中的干预单元。"""

    unit: UnitSelection
    head_z: float
    quadrant: str
    mi_threshold: float
    head_threshold: float


def zscore_profile(profile: np.ndarray) -> np.ndarray:
    """把 attention 或 gradient profile 标准化到可比较的 z 分数。"""
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Head profile must be finite [patch], got {values.shape}.")
    return ((values - values.mean()) / max(float(values.std()), 1e-12)).astype(np.float32)


def select_head_aware_units(
    mi_z: np.ndarray,
    head_profile: np.ndarray,
    *,
    threshold_quantile: float = 0.5,
) -> list[HeadAwareSelection]:
    """在四个 MI/head 象限各选一个 unit，head 分数按 patch 广播到各层。"""
    mi = np.asarray(mi_z, dtype=np.float64)
    if mi.ndim != 2 or not np.isfinite(mi).all():
        raise ValueError(f"MI profile must be finite [layer, patch], got {mi.shape}.")
    head = zscore_profile(head_profile)
    if len(head) != mi.shape[1]:
        raise ValueError(f"Head profile has {len(head)} patches but MI has {mi.shape[1]}.")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be strictly between zero and one.")
    mi_threshold = float(np.quantile(mi, threshold_quantile))
    head_threshold = float(np.quantile(head, threshold_quantile))
    high_mi = mi >= mi_threshold
    high_head_patch = head >= head_threshold
    high_head = np.broadcast_to(high_head_patch[None, :], mi.shape)
    masks = {
        "mi_high_head_high": high_mi & high_head,
        "mi_high_head_low": high_mi & ~high_head,
        "mi_low_head_high": ~high_mi & high_head,
        "mi_low_head_low": ~high_mi & ~high_head,
    }
    objectives = {
        "mi_high_head_high": mi + head[None, :],
        "mi_high_head_low": mi - head[None, :],
        "mi_low_head_high": head[None, :] - mi,
        "mi_low_head_low": -(mi + head[None, :]),
    }
    selections: list[HeadAwareSelection] = []
    for quadrant, mask in masks.items():
        candidate_indices = np.argwhere(mask)
        if len(candidate_indices) == 0:
            # 奇数 patch 或并列中位数可能让某个象限为空；选择最接近该象限
            # 的 cell 并记录真实分数，避免静默改变实验条件。
            if quadrant == "mi_high_head_high":
                score = mi + head[None, :]
            elif quadrant == "mi_high_head_low":
                score = mi - head[None, :]
            elif quadrant == "mi_low_head_high":
                score = head[None, :] - mi
            else:
                score = -(mi + head[None, :])
            flat_index = int(np.argmax(score))
            layer, patch = np.unravel_index(flat_index, mi.shape)
        else:
            scores = objectives[quadrant][mask]
            layer, patch = candidate_indices[int(np.argmax(scores))]
        layer, patch = int(layer), int(patch)
        selections.append(
            HeadAwareSelection(
                unit=UnitSelection(quadrant, layer, patch, float(mi[layer, patch])),
                head_z=float(head[patch]),
                quadrant=quadrant,
                mi_threshold=mi_threshold,
                head_threshold=head_threshold,
            )
        )
    return selections
