"""Deterministic, history-only temporal motif labels for MI peak mapping."""

from __future__ import annotations

import numpy as np


MOTIF_NAMES = (
    "rising_trend",
    "falling_trend",
    "volatile_burst",
    "smooth_segment",
    "turning_up",
    "turning_down",
    "level_jump",
    "local_peak",
    "local_trough",
    "cross_channel_slope_sync",
    "cross_channel_volatility_burst",
)


def _patch_statistics(
    histories: np.ndarray,
    *,
    patch_len: int,
    patch_stride: int,
    num_patches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected histories [sample, time, channel], got {values.shape}")
    if patch_len < 2 or patch_stride < 1 or num_patches < 1:
        raise ValueError("patch_len must be >=2; patch_stride and num_patches must be positive")
    n_samples, history_length, _ = values.shape
    time = np.linspace(-1.0, 1.0, patch_len, dtype=np.float64)
    time_norm = max(float(np.dot(time, time)), 1e-12)
    means = []
    slopes = []
    diff_stds = []
    roughness = []
    for patch in range(num_patches):
        start = patch * patch_stride
        stop = start + patch_len
        if stop > history_length:
            raise ValueError(f"Patch {patch} spans [{start}, {stop}) outside history={history_length}")
        segment = values[:, start:stop, :]
        mean = segment.mean(axis=1)
        centered = segment - mean[:, None, :]
        slope = np.einsum("ntc,t->nc", centered, time) / time_norm
        diff_std = np.diff(segment, axis=1).std(axis=1)
        second = np.diff(segment, n=2, axis=1)
        rough = np.sqrt(np.mean(np.square(second), axis=1)) if second.shape[1] else np.zeros_like(mean)
        means.append(mean)
        slopes.append(slope)
        diff_stds.append(diff_std)
        roughness.append(rough)
    arrays = tuple(np.stack(items, axis=1) for items in (means, slopes, diff_stds, roughness))
    if not all(np.isfinite(item).all() for item in arrays):
        raise ValueError("History motif statistics contain non-finite values")
    return arrays  # type: ignore[return-value]


def build_history_motif_labels(
    histories: np.ndarray,
    *,
    patch_len: int,
    patch_stride: int,
    num_patches: int,
    target_channel: int = 0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return boolean ``[sample, patch, motif]`` labels from history only.

    Thresholds are computed within each origin across its patches. This makes
    the labels describe local structure rather than absolute scale or position.
    """
    means, slopes, diff_stds, roughness = _patch_statistics(
        histories,
        patch_len=patch_len,
        patch_stride=patch_stride,
        num_patches=num_patches,
    )
    if not 0 <= target_channel < means.shape[2]:
        raise ValueError(f"target_channel={target_channel} outside {means.shape[2]} channels")
    eps = 1e-8
    target_mean = means[:, :, target_channel]
    target_slope = slopes[:, :, target_channel]
    target_diff = diff_stds[:, :, target_channel]
    target_rough = roughness[:, :, target_channel]
    q25_slope = np.quantile(target_slope, 0.25, axis=1, keepdims=True)
    q75_slope = np.quantile(target_slope, 0.75, axis=1, keepdims=True)
    q25_diff = np.quantile(target_diff, 0.25, axis=1, keepdims=True)
    q75_diff = np.quantile(target_diff, 0.75, axis=1, keepdims=True)
    q75_rough = np.quantile(target_rough, 0.75, axis=1, keepdims=True)

    labels = []
    labels.append(target_slope >= q75_slope)
    labels.append(target_slope <= q25_slope)
    labels.append(target_diff >= q75_diff)
    labels.append(target_diff <= q25_diff)

    turning_up = np.zeros_like(target_slope, dtype=bool)
    turning_down = np.zeros_like(target_slope, dtype=bool)
    if num_patches >= 2:
        turning_up[:, :-1] = (target_slope[:, :-1] < 0.0) & (target_slope[:, 1:] > 0.0)
        turning_down[:, :-1] = (target_slope[:, :-1] > 0.0) & (target_slope[:, 1:] < 0.0)
    labels.extend([turning_up, turning_down])

    jumps = np.abs(np.diff(target_mean, axis=1))
    q75_jump = np.quantile(jumps, 0.75, axis=1, keepdims=True) if jumps.shape[1] else np.zeros((len(target_mean), 1))
    level_jump = np.zeros_like(target_mean, dtype=bool)
    if num_patches >= 2:
        level_jump[:, 1:] = jumps >= (q75_jump + eps)
    labels.append(level_jump)

    local_peak = np.zeros_like(target_mean, dtype=bool)
    local_trough = np.zeros_like(target_mean, dtype=bool)
    if num_patches >= 3:
        local_peak[:, 1:-1] = (
            (target_mean[:, 1:-1] >= target_mean[:, :-2])
            & (target_mean[:, 1:-1] >= target_mean[:, 2:])
            & (target_mean[:, 1:-1] > np.quantile(target_mean, 0.75, axis=1, keepdims=True))
        )
        local_trough[:, 1:-1] = (
            (target_mean[:, 1:-1] <= target_mean[:, :-2])
            & (target_mean[:, 1:-1] <= target_mean[:, 2:])
            & (target_mean[:, 1:-1] < np.quantile(target_mean, 0.25, axis=1, keepdims=True))
        )
    labels.extend([local_peak, local_trough])

    slope_sign = np.sign(slopes)
    channel_mean_sign = np.mean(slope_sign, axis=2)
    cross_slope_sync = np.abs(channel_mean_sign) >= 0.75
    labels.append(cross_slope_sync)

    channel_diff_median = np.median(diff_stds, axis=2)
    q75_cross_diff = np.quantile(channel_diff_median, 0.75, axis=1, keepdims=True)
    labels.append(channel_diff_median >= q75_cross_diff)

    output = np.stack(labels, axis=2).astype(bool)
    if output.shape[2] != len(MOTIF_NAMES):
        raise RuntimeError(f"Motif shape {output.shape} disagrees with names={len(MOTIF_NAMES)}")
    return output, MOTIF_NAMES

