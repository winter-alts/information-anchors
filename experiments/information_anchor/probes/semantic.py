from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.information_anchor.targets.future_summary import build_future_summary


@dataclass(frozen=True)
class SemanticTargets:
    values: np.ndarray
    names: tuple[str, ...]


SEMANTIC_PROTOCOL = "future-semantics-v6.1-12-nonredundant"
SEMANTIC_NAMES = (
    "level_q1",
    "level_q2",
    "level_q3",
    "level_q4",
    "global_change",
    "linear_trend",
    "mean_absolute_change",
    "diff_std",
    "roughness",
    "low_frequency_energy",
    "mid_frequency_energy",
    "high_frequency_energy",
)


def build_global_semantic_targets(
    future_normalized: np.ndarray,
    channel_names: tuple[str, ...] | None = None,
) -> SemanticTargets:
    """Macro-average every registered semantic across the channel system."""
    future = np.asarray(future_normalized, dtype=np.float32)
    if future.ndim != 3:
        raise ValueError(f"Global semantics require [samples, horizon, channels], got {future.shape}")
    channel_targets = build_semantic_targets(future, channel_names)
    n_channels = future.shape[-1]
    values = channel_targets.values.reshape(len(future), n_channels, -1)
    semantic_names = tuple(name.split(":", 1)[1] for name in channel_targets.names[: values.shape[-1]])
    output = values.mean(axis=1).astype(np.float32)
    names = tuple(f"cross_channel_mean:{name}" for name in semantic_names)
    if not np.isfinite(output).all():
        raise ValueError("Global semantic targets contain non-finite values.")
    return SemanticTargets(values=output, names=names)


def _normalized_spectral_energy(future: np.ndarray) -> np.ndarray:
    centered = future - future.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1))[:, 1:] ** 2
    if power.shape[1] == 0:
        return np.zeros((len(future), 3), dtype=np.float32)
    chunks = np.array_split(np.arange(power.shape[1]), 3)
    band_power = np.stack(
        [
            power[:, indices].sum(axis=1)
            if len(indices)
            else np.zeros(len(future), dtype=np.float64)
            for indices in chunks
        ],
        axis=1,
    )
    total = band_power.sum(axis=1, keepdims=True)
    return np.divide(
        band_power,
        total,
        out=np.zeros_like(band_power),
        where=total > np.finfo(np.float64).eps,
    ).astype(np.float32)


def _single_channel_targets(future: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct the preregistered 12-property future-semantic vector."""
    future = np.asarray(future, dtype=np.float32)
    if future.ndim != 2:
        raise ValueError(f"Expected [samples, horizon], got {future.shape}.")
    if future.shape[1] < 4:
        raise ValueError("The 12-property semantic protocol requires horizon >= 4.")

    quarter_levels = build_future_summary(
        future,
        bins=4,
        spectral_bands=1,
    ).values[:, :4]
    time = np.linspace(-1.0, 1.0, future.shape[1], dtype=np.float32)
    centered = future - future.mean(axis=1, keepdims=True)
    slope = np.einsum("nt,t->n", centered, time) / float(np.dot(time, time))
    first_diff = np.diff(future, axis=1)
    second_diff = np.diff(future, n=2, axis=1)
    scalar = np.stack(
        [
            future[:, -1] - future[:, 0],
            slope,
            np.abs(first_diff).mean(axis=1),
            first_diff.std(axis=1),
            np.sqrt(np.mean(np.square(second_diff, dtype=np.float64), axis=1)),
        ],
        axis=1,
    ).astype(np.float32)
    values = np.concatenate(
        [quarter_levels, scalar, _normalized_spectral_energy(future)],
        axis=1,
    ).astype(np.float32)
    if values.shape[1] != len(SEMANTIC_NAMES):
        raise RuntimeError(f"Semantic shape {values.shape} does not match the registered protocol.")
    return values, SEMANTIC_NAMES


def build_semantic_targets(
    future_normalized: np.ndarray,
    channel_names: tuple[str, ...] | None = None,
) -> SemanticTargets:
    """构造预注册的连续未来语义，并支持单变量和多变量窗口。"""
    future = np.asarray(future_normalized, dtype=np.float32)
    if future.ndim == 2:
        values, names = _single_channel_targets(future)
    elif future.ndim == 3:
        n_channels = future.shape[-1]
        if channel_names is None:
            channel_names = tuple(f"var_{index}" for index in range(n_channels))
        if len(channel_names) != n_channels:
            raise ValueError(f"Got {len(channel_names)} channel names for {n_channels} channels.")
        channel_values = []
        channel_target_names = []
        for index, channel_name in enumerate(channel_names):
            values_for_channel, names_for_channel = _single_channel_targets(future[:, :, index])
            channel_values.append(values_for_channel)
            channel_target_names.extend(f"{channel_name}:{name}" for name in names_for_channel)
        values = np.concatenate(channel_values, axis=1).astype(np.float32)
        names = tuple(channel_target_names)
    else:
        raise ValueError(f"Expected [samples, horizon] or [samples, horizon, channels], got {future.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Semantic targets contain non-finite values.")
    return SemanticTargets(values=values, names=names)
