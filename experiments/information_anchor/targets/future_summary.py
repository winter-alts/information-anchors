from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FutureSummary:
    values: np.ndarray
    feature_names: tuple[str, ...]


def build_raw_future_target(
    future_normalized: np.ndarray,
    *,
    channel_names: tuple[str, ...] | None = None,
) -> FutureSummary:
    """Flatten the observed future trajectory without handcrafted summaries."""
    future = np.asarray(future_normalized, dtype=np.float32)
    if future.ndim == 2:
        values = future
        names = tuple(f"step_{step:03d}" for step in range(future.shape[1]))
    elif future.ndim == 3:
        n_samples, horizon, n_channels = future.shape
        if channel_names is None:
            channel_names = tuple(f"var_{index}" for index in range(n_channels))
        if len(channel_names) != n_channels:
            raise ValueError(f"Got {len(channel_names)} names for {n_channels} future channels.")
        values = future.reshape(n_samples, horizon * n_channels)
        names = tuple(
            f"step_{step:03d}:{channel}"
            for step in range(horizon)
            for channel in channel_names
        )
    else:
        raise ValueError(
            f"Expected [samples, horizon] or [samples, horizon, channels], got {future.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Raw future target contains non-finite values.")
    return FutureSummary(values=values.astype(np.float32), feature_names=names)


def _relative_bin_means(future: np.ndarray, bins: int) -> np.ndarray:
    n_samples, horizon = future.shape
    if not 1 <= bins <= horizon:
        raise ValueError(f"bins={bins} must be in [1, horizon={horizon}]")
    edges = np.rint(np.linspace(0, horizon, bins + 1)).astype(np.int64)
    output = np.empty((n_samples, bins), dtype=np.float32)
    for index in range(bins):
        start, stop = int(edges[index]), int(edges[index + 1])
        if stop <= start:
            stop = min(horizon, start + 1)
        output[:, index] = future[:, start:stop].mean(axis=1)
    return output


def _spectral_band_energy(future: np.ndarray, bands: int) -> np.ndarray:
    if bands < 1:
        raise ValueError("spectral_bands must be positive")
    centered = future - future.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum = spectrum[:, 1:]
    if spectrum.shape[1] == 0:
        return np.zeros((future.shape[0], bands), dtype=np.float32)
    chunks = np.array_split(np.arange(spectrum.shape[1]), bands)
    energies = []
    for indices in chunks:
        if len(indices) == 0:
            energies.append(np.zeros(future.shape[0], dtype=np.float64))
        else:
            energies.append(spectrum[:, indices].mean(axis=1))
    return np.log1p(np.stack(energies, axis=1)).astype(np.float32)


def _build_univariate_future_summary(
    future: np.ndarray,
    *,
    bins: int,
    spectral_bands: int,
) -> FutureSummary:
    time = np.linspace(-1.0, 1.0, future.shape[1], dtype=np.float32)
    time_norm = float(np.dot(time, time))

    bin_means = _relative_bin_means(future, bins)
    future_mean = future.mean(axis=1)
    future_std = future.std(axis=1)
    slope = np.einsum("nt,t->n", future - future_mean[:, None], time) / time_norm
    last_value = future[:, -1]
    total_change = future[:, -1] - future[:, 0]
    diff_std = np.diff(future, axis=1).std(axis=1)
    spectral = _spectral_band_energy(future, spectral_bands)

    scalar = np.stack(
        [future_mean, future_std, slope, last_value, total_change, diff_std],
        axis=1,
    ).astype(np.float32)
    values = np.concatenate([bin_means, scalar, spectral], axis=1)
    names = (
        tuple(f"relative_bin_mean_{index:02d}" for index in range(bins))
        + (
            "future_mean",
            "future_std",
            "future_slope",
            "future_last",
            "future_total_change",
            "future_diff_std",
        )
        + tuple(f"spectral_band_energy_{index:02d}" for index in range(spectral_bands))
    )
    return FutureSummary(values=values.astype(np.float32), feature_names=names)


def build_future_summary(
    future_normalized: np.ndarray,
    *,
    bins: int,
    spectral_bands: int,
    channel_names: tuple[str, ...] | None = None,
) -> FutureSummary:
    future = np.asarray(future_normalized, dtype=np.float32)
    if future.ndim == 2:
        summary = _build_univariate_future_summary(
            future, bins=bins, spectral_bands=spectral_bands
        )
        if not np.isfinite(summary.values).all():
            raise ValueError("Future summary contains non-finite values.")
        return summary
    if future.ndim != 3:
        raise ValueError(f"Expected [samples, horizon] or [samples, horizon, channels], got {future.shape}")

    n_samples, _horizon, n_channels = future.shape
    if channel_names is None:
        channel_names = tuple(f"var_{index}" for index in range(n_channels))
    if len(channel_names) != n_channels:
        raise ValueError(f"Got {len(channel_names)} names for {n_channels} future channels.")

    channel_values = []
    feature_names: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        summary = _build_univariate_future_summary(
            future[:, :, channel_index],
            bins=bins,
            spectral_bands=spectral_bands,
        )
        channel_values.append(summary.values)
        feature_names.extend(f"{channel_name}:{name}" for name in summary.feature_names)

    values = np.concatenate(channel_values, axis=1).reshape(n_samples, -1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Future summary contains non-finite values.")
    return FutureSummary(values=values, feature_names=tuple(feature_names))
