from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.information_anchor.config import DataConfig


@dataclass(frozen=True)
class WindowBatch:
    origins: np.ndarray
    history_raw: np.ndarray
    future_raw: np.ndarray
    history_normalized: np.ndarray
    future_normalized: np.ndarray
    history_mean: np.ndarray
    history_scale: np.ndarray
    columns: tuple[str, ...]


def load_benchmark_frame(config: DataConfig, offline: bool) -> pd.DataFrame:
    if config.local_path:
        local_path = Path(config.local_path)
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        frame = pd.read_csv(local_path)
    else:
        if offline:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import Dataset, load_dataset

        try:
            dataset = load_dataset(config.source, name=config.dataset)
            split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
            frame = dataset[split_name].to_pandas()
        except ConnectionError:
            if not offline:
                raise
            cache_root = Path(
                os.environ.get(
                    "HF_DATASETS_CACHE",
                    Path.home() / ".cache" / "huggingface" / "datasets",
                )
            )
            source_key = config.source.lower().replace("/", "___")
            candidates = sorted(
                (cache_root / source_key / config.dataset).glob(
                    "*/*/*-train.arrow"
                )
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    "Offline dataset loading failed and the Arrow cache fallback "
                    f"found {len(candidates)} candidates for "
                    f"source={config.source!r}, dataset={config.dataset!r}: "
                    f"{[str(path) for path in candidates]}"
                )
            frame = Dataset.from_file(str(candidates[0])).to_pandas()

    required = {"date", config.target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark {config.dataset} data is missing columns: {sorted(missing)}")
    if config.target_columns:
        missing_columns = set(config.target_columns).difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Benchmark {config.dataset} data is missing target_columns: {sorted(missing_columns)}"
            )
    if len(frame) < config.test_end:
        raise ValueError(
            f"{config.dataset} has {len(frame)} rows, shorter than test_end={config.test_end}."
        )
    # The Weather benchmark stores missing measurements as large negative
    # sentinels (notably -9999 in OT). They must not enter normalization, MI,
    # or forecast windows; interpolate each numeric series chronologically.
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    if len(numeric_columns):
        numeric = frame.loc[:, numeric_columns].replace([np.inf, -np.inf, -9999.0, -999.0], np.nan)
        numeric = numeric.interpolate(limit_direction="both")
        frame = frame.copy()
        frame.loc[:, numeric_columns] = numeric
    return frame


def _origin_bounds(config: DataConfig) -> tuple[int, int]:
    seq_len = config.origin_seq_len or config.seq_len
    if config.split == "discovery":
        return seq_len, config.discovery_end
    if config.split == "probe":
        start = config.discovery_end + seq_len if config.strict_guard else config.discovery_end
        return start, config.train_end
    if config.split == "validation":
        start = config.train_end + seq_len if config.strict_guard else config.train_end
        return start, config.validation_end
    if config.split == "test":
        start = config.validation_end + seq_len if config.strict_guard else config.validation_end
        return start, config.test_end
    raise ValueError(f"Unsupported analysis split: {config.split}")


def build_forecast_origins(config: DataConfig) -> np.ndarray:
    first_origin, segment_end = _origin_bounds(config)
    origin_pred_len = config.origin_pred_len or config.pred_len
    last_origin = segment_end - origin_pred_len
    if first_origin > last_origin:
        raise ValueError(
            "No valid forecast origins. Reduce seq_len/pred_len or change split bounds."
        )
    origins = np.arange(
        first_origin,
        last_origin + 1,
        config.origin_stride,
        dtype=np.int64,
    )
    if len(origins) > config.max_origins:
        selected = np.linspace(0, len(origins) - 1, config.max_origins)
        selected = np.unique(np.rint(selected).astype(np.int64))
        origins = origins[selected]
    return origins


def select_value_columns(frame: pd.DataFrame, config: DataConfig) -> tuple[str, ...]:
    if config.target_columns:
        return tuple(config.target_columns)
    if config.features == "S":
        return (config.target,)
    if config.features == "M":
        columns = tuple(column for column in frame.columns if column != "date")
        if config.target not in columns:
            raise ValueError(f"target={config.target!r} is not part of multivariate columns.")
        return columns
    raise ValueError(f"Unsupported features={config.features!r}; expected S, M, or custom.")


def make_windows(
    values: np.ndarray,
    origins: np.ndarray,
    seq_len: int,
    pred_len: int,
    eps: float = 1e-6,
    columns: tuple[str, ...] | None = None,
) -> WindowBatch:
    values = np.asarray(values, dtype=np.float32)
    single_channel = values.ndim == 1
    if single_channel:
        values_2d = values[:, None]
    elif values.ndim == 2:
        values_2d = values
    else:
        raise ValueError(f"Expected values with shape [time] or [time, channels], got {values.shape}.")
    if columns is None:
        columns = tuple(f"var_{index}" for index in range(values_2d.shape[1]))
    if len(columns) != values_2d.shape[1]:
        raise ValueError(f"Got {len(columns)} column names for {values_2d.shape[1]} channels.")

    history = np.stack([values_2d[tau - seq_len : tau, :] for tau in origins])
    future = np.stack([values_2d[tau : tau + pred_len, :] for tau in origins])
    mean = history.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = history.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(eps))
    history_normalized = (history - mean) / scale
    future_normalized = (future - mean) / scale

    if single_channel:
        history = history[:, :, 0]
        future = future[:, :, 0]
        history_normalized = history_normalized[:, :, 0]
        future_normalized = future_normalized[:, :, 0]
        mean = mean[:, 0, 0]
        scale = scale[:, 0, 0]
    else:
        mean = mean[:, 0, :]
        scale = scale[:, 0, :]

    return WindowBatch(
        origins=np.asarray(origins, dtype=np.int64),
        history_raw=history.astype(np.float32),
        future_raw=future.astype(np.float32),
        history_normalized=history_normalized.astype(np.float32),
        future_normalized=future_normalized.astype(np.float32),
        history_mean=mean.astype(np.float32),
        history_scale=scale.astype(np.float32),
        columns=tuple(columns),
    )


def train_split_normalize_future(
    future_raw: np.ndarray,
    reference_values: np.ndarray,
    train_end: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize future values with channel statistics fitted on the train segment only."""
    future = np.asarray(future_raw, dtype=np.float32)
    reference = np.asarray(reference_values, dtype=np.float32)
    future_single_channel = future.ndim == 2
    reference_single_channel = reference.ndim == 1
    if future_single_channel:
        future = future[:, :, None]
    elif future.ndim != 3:
        raise ValueError(
            f"Expected future_raw with shape [samples, horizon] or [samples, horizon, channels], got {future.shape}."
        )
    if reference_single_channel:
        reference = reference[:, None]
    elif reference.ndim != 2:
        raise ValueError(
            f"Expected reference_values with shape [time] or [time, channels], got {reference.shape}."
        )
    if future.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"Future has {future.shape[-1]} channels but reference has {reference.shape[-1]}."
        )
    if not 1 <= train_end <= len(reference):
        raise ValueError(f"train_end={train_end} must be in [1, {len(reference)}].")

    train = reference[:train_end]
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(eps))
    normalized = ((future - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)
    if future_single_channel and reference_single_channel:
        return normalized[:, :, 0]
    return normalized


def robust_history_normalize_future(
    windows: WindowBatch,
    reference_values: np.ndarray,
    train_end: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Center on each history and floor its scale by the train-only channel scale."""
    future = np.asarray(windows.future_raw, dtype=np.float32)
    reference = np.asarray(reference_values, dtype=np.float32)
    future_single_channel = future.ndim == 2
    reference_single_channel = reference.ndim == 1
    if future_single_channel:
        future = future[:, :, None]
    elif future.ndim != 3:
        raise ValueError(f"Unsupported future shape {future.shape}.")
    if reference_single_channel:
        reference = reference[:, None]
    elif reference.ndim != 2:
        raise ValueError(f"Unsupported reference shape {reference.shape}.")
    if future.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"Future has {future.shape[-1]} channels but reference has {reference.shape[-1]}."
        )
    if not 1 <= train_end <= len(reference):
        raise ValueError(f"train_end={train_end} must be in [1, {len(reference)}].")

    train_scale = reference[:train_end].std(axis=0, dtype=np.float64).astype(np.float32)
    train_scale = np.maximum(train_scale, np.float32(eps))
    history_mean = np.asarray(windows.history_mean, dtype=np.float32)
    history_scale = np.asarray(windows.history_scale, dtype=np.float32)
    if history_mean.ndim == 1:
        history_mean = history_mean[:, None]
        history_scale = history_scale[:, None]
    scale = np.maximum(history_scale, train_scale[None, :])
    normalized = (
        (future - history_mean[:, None, :]) / scale[:, None, :]
    ).astype(np.float32)
    if future_single_channel and reference_single_channel:
        return normalized[:, :, 0]
    return normalized


def analysis_future_values(
    windows: WindowBatch,
    reference_values: np.ndarray,
    *,
    train_end: int,
    normalization: str,
) -> np.ndarray:
    """Return the future view used by MI targets and semantic probes."""
    if normalization == "history_window":
        return windows.future_normalized
    if normalization == "train_split":
        return train_split_normalize_future(
            windows.future_raw,
            reference_values,
            train_end,
        )
    if normalization == "robust_history_window":
        return robust_history_normalize_future(
            windows,
            reference_values,
            train_end,
        )
    raise ValueError(f"Unsupported future-target normalization={normalization!r}.")


def origins_hash(origins: np.ndarray) -> str:
    payload = np.asarray(origins, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def window_spans(origins: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    return np.stack([origins - seq_len, origins + pred_len], axis=1)
