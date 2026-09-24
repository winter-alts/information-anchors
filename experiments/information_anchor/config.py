from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _coerce_columns(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    patch_len: int
    device: str = "cuda:0"
    cache_dtype: str = "float16"
    revision: str = ""
    # 多变量缓存默认保持历史拼接协议，高维数据可显式选择 mean 或 mean_std。
    channel_aggregation: str = "concat_same_time_patch"


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    source: str
    target: str
    seq_len: int
    pred_len: int
    split: str
    origin_stride: int
    max_origins: int
    strict_guard: bool
    train_end: int
    validation_end: int
    test_end: int
    discovery_end: int
    local_path: str = ""
    origin_seq_len: int = 0
    origin_pred_len: int = 0
    features: str = "S"
    target_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FutureTargetConfig:
    bins: int = 16
    spectral_bands: int = 4
    pca_dim: int = 8
    normalization: str = "history_window"
    # MI 目标可按指定变量计算，也可保留全部输入变量的未来摘要。
    scope: str = "all"


@dataclass(frozen=True)
class MIConfig:
    hidden_pca_dim: int = 8
    k: int = 5
    null_permutations: int = 100
    null_mode: str = "sampled"
    seed: int = 2021
    jitter: float = 1e-8
    estimator: str = "ksg_gpu"
    shift_batch_size: int = 8


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    output_root: str
    cache_root: str
    offline: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    future_target: FutureTargetConfig
    mi: MIConfig
    runtime: RuntimeConfig
    protocol_version: str = "information-anchor-v5"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(raw: dict[str, Any]) -> ExperimentConfig:
    data_raw = dict(raw["data"])
    data_raw["target_columns"] = _coerce_columns(data_raw.get("target_columns", ()))
    data_raw.setdefault("features", "S")
    config = ExperimentConfig(
        schema_version=int(raw["schema_version"]),
        experiment_name=str(raw["experiment_name"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**data_raw),
        future_target=FutureTargetConfig(**raw["future_target"]),
        mi=MIConfig(**raw["mi"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        protocol_version=str(raw.get("protocol_version", "information-anchor-v5")),
    )
    validate_config(config)
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _construct_config(json.load(handle))


def validate_config(config: ExperimentConfig) -> None:
    if config.schema_version != 1:
        raise ValueError(f"Unsupported schema_version={config.schema_version}")
    if not config.protocol_version.strip():
        raise ValueError("protocol_version must be non-empty.")
    supported_datasets = {
        "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather",
        "traffic", "electricity", "exchange_rate", "solar",
    }
    if config.data.dataset not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset={config.data.dataset!r}; expected one of {sorted(supported_datasets)}."
        )
    if config.model.name not in {
        "TimesFM",
        "TimesFM2.5",
        "TimesFM3",
        "Chronos2",
        "ChronosBolt",
        "Moirai2",
        "Toto2",
        "TTM",
    }:
        raise ValueError(
            "Supported information-anchor models are TimesFM, TimesFM2.5, TimesFM3, Chronos2, "
            "ChronosBolt, Moirai2, Toto2, and TTM."
        )
    if config.data.features not in {"S", "M", "custom"}:
        raise ValueError("features must be one of S, M, or custom for information-anchor experiments.")
    if config.data.features == "custom" and not config.data.target_columns:
        raise ValueError("features=custom requires target_columns.")
    if config.data.seq_len <= 0 or config.data.pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive.")
    if config.model.patch_len <= 0:
        raise ValueError("model.patch_len must be positive.")
    if config.model.name == "TimesFM3" and config.model.patch_len != 32:
        raise ValueError("TimesFM3 requires model.patch_len=32.")
    origin_seq_len = config.data.origin_seq_len or config.data.seq_len
    origin_pred_len = config.data.origin_pred_len or config.data.pred_len
    if origin_seq_len < config.data.seq_len:
        raise ValueError("origin_seq_len cannot be shorter than seq_len.")
    if origin_pred_len < config.data.pred_len:
        raise ValueError("origin_pred_len cannot be shorter than pred_len.")
    if config.data.origin_stride <= 0:
        raise ValueError("origin_stride must be positive.")
    if config.data.max_origins <= config.mi.k + 2:
        raise ValueError("max_origins is too small for the requested KSG k.")
    if not 1 <= config.future_target.bins <= config.data.pred_len:
        raise ValueError("future target bins must be between 1 and pred_len.")
    if config.future_target.normalization not in {
        "history_window",
        "robust_history_window",
        "train_split",
    }:
        raise ValueError(
            "future_target.normalization must be history_window, robust_history_window, or train_split."
        )
    if config.future_target.scope not in {"target", "all"}:
        raise ValueError("future_target.scope must be target or all.")
    if config.mi.estimator not in {"ksg_cpu", "ksg_gpu", "gcmi"}:
        raise ValueError("estimator must be ksg_cpu, ksg_gpu, or gcmi.")
    if config.future_target.pca_dim < 1 or config.mi.hidden_pca_dim < 1:
        raise ValueError("PCA dimensions must be positive.")
    if config.mi.shift_batch_size < 1:
        raise ValueError("shift_batch_size must be positive.")
    if config.mi.k < 1:
        raise ValueError("KSG k must be positive.")
    if config.mi.null_mode not in {"sampled", "all"}:
        raise ValueError("null_mode must be sampled or all.")
    if config.mi.null_permutations < 1:
        raise ValueError("At least one null permutation is required.")
    if config.model.cache_dtype not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be float16 or float32.")
    if config.model.name == "Toto2" and config.model.cache_dtype != "float32":
        raise ValueError(
            "Toto2 multivariate hidden must use float32 cache to avoid unit-scaled overflow."
        )
    if config.model.channel_aggregation not in {"concat_same_time_patch", "mean", "mean_std"}:
        raise ValueError(
            "model.channel_aggregation must be concat_same_time_patch, mean, or mean_std."
        )
    if config.model.name not in {
        "Chronos2",
        "ChronosBolt",
        "Moirai2",
        "Toto2",
        "TimesFM2.5",
        "TimesFM3",
        "TTM",
    } and config.model.channel_aggregation != "concat_same_time_patch":
        raise ValueError(
            "channel_aggregation overrides are supported for multivariate Chronos2, "
            "ChronosBolt, Moirai2, Toto2, TimesFM2.5, and TTM."
        )
    if not (
        0 < config.data.discovery_end <= config.data.train_end
        < config.data.validation_end < config.data.test_end
    ):
        raise ValueError("Invalid chronological split boundaries.")


def with_overrides(
    config: ExperimentConfig,
    *,
    device: str | None = None,
    max_origins: int | None = None,
    null_permutations: int | None = None,
    output_root: str | None = None,
    estimator: str | None = None,
    k: int | None = None,
    hidden_pca_dim: int | None = None,
    future_pca_dim: int | None = None,
    seed: int | None = None,
    null_mode: str | None = None,
    split: str | None = None,
    origin_stride: int | None = None,
    strict_guard: bool | None = None,
    future_scope: str | None = None,
) -> ExperimentConfig:
    """返回带运行时覆盖的配置，避免复制多套容易漂移的 JSON。"""
    raw = config.to_dict()
    if device is not None:
        raw["model"]["device"] = device
    if max_origins is not None:
        raw["data"]["max_origins"] = max_origins
    if null_permutations is not None:
        raw["mi"]["null_permutations"] = null_permutations
    if output_root is not None:
        raw["runtime"]["output_root"] = output_root
    if estimator is not None:
        raw["mi"]["estimator"] = estimator
    if k is not None:
        raw["mi"]["k"] = k
    if hidden_pca_dim is not None:
        raw["mi"]["hidden_pca_dim"] = hidden_pca_dim
    if future_pca_dim is not None:
        raw["future_target"]["pca_dim"] = future_pca_dim
    if seed is not None:
        raw["mi"]["seed"] = seed
    if null_mode is not None:
        raw["mi"]["null_mode"] = null_mode
    if split is not None:
        raw["data"]["split"] = split
    if origin_stride is not None:
        raw["data"]["origin_stride"] = origin_stride
    if strict_guard is not None:
        raw["data"]["strict_guard"] = strict_guard
    if future_scope is not None:
        raw["future_target"]["scope"] = future_scope
    return _construct_config(raw)
