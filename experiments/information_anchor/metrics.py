"""可复用的 TSLib 风格预测指标。

这里把训练段拟合的全局 StandardScaler 与逐样本聚合分开，避免不同实验
脚本因为数组维度或缩放时机不同而产生不可比较的 MSE/MAE。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainStandardScaler:
    """保存训练段逐变量均值和标准差。"""

    mean: np.ndarray
    scale: np.ndarray


def _as_forecast_array(values: np.ndarray) -> np.ndarray:
    """将预测数组统一成 [sample, horizon, variable]。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(
            f"Expected [sample, horizon] or [sample, horizon, variable], got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Forecast array contains non-finite values.")
    return array


def fit_train_standard_scaler(values: np.ndarray, train_end: int) -> TrainStandardScaler:
    """只用数据前 train_end 个时间点拟合 TSLib 的逐变量 scaler。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected [time, variable] values, got {array.shape}.")
    if not 1 <= train_end <= len(array):
        raise ValueError(f"train_end must be in [1, {len(array)}], got {train_end}.")
    train = array[:train_end]
    if not np.isfinite(train).all():
        raise ValueError("Training values contain non-finite entries.")
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-12)
    return TrainStandardScaler(mean=mean.astype(np.float64), scale=scale.astype(np.float64))


def standardize_forecasts(values: np.ndarray, scaler: TrainStandardScaler) -> np.ndarray:
    """用训练段 scaler 标准化预测或真实未来数组。"""
    array = _as_forecast_array(values)
    mean = np.asarray(scaler.mean, dtype=np.float64).reshape(1, 1, -1)
    scale = np.asarray(scaler.scale, dtype=np.float64).reshape(1, 1, -1)
    if array.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Scaler has {mean.shape[-1]} variables but values have {array.shape[-1]}."
        )
    return (array - mean) / scale


def per_origin_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    scaler: TrainStandardScaler,
    *,
    target_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回每个 origin 的标准化 MSE/MAE。

    ``target_index=None`` 对 horizon 和全部变量共同平均；显式传入变量索引时，
    只对该目标变量的 horizon 平均。调用方再对 origins 汇总。
    """
    prediction_z = standardize_forecasts(prediction, scaler)
    truth_z = standardize_forecasts(truth, scaler)
    if prediction_z.shape != truth_z.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction_z.shape} vs {truth_z.shape}.")
    error = prediction_z - truth_z
    if target_index is not None:
        try:
            error = error[:, :, [target_index]]
        except IndexError as exc:
            raise ValueError(
                f"target_index={target_index} is invalid for {error.shape[-1]} variables."
            ) from exc
    # TSLib 的多变量指标对 horizon 和 variable 共同求平均，origin 只在最后汇总。
    return np.mean(np.square(error), axis=(1, 2)), np.mean(np.abs(error), axis=(1, 2))


def forecast_change_metrics(
    prediction: np.ndarray,
    baseline: np.ndarray,
    scaler: TrainStandardScaler,
    *,
    target_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算干预预测相对 baseline 的标准化变化量，可限定目标变量。"""
    prediction_z = standardize_forecasts(prediction, scaler)
    baseline_z = standardize_forecasts(baseline, scaler)
    if prediction_z.shape != baseline_z.shape:
        raise ValueError(f"Prediction/baseline shape mismatch: {prediction_z.shape} vs {baseline_z.shape}.")
    difference = prediction_z - baseline_z
    if target_index is not None:
        try:
            difference = difference[:, :, [target_index]]
        except IndexError as exc:
            raise ValueError(
                f"target_index={target_index} is invalid for {difference.shape[-1]} variables."
            ) from exc
    return np.mean(np.square(difference), axis=(1, 2)), np.mean(np.abs(difference), axis=(1, 2))


def forecast_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    scaler: TrainStandardScaler,
    history_scale: np.ndarray | None = None,
) -> dict[str, float]:
    """同时返回 TSLib 主指标和窗口标准化诊断指标。"""
    prediction_array = _as_forecast_array(prediction)
    truth_array = _as_forecast_array(truth)
    if prediction_array.shape != truth_array.shape:
        raise ValueError(
            f"Prediction/truth shape mismatch: {prediction_array.shape} vs {truth_array.shape}."
        )
    tslib_mse, tslib_mae = per_origin_metrics(prediction_array, truth_array, scaler)
    residual = prediction_array - truth_array
    result = {
        "raw_mse": float(np.square(residual).mean()),
        "raw_mae": float(np.abs(residual).mean()),
        "tslib_mse": float(tslib_mse.mean()),
        "tslib_mae": float(tslib_mae.mean()),
        # 保留旧字段名，数值改为 TSLib 主口径，避免下游汇总脚本误读。
        "normalized_mse": float(tslib_mse.mean()),
        "normalized_mae": float(tslib_mae.mean()),
    }
    if history_scale is not None:
        # 窗口尺度只作为诊断，不能替代训练段尺度。
        scale = np.asarray(history_scale, dtype=np.float64)
        if scale.ndim == 1:
            scale = scale[:, None, None]
        elif scale.ndim == 2:
            scale = scale[:, None, :]
        else:
            raise ValueError(f"History scale must be [sample] or [sample, variable], got {scale.shape}.")
        if scale.shape[0] != residual.shape[0] or scale.shape[-1] != residual.shape[-1]:
            raise ValueError(f"History scale shape {scale.shape} does not match residual {residual.shape}.")
        history_normalized = residual / np.maximum(scale, 1e-12)
        result["history_window_normalized_mse"] = float(np.square(history_normalized).mean())
        result["history_window_normalized_mae"] = float(np.abs(history_normalized).mean())
    return result


def aggregate_metrics(mse: np.ndarray, mae: np.ndarray) -> dict[str, float]:
    """把逐 origin 指标汇总为论文表格使用的标量。"""
    mse_array = np.asarray(mse, dtype=np.float64)
    mae_array = np.asarray(mae, dtype=np.float64)
    if mse_array.ndim != 1 or mae_array.ndim != 1 or mse_array.shape != mae_array.shape:
        raise ValueError("Per-origin MSE and MAE must be one-dimensional arrays of equal length.")
    return {"mse": float(mse_array.mean()), "mae": float(mae_array.mean())}
