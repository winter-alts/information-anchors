from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class RidgeProbeResult:
    alpha: np.ndarray
    validation_r2: np.ndarray
    test_r2: np.ndarray
    test_mae_standardized: np.ndarray
    validation_prediction_standardized: np.ndarray
    test_prediction_standardized: np.ndarray
    test_target_standardized: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray


def r2_per_target(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """计算每个目标的 R²；测试方差退化时返回 NaN。"""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.square(target - prediction).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    valid = total > 1e-8
    result[valid] = 1.0 - residual[valid] / total[valid]
    return result.astype(np.float32)


def fit_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alphas: tuple[float, ...],
) -> RidgeProbeResult:
    """Select alpha per preregistered target on validation, then test once."""
    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    validation_y = np.asarray(validation_y, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.float32)
    mean = train_y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = train_y.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    train_z = (train_y - mean) / scale
    validation_z = (validation_y - mean) / scale
    test_z = (test_y - mean) / scale

    n_targets = train_z.shape[1]
    best_alpha = np.full(n_targets, np.nan, dtype=np.float32)
    best_loss = np.full(n_targets, np.inf, dtype=np.float64)
    validation_prediction = np.empty_like(validation_z)
    test_prediction = np.empty_like(test_z)
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_z)
        candidate_validation = model.predict(validation_x)
        candidate_test = model.predict(test_x)
        loss = np.square(validation_z - candidate_validation).mean(axis=0)
        improved = loss < best_loss
        best_loss[improved] = loss[improved]
        best_alpha[improved] = float(alpha)
        validation_prediction[:, improved] = candidate_validation[:, improved]
        test_prediction[:, improved] = candidate_test[:, improved]
    if not np.isfinite(best_alpha).all():
        raise RuntimeError("No Ridge probe was fitted.")

    return RidgeProbeResult(
        alpha=best_alpha,
        validation_r2=r2_per_target(validation_z, validation_prediction),
        test_r2=r2_per_target(test_z, test_prediction),
        test_mae_standardized=np.abs(test_z - test_prediction).mean(axis=0).astype(np.float32),
        validation_prediction_standardized=validation_prediction.astype(np.float32),
        test_prediction_standardized=test_prediction.astype(np.float32),
        test_target_standardized=test_z.astype(np.float32),
        target_mean=mean.reshape(-1),
        target_scale=scale.reshape(-1),
    )
