"""MI 的 raw-recency 与残差化未来信息控制实验。

这些控制不把时间位置本身当作可解释变量，而是用同一 patch 的原始统计量
解释未来摘要，再在交叉拟合残差上计算 hidden MI。这样可以区分“patch 本身
携带的可预测统计量”和“模型表示额外编码的未来信息”。后者是条件 MI 的
可复现线性残差代理，不冒充严格的非线性条件 MI。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.information_anchor.estimators.gcmi import (
    gaussian_copula_transform,
    gcmi_from_gaussianized,
)
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter, ksg_mi
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import benjamini_hochberg, robust_null_score
from experiments.information_anchor.estimators.projection import fit_projection


@dataclass(frozen=True)
class ControlProfile:
    """保存一个 profile 及其时间置换 null。"""

    observed: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    q_values: np.ndarray
    null_values: np.ndarray


def history_patch_statistics(
    histories: np.ndarray,
    *,
    patch_len: int,
    patch_stride: int,
    num_patches: int,
) -> np.ndarray:
    """为每个历史 patch 计算跨变量的均值、波动、末值和局部斜率。"""
    values = np.asarray(histories, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected [sample, history, variable], got {values.shape}.")
    if patch_len < 1 or patch_stride < 1 or num_patches < 1:
        raise ValueError("patch_len, patch_stride, and num_patches must be positive.")
    n_samples, history_length, n_channels = values.shape
    time = np.linspace(-1.0, 1.0, patch_len, dtype=np.float32)
    time_norm = max(float(np.dot(time, time)), 1e-12)
    features = []
    for patch_index in range(num_patches):
        start = patch_index * patch_stride
        stop = start + patch_len
        if start < 0 or stop > history_length:
            raise ValueError(
                f"Patch {patch_index} spans [{start}, {stop}) outside history={history_length}."
            )
        patch = values[:, start:stop, :]
        mean = patch.mean(axis=1)
        std = patch.std(axis=1)
        last = patch[:, -1, :]
        centered = patch - mean[:, None, :]
        slope = np.einsum("ntc,t->nc", centered, time) / time_norm
        diff_std = np.diff(patch, axis=1).std(axis=1) if patch_len > 1 else np.zeros_like(mean)
        # 每个变量保留相同的统计量，最后交给 PCA 压缩到 MI 维度。
        features.append(np.concatenate([mean, std, last, slope, diff_std], axis=1))
    output = np.stack(features, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("History patch statistics contain non-finite values.")
    return output


def cross_fitted_residuals(
    target: np.ndarray,
    covariates: np.ndarray,
    *,
    folds: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    """用时间连续 folds 的 Ridge 交叉拟合未来摘要残差。"""
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 2 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"Residual inputs must share samples, got x={x.shape}, y={y.shape}.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Residual inputs contain non-finite values.")
    folds = max(2, min(int(folds), len(y)))
    residual = np.empty_like(y)
    for test_indices in np.array_split(np.arange(len(y)), folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_indices] = False
        scaler = StandardScaler().fit(x[train_mask])
        model = Ridge(alpha=float(alpha)).fit(scaler.transform(x[train_mask]), y[train_mask])
        prediction = model.predict(scaler.transform(x[test_indices]))
        residual[test_indices] = y[test_indices] - prediction
    return residual.astype(np.float32)


def _profile_from_null(
    observed: np.ndarray, null_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐 profile 单元用相同的 circular-shift null 校准 z 和 p。"""
    observed = np.asarray(observed, dtype=np.float64)
    null_values = np.asarray(null_values, dtype=np.float64)
    z_scores = np.empty_like(observed, dtype=np.float32)
    p_values = np.empty_like(observed, dtype=np.float32)
    for index in np.ndindex(observed.shape):
        z_scores[index], p_values[index] = robust_null_score(
            observed[index], null_values[(slice(None),) + index]
        )
    q_values = benjamini_hochberg(p_values.reshape(-1)).reshape(p_values.shape).astype(np.float32)
    return z_scores, p_values, q_values


def _estimate_one(
    x: np.ndarray,
    future: np.ndarray,
    shifts: np.ndarray,
    *,
    estimator: str,
    k: int,
    jitter: float,
    seed: int,
    device: str,
    shift_batch_size: int,
) -> tuple[float, np.ndarray]:
    """对一个 patch 计算观测 MI 和 circular-shift null。"""
    if estimator in {"ksg_cpu", "ksg_gpu"}:
        x_ready = add_deterministic_jitter(x, jitter, seed)
        future_ready = add_deterministic_jitter(future, jitter, seed + 1)
        if estimator == "ksg_gpu":
            gpu_estimator = TorchKSG(
                future_ready,
                k=k,
                device=device,
                shift_batch_size=shift_batch_size,
            )
            return gpu_estimator.estimate_observed_and_shifts(x_ready, shifts)
        observed = ksg_mi(x_ready, future_ready, k=k)
        null_values = np.asarray(
            [ksg_mi(x_ready, np.roll(future_ready, int(shift), axis=0), k=k) for shift in shifts],
            dtype=np.float64,
        )
        return observed, null_values
    if estimator != "gcmi":
        raise ValueError(f"Unsupported control estimator={estimator!r}.")
    x_ready = gaussian_copula_transform(x)
    future_ready = gaussian_copula_transform(future)
    observed = gcmi_from_gaussianized(x_ready, future_ready)
    null_values = np.asarray(
        [gcmi_from_gaussianized(x_ready, np.roll(future_ready, int(shift), axis=0)) for shift in shifts],
        dtype=np.float64,
    )
    return observed, null_values


def raw_patch_profile(
    patch_features: np.ndarray,
    future_projected: np.ndarray,
    shifts: np.ndarray,
    *,
    estimator: str,
    k: int,
    pca_dim: int,
    jitter: float,
    seed: int,
    device: str,
    shift_batch_size: int,
) -> ControlProfile:
    """计算每个 raw patch statistics 与未来摘要的 MI profile。"""
    features = np.asarray(patch_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected [sample, patch, feature], got {features.shape}.")
    observed = np.empty(features.shape[1], dtype=np.float32)
    null_values = np.empty((len(shifts), features.shape[1]), dtype=np.float32)
    for patch_index in range(features.shape[1]):
        _, projected = fit_projection(
            features[:, patch_index],
            pca_dim,
            seed=seed + patch_index,
            standardize_features=True,
        )
        observed[patch_index], null_values[:, patch_index] = _estimate_one(
            projected,
            future_projected,
            shifts,
            estimator=estimator,
            k=k,
            jitter=jitter,
            seed=seed + 10_000 + patch_index,
            device=device,
            shift_batch_size=shift_batch_size,
        )
    z_scores, p_values, q_values = _profile_from_null(observed, null_values)
    return ControlProfile(observed, z_scores, p_values, q_values, null_values)


def residual_hidden_profile(
    hidden: np.ndarray,
    projection_paths: list[str],
    patch_features: np.ndarray,
    future_projected: np.ndarray,
    shifts: np.ndarray,
    *,
    estimator: str,
    k: int,
    hidden_pca_dim: int,
    residual_folds: int,
    residual_alpha: float,
    jitter: float,
    seed: int,
    device: str,
    shift_batch_size: int,
    layer_indices: np.ndarray | None = None,
) -> ControlProfile:
    """逐 patch 残差化未来摘要，再计算 hidden 的 residual-future MI。"""
    from experiments.information_anchor.estimators.projection import load_projection

    hidden_array = np.asarray(hidden)
    if hidden_array.ndim != 4:
        raise ValueError(f"Expected hidden [sample, layer, patch, dim], got {hidden_array.shape}.")
    n_samples, n_layers, n_patches, _ = hidden_array.shape
    if patch_features.shape[:2] != (n_samples, n_patches):
        raise ValueError("Patch features and hidden cache have incompatible sample/patch axes.")
    if len(projection_paths) != n_layers:
        raise ValueError("Need one hidden projection file per selected layer.")
    if layer_indices is None:
        layer_indices = np.arange(n_layers, dtype=np.int64)
    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    # residual target 只依赖 patch 的原始统计量，与 layer 无关；先按 patch 计算一次，
    # 避免高维多变量数据在每一层重复拟合同一个 Ridge residual。
    residual_targets = [
        cross_fitted_residuals(
            future_projected,
            patch_features[:, patch_index],
            folds=residual_folds,
            alpha=residual_alpha,
        )
        for patch_index in range(n_patches)
    ]
    observed = np.empty((len(layer_indices), n_patches), dtype=np.float32)
    null_values = np.empty((len(shifts), len(layer_indices), n_patches), dtype=np.float32)
    for output_layer, layer_index in enumerate(layer_indices):
        layer_projection = load_projection(projection_paths[int(layer_index)])
        layer_values = np.asarray(hidden_array[:, int(layer_index)], dtype=np.float32)
        flattened = layer_values.reshape(n_samples * n_patches, -1)
        projected = layer_projection.transform(flattened).reshape(n_samples, n_patches, -1)
        for patch_index in range(n_patches):
            observed[output_layer, patch_index], null_values[:, output_layer, patch_index] = _estimate_one(
                projected[:, patch_index],
                residual_targets[patch_index],
                shifts,
                estimator=estimator,
                k=k,
                jitter=jitter,
                seed=seed + 20_000 + int(layer_index) * n_patches + patch_index,
                device=device,
                shift_batch_size=shift_batch_size,
            )
    z_scores, p_values, q_values = _profile_from_null(observed, null_values)
    return ControlProfile(observed, z_scores, p_values, q_values, null_values)
