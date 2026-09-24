from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.information_anchor.probes.linear import r2_per_target
from experiments.information_anchor.probes.semantic import SemanticTargets


TARGET_EXCLUSIVE_PROTOCOL = "target-exclusive-residual-v1.1"
UNIVARIATE_EXCLUSIVE_NAMES = (
    "first_difference_skewness",
    "first_difference_excess_kurtosis",
    "turning_point_rate",
    "detrended_zero_crossing_rate",
    "permutation_entropy_order3",
)
GLOBAL_EXCLUSIVE_NAMES = UNIVARIATE_EXCLUSIVE_NAMES + (
    "cross_channel_correlation_strength",
    "cross_channel_covariance_leading_eigenvalue_ratio",
)


@dataclass(frozen=True)
class ResidualizedTargets:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    alpha: np.ndarray
    validation_r2: np.ndarray
    test_r2: np.ndarray
    train_residual_variance_ratio: np.ndarray
    fold_train_sizes: tuple[int, ...]


@dataclass(frozen=True)
class BlockBootstrapResult:
    observed: np.ndarray
    estimates: np.ndarray
    ci95_lower: np.ndarray
    ci95_upper: np.ndarray
    p_greater: np.ndarray
    block_length: int


def _standardized_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=1, keepdims=True)
    variance = np.mean(np.square(centered, dtype=np.float64), axis=1)
    valid = variance > 1e-12
    skewness = np.zeros(len(values), dtype=np.float64)
    kurtosis = np.zeros(len(values), dtype=np.float64)
    skewness[valid] = (
        np.mean(np.power(centered[valid], 3), axis=1) / np.power(variance[valid], 1.5)
    )
    kurtosis[valid] = (
        np.mean(np.power(centered[valid], 4), axis=1) / np.square(variance[valid]) - 3.0
    )
    return skewness.astype(np.float32), kurtosis.astype(np.float32)


def _permutation_entropy(values: np.ndarray, order: int = 3) -> np.ndarray:
    if values.shape[1] < order:
        raise ValueError(f"Permutation entropy order={order} requires at least {order} steps.")
    windows = np.lib.stride_tricks.sliding_window_view(values, order, axis=1)
    patterns = np.argsort(windows, axis=-1, kind="stable")
    weights = np.power(order, np.arange(order, dtype=np.int64))
    codes = np.einsum("nwo,o->nw", patterns, weights)
    valid_codes = np.asarray(
        [sum(value * weight for value, weight in zip(item, weights)) for item in itertools.permutations(range(order))],
        dtype=np.int64,
    )
    probabilities = np.stack([(codes == code).mean(axis=1) for code in valid_codes], axis=1)
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities, where=probabilities > 0.0), 0.0),
        axis=1,
    )
    return (entropy / math.log(math.factorial(order))).astype(np.float32)


def _univariate_exclusive_targets(future: np.ndarray) -> np.ndarray:
    values = np.asarray(future, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Exclusive univariate targets require [samples, horizon>=4], got {values.shape}.")
    first_difference = np.diff(values, axis=1)
    skewness, kurtosis = _standardized_moments(first_difference)
    turning_points = np.mean(first_difference[:, :-1] * first_difference[:, 1:] < 0.0, axis=1)

    time = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    slope = np.einsum("nt,t->n", centered, time) / float(np.dot(time, time))
    detrended = centered - slope[:, None] * time[None, :]
    zero_crossings = np.mean(detrended[:, :-1] * detrended[:, 1:] < 0.0, axis=1)
    output = np.column_stack(
        [
            skewness,
            kurtosis,
            turning_points,
            zero_crossings,
            _permutation_entropy(values),
        ]
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Exclusive univariate targets contain non-finite values.")
    return output


def _cross_channel_targets(future: np.ndarray) -> np.ndarray:
    if future.shape[-1] < 2:
        raise ValueError("Cross-channel targets require at least two channels.")
    centered = future - future.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(np.square(centered, dtype=np.float64), axis=1))
    standardized = np.divide(
        centered,
        norms[:, None, :],
        out=np.zeros_like(centered, dtype=np.float64),
        where=norms[:, None, :] > 1e-12,
    )
    correlations = np.einsum("ntc,ntd->ncd", standardized, standardized)
    upper = np.triu_indices(future.shape[-1], k=1)
    correlation_strength = np.abs(correlations[:, upper[0], upper[1]]).mean(axis=1)

    covariance = np.einsum("ntc,ntd->ncd", centered, centered) / max(1, future.shape[1] - 1)
    leading = np.linalg.eigvalsh(covariance.astype(np.float64))[:, -1]
    trace = np.trace(covariance, axis1=1, axis2=2)
    leading_ratio = np.divide(
        leading,
        trace,
        out=np.zeros_like(leading),
        where=trace > 1e-12,
    )
    output = np.column_stack([correlation_strength, leading_ratio]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Exclusive cross-channel targets contain non-finite values.")
    return output


def build_target_exclusive_targets(
    future_normalized: np.ndarray,
    *,
    scope: str,
    target_index: int,
) -> SemanticTargets:
    future = np.asarray(future_normalized, dtype=np.float32)
    if future.ndim != 3:
        raise ValueError(f"Exclusive targets require [samples, horizon, channels], got {future.shape}.")
    if not 0 <= target_index < future.shape[-1]:
        raise ValueError(f"target_index={target_index} is outside {future.shape[-1]} channels.")
    if scope == "target":
        values = _univariate_exclusive_targets(future[:, :, target_index])
        names = UNIVARIATE_EXCLUSIVE_NAMES
    elif scope == "global":
        per_channel = np.stack(
            [_univariate_exclusive_targets(future[:, :, channel]) for channel in range(future.shape[-1])],
            axis=1,
        )
        values = np.concatenate(
            [per_channel.mean(axis=1), _cross_channel_targets(future)], axis=1
        ).astype(np.float32)
        names = GLOBAL_EXCLUSIVE_NAMES
    else:
        raise ValueError(f"Unsupported target-exclusive scope={scope!r}.")
    return SemanticTargets(values=values, names=names)


def cross_fitted_residualize(
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    train_covariates: np.ndarray,
    validation_covariates: np.ndarray,
    test_covariates: np.ndarray,
    train_origins: np.ndarray,
    *,
    alphas: tuple[float, ...],
    folds: int,
    purge_gap: int,
) -> ResidualizedTargets:
    y = np.asarray(train_y, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    test_y = np.asarray(test_y, dtype=np.float64)
    x = np.asarray(train_covariates, dtype=np.float64)
    validation_x = np.asarray(validation_covariates, dtype=np.float64)
    test_x = np.asarray(test_covariates, dtype=np.float64)
    origins = np.asarray(train_origins, dtype=np.int64)
    if y.ndim != 2 or x.ndim != 2 or len(y) != len(x) or len(y) != len(origins):
        raise ValueError(f"Cross-fitting inputs are incompatible: x={x.shape}, y={y.shape}, origins={origins.shape}.")
    if validation_y.shape[1:] != y.shape[1:] or test_y.shape[1:] != y.shape[1:]:
        raise ValueError("Nuisance targets differ across chronological splits.")
    if folds < 2 or folds > len(y):
        raise ValueError(f"folds={folds} must be in [2, {len(y)}].")
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative.")
    if not all(np.isfinite(item).all() for item in (x, validation_x, test_x, y, validation_y, test_y)):
        raise ValueError("Cross-fitting inputs contain non-finite values.")

    fold_indices = np.array_split(np.arange(len(y)), folds)
    predictions = np.empty((len(alphas), *y.shape), dtype=np.float64)
    fold_train_sizes: list[int] = []
    for fold_index, held_out in enumerate(fold_indices):
        first, last = origins[held_out[0]], origins[held_out[-1]]
        train_mask = (origins <= first - purge_gap) | (origins >= last + purge_gap)
        train_mask[held_out] = False
        if int(train_mask.sum()) <= x.shape[1] // 4:
            raise ValueError(
                f"Purged fold {fold_index} leaves only {int(train_mask.sum())} nuisance-training origins."
            )
        fold_train_sizes.append(int(train_mask.sum()))
        scaler = StandardScaler().fit(x[train_mask])
        x_fit = scaler.transform(x[train_mask])
        x_held_out = scaler.transform(x[held_out])
        mean = y[train_mask].mean(axis=0, keepdims=True)
        scale = np.maximum(y[train_mask].std(axis=0, keepdims=True), 1e-6)
        standardized_y = (y[train_mask] - mean) / scale
        for alpha_index, alpha in enumerate(alphas):
            prediction = np.asarray(
                Ridge(alpha=float(alpha)).fit(x_fit, standardized_y).predict(x_held_out)
            ).reshape(len(held_out), -1)
            predictions[alpha_index, held_out] = prediction * scale + mean

    losses = np.square(predictions - y[None, :, :]).mean(axis=1)
    selected_indices = np.argmin(losses, axis=0)
    selected_alpha = np.asarray(alphas, dtype=np.float64)[selected_indices]
    train_prediction = np.column_stack(
        [predictions[selected_indices[target], :, target] for target in range(y.shape[1])]
    )

    scaler = StandardScaler().fit(x)
    train_x_scaled = scaler.transform(x)
    validation_x_scaled = scaler.transform(validation_x)
    test_x_scaled = scaler.transform(test_x)
    mean = y.mean(axis=0, keepdims=True)
    scale = np.maximum(y.std(axis=0, keepdims=True), 1e-6)
    validation_prediction = np.empty_like(validation_y)
    test_prediction = np.empty_like(test_y)
    for alpha in np.unique(selected_alpha):
        target_mask = selected_alpha == alpha
        model = Ridge(alpha=float(alpha)).fit(
            train_x_scaled, ((y - mean) / scale)[:, target_mask]
        )
        validation_standardized = np.asarray(model.predict(validation_x_scaled)).reshape(
            len(validation_x), -1
        )
        test_standardized = np.asarray(model.predict(test_x_scaled)).reshape(len(test_x), -1)
        validation_prediction[:, target_mask] = (
            validation_standardized * scale[:, target_mask] + mean[:, target_mask]
        )
        test_prediction[:, target_mask] = (
            test_standardized * scale[:, target_mask] + mean[:, target_mask]
        )

    train_residual = y - train_prediction
    raw_variance = np.var(y, axis=0)
    residual_variance = np.var(train_residual, axis=0)
    variance_ratio = np.divide(
        residual_variance,
        raw_variance,
        out=np.full_like(raw_variance, np.nan),
        where=raw_variance > 1e-12,
    )
    return ResidualizedTargets(
        train=train_residual.astype(np.float32),
        validation=(validation_y - validation_prediction).astype(np.float32),
        test=(test_y - test_prediction).astype(np.float32),
        alpha=selected_alpha.astype(np.float32),
        validation_r2=r2_per_target(validation_y, validation_prediction),
        test_r2=r2_per_target(test_y, test_prediction),
        train_residual_variance_ratio=variance_ratio.astype(np.float32),
        fold_train_sizes=tuple(fold_train_sizes),
    )


def _moving_block_indices(
    rng: np.random.Generator, sample_count: int, block_length: int
) -> np.ndarray:
    block_count = math.ceil(sample_count / block_length)
    starts = rng.integers(0, sample_count, size=block_count)
    return np.concatenate(
        [(start + np.arange(block_length, dtype=np.int64)) % sample_count for start in starts]
    )[:sample_count]


def paired_r2_block_bootstrap(
    target: np.ndarray,
    high_prediction: np.ndarray,
    low_prediction: np.ndarray,
    origins: np.ndarray,
    *,
    dependence_span: int,
    repetitions: int,
    seed: int,
) -> BlockBootstrapResult:
    target = np.asarray(target, dtype=np.float64)
    high_prediction = np.asarray(high_prediction, dtype=np.float64)
    low_prediction = np.asarray(low_prediction, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    if target.shape != high_prediction.shape or target.shape != low_prediction.shape:
        raise ValueError("Paired bootstrap target and predictions must have identical shapes.")
    if len(target) != len(origins) or repetitions < 1:
        raise ValueError("Paired bootstrap requires matching origins and at least one repetition.")
    step = max(1.0, float(np.median(np.diff(origins))))
    block_length = min(len(origins), max(2, int(math.ceil(dependence_span / step))))
    observed = r2_per_target(target, high_prediction) - r2_per_target(target, low_prediction)
    estimates = np.empty((repetitions, target.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        indices = _moving_block_indices(rng, len(target), block_length)
        estimates[repetition] = r2_per_target(
            target[indices], high_prediction[indices]
        ) - r2_per_target(target[indices], low_prediction[indices])
    centered = estimates - observed[None, :]
    p_greater = (1.0 + np.sum(centered >= observed[None, :], axis=0)) / (repetitions + 1.0)
    lower, upper = np.nanquantile(estimates, (0.025, 0.975), axis=0)
    return BlockBootstrapResult(
        observed=observed.astype(np.float32),
        estimates=estimates,
        ci95_lower=lower.astype(np.float32),
        ci95_upper=upper.astype(np.float32),
        p_greater=p_greater.astype(np.float32),
        block_length=block_length,
    )
