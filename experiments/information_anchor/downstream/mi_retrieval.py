"""MI-guided retrieval and residual correction for frozen forecasters.

The functions in this module are deliberately model-agnostic.  Every retrieval
variant shares the same candidate pool, rank fusion, residual bank, and gain
calibration; only the hidden-state patch set is changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from experiments.information_anchor.estimators.gcmi import (
    gaussian_copula_transform,
    gcmi_from_gaussianized,
)
from experiments.information_anchor.estimators.nulls import robust_null_score
from experiments.information_anchor.estimators.projection import FittedProjection


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray
    scores: np.ndarray
    recent_distances: np.ndarray
    hidden_distances: np.ndarray


@dataclass(frozen=True)
class HorizonMIProfile:
    observed: np.ndarray
    weights: np.ndarray
    global_observed: np.ndarray
    global_weights: np.ndarray


def _as_3d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [sample,time,channel], got {array.shape}.")
    return array


def row_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite 2D matrix, got {array.shape}.")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, np.float32(1e-12))


def recent_keys(history_normalized: np.ndarray, length: int = 96) -> np.ndarray:
    history = _as_3d(history_normalized, "history_normalized")
    if not 1 <= length <= history.shape[1]:
        raise ValueError(f"Recent length must be in [1,{history.shape[1]}], got {length}.")
    return row_normalize(history[:, -length:].reshape(len(history), -1))


def raw_history_keys(history_normalized: np.ndarray, length: int = 96) -> np.ndarray:
    """Return a causal, unweighted recent history key.

    The runner fits PCA on the discovery bank before calling this helper.  The
    function intentionally does no fitting so a query can never influence the
    representation.
    """
    return recent_keys(history_normalized, length=length)


def hidden_mean_keys(hidden: np.ndarray, *, layer: int) -> np.ndarray:
    """Mean-pool one functional hidden layer without using MI weights."""
    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim == 4:
        if not 0 <= int(layer) < values.shape[1]:
            raise ValueError(f"Invalid layer {layer} for hidden shape {values.shape}")
        values = values[:, int(layer)]
    if values.ndim != 3:
        raise ValueError(f"hidden must be [sample,layer,patch,width] or [sample,patch,width], got {values.shape}")
    return row_normalize(values.mean(axis=1))


def multivariate_dtw_distance(query: np.ndarray, database: np.ndarray) -> float:
    """Squared-Euclidean DTW distance for two [time, feature] sequences."""
    x = np.asarray(query, dtype=np.float32)
    y = np.asarray(database, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"DTW sequences must be [time,feature], got {x.shape}, {y.shape}")
    costs = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    table = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        table[i, 1:] = costs[i - 1] + np.minimum(
            table[i - 1, 1:], np.minimum(table[i, :-1], table[i - 1, :-1])
        )
    return float(table[-1, -1] / max(len(x) + len(y), 1))


def dtw_distance_matrix(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute a finite pairwise DTW matrix; intended for a fixed candidate pool."""
    q = np.asarray(query, dtype=np.float32)
    d = np.asarray(database, dtype=np.float32)
    if q.ndim != 3 or d.ndim != 3 or q.shape[2] != d.shape[2]:
        raise ValueError(f"DTW batches must be [sample,time,feature], got {q.shape}, {d.shape}")
    return np.asarray(
        [[multivariate_dtw_distance(qi, di) for di in d] for qi in q],
        dtype=np.float32,
    )


def projected_patch_keys(
    hidden: np.ndarray,
    projection: FittedProjection,
    *,
    layer: int,
    patches: tuple[int, ...],
    patch_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the discovery-fitted per-patch PCA, then concatenate selected patches."""
    values = np.asarray(hidden)
    if values.ndim != 4:
        raise ValueError(f"Hidden cache must be [sample,layer,patch,width], got {values.shape}.")
    if not 0 <= layer < values.shape[1]:
        raise ValueError(f"Layer {layer} is invalid for hidden shape {values.shape}.")
    patch_indices = np.asarray(patches, dtype=np.int64)
    if patch_indices.ndim != 1 or len(patch_indices) == 0:
        raise ValueError("At least one patch is required.")
    if patch_indices.min() < 0 or patch_indices.max() >= values.shape[2]:
        raise ValueError(f"Patch set {patches} is invalid for hidden shape {values.shape}.")

    selected = np.asarray(values[:, layer, patch_indices, :], dtype=np.float32)
    sample_count, patch_count, width = selected.shape
    projected = projection.transform(selected.reshape(sample_count * patch_count, width))
    projected = projected.reshape(sample_count, patch_count, -1)
    if patch_weights is not None:
        weights = np.asarray(patch_weights, dtype=np.float32).reshape(-1)
        if weights.shape != (patch_count,) or np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("patch_weights must be finite, positive, and match patches.")
        # Euclidean/cosine energy is quadratic, so sqrt implements linear
        # importance weighting in the resulting distance.
        projected = projected * np.sqrt(weights)[None, :, None]
    return row_normalize(projected.reshape(sample_count, -1))


def mi_rank_weights(mi_scores: np.ndarray) -> np.ndarray:
    """Map discovery MI order to parameter-free positive weights in [1/P, 1]."""
    scores = np.asarray(mi_scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("mi_scores must be a non-empty finite vector.")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    return ranks / np.float32(len(scores))


def _normalized_rank_weights(scores: np.ndarray) -> np.ndarray:
    ranks = mi_rank_weights(scores).astype(np.float32)
    return ranks / np.maximum(ranks.sum(), np.float32(1e-12))


def fit_horizon_mi_profile(
    patch_features: np.ndarray,
    future_target: np.ndarray,
    *,
    block_size: int = 16,
) -> HorizonMIProfile:
    features = np.asarray(patch_features, dtype=np.float64)
    future = np.asarray(future_target, dtype=np.float64)
    if features.ndim != 3 or future.ndim != 2 or len(features) != len(future):
        raise ValueError("patch features [N,P,D] and future [N,H] must share samples")
    if not np.isfinite(features).all() or not np.isfinite(future).all():
        raise ValueError("horizon MI inputs must be finite")
    if int(block_size) < 1 or future.shape[1] % int(block_size):
        raise ValueError("future length must be divisible by block_size")
    blocks = future.shape[1] // int(block_size)
    patches = features.shape[1]
    gaussian_patches = [
        gaussian_copula_transform(features[:, patch]) for patch in range(patches)
    ]
    observed = np.empty((blocks, patches), dtype=np.float32)
    for block in range(blocks):
        target = gaussian_copula_transform(
            future[:, block * block_size : (block + 1) * block_size]
        )
        for patch, values in enumerate(gaussian_patches):
            observed[block, patch] = gcmi_from_gaussianized(values, target)
    global_target = gaussian_copula_transform(future)
    global_observed = np.asarray(
        [gcmi_from_gaussianized(values, global_target) for values in gaussian_patches],
        dtype=np.float32,
    )
    weights = np.stack([_normalized_rank_weights(row) for row in observed])
    return HorizonMIProfile(
        observed=observed,
        weights=weights.astype(np.float32),
        global_observed=global_observed,
        global_weights=_normalized_rank_weights(global_observed),
    )


def horizon_control_weights(
    profile: HorizonMIProfile, *, mode: str, seed: int = 2021
) -> np.ndarray:
    if mode == "horizon":
        return profile.weights.copy()
    if mode == "global":
        return np.broadcast_to(profile.global_weights[None], profile.weights.shape).copy()
    if mode == "uniform":
        return np.full_like(profile.weights, 1.0 / profile.weights.shape[1])
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return np.stack([row[rng.permutation(len(row))] for row in profile.weights])
    raise ValueError(f"unknown horizon control mode: {mode}")


def horizon_weighted_hidden_distances(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
    patch_weights: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_tokens, dtype=np.float64)
    candidates = np.asarray(candidate_tokens, dtype=np.float64)
    weights = np.asarray(patch_weights, dtype=np.float64)
    if query.ndim != 3 or candidates.ndim != 4 or weights.ndim != 2:
        raise ValueError("expected query [N,P,D], candidates [N,K,P,D], weights [B,P]")
    if query.shape[0] != candidates.shape[0] or query.shape[1:] != candidates.shape[2:]:
        raise ValueError("query and candidate token shapes are incompatible")
    if weights.shape[1] != query.shape[1]:
        raise ValueError("horizon weights do not match the history patch count")
    if not all(np.isfinite(item).all() for item in (query, candidates, weights)):
        raise ValueError("horizon distance inputs must be finite")
    if np.any(weights <= 0):
        raise ValueError("horizon weights must be strictly positive")
    patch_cost = np.sum(np.square(candidates - query[:, None]), axis=3, dtype=np.float64)
    distances = np.einsum("bp,nkp->nbk", weights, patch_cost, optimize=True)
    if not np.isfinite(distances).all():
        raise OverflowError("horizon weighted hidden distances overflowed")
    if distances.size and float(np.max(distances)) > np.finfo(np.float32).max:
        raise OverflowError("horizon weighted hidden distances overflowed")
    return distances.astype(np.float32)


def residual_mi_scores(
    patch_features: np.ndarray,
    residual_target: np.ndarray,
    shifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score each patch against a discovery-only conditional residual target.

    This is a Gaussian-copula MI profile with a temporal circular-shift null.  It
    is used as a practical residual-information proxy, not as an exact nonlinear
    conditional-MI estimator.
    """
    features = np.asarray(patch_features, dtype=np.float64)
    target = np.asarray(residual_target, dtype=np.float64)
    offsets = np.asarray(shifts, dtype=np.int64).reshape(-1)
    if features.ndim != 3:
        raise ValueError(
            f"patch_features must be [sample,patch,feature], got {features.shape}."
        )
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != features.shape[0]:
        raise ValueError(
            f"Residual target must share samples, got x={features.shape}, y={target.shape}."
        )
    if len(offsets) == 0 or np.any(offsets <= 0) or np.any(offsets >= len(features)):
        raise ValueError("shifts must contain offsets in [1, sample_count).")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Residual MI inputs contain non-finite values.")

    target_gaussian = gaussian_copula_transform(target)
    patch_count = features.shape[1]
    observed = np.empty(patch_count, dtype=np.float32)
    z_scores = np.empty(patch_count, dtype=np.float32)
    p_values = np.empty(patch_count, dtype=np.float32)
    null_values = np.empty((len(offsets), patch_count), dtype=np.float32)
    for patch_index in range(patch_count):
        patch_gaussian = gaussian_copula_transform(features[:, patch_index])
        observed[patch_index] = gcmi_from_gaussianized(
            patch_gaussian, target_gaussian
        )
        for shift_index, shift in enumerate(offsets):
            null_values[shift_index, patch_index] = gcmi_from_gaussianized(
                patch_gaussian, np.roll(target_gaussian, int(shift), axis=0)
            )
        z_scores[patch_index], p_values[patch_index] = robust_null_score(
            observed[patch_index], null_values[:, patch_index]
        )
    return observed, z_scores, p_values, null_values


def _conditional_gcmi_from_gaussianized(
    x: np.ndarray, y: np.ndarray, condition: np.ndarray
) -> float:
    joint = np.concatenate([x, y, condition], axis=1)
    covariance = OAS(store_precision=False).fit(joint).covariance_
    x_stop = x.shape[1]
    y_stop = x_stop + y.shape[1]
    x_indices = np.arange(x_stop)
    y_indices = np.arange(x_stop, y_stop)
    z_indices = np.arange(y_stop, joint.shape[1])

    def logdet(*parts: np.ndarray) -> float:
        indices = np.concatenate(parts)
        sign, value = np.linalg.slogdet(covariance[np.ix_(indices, indices)])
        if sign <= 0:
            raise RuntimeError("Conditional-GCMI covariance is not positive definite.")
        return float(value)

    estimate = 0.5 * (
        logdet(x_indices, z_indices)
        + logdet(y_indices, z_indices)
        - logdet(z_indices)
        - logdet(x_indices, y_indices, z_indices)
    )
    return float(max(estimate, 0.0))


def conditional_residual_mi_scores(
    patch_features: np.ndarray,
    residual_target: np.ndarray,
    recent_condition: np.ndarray,
    shifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ``I(patch; residual | recent_key)`` with a temporal null."""
    features = np.asarray(patch_features, dtype=np.float64)
    target = np.asarray(residual_target, dtype=np.float64)
    condition = np.asarray(recent_condition, dtype=np.float64)
    offsets = np.asarray(shifts, dtype=np.int64).reshape(-1)
    if target.ndim == 1:
        target = target[:, None]
    if condition.ndim == 1:
        condition = condition[:, None]
    if features.ndim != 3:
        raise ValueError(
            f"patch_features must be [sample,patch,feature], got {features.shape}."
        )
    if target.ndim != 2 or condition.ndim != 2:
        raise ValueError("Residual target and recent condition must be two-dimensional.")
    if not (len(features) == len(target) == len(condition)):
        raise ValueError("Conditional residual-MI inputs must share samples.")
    if len(offsets) == 0 or np.any(offsets <= 0) or np.any(offsets >= len(features)):
        raise ValueError("shifts must contain offsets in [1, sample_count).")
    if not all(np.isfinite(item).all() for item in (features, target, condition)):
        raise ValueError("Conditional residual-MI inputs contain non-finite values.")

    target_gaussian = gaussian_copula_transform(target)
    condition_gaussian = gaussian_copula_transform(condition)
    patch_count = features.shape[1]
    observed = np.empty(patch_count, dtype=np.float32)
    z_scores = np.empty(patch_count, dtype=np.float32)
    p_values = np.empty(patch_count, dtype=np.float32)
    null_values = np.empty((len(offsets), patch_count), dtype=np.float32)
    for patch_index in range(patch_count):
        patch_gaussian = gaussian_copula_transform(features[:, patch_index])
        observed[patch_index] = _conditional_gcmi_from_gaussianized(
            patch_gaussian, target_gaussian, condition_gaussian
        )
        for shift_index, shift in enumerate(offsets):
            null_values[shift_index, patch_index] = (
                _conditional_gcmi_from_gaussianized(
                    patch_gaussian,
                    np.roll(target_gaussian, int(shift), axis=0),
                    condition_gaussian,
                )
            )
        z_scores[patch_index], p_values[patch_index] = robust_null_score(
            observed[patch_index], null_values[:, patch_index]
        )
    return observed, z_scores, p_values, null_values


def fit_supervised_retrieval_keys(
    database_features: np.ndarray,
    query_features: np.ndarray,
    residual_target: np.ndarray,
    *,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Learn one discovery-only ridge metric for residual-semantic retrieval."""
    database = np.asarray(database_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    target = np.asarray(residual_target, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if database.ndim != 2 or query.ndim != 2 or target.ndim != 2:
        raise ValueError("Supervised retrieval inputs must be two-dimensional.")
    if database.shape[0] != target.shape[0]:
        raise ValueError("Database features and residual target must share samples.")
    if database.shape[1] != query.shape[1]:
        raise ValueError("Database/query retrieval features have different widths.")
    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")
    if not all(np.isfinite(item).all() for item in (database, query, target)):
        raise ValueError("Supervised retrieval inputs contain non-finite values.")

    scaler = StandardScaler().fit(database)
    database_scaled = scaler.transform(database)
    model = Ridge(alpha=float(alpha)).fit(database_scaled, target)
    database_prediction = model.predict(database_scaled)
    query_prediction = model.predict(scaler.transform(query))
    metadata: dict[str, float | int] = {
        "alpha": float(alpha),
        "input_width": int(database.shape[1]),
        "target_width": int(target.shape[1]),
        "discovery_r2": float(model.score(database_scaled, target)),
    }
    return (
        row_normalize(database_prediction),
        row_normalize(query_prediction),
        metadata,
    )


def temporal_valid_mask(
    query_origins: np.ndarray,
    database_origins: np.ndarray,
    minimum_separation: int,
) -> np.ndarray:
    """Mask overlapping examples for discovery-only leave-neighborhood-out calibration."""
    query = np.asarray(query_origins, dtype=np.int64).reshape(-1, 1)
    database = np.asarray(database_origins, dtype=np.int64).reshape(1, -1)
    if minimum_separation < 1:
        raise ValueError("minimum_separation must be positive.")
    return np.abs(query - database) >= int(minimum_separation)


def _stable_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = np.float32((start + stop - 1) / 2.0)
        start = stop
    if len(values) > 1:
        ranks /= np.float32(len(values) - 1)
    return ranks


def retrieve_neighbors(
    query_recent: np.ndarray,
    database_recent: np.ndarray,
    *,
    k: int,
    candidate_count: int,
    query_hidden: np.ndarray | None = None,
    database_hidden: np.ndarray | None = None,
    hidden_weight: float = 0.5,
    query_anti_hidden: np.ndarray | None = None,
    database_anti_hidden: np.ndarray | None = None,
    anti_hidden_weight: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> NeighborResult:
    """Recent-key recall followed by deterministic recent/hidden rank fusion."""
    query_recent = row_normalize(query_recent)
    database_recent = row_normalize(database_recent)
    if query_recent.shape[1] != database_recent.shape[1]:
        raise ValueError("Recent query/database keys have different widths.")
    if not 0.0 <= hidden_weight <= 1.0:
        raise ValueError("hidden_weight must be in [0,1].")
    if anti_hidden_weight < 0.0:
        raise ValueError("anti_hidden_weight must be non-negative.")
    uses_hidden = query_hidden is not None or database_hidden is not None
    if uses_hidden:
        if query_hidden is None or database_hidden is None:
            raise ValueError("Both query_hidden and database_hidden are required.")
        query_hidden = row_normalize(query_hidden)
        database_hidden = row_normalize(database_hidden)
        if query_hidden.shape[0] != query_recent.shape[0]:
            raise ValueError("Query hidden/recent sample counts differ.")
        if database_hidden.shape[0] != database_recent.shape[0]:
            raise ValueError("Database hidden/recent sample counts differ.")
        if query_hidden.shape[1] != database_hidden.shape[1]:
            raise ValueError("Hidden query/database keys have different widths.")
    uses_anti_hidden = query_anti_hidden is not None or database_anti_hidden is not None
    if uses_anti_hidden:
        if query_anti_hidden is None or database_anti_hidden is None:
            raise ValueError("Both query_anti_hidden and database_anti_hidden are required.")
        query_anti_hidden = row_normalize(query_anti_hidden)
        database_anti_hidden = row_normalize(database_anti_hidden)
        if query_anti_hidden.shape[0] != query_recent.shape[0]:
            raise ValueError("Query anti-hidden/recent sample counts differ.")
        if database_anti_hidden.shape[0] != database_recent.shape[0]:
            raise ValueError("Database anti-hidden/recent sample counts differ.")
        if query_anti_hidden.shape[1] != database_anti_hidden.shape[1]:
            raise ValueError("Anti-hidden query/database keys have different widths.")

    query_count, database_count = query_recent.shape[0], database_recent.shape[0]
    if valid_mask is None:
        valid = np.ones((query_count, database_count), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (query_count, database_count):
            raise ValueError(
                f"valid_mask must have shape {(query_count, database_count)}, got {valid.shape}."
            )
    valid_counts = valid.sum(axis=1)
    if np.any(valid_counts < k):
        raise ValueError(f"At least one query has fewer than k={k} valid database items.")
    recall_count = min(int(candidate_count), database_count)
    if recall_count < k:
        raise ValueError("candidate_count must be at least k.")

    recent_matrix = 1.0 - query_recent @ database_recent.T
    recent_matrix = np.where(valid, recent_matrix, np.inf)
    indices = np.empty((query_count, k), dtype=np.int64)
    scores = np.empty((query_count, k), dtype=np.float32)
    selected_recent = np.empty_like(scores)
    selected_hidden = np.full_like(scores, np.nan)

    for query_index in range(query_count):
        usable_count = min(recall_count, int(valid_counts[query_index]))
        candidates = np.argpartition(recent_matrix[query_index], usable_count - 1)[:usable_count]
        candidate_recent = recent_matrix[query_index, candidates]
        recent_rank = _stable_ranks(candidate_recent)
        if uses_hidden:
            candidate_hidden = 1.0 - database_hidden[candidates] @ query_hidden[query_index]
            fused = (1.0 - hidden_weight) * recent_rank + hidden_weight * _stable_ranks(
                candidate_hidden
            )
        else:
            candidate_hidden = np.full(usable_count, np.nan, dtype=np.float32)
            fused = recent_rank
        if uses_anti_hidden:
            candidate_anti_hidden = (
                1.0
                - database_anti_hidden[candidates] @ query_anti_hidden[query_index]
            )
            # Favor future-semantic agreement that cannot be explained by
            # nuisance similarity in the matched low-MI representation.
            fused = fused - anti_hidden_weight * _stable_ranks(candidate_anti_hidden)
        chosen_local = np.argsort(fused, kind="stable")[:k]
        indices[query_index] = candidates[chosen_local]
        scores[query_index] = fused[chosen_local]
        selected_recent[query_index] = candidate_recent[chosen_local]
        selected_hidden[query_index] = candidate_hidden[chosen_local]

    return NeighborResult(
        indices=indices,
        scores=scores,
        recent_distances=selected_recent,
        hidden_distances=selected_hidden,
    )


def softmax_neighbor_weights(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Neighbor scores must be finite [query,k], got {values.shape}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = -values / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def neighbor_future_mse(
    query_future_normalized: np.ndarray,
    database_future_normalized: np.ndarray,
    neighbors: NeighborResult,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    query = _as_3d(query_future_normalized, "query_future_normalized")
    database = _as_3d(database_future_normalized, "database_future_normalized")
    selected = database[neighbors.indices]
    if selected.shape[0] != len(query) or selected.shape[2:] != query.shape[1:]:
        raise ValueError("Query and selected database futures have incompatible shapes.")
    distance = np.square(selected - query[:, None]).mean(axis=(2, 3))
    weights = softmax_neighbor_weights(neighbors.scores, temperature)
    return np.sum(distance * weights, axis=1).astype(np.float32)


def residual_correction(
    database_prediction: np.ndarray,
    database_truth: np.ndarray,
    database_history_scale: np.ndarray,
    query_history_scale: np.ndarray,
    neighbors: NeighborResult,
    *,
    temperature: float = 1.0,
    scale_ratio_bounds: tuple[float, float] = (0.25, 4.0),
) -> tuple[np.ndarray, np.ndarray]:
    prediction = _as_3d(database_prediction, "database_prediction")
    truth = _as_3d(database_truth, "database_truth")
    if prediction.shape != truth.shape:
        raise ValueError("Database prediction and truth shapes differ.")
    database_scale = np.asarray(database_history_scale, dtype=np.float32)
    query_scale = np.asarray(query_history_scale, dtype=np.float32)
    if database_scale.ndim == 1:
        database_scale = database_scale[:, None]
    if query_scale.ndim == 1:
        query_scale = query_scale[:, None]
    if database_scale.shape != (len(prediction), prediction.shape[-1]):
        raise ValueError("Database history scale has an incompatible shape.")
    if query_scale.shape != (len(neighbors.indices), prediction.shape[-1]):
        raise ValueError("Query history scale has an incompatible shape.")
    lower, upper = scale_ratio_bounds
    if not 0 < lower <= upper:
        raise ValueError("Invalid scale_ratio_bounds.")

    residual = truth - prediction
    selected = residual[neighbors.indices]
    selected_scale = database_scale[neighbors.indices]
    ratio = query_scale[:, None, :] / np.maximum(selected_scale, np.float32(1e-6))
    ratio = np.clip(ratio, lower, upper)
    aligned = selected * ratio[:, :, None, :]
    weights = softmax_neighbor_weights(neighbors.scores, temperature)
    correction = np.sum(aligned * weights[:, :, None, None], axis=1)
    dispersion = np.mean(np.var(aligned, axis=1), axis=(1, 2))
    return correction.astype(np.float32), dispersion.astype(np.float32)


def fit_residual_gain(
    baseline: np.ndarray,
    truth: np.ndarray,
    correction: np.ndarray,
    train_scale: np.ndarray,
    *,
    maximum: float = 1.0,
) -> float:
    """Fit one non-negative scalar using discovery data only."""
    baseline = _as_3d(baseline, "baseline")
    truth = _as_3d(truth, "truth")
    correction = _as_3d(correction, "correction")
    if baseline.shape != truth.shape or baseline.shape != correction.shape:
        raise ValueError("Baseline, truth, and correction shapes differ.")
    scale = np.asarray(train_scale, dtype=np.float64).reshape(1, 1, -1)
    error = (baseline.astype(np.float64) - truth) / scale
    direction = correction.astype(np.float64) / scale
    denominator = float(np.sum(np.square(direction)))
    if denominator <= 1e-12:
        return 0.0
    gain = -float(np.sum(error * direction)) / denominator
    return float(np.clip(gain, 0.0, maximum))


def fit_horizon_gain(
    baseline: np.ndarray,
    truth: np.ndarray,
    correction: np.ndarray,
    train_scale: np.ndarray,
    *,
    maximum: float = 1.0,
) -> np.ndarray:
    """Fit one discovery-only scalar gain per forecast horizon.

    The gain is shared across channels at a horizon, which keeps the adapter
    parameter-free with respect to the retrieval method and prevents channel
    specific overfitting.  Values are clipped to ``[0, maximum]``.
    """
    base = _as_3d(baseline, "baseline")
    target = _as_3d(truth, "truth")
    delta = _as_3d(correction, "correction")
    if base.shape != target.shape or base.shape != delta.shape:
        raise ValueError("Baseline, truth, and correction shapes differ.")
    scale = np.asarray(train_scale, dtype=np.float64).reshape(1, 1, -1)
    error = (base.astype(np.float64) - target) / np.maximum(scale, 1e-8)
    direction = delta.astype(np.float64) / np.maximum(scale, 1e-8)
    numerator = -np.sum(error * direction, axis=(0, 2))
    denominator = np.sum(direction * direction, axis=(0, 2))
    gains = np.divide(
        numerator,
        np.maximum(denominator, 1e-12),
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return np.clip(gains, 0.0, float(maximum)).astype(np.float32)


@dataclass
class LowRankResidualAdapter:
    """Discovery-only PCA + Ridge residual adapter."""

    pca: PCA
    model: Ridge
    channel_count: int
    horizon: int

    def transform(self, correction: np.ndarray) -> np.ndarray:
        values = _as_3d(correction, "correction")
        if values.shape[1:] != (self.horizon, self.channel_count):
            raise ValueError("Correction shape does not match fitted adapter.")
        coeff = self.model.predict(self.pca.transform(values.reshape(len(values), -1)))
        reconstructed = self.pca.inverse_transform(coeff)
        return reconstructed.reshape(values.shape).astype(np.float32)


def fit_low_rank_residual_adapter(
    discovery_correction: np.ndarray,
    discovery_truth_residual: np.ndarray,
    *,
    rank: int = 16,
    alpha: float = 10.0,
    seed: int = 2021,
) -> LowRankResidualAdapter:
    """Fit PCA(rank) and Ridge(alpha) using discovery origins only."""
    correction = _as_3d(discovery_correction, "discovery_correction")
    target = _as_3d(discovery_truth_residual, "discovery_truth_residual")
    if correction.shape != target.shape:
        raise ValueError("Discovery correction and residual shapes differ.")
    flat_correction = correction.reshape(len(correction), -1).astype(np.float64)
    flat_target = target.reshape(len(target), -1).astype(np.float64)
    components = min(int(rank), flat_target.shape[0], flat_target.shape[1])
    if components < 1:
        raise ValueError("Residual adapter needs at least one PCA component.")
    pca = PCA(n_components=components, random_state=int(seed)).fit(flat_target)
    target_coeff = pca.transform(flat_target)
    # Ridge maps the retrieved residual direction to the target residual in the
    # same discovery coordinate system.  No query future is used here.
    correction_coeff = pca.transform(flat_correction)
    model = Ridge(alpha=float(alpha)).fit(correction_coeff, target_coeff)
    return LowRankResidualAdapter(
        pca=pca,
        model=model,
        channel_count=int(correction.shape[2]),
        horizon=int(correction.shape[1]),
    )


def reliability_gate(high_dispersion: np.ndarray, low_dispersion: np.ndarray) -> np.ndarray:
    high = np.asarray(high_dispersion, dtype=np.float64)
    low = np.asarray(low_dispersion, dtype=np.float64)
    if high.shape != low.shape or high.ndim != 1:
        raise ValueError("High/low dispersion must be one-dimensional arrays of equal shape.")
    return np.clip(1.0 - high / np.maximum(low, 1e-12), 0.0, 1.0).astype(np.float32)
