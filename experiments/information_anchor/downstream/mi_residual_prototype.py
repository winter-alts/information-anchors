"""MI-weighted hidden retrieval and residual-prototype correction.

The selector-side functions consume history representations and discovery
scores only.  The residual-side functions are called after a candidate future
has already been retrieved; they do not inspect a query future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = np.float64(1e-6)


@dataclass(frozen=True)
class ResidualPrototypeResult:
    """Residual consensus and its history-distance diagnostics."""

    correction: np.ndarray
    weights: np.ndarray
    normalized_distances: np.ndarray
    distance: np.ndarray
    consistency: np.ndarray
    # Kept as a shape-compatible diagnostic for existing analysis artifacts.
    # The current residual-prototype fusion does not use this value as a gate.
    confidence: np.ndarray


@dataclass(frozen=True)
class MIReliabilityResult:
    """History-only reliability diagnostics for an MI candidate ranking."""

    gate: np.ndarray
    mi_confidence: np.ndarray
    top_margin: np.ndarray
    rank_agreement: np.ndarray
    weights: np.ndarray


def _finite(name: str, values: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite, got {array.shape}")
    return array


def mi_patch_weights(mi_scores: np.ndarray) -> np.ndarray:
    """Normalize non-negative discovery MI mass independently per query."""

    scores = _finite("mi_scores", mi_scores, ndim=2)
    if scores.shape[1] < 1:
        raise ValueError("mi_scores must contain at least one patch")
    positive = np.maximum(scores, 0.0)
    total = positive.sum(axis=1, keepdims=True)
    uniform = np.full_like(positive, 1.0 / positive.shape[1])
    weights = np.divide(
        positive,
        total,
        out=uniform,
        where=total > _EPS,
    )
    return weights.astype(np.float32)


def mi_weighted_squared_hidden_distances(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
    query_mi_scores: np.ndarray,
) -> np.ndarray:
    """Compute MI-weighted squared-Euclidean distances inside a fixed pool.

    Parameters use ``query_tokens=[N,P,D]``, ``candidate_tokens=[N,K,P,D]``
    and ``query_mi_scores=[N,P]``.  The query-side profile is the
    discovery-fitted estimate of ``I(H_j;E)``; candidate-side future values are
    intentionally not part of this API.
    """

    query = _finite("query_tokens", query_tokens, ndim=3)
    candidates = _finite("candidate_tokens", candidate_tokens, ndim=4)
    scores = _finite("query_mi_scores", query_mi_scores, ndim=2)
    if query.shape[0] != candidates.shape[0] or query.shape[1:] != candidates.shape[2:]:
        raise ValueError(
            "query_tokens and candidate_tokens must align as [N,P,D] and [N,K,P,D]"
        )
    if scores.shape != query.shape[:2]:
        raise ValueError(
            f"query_mi_scores must have shape {query.shape[:2]}, got {scores.shape}"
        )

    patch_distance = np.square(query[:, None, :, :] - candidates).sum(axis=-1)
    weights = mi_patch_weights(scores).astype(np.float64)
    return np.sum(weights[:, None, :] * patch_distance, axis=2).astype(np.float32)


def _residuals(values: np.ndarray) -> tuple[np.ndarray, int]:
    array = _finite("candidate_residuals", values)
    if array.ndim not in {3, 4}:
        raise ValueError(
            "candidate_residuals must have rank 3 [N,K,H] or rank 4 [N,K,H,C]"
        )
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError("candidate_residuals must contain candidates and horizons")
    return (array[..., None], array.ndim) if array.ndim == 3 else (array, array.ndim)


def _distance_weights(
    distances: np.ndarray,
    reference_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite("distances", distances, ndim=2)
    if values.shape[1] < 1:
        raise ValueError("distances must contain at least one candidate")
    if np.any(values < 0.0):
        raise ValueError("distances must be non-negative")
    reference = values if reference_distances is None else _finite(
        "reference_distances", reference_distances, ndim=2
    )
    if reference.shape[0] != values.shape[0] or reference.shape[1] < 1:
        raise ValueError(
            "reference_distances must share the batch and contain candidates"
        )
    if np.any(reference < 0.0):
        raise ValueError("reference_distances must be non-negative")
    normalized = values / (np.median(reference, axis=1, keepdims=True) + _EPS)
    logits = -normalized
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    weights /= weights.sum(axis=1, keepdims=True)
    return normalized, weights


def _ranking_agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return one minus normalized mean rank disagreement per query."""

    left = _finite("first distances", first, ndim=2)
    right = _finite("second distances", second, ndim=2)
    if left.shape != right.shape:
        raise ValueError(
            f"distance rankings must have the same shape, got {left.shape} and {right.shape}"
        )
    candidate_count = left.shape[1]
    if candidate_count == 1:
        return np.ones(left.shape[0], dtype=np.float32)

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, axis=1, kind="stable")
        output = np.empty_like(order, dtype=np.float64)
        output[np.arange(values.shape[0])[:, None], order] = np.arange(
            candidate_count, dtype=np.float64
        )[None, :]
        return output

    disagreement = np.abs(ranks(left) - ranks(right)).mean(axis=1)
    return np.clip(1.0 - disagreement / float(candidate_count - 1), 0.0, 1.0).astype(
        np.float32
    )


def mi_reliability_gate(
    mi_distances: np.ndarray,
    official_distances: np.ndarray,
    *,
    reference_distances: np.ndarray | None = None,
    threshold: float = 0.05,
    power: float = 1.0,
) -> MIReliabilityResult:
    """Estimate when an MI-weighted residual correction is trustworthy.

    The gate consumes only candidate distances.  It combines MI concentration
    (one minus normalized entropy), the top-two MI weight margin, and ranking
    agreement with the official history-distance ordering.  The threshold is
    applied to concentration, so a flat/Uniform MI row has an exactly zero
    gate.  ``reference_distances`` optionally supplies the complete MI pool
    for scale normalization; it does not affect the rank diagnostics.
    """

    mi = _finite("mi_distances", mi_distances, ndim=2)
    official = _finite("official_distances", official_distances, ndim=2)
    if mi.shape != official.shape:
        raise ValueError(
            f"mi_distances and official_distances must have the same shape, got {mi.shape} and {official.shape}"
        )
    if np.any(mi < 0.0) or np.any(official < 0.0):
        raise ValueError("MI and official distances must be non-negative")
    threshold_value = float(threshold)
    power_value = float(power)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value < 1.0:
        raise ValueError("threshold must be finite and lie in [0,1)")
    if not np.isfinite(power_value) or power_value <= 0.0:
        raise ValueError("power must be finite and positive")

    _, weights = _distance_weights(mi, reference_distances)
    candidate_count = weights.shape[1]
    if candidate_count == 1:
        mi_confidence = np.ones(weights.shape[0], dtype=np.float32)
        top_margin = np.ones(weights.shape[0], dtype=np.float32)
    else:
        entropy = -np.sum(weights * np.log(np.maximum(weights, _EPS)), axis=1)
        mi_confidence = np.clip(
            1.0 - entropy / np.log(float(candidate_count)), 0.0, 1.0
        ).astype(np.float32)
        top_two = np.sort(weights, axis=1)[:, -2:]
        top_margin = np.clip(
            (top_two[:, 1] - top_two[:, 0])
            * float(candidate_count)
            / float(candidate_count - 1),
            0.0,
            1.0,
        ).astype(np.float32)

    rank_agreement = _ranking_agreement(mi, official)
    activation = np.clip(
        (mi_confidence.astype(np.float64) - threshold_value)
        / max(1.0 - threshold_value, _EPS),
        0.0,
        1.0,
    )
    margin_factor = 0.5 + 0.5 * top_margin.astype(np.float64)
    agreement_factor = 0.5 + 0.5 * rank_agreement.astype(np.float64)
    gate = np.power(activation, power_value) * margin_factor * agreement_factor
    # Make the Uniform control exact even when entropy arithmetic leaves a
    # tiny negative/positive round-off around zero concentration.
    gate[mi_confidence <= threshold_value + 1e-7] = 0.0
    return MIReliabilityResult(
        gate=np.clip(gate, 0.0, 1.0).astype(np.float32),
        mi_confidence=mi_confidence,
        top_margin=top_margin,
        rank_agreement=rank_agreement,
        weights=weights.astype(np.float32),
    )


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a weighted median along the candidate axis.

    ``values`` is shaped ``[N,K,...]`` and ``weights`` is ``[N,K]``.  The
    remaining dimensions are treated independently, so the result has shape
    ``values.shape[2:]`` per query.
    """

    samples = _finite("values", values)
    candidate_weights = _finite("weights", weights, ndim=2)
    if samples.ndim < 2:
        raise ValueError("values must have rank at least 2")
    if samples.shape[:2] != candidate_weights.shape:
        raise ValueError(
            f"values and weights must agree on [N,K], got {samples.shape[:2]} "
            f"and {candidate_weights.shape}"
        )
    if np.any(candidate_weights < 0.0):
        raise ValueError("weights must be non-negative")
    totals = candidate_weights.sum(axis=1, keepdims=True)
    if np.any(totals <= _EPS):
        raise ValueError("each query must have positive total weight")

    normalized_weights = candidate_weights / totals
    expanded_weights = np.broadcast_to(
        normalized_weights[(...,) + (None,) * (samples.ndim - 2)],
        samples.shape,
    )
    order = np.argsort(samples, axis=1, kind="stable")
    sorted_values = np.take_along_axis(samples, order, axis=1)
    sorted_weights = np.take_along_axis(expanded_weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    median_index = np.argmax(cumulative >= 0.5, axis=1)
    return np.take_along_axis(
        sorted_values, median_index[:, None, ...], axis=1
    ).squeeze(axis=1)


def _direction_consistency(residuals: np.ndarray) -> np.ndarray:
    candidate_count = residuals.shape[1]
    if candidate_count == 1:
        return np.ones(residuals.shape[0], dtype=np.float64)
    flat = residuals.reshape(residuals.shape[0], candidate_count, -1)
    norms = np.linalg.norm(flat, axis=2, keepdims=True)
    unit = flat / np.maximum(norms, _EPS)
    cosine = np.einsum("nkd,nld->nkl", unit, unit)
    upper = np.triu_indices(candidate_count, k=1)
    return cosine[:, upper[0], upper[1]].mean(axis=1)


def residual_prototype_correction(
    candidate_residuals: np.ndarray,
    distances: np.ndarray,
    *,
    reference_distances: np.ndarray | None = None,
    estimator: str = "mean",
    temperature: float = 1.0,
    weighting: str = "median_pool",
) -> ResidualPrototypeResult:
    """Estimate a distance-weighted residual prototype.

    ``candidate_residuals`` may be univariate ``[N,K,H]`` or multivariate
    ``[N,K,H,C]``.  Selected distances are normalized by each query's
    ``reference_distances`` median when supplied, otherwise by their own
    median, then used in ``softmax(-distance)``.  With ``weighting=\"standardized\"``
    the weights instead use ``softmax(-z(distance)/temperature)`` on the
    selected candidates. ``estimator`` selects either
    the weighted mean or the coordinate-wise weighted median. Direction
    consistency and weighted distance remain available as diagnostics, but no
    confidence gate is applied to the returned correction.
    """

    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")
    if weighting not in {"median_pool", "standardized"}:
        raise ValueError("weighting must be 'median_pool' or 'standardized'")
    temperature_value = float(temperature)
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    residuals, original_rank = _residuals(candidate_residuals)
    values = _finite("distances", distances, ndim=2)
    if values.shape != residuals.shape[:2]:
        raise ValueError(
            f"distances must have shape {residuals.shape[:2]}, got {values.shape}"
        )
    normalized, _ = _distance_weights(values, reference_distances)
    if weighting == "standardized":
        centered = values - values.mean(axis=1, keepdims=True)
        scale = np.maximum(values.std(axis=1, keepdims=True), _EPS)
        weight_distances = centered / scale
    else:
        weight_distances = normalized
    logits = -weight_distances / temperature_value
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    weights /= weights.sum(axis=1, keepdims=True)
    if estimator == "mean":
        correction = np.sum(
            residuals * weights[:, :, None, None], axis=1
        )
    else:
        correction = weighted_median(residuals, weights)
    consistency = _direction_consistency(residuals)
    distance = np.sum(weights * weight_distances, axis=1)
    # Preserve the historical output field for downstream analysis files.  A
    # constant one records that this no-gate branch applies the full lambda
    # rather than multiplying it by g_q.
    confidence = np.ones(residuals.shape[0], dtype=np.float64)
    correction = correction[..., 0] if original_rank == 3 else correction
    return ResidualPrototypeResult(
        correction=correction.astype(np.float32),
        weights=weights.astype(np.float32),
        normalized_distances=normalized.astype(np.float32),
        distance=distance.astype(np.float32),
        consistency=consistency.astype(np.float32),
        confidence=confidence.astype(np.float32),
    )


def residual_prototype_confidence(
    candidate_residuals: np.ndarray,
    distances: np.ndarray,
    *,
    reference_distances: np.ndarray | None = None,
    weighting: str = "median_pool",
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute the legacy ``max(0, C_q) * exp(-D_q)`` confidence."""

    result = residual_prototype_correction(
        candidate_residuals,
        distances,
        reference_distances=reference_distances,
        weighting=weighting,
        temperature=temperature,
    )
    confidence = np.maximum(result.consistency, 0.0) * np.exp(-result.distance)
    return confidence.astype(np.float32)


def hybrid_residual_prototype_correction(
    candidate_residuals: np.ndarray,
    mi_distances: np.ndarray,
    official_distances: np.ndarray,
    *,
    beta: float = 0.5,
    reference_mi_distances: np.ndarray | None = None,
    reference_official_distances: np.ndarray | None = None,
    estimator: str = "mean",
    temperature: float = 1.0,
) -> ResidualPrototypeResult:
    """Estimate a residual prototype from MI and official history distances.

    Each distance source is normalized by its own query-wise median before the
    two signals are mixed.  ``beta=0`` is official-distance weighting and
    ``beta=1`` is MI-distance weighting.  The candidate futures are used only
    to form residuals; both weighting signals remain history-only.
    """

    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")
    beta_value = float(beta)
    if not np.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
        raise ValueError("beta must be finite and lie in [0,1]")
    temperature_value = float(temperature)
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")

    residuals, original_rank = _residuals(candidate_residuals)
    mi_values = _finite("mi_distances", mi_distances, ndim=2)
    official_values = _finite("official_distances", official_distances, ndim=2)
    expected_shape = residuals.shape[:2]
    if mi_values.shape != expected_shape or official_values.shape != expected_shape:
        raise ValueError(
            "MI and official distances must match candidate residual shape "
            f"{expected_shape}, got {mi_values.shape} and {official_values.shape}"
        )

    mi_normalized, _ = _distance_weights(mi_values, reference_mi_distances)
    official_normalized, _ = _distance_weights(
        official_values, reference_official_distances
    )
    normalized = (
        beta_value * mi_normalized
        + (1.0 - beta_value) * official_normalized
    )
    logits = -normalized / temperature_value
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    weights /= weights.sum(axis=1, keepdims=True)
    if estimator == "mean":
        correction = np.sum(residuals * weights[:, :, None, None], axis=1)
    else:
        correction = weighted_median(residuals, weights)
    consistency = _direction_consistency(residuals)
    distance = np.sum(weights * normalized, axis=1)
    confidence = np.ones(residuals.shape[0], dtype=np.float64)
    correction = correction[..., 0] if original_rank == 3 else correction
    return ResidualPrototypeResult(
        correction=correction.astype(np.float32),
        weights=weights.astype(np.float32),
        normalized_distances=normalized.astype(np.float32),
        distance=distance.astype(np.float32),
        consistency=consistency.astype(np.float32),
        confidence=confidence.astype(np.float32),
    )


def apply_residual_prototype_correction(
    base_prediction: np.ndarray,
    candidate_residuals: np.ndarray,
    distances: np.ndarray,
    *,
    lambda_: float,
    reference_distances: np.ndarray | None = None,
    estimator: str = "mean",
) -> np.ndarray:
    """Apply ``lambda * E_hat_q`` to a base forecast without a confidence gate."""

    prediction = _finite("base_prediction", base_prediction)
    if prediction.ndim not in {2, 3}:
        raise ValueError(
            "base_prediction must have rank 2 [N,H] or rank 3 [N,H,C]"
        )
    gain = np.asarray(lambda_, dtype=np.float64)
    if gain.ndim != 0 or not np.isfinite(gain) or float(gain) < 0.0:
        raise ValueError("lambda_ must be finite and non-negative")
    result = residual_prototype_correction(
        candidate_residuals,
        distances,
        reference_distances=reference_distances,
        estimator=estimator,
    )
    if prediction.shape != result.correction.shape:
        raise ValueError(
            f"base_prediction shape {prediction.shape} does not match correction "
            f"shape {result.correction.shape}"
        )
    correction = np.asarray(result.correction, dtype=np.float64)
    output = prediction + float(gain) * correction
    return output.astype(np.float32)


__all__ = [
    "MIReliabilityResult",
    "ResidualPrototypeResult",
    "apply_residual_prototype_correction",
    "hybrid_residual_prototype_correction",
    "mi_patch_weights",
    "mi_weighted_squared_hidden_distances",
    "mi_reliability_gate",
    "residual_prototype_confidence",
    "residual_prototype_correction",
    "weighted_median",
]
