"""History-only confidence gates for MI forecast fusion.

The gate is deliberately computed from the candidate-history distance rows
only.  It never accepts a future window, a forecast residual, or a target
value, so it can be used at test time without changing the MI leakage
boundary.  The default confidence is the agreement between the MI ranking and
the official history-distance ranking.  A low agreement means that the MI
retriever is making a disruptive choice, in which case the downstream
forecast correction is shrunk toward the frozen official forecast.
"""

from __future__ import annotations

import math

import numpy as np


def _finite_matrix(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got {array.shape}")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one candidate")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    # Distance sidecars are written through float32 arithmetic.  Values that
    # should be zero can therefore arrive as tiny negative round-off (e.g.
    # -2.38e-7).  Keep the non-negative-distance contract while accepting and
    # normalizing that harmless numerical noise.
    if np.any(array < -1e-6):
        raise ValueError(f"{name} must be non-negative")
    return np.maximum(array, 0.0)


def _row_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.float64)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float64)[None, :]
    return ranks


def rank_agreement(
    mi_distances: np.ndarray,
    official_distances: np.ndarray,
) -> np.ndarray:
    """Return row-wise MI/official ranking agreement in ``[0, 1]``.

    Both inputs are history-only lower-is-better distance matrices with shape
    ``[batch, candidates]``.  Agreement is one minus the normalized Spearman
    footrule distance.  The normalization uses the maximum possible footrule,
    so a rank reversal maps to zero and an identical ranking maps to one.
    Stable sorting makes ties deterministic.
    """

    mi = _finite_matrix("mi_distances", mi_distances)
    official = _finite_matrix("official_distances", official_distances)
    if mi.shape != official.shape:
        raise ValueError(
            "mi_distances and official_distances must have the same shape, "
            f"got {mi.shape} and {official.shape}"
        )
    candidates = mi.shape[1]
    if candidates == 1:
        return np.ones(mi.shape[0], dtype=np.float64)
    max_footrule = float((candidates * candidates) // 2)
    distance = np.abs(_row_ranks(mi) - _row_ranks(official)).sum(axis=1)
    return np.clip(1.0 - distance / max_footrule, 0.0, 1.0)


def _normalized_entropy_confidence(
    mi_distances: np.ndarray,
    temperature: float,
) -> np.ndarray:
    centered = mi_distances - mi_distances.mean(axis=1, keepdims=True)
    scale = mi_distances.std(axis=1, keepdims=True)
    logits = -centered / np.maximum(scale, 1e-12) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    ) / math.log(float(mi_distances.shape[1]))
    return np.clip(1.0 - entropy, 0.0, 1.0)


def history_only_mi_confidence(
    mi_distances: np.ndarray,
    official_distances: np.ndarray,
    *,
    entropy_power: float = 0.0,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Build a history-only confidence vector for MI forecast fusion.

    The default is rank agreement alone.  Setting ``entropy_power`` above
    zero additionally requires the MI distances to have a concentrated
    within-query distribution; this optional factor is also history-only.
    ``floor`` provides a bounded minimum amount of MI correction.
    """

    mi = _finite_matrix("mi_distances", mi_distances)
    agreement = rank_agreement(mi, official_distances)
    power = float(entropy_power)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("entropy_power must be finite and non-negative")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("temperature must be finite and positive")
    minimum = float(floor)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("floor must be finite and in [0, 1]")
    if power == 0.0:
        raw = agreement
    else:
        entropy_confidence = _normalized_entropy_confidence(mi, tau)
        raw = agreement * np.power(entropy_confidence, power)
    return (minimum + (1.0 - minimum) * raw).astype(np.float32)


def history_only_mi_branch_gate(
    mi_distances: np.ndarray,
    official_distances: np.ndarray,
    *,
    threshold: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Admit the MI retrieval branch using history-only rank agreement.

    Queries whose MI and official candidate rankings agree no more than
    ``threshold`` receive zero MI branch weight and therefore fall back to
    the official forecast.  Above the threshold, the normalized agreement is
    raised to ``power``.  The gate consumes only sidecar history distances;
    no retrieved future or target value is accepted.
    """

    agreement = rank_agreement(mi_distances, official_distances)
    cutoff = float(threshold)
    exponent = float(power)
    if not math.isfinite(cutoff) or not 0.0 <= cutoff < 1.0:
        raise ValueError("threshold must be finite and lie in [0,1)")
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("power must be finite and positive")
    normalized = np.clip((agreement - cutoff) / (1.0 - cutoff), 0.0, 1.0)
    return np.power(normalized, exponent).astype(np.float32)


def mi_distance_match_multiplier(
    mi_distances: np.ndarray,
    *,
    center: float,
    scale: float,
    slope: float,
    floor: float = 0.5,
    ceiling: float = 1.5,
) -> np.ndarray:
    """Return a bounded query-wise multiplier from the best MI match.

    ``min(mi_distances[row])`` is a history-only measure of whether the query
    has at least one close MI candidate.  The calibration statistics and
    slope are frozen on validation data; no forecast or target is accepted.
    A negative slope therefore gives a modestly larger residual step to
    queries with a particularly close MI match and shrinks uncertain queries.
    """

    mi = _finite_matrix("mi_distances", mi_distances)
    location = float(center)
    spread = float(scale)
    coefficient = float(slope)
    lower = float(floor)
    upper = float(ceiling)
    if not math.isfinite(location):
        raise ValueError("center must be finite")
    if not math.isfinite(spread) or spread <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(coefficient):
        raise ValueError("slope must be finite")
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or lower > upper
    ):
        raise ValueError("multiplier bounds must be finite and ordered")
    standardized = (mi.min(axis=1) - location) / spread
    return np.clip(1.0 + coefficient * standardized, lower, upper).astype(
        np.float32
    )


def apply_history_only_gate(
    official_prediction: np.ndarray,
    consensus: np.ndarray,
    confidence: np.ndarray,
    strength: float | np.ndarray,
) -> np.ndarray:
    """Apply a bounded history-only residual correction.

    This is the NumPy equivalent of
    ``y_off + confidence * alpha * (y_mi - y_off)``.  ``strength`` may be a
    scalar or a frozen horizon vector.  A zero strength or zero confidence is
    an exact identity on the official prediction.
    """

    official = np.asarray(official_prediction, dtype=np.float64)
    retrieved = np.asarray(consensus, dtype=np.float64)
    if official.ndim != 2 or retrieved.ndim != 2:
        raise ValueError("official_prediction and consensus must be rank-2")
    if official.shape != retrieved.shape:
        raise ValueError(
            "official_prediction and consensus must have the same shape, "
            f"got {official.shape} and {retrieved.shape}"
        )
    if not np.isfinite(official).all() or not np.isfinite(retrieved).all():
        raise ValueError("official_prediction and consensus must be finite")

    gate = np.asarray(confidence, dtype=np.float64)
    if gate.ndim != 1 or gate.shape[0] != official.shape[0]:
        raise ValueError("confidence must be rank-1 with one value per query")
    if not np.isfinite(gate).all() or np.any((gate < 0.0) | (gate > 1.0)):
        raise ValueError("confidence must be finite and in [0, 1]")

    alpha = np.asarray(strength, dtype=np.float64)
    if alpha.ndim == 0:
        valid = np.isfinite(alpha) and 0.0 <= float(alpha) <= 1.0
    elif alpha.ndim == 1:
        valid = (
            alpha.shape[0] == official.shape[1]
            and np.isfinite(alpha).all()
            and np.all((alpha >= 0.0) & (alpha <= 1.0))
        )
    else:
        valid = False
    if not valid:
        raise ValueError("strength must be a scalar or horizon vector in [0, 1]")

    return (official + gate[:, None] * alpha * (retrieved - official)).astype(
        official_prediction.dtype if np.issubdtype(np.asarray(official_prediction).dtype, np.floating)
        else np.float32
    )


__all__ = [
    "apply_history_only_gate",
    "history_only_mi_branch_gate",
    "history_only_mi_confidence",
    "mi_distance_match_multiplier",
    "rank_agreement",
]
