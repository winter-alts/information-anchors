"""Discovery-fitted abstention for residual-MI candidate selection.

The gate predicts the discovery-time gain of an MI-prior selection over an
official-history fallback.  At evaluation time it consumes only history-side
distances, student scores, and MI profiles; a non-positive conformal lower
bound routes the query to the fallback selection.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.information_anchor.downstream.conformal_risk_gate import (
    ConformalRiskGate,
    fit_conformal_risk_gate,
)
from experiments.information_anchor.downstream.mi_candidate_selector import (
    select_candidates,
)


GATE_FEATURE_NAMES = (
    "mi_official_distance_gap",
    "mi_high_distance_gap",
    "mi_student_utility_gap",
    "mi_profile_dispersion",
    "mi_official_overlap",
)


def _finite(name: str, values: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _row_zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(values.std(axis=1, keepdims=True), 1e-6)


def _mean_at(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices, axis=1).mean(axis=1)


def build_gate_features(
    official_distances: np.ndarray,
    high_mi_distances: np.ndarray,
    student_scores: np.ndarray,
    query_mi_scores: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Build query-level features from history-only candidate scores."""
    official = _finite("official_distances", official_distances, 2)
    high = _finite("high_mi_distances", high_mi_distances, 2)
    student = _finite("student_scores", student_scores, 2)
    query_mi = _finite("query_mi_scores", query_mi_scores, 2)
    if official.shape != high.shape or official.shape != student.shape:
        raise ValueError("candidate score matrices must have the same shape")
    if len(query_mi) != len(official):
        raise ValueError("query MI profiles must align with candidate scores")
    if not 1 <= int(top_k) <= official.shape[1]:
        raise ValueError("top_k must lie inside the candidate pool")

    k = int(top_k)
    fallback = np.argsort(official, axis=1, kind="stable")[:, :k]
    mi_objective = -_row_zscore(student) + _row_zscore(high)
    treatment = np.argsort(mi_objective, axis=1, kind="stable")[:, :k]
    overlap = np.zeros(len(official), dtype=np.float32)
    for row in range(len(overlap)):
        overlap[row] = len(set(fallback[row].tolist()) & set(treatment[row].tolist())) / k
    features = np.column_stack([
        _mean_at(official, treatment) - _mean_at(official, fallback),
        _mean_at(high, fallback) - _mean_at(high, treatment),
        _mean_at(student, treatment) - _mean_at(student, fallback),
        query_mi.std(axis=1),
        overlap,
    ]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("gate features are non-finite")
    return features


def route_candidate_indices(
    treatment_indices: np.ndarray,
    fallback_indices: np.ndarray,
    use_treatment: np.ndarray,
) -> np.ndarray:
    """Select treatment or fallback indices independently for each query."""
    treatment = np.asarray(treatment_indices, dtype=np.int64)
    fallback = np.asarray(fallback_indices, dtype=np.int64)
    route = np.asarray(use_treatment, dtype=bool)
    if treatment.shape != fallback.shape or treatment.ndim != 2:
        raise ValueError("treatment and fallback indices must share a 2-D shape")
    if route.shape != (len(treatment),):
        raise ValueError("use_treatment must align with candidate rows")
    return np.where(route[:, None], treatment, fallback).astype(np.int64)


def history_cosine_distances(
    query_histories: np.ndarray,
    candidate_histories: np.ndarray,
) -> np.ndarray:
    """Return lower-is-better cosine distances for a causal history pool."""
    query = _finite("query_histories", query_histories, 2)
    candidates = _finite("candidate_histories", candidate_histories, 3)
    if query.shape[0] != candidates.shape[0] or query.shape[1] != candidates.shape[2]:
        raise ValueError("query/candidate history shapes are incompatible")
    q_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    c_norm = candidates / np.maximum(np.linalg.norm(candidates, axis=2, keepdims=True), 1e-8)
    return (1.0 - np.einsum("nd,nkd->nk", q_norm, c_norm)).astype(np.float32)


def fit_selective_gate_from_discovery(
    query_histories: np.ndarray,
    candidate_histories: np.ndarray,
    high_mi_distances: np.ndarray,
    student_scores: np.ndarray,
    query_mi_scores: np.ndarray,
    candidate_utility: np.ndarray,
    *,
    top_k: int,
    alpha: float = 0.2,
    fit_fraction: float = 0.6,
    ridge_alpha: float = 10.0,
    mi_prior_weight: float = 1.0,
) -> tuple[ConformalRiskGate, dict[str, object]]:
    """Fit a conformal MI treatment gate using discovery-only utility labels."""
    official = history_cosine_distances(query_histories, candidate_histories)
    high = _finite("high_mi_distances", high_mi_distances, 2)
    student = _finite("student_scores", student_scores, 2)
    utility = _finite("candidate_utility", candidate_utility, 2)
    if official.shape != high.shape or official.shape != student.shape or official.shape != utility.shape:
        raise ValueError("discovery gate arrays must share the same shape")
    treatment = select_candidates(
        official, top_k=top_k, method="mi_prior",
        high_mi_distances=high, student_scores=student,
        mi_prior_weight=mi_prior_weight,
    ).indices
    fallback = select_candidates(official, top_k=top_k, method="official").indices
    features = build_gate_features(
        official, high, student, query_mi_scores, top_k=top_k,
    )
    treatment_gain = (
        _mean_at(utility, treatment) - _mean_at(utility, fallback)
    ).astype(np.float32)
    gate = fit_conformal_risk_gate(
        features,
        treatment_gain,
        alpha=alpha,
        fit_fraction=fit_fraction,
        ridge_alpha=ridge_alpha,
        feature_names=GATE_FEATURE_NAMES,
    )
    route = gate.route(features)
    diagnostics = {
        "gate_alpha": float(alpha),
        "gate_fit_fraction": float(fit_fraction),
        "gate_ridge_alpha": float(ridge_alpha),
        "gate_top_k": int(top_k),
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "gate_fit_count": int(gate.fit_count),
        "gate_calibration_count": int(gate.calibration_count),
        "gate_radius": float(gate.radius),
        "gate_fit_rmse": float(gate.fit_rmse),
        "gate_calibration_rmse": float(gate.calibration_rmse),
        "discovery_treatment_gain_mean": float(treatment_gain.mean()),
        "discovery_treatment_gain_positive_rate": float(np.mean(treatment_gain > 0.0)),
        "discovery_gate_use_treatment_rate": float(np.mean(route.use_treatment)),
        "discovery_query_future_loaded": True,
    }
    return gate, diagnostics


def gate_to_state(gate: ConformalRiskGate, *, top_k: int) -> dict[str, np.ndarray]:
    """Serialize a fitted gate into arrays accepted by discovery sidecars."""
    return {
        "selective_gate_scaler_mean": np.asarray(gate.scaler.mean_, dtype=np.float32),
        "selective_gate_scaler_scale": np.maximum(gate.scaler.scale_, 1e-12).astype(np.float32),
        "selective_gate_coef": np.asarray(gate.model.coef_, dtype=np.float32),
        "selective_gate_intercept": np.asarray(gate.model.intercept_, dtype=np.float32),
        "selective_gate_radius": np.asarray(gate.radius, dtype=np.float32),
        "selective_gate_top_k": np.asarray(int(top_k), dtype=np.int64),
    }


def route_from_state(
    state: Mapping[str, np.ndarray],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a serialized gate and return use flag, prediction, and lower bound."""
    values = _finite("gate_features", features, 2)
    mean = _finite("selective_gate_scaler_mean", state["selective_gate_scaler_mean"], 1)
    scale = _finite("selective_gate_scaler_scale", state["selective_gate_scaler_scale"], 1)
    coef = _finite("selective_gate_coef", state["selective_gate_coef"], 1)
    intercept = float(np.asarray(state["selective_gate_intercept"]).reshape(-1)[0])
    if values.shape[1] != len(mean) or len(mean) != len(coef):
        raise ValueError("serialized gate and feature widths differ")
    predicted = (((values - mean) / np.maximum(scale, 1e-12)) @ coef + intercept).astype(np.float32)
    lower = predicted - np.float32(np.asarray(state["selective_gate_radius"]).reshape(-1)[0])
    return (lower > 0.0), predicted, lower


__all__ = [
    "GATE_FEATURE_NAMES",
    "build_gate_features",
    "fit_selective_gate_from_discovery",
    "gate_to_state",
    "history_cosine_distances",
    "route_candidate_indices",
    "route_from_state",
]
