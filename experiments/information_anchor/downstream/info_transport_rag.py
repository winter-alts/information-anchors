"""Horizon-conditioned MI residual transport for frozen forecasters.

The module is deliberately model-agnostic.  Candidate residuals are built by
the caller from historical donor futures and a frozen base forecast; query
future labels never enter this API.  MI acts on a ``[query, horizon block,
candidate]`` distance tensor, so it can route different forecast blocks to
different residual donors instead of only changing one scalar consensus weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class HorizonResidualTransportResult:
    """Auditable output of the history-only blockwise transport step."""

    correction: np.ndarray
    raw_correction: np.ndarray
    weights: np.ndarray
    normalized_distances: np.ndarray
    gate: np.ndarray


def _finite(name: str, values: np.ndarray, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite, got {array.shape}")
    return array


def _as_residuals(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = _finite("candidate_residuals", values)
    if array.ndim == 3:
        return array[..., None], True
    if array.ndim == 4:
        return array, False
    raise ValueError(
        "candidate_residuals must be [query,candidate,horizon] or "
        "[query,candidate,horizon,channel]"
    )


def _block_values(values: np.ndarray, block_size: int) -> tuple[int, int]:
    if int(block_size) != block_size or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    horizon = values.shape[2]
    if horizon % int(block_size):
        raise ValueError(
            f"horizon {horizon} must be divisible by block_size {int(block_size)}"
        )
    return horizon // int(block_size), int(block_size)


def _validate_gate(gate: np.ndarray | None, samples: int, blocks: int) -> np.ndarray:
    if gate is None:
        return np.ones((samples, blocks), dtype=np.float64)
    values = _finite("gate", gate, ndim=2)
    if values.shape != (samples, blocks):
        raise ValueError(
            f"gate must have shape {(samples, blocks)}, got {values.shape}"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("gate must lie in [0,1]")
    return values


def _reference_scale(
    distances: np.ndarray, reference_distances: np.ndarray | None
) -> np.ndarray:
    reference = distances if reference_distances is None else _finite(
        "reference_distances", reference_distances, ndim=3
    )
    if reference.shape[:2] != distances.shape[:2] or reference.shape[2] < 1:
        raise ValueError(
            "reference_distances must share [query,block] and contain candidates"
        )
    scale = np.median(reference, axis=2, keepdims=True)
    # A zero median can occur for a tied synthetic/control row.  The row-wise
    # epsilon keeps the operation finite while preserving exact equal weights.
    return np.maximum(scale, _EPS)


def _softmax_weights(normalized_distances: np.ndarray, temperature: float) -> np.ndarray:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = -normalized_distances / value
    logits -= logits.max(axis=2, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    return weights / np.maximum(weights.sum(axis=2, keepdims=True), _EPS)


def mi_concentration_gate(
    horizon_distances: np.ndarray,
    *,
    reference_distances: np.ndarray | None = None,
    threshold: float = 0.0,
    power: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return a history-only gate from per-block MI concentration.

    A uniform distance row has exactly zero concentration.  A caller may
    multiply this gate by a discovery-frozen real-vs-null MI margin; keeping
    that margin outside this primitive makes shuffled/null controls explicit.
    """

    distances = _finite("horizon_distances", horizon_distances, ndim=3)
    threshold = float(threshold)
    power = float(power)
    if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be finite and lie in [0,1)")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and positive")
    normalized = distances / _reference_scale(distances, reference_distances)
    weights = _softmax_weights(normalized, temperature)
    count = weights.shape[2]
    if count == 1:
        concentration = np.ones(weights.shape[:2], dtype=np.float64)
    else:
        entropy = -np.sum(weights * np.log(np.maximum(weights, _EPS)), axis=2)
        concentration = np.clip(1.0 - entropy / np.log(float(count)), 0.0, 1.0)
    active = np.clip(
        (concentration - threshold) / max(1.0 - threshold, _EPS), 0.0, 1.0
    )
    gate = np.power(active, power)
    gate[concentration <= threshold + 1e-12] = 0.0
    return gate.astype(np.float32)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Coordinate-wise weighted median over the candidate axis."""

    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    expanded = np.broadcast_to(weights[:, :, None, None], values.shape)
    sorted_weights = np.take_along_axis(expanded, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    index = np.argmax(cumulative >= 0.5, axis=1)
    return np.take_along_axis(sorted_values, index[:, None, ...], axis=1).squeeze(1)


def horizon_residual_transport(
    candidate_residuals: np.ndarray,
    horizon_distances: np.ndarray,
    *,
    block_size: int,
    reference_distances: np.ndarray | None = None,
    temperature: float = 1.0,
    estimator: str = "mean",
    gate: np.ndarray | None = None,
) -> HorizonResidualTransportResult:
    """Aggregate historical residuals with one MI kernel per horizon block.

    ``candidate_residuals`` may be ``[N,K,H]`` or ``[N,K,H,C]`` and
    ``horizon_distances`` must be ``[N,B,K]`` with ``H=B*block_size``.  The
    distance tensor is history-only; candidate residuals are legal only after
    causal retrieval because they contain historical donor futures.
    """

    residuals, squeeze_channel = _as_residuals(candidate_residuals)
    distances = _finite("horizon_distances", horizon_distances, ndim=3)
    blocks, block_size = _block_values(residuals, block_size)
    if distances.shape != (residuals.shape[0], blocks, residuals.shape[1]):
        raise ValueError(
            "horizon_distances must have shape "
            f"{(residuals.shape[0], blocks, residuals.shape[1])}, got {distances.shape}"
        )
    if np.any(distances < 0.0):
        raise ValueError("horizon_distances must be non-negative")
    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")

    normalized = distances / _reference_scale(distances, reference_distances)
    weights = _softmax_weights(normalized, temperature)
    block_gate = _validate_gate(gate, residuals.shape[0], blocks)
    raw = np.empty((residuals.shape[0], residuals.shape[2], residuals.shape[3]))
    for block in range(blocks):
        start = block * block_size
        stop = start + block_size
        values = residuals[:, :, start:stop, :]
        if estimator == "mean":
            estimate = np.einsum("nk,nk...->n...", weights[:, block], values)
        else:
            estimate = _weighted_median(values, weights[:, block])
        raw[:, start:stop, :] = estimate
    correction = raw * np.repeat(block_gate, block_size, axis=1)[:, :, None]
    if squeeze_channel:
        correction = correction[..., 0]
        raw = raw[..., 0]
    return HorizonResidualTransportResult(
        correction=correction.astype(np.float32),
        raw_correction=raw.astype(np.float32),
        weights=weights.astype(np.float32),
        normalized_distances=normalized.astype(np.float32),
        gate=block_gate.astype(np.float32),
    )


def apply_horizon_residual_transport(
    base_prediction: np.ndarray,
    candidate_residuals: np.ndarray,
    horizon_distances: np.ndarray,
    *,
    block_size: int,
    strength: float | np.ndarray = 1.0,
    reference_distances: np.ndarray | None = None,
    temperature: float = 1.0,
    estimator: str = "mean",
    gate: np.ndarray | None = None,
) -> np.ndarray:
    """Add the transported correction to a frozen base forecast."""

    base = _finite("base_prediction", base_prediction)
    result = horizon_residual_transport(
        candidate_residuals,
        horizon_distances,
        block_size=block_size,
        reference_distances=reference_distances,
        temperature=temperature,
        estimator=estimator,
        gate=gate,
    )
    if base.shape != result.correction.shape:
        raise ValueError(
            f"base_prediction shape {base.shape} does not match correction "
            f"shape {result.correction.shape}"
        )
    blocks = horizon_distances.shape[1]
    block_strength = np.asarray(strength, dtype=np.float64)
    if block_strength.ndim == 0:
        block_strength = np.full(blocks, float(block_strength))
    if block_strength.shape != (blocks,) or not np.isfinite(block_strength).all():
        raise ValueError(f"strength must be a finite scalar or shape {(blocks,)}")
    if np.any(block_strength < 0.0):
        raise ValueError("strength must be non-negative")
    expanded = np.repeat(block_strength, int(block_size))
    if base.ndim == 2:
        return (base + result.correction * expanded[None, :]).astype(np.float32)
    return (base + result.correction * expanded[None, :, None]).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(2021)
    residuals = rng.normal(size=(3, 4, 8)).astype(np.float32)
    distances = rng.uniform(size=(3, 2, 4)).astype(np.float32)
    result = horizon_residual_transport(residuals, distances, block_size=4)
    assert result.correction.shape == residuals.shape[:1] + residuals.shape[2:]
    assert np.allclose(result.weights.sum(axis=2), 1.0)
    print("info_transport_rag self-check passed")
