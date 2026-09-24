from __future__ import annotations

import numpy as np


def circular_shift_offsets(
    n_samples: int,
    n_permutations: int,
    min_shift: int,
    seed: int,
) -> np.ndarray:
    if n_samples < 4:
        raise ValueError("At least four samples are required for circular-shift nulls.")
    min_shift = max(1, int(min_shift))
    if 2 * min_shift >= n_samples:
        min_shift = max(1, n_samples // 4)
    candidates = np.arange(min_shift, n_samples - min_shift + 1, dtype=np.int64)
    candidates = candidates[candidates % n_samples != 0]
    if len(candidates) == 0:
        candidates = np.arange(1, n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    replace = n_permutations > len(candidates)
    return rng.choice(candidates, size=n_permutations, replace=replace)


def all_temporal_circular_shift_offsets(
    origins: np.ndarray,
    min_temporal_separation: int,
) -> np.ndarray:
    """Return every circular shift whose paired origins are sufficiently separated."""
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or len(origins) < 4:
        raise ValueError("At least four one-dimensional origins are required.")
    if not np.all(np.diff(origins) > 0):
        raise ValueError("Forecast origins must be strictly increasing.")
    if min_temporal_separation < 1:
        raise ValueError("min_temporal_separation must be positive.")
    candidates = [
        shift
        for shift in range(1, len(origins))
        if int(np.min(np.abs(origins - np.roll(origins, shift)))) >= min_temporal_separation
    ]
    if not candidates:
        raise ValueError(
            "No circular shift can satisfy the requested temporal separation. "
            "Use more origins or a wider chronological analysis interval."
        )
    return np.asarray(candidates, dtype=np.int64)


def temporal_circular_shift_offsets(
    origins: np.ndarray,
    n_permutations: int,
    min_temporal_separation: int,
    seed: int,
) -> np.ndarray:
    """Sample circular shifts whose matched origins are all far enough apart."""

    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or len(origins) < 4:
        raise ValueError("At least four one-dimensional origins are required.")
    if not np.all(np.diff(origins) > 0):
        raise ValueError("Forecast origins must be strictly increasing.")
    if min_temporal_separation < 1:
        raise ValueError("min_temporal_separation must be positive.")

    candidates = []
    for shift in range(1, len(origins)):
        paired = np.roll(origins, shift)
        realized = int(np.min(np.abs(origins - paired)))
        if realized >= min_temporal_separation:
            candidates.append(shift)
    if not candidates:
        raise ValueError(
            "No circular shift can satisfy the requested temporal separation. "
            "Use more origins or a wider chronological analysis interval."
        )

    rng = np.random.default_rng(seed)
    candidates_array = np.asarray(candidates, dtype=np.int64)
    replace = n_permutations > len(candidates_array)
    return rng.choice(candidates_array, size=n_permutations, replace=replace)


def robust_null_score(observed: float, null_values: np.ndarray) -> tuple[float, float]:
    null_values = np.asarray(null_values, dtype=np.float64)
    median = float(np.median(null_values))
    mad = float(np.median(np.abs(null_values - median)))
    denominator = 1.4826 * mad
    if denominator < 1e-12:
        denominator = max(float(null_values.std(ddof=1)), 1e-12)
    z_score = (observed - median) / denominator
    p_value = (1.0 + float(np.sum(null_values >= observed))) / (len(null_values) + 1.0)
    return float(z_score), float(p_value)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    flat = p_values.reshape(-1)
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.reshape(p_values.shape)
