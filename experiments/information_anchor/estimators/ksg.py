from __future__ import annotations

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors


def add_deterministic_jitter(values: np.ndarray, scale: float, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if scale <= 0:
        return values
    rng = np.random.default_rng(seed)
    feature_scale = np.maximum(values.std(axis=0, keepdims=True), 1.0)
    return values + rng.normal(size=values.shape) * feature_scale * scale


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """KSG-1 mutual-information estimator using the Chebyshev metric."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Incompatible KSG inputs: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("KSG input contains non-finite values.")
    n_samples = x.shape[0]
    if not 1 <= k < n_samples:
        raise ValueError(f"KSG requires 1 <= k < n_samples, got k={k}, n={n_samples}")

    joint = np.concatenate([x, y], axis=1)
    neighbors = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1)
    neighbors.fit(joint)
    # Query the fitted samples explicitly: column zero is self, so column k is the k-th other neighbour.
    distances = neighbors.kneighbors(joint, return_distance=True)[0][:, k]
    radii = np.nextafter(distances, np.zeros_like(distances))

    x_counts = KDTree(x, metric="chebyshev").query_radius(
        x, radii, count_only=True
    ) - 1
    y_counts = KDTree(y, metric="chebyshev").query_radius(
        y, radii, count_only=True
    ) - 1

    estimate = (
        digamma(k)
        + digamma(n_samples)
        - np.mean(digamma(x_counts + 1) + digamma(y_counts + 1))
    )
    return float(estimate)
