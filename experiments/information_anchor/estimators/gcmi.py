from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.covariance import OAS


def gaussian_copula_transform(values: np.ndarray) -> np.ndarray:
    """Apply a marginal rank-to-Gaussian transform feature by feature."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError(f"Expected [samples, features] with >=4 samples, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("GCMI input contains non-finite values.")
    ranks = np.empty_like(values, dtype=np.float64)
    for feature_index in range(values.shape[1]):
        ranks[:, feature_index] = rankdata(values[:, feature_index], method="average")
    uniforms = (ranks - 0.5) / values.shape[0]
    return ndtri(np.clip(uniforms, 1e-6, 1.0 - 1e-6))


def gcmi_from_gaussianized(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian-copula MI in nats using an OAS-shrunk joint covariance."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Incompatible GCMI inputs: x={x.shape}, y={y.shape}")
    joint = np.concatenate([x, y], axis=1)
    covariance = OAS(store_precision=False).fit(joint).covariance_
    split = x.shape[1]
    sign_x, logdet_x = np.linalg.slogdet(covariance[:split, :split])
    sign_y, logdet_y = np.linalg.slogdet(covariance[split:, split:])
    sign_joint, logdet_joint = np.linalg.slogdet(covariance)
    if sign_x <= 0 or sign_y <= 0 or sign_joint <= 0:
        raise RuntimeError("OAS covariance is not positive definite.")
    estimate = 0.5 * (logdet_x + logdet_y - logdet_joint)
    return float(max(estimate, 0.0))


def gcmi(x: np.ndarray, y: np.ndarray) -> float:
    return gcmi_from_gaussianized(
        gaussian_copula_transform(x), gaussian_copula_transform(y)
    )
