from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FittedProjection:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray
    whiten_scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - self.scaler_mean) / self.scaler_scale
        centered = standardized - self.pca_mean
        projected = centered @ self.pca_components.T
        projected = projected / self.whiten_scale
        return projected.astype(np.float32)


def load_projection(path: str | Path) -> FittedProjection:
    """从 run 目录读取已锁定的 PCA，避免控制实验重新拟合投影。"""
    payload = np.load(Path(path))
    return FittedProjection(
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float32),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float32),
        pca_mean=np.asarray(payload["pca_mean"], dtype=np.float32),
        pca_components=np.asarray(payload["pca_components"], dtype=np.float32),
        explained_variance_ratio=np.asarray(
            payload["explained_variance_ratio"], dtype=np.float32
        ),
        whiten_scale=np.asarray(payload["whiten_scale"], dtype=np.float32),
    )


def fit_projection(
    values: np.ndarray,
    n_components: int,
    *,
    seed: int,
    standardize_features: bool,
) -> tuple[FittedProjection, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Projection expects a 2D matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Projection input contains non-finite values.")
    n_components = min(n_components, values.shape[0] - 1, values.shape[1])
    if n_components < 1:
        raise ValueError("Not enough samples/features for PCA.")

    if standardize_features:
        scaler = StandardScaler().fit(values)
        standardized = scaler.transform(values)
        scaler_mean = scaler.mean_.astype(np.float32)
        scaler_scale = np.maximum(scaler.scale_, 1e-12).astype(np.float32)
    else:
        scaler_mean = np.zeros(values.shape[1], dtype=np.float32)
        scaler_scale = np.ones(values.shape[1], dtype=np.float32)
        standardized = values

    solver = "randomized" if n_components < min(standardized.shape) else "full"
    pca = PCA(
        n_components=n_components,
        whiten=False,
        svd_solver=solver,
        random_state=seed,
    ).fit(standardized)
    whiten_scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12)).astype(np.float32)
    fitted = FittedProjection(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        whiten_scale=whiten_scale,
    )
    return fitted, fitted.transform(values)
