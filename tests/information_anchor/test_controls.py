from __future__ import annotations

import numpy as np

from experiments.information_anchor.controls import (
    cross_fitted_residuals,
    history_patch_statistics,
)


def test_history_patch_statistics_are_finite_and_have_expected_shape() -> None:
    rng = np.random.default_rng(19)
    histories = rng.normal(size=(20, 16, 3)).astype(np.float32)
    features = history_patch_statistics(
        histories,
        patch_len=4,
        patch_stride=4,
        num_patches=4,
    )
    assert features.shape == (20, 4, 15)
    assert np.isfinite(features).all()


def test_cross_fitted_residual_removes_linear_covariate_signal() -> None:
    rng = np.random.default_rng(23)
    covariates = rng.normal(size=(120, 3))
    target = np.stack(
        [covariates[:, 0] + 0.05 * rng.normal(size=120), 2.0 * covariates[:, 1]],
        axis=1,
    )
    residual = cross_fitted_residuals(target, covariates, folds=6, alpha=1e-3)
    correlation = np.corrcoef(residual[:, 0], covariates[:, 0])[0, 1]
    assert abs(correlation) < 0.2
    assert residual.shape == target.shape
