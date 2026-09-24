from __future__ import annotations

import numpy as np
import pytest

from experiments.information_anchor.metrics import (
    aggregate_metrics,
    fit_train_standard_scaler,
    forecast_metrics,
    forecast_change_metrics,
    per_origin_metrics,
    standardize_forecasts,
)


def test_train_scaler_uses_only_training_prefix_and_multivariate_axes() -> None:
    values = np.asarray(
        [[0.0, 10.0], [2.0, 14.0], [4.0, 18.0], [100.0, 1000.0]],
        dtype=np.float64,
    )
    scaler = fit_train_standard_scaler(values, train_end=3)
    np.testing.assert_allclose(scaler.mean, [2.0, 14.0])
    np.testing.assert_allclose(scaler.scale, np.std(values[:3], axis=0))


def test_tslib_metrics_average_horizon_and_variables_before_origins() -> None:
    train = np.stack([np.arange(20), 10.0 * np.arange(20)], axis=1)
    scaler = fit_train_standard_scaler(train, train_end=20)
    truth = np.zeros((2, 2, 2), dtype=np.float64)
    prediction = np.ones_like(truth)
    mse, mae = per_origin_metrics(prediction, truth, scaler)
    expected_mse = np.mean((1.0 / scaler.scale) ** 2)
    expected_mae = np.mean(1.0 / scaler.scale)
    np.testing.assert_allclose(mse, [expected_mse, expected_mse])
    np.testing.assert_allclose(mae, [expected_mae, expected_mae])
    assert aggregate_metrics(mse, mae) == {
        "mse": pytest.approx(expected_mse),
        "mae": pytest.approx(expected_mae),
    }


def test_single_channel_2d_arrays_are_supported() -> None:
    train = np.arange(10, dtype=np.float64)[:, None]
    scaler = fit_train_standard_scaler(train, train_end=10)
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    transformed = standardize_forecasts(values, scaler)
    assert transformed.shape == (2, 3, 1)
    change_mse, change_mae = forecast_change_metrics(values + 1.0, values, scaler)
    assert change_mse.shape == (2,)
    assert change_mae.shape == (2,)


def test_target_metrics_do_not_average_unrelated_variables() -> None:
    train = np.stack([np.arange(20), 10.0 * np.arange(20)], axis=1)
    scaler = fit_train_standard_scaler(train, train_end=20)
    truth = np.zeros((2, 3, 2), dtype=np.float64)
    prediction = np.zeros_like(truth)
    prediction[:, :, 1] = 10.0

    all_mse, all_mae = per_origin_metrics(prediction, truth, scaler)
    target_mse, target_mae = per_origin_metrics(
        prediction, truth, scaler, target_index=1
    )
    np.testing.assert_allclose(target_mse, 2.0 * all_mse)
    np.testing.assert_allclose(target_mae, 2.0 * all_mae)

    change_mse, change_mae = forecast_change_metrics(
        prediction, truth, scaler, target_index=1
    )
    np.testing.assert_allclose(change_mse, target_mse)
    np.testing.assert_allclose(change_mae, target_mae)


def test_target_metrics_reject_invalid_variable_index() -> None:
    scaler = fit_train_standard_scaler(np.ones((8, 2)), train_end=8)
    values = np.ones((2, 3, 2))
    with pytest.raises(ValueError, match="target_index"):
        per_origin_metrics(values, values, scaler, target_index=2)


def test_scaler_rejects_wrong_variable_count() -> None:
    scaler = fit_train_standard_scaler(np.ones((8, 2)), train_end=8)
    with pytest.raises(ValueError, match="variables"):
        standardize_forecasts(np.ones((2, 3, 1)), scaler)


def test_forecast_metrics_separates_tslib_and_history_window_scales() -> None:
    """训练段尺度是主指标，窗口尺度只能作为诊断输出。"""
    values = np.stack([np.arange(40), 10.0 * np.arange(40)], axis=1).astype(np.float64)
    scaler = fit_train_standard_scaler(values, train_end=20)
    truth = np.zeros((2, 3, 2), dtype=np.float64)
    prediction = np.ones_like(truth)
    history_scale = np.full((2, 2), 0.5, dtype=np.float64)
    metrics = forecast_metrics(prediction, truth, scaler, history_scale=history_scale)
    assert metrics["tslib_mse"] == pytest.approx(metrics["normalized_mse"])
    assert metrics["tslib_mae"] == pytest.approx(metrics["normalized_mae"])
    assert metrics["history_window_normalized_mse"] == pytest.approx(4.0)
    assert metrics["history_window_normalized_mae"] == pytest.approx(2.0)
