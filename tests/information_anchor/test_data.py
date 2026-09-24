from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.information_anchor.config import DataConfig
from experiments.information_anchor.data import (
    analysis_future_values,
    build_forecast_origins,
    make_windows,
    robust_history_normalize_future,
    select_value_columns,
    train_split_normalize_future,
    window_spans,
)
from experiments.information_anchor.stability import contiguous_leave_one_out_blocks


def _data_config(split: str) -> DataConfig:
    return DataConfig(
        dataset="ETTh1",
        source="thuml/Time-Series-Library",
        target="OT",
        seq_len=16,
        pred_len=4,
        split=split,
        origin_stride=2,
        max_origins=1000,
        strict_guard=True,
        train_end=100,
        validation_end=140,
        test_end=180,
        discovery_end=60,
        local_path="",
    )


def test_strict_analysis_splits_do_not_share_raw_points() -> None:
    discovery = build_forecast_origins(_data_config("discovery"))
    probe = build_forecast_origins(_data_config("probe"))
    validation = build_forecast_origins(_data_config("validation"))
    test = build_forecast_origins(_data_config("test"))

    discovery_span = window_spans(discovery, 16, 4)
    probe_span = window_spans(probe, 16, 4)
    validation_span = window_spans(validation, 16, 4)
    test_span = window_spans(test, 16, 4)
    assert discovery_span[:, 1].max() <= probe_span[:, 0].min()
    assert probe_span[:, 1].max() <= validation_span[:, 0].min()
    assert validation_span[:, 1].max() <= test_span[:, 0].min()


def test_length_sweep_uses_identical_intersection_origins() -> None:
    base = replace(
        _data_config("discovery"),
        seq_len=16,
        pred_len=4,
        origin_seq_len=32,
        origin_pred_len=12,
    )
    longer_context = replace(base, seq_len=32)
    longer_horizon = replace(base, pred_len=12)
    np.testing.assert_array_equal(
        build_forecast_origins(base), build_forecast_origins(longer_context)
    )
    np.testing.assert_array_equal(
        build_forecast_origins(base), build_forecast_origins(longer_horizon)
    )


def test_jackknife_blocks_are_contiguous_and_cover_all_samples() -> None:
    blocks = contiguous_leave_one_out_blocks(17, 4)
    np.testing.assert_array_equal(np.concatenate(blocks), np.arange(17))
    assert all(len(block) > 0 for block in blocks)


def test_history_normalization_is_applied_to_future_with_history_statistics() -> None:
    values = np.arange(100, dtype=np.float32)
    batch = make_windows(values, np.array([20, 30]), seq_len=10, pred_len=5)
    np.testing.assert_allclose(batch.history_normalized.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(batch.history_normalized.std(axis=1), 1.0, atol=1e-6)
    expected = (batch.future_raw - batch.history_mean[:, None]) / batch.history_scale[:, None]
    np.testing.assert_allclose(batch.future_normalized, expected)


def test_multivariate_windows_preserve_channel_axis() -> None:
    values = np.stack(
        [np.arange(100, dtype=np.float32), np.arange(100, dtype=np.float32) + 100],
        axis=1,
    )
    batch = make_windows(
        values,
        np.array([20, 30]),
        seq_len=10,
        pred_len=5,
        columns=("a", "b"),
    )
    assert batch.history_normalized.shape == (2, 10, 2)
    assert batch.future_normalized.shape == (2, 5, 2)
    assert batch.history_mean.shape == (2, 2)
    assert batch.history_scale.shape == (2, 2)
    assert batch.columns == ("a", "b")
    np.testing.assert_allclose(batch.history_normalized.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(batch.history_normalized.std(axis=1), 1.0, atol=1e-6)


def test_train_split_future_normalization_avoids_flat_history_explosion() -> None:
    values = np.concatenate(
        [np.zeros(20, dtype=np.float32), np.arange(1, 21, dtype=np.float32)]
    )
    batch = make_windows(values, np.array([20]), seq_len=16, pred_len=4)
    assert float(np.abs(batch.future_normalized).max()) > 1e6

    normalized = train_split_normalize_future(
        batch.future_raw,
        values,
        train_end=len(values),
    )
    assert np.isfinite(normalized).all()
    assert float(np.abs(normalized).max()) < 2.0
    np.testing.assert_allclose(
        analysis_future_values(
            batch,
            values,
            train_end=len(values),
            normalization="train_split",
        ),
        normalized,
    )


def test_robust_history_future_keeps_local_center_and_train_scale_floor() -> None:
    values = np.concatenate(
        [np.zeros(20, dtype=np.float32), np.arange(1, 21, dtype=np.float32)]
    )
    batch = make_windows(values, np.array([20]), seq_len=16, pred_len=4)
    normalized = robust_history_normalize_future(
        batch,
        values,
        train_end=len(values),
    )
    train_scale = values.std(dtype=np.float64)
    expected = batch.future_raw / train_scale
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        analysis_future_values(
            batch,
            values,
            train_end=len(values),
            normalization="robust_history_window",
        ),
        normalized,
    )


def test_select_value_columns_supports_multivariate_frame() -> None:
    import pandas as pd

    frame = pd.DataFrame({"date": [0, 1], "a": [1.0, 2.0], "OT": [3.0, 4.0]})
    config = replace(_data_config("discovery"), features="M")
    assert select_value_columns(frame, config) == ("a", "OT")
    custom = replace(config, features="custom", target_columns=("OT",))
    assert select_value_columns(frame, custom) == ("OT",)
