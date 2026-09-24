import numpy as np

from experiments.information_anchor.data import make_windows
from experiments.information_anchor.probes.linear import fit_ridge_probe, r2_per_target
from experiments.information_anchor.probes.run_linear import _semantic_splits
from experiments.information_anchor.probes.semantic import (
    build_global_semantic_targets,
    build_semantic_targets,
)


def test_semantic_targets_are_fixed_and_finite():
    rng = np.random.default_rng(3)
    future = rng.normal(size=(32, 96)).astype(np.float32)
    targets = build_semantic_targets(future)
    assert targets.values.shape == (32, 12)
    assert len(targets.names) == 12
    assert targets.names == (
        "level_q1",
        "level_q2",
        "level_q3",
        "level_q4",
        "global_change",
        "linear_trend",
        "mean_absolute_change",
        "diff_std",
        "roughness",
        "low_frequency_energy",
        "mid_frequency_energy",
        "high_frequency_energy",
    )
    np.testing.assert_allclose(targets.values[:, -3:].sum(axis=1), 1.0, atol=1e-6)


def test_change_semantics_are_not_duplicate_coordinates() -> None:
    rng = np.random.default_rng(31)
    future = rng.normal(size=(64, 96)).astype(np.float32)
    targets = build_semantic_targets(future)
    global_change = targets.values[:, targets.names.index("global_change")]
    local_motion = targets.values[:, targets.names.index("mean_absolute_change")]
    assert not np.allclose(global_change, local_motion)
    assert np.isfinite(targets.values).all()


def test_multivariate_semantic_targets_keep_channel_identity():
    """多变量 probe 目标应按变量展开，避免把变量维误当作时间维。"""
    rng = np.random.default_rng(4)
    future = rng.normal(size=(16, 96, 3)).astype(np.float32)
    targets = build_semantic_targets(future, ("a", "b", "c"))
    assert targets.values.shape == (16, 36)
    assert targets.names[0] == "a:level_q1"
    assert targets.names[-1] == "c:high_frequency_energy"
    assert np.isfinite(targets.values).all()


def test_global_semantics_are_channel_permutation_invariant():
    rng = np.random.default_rng(14)
    future = rng.normal(size=(16, 96, 3)).astype(np.float32)
    first = build_global_semantic_targets(future, ("a", "b", "c"))
    second = build_global_semantic_targets(future[:, :, (2, 0, 1)], ("c", "a", "b"))
    assert first.values.shape == (16, 12)
    assert len(first.names) == 12
    np.testing.assert_allclose(first.values, second.values, rtol=1e-6, atol=1e-6)


def test_ridge_probe_recovers_linear_signal():
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(8, 3))

    def sample(n):
        x = rng.normal(size=(n, 8)).astype(np.float32)
        y = (x @ weight + 0.01 * rng.normal(size=(n, 3))).astype(np.float32)
        return x, y

    train_x, train_y = sample(256)
    validation_x, validation_y = sample(128)
    test_x, test_y = sample(128)
    result = fit_ridge_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
        alphas=(1e-4, 1e-2, 1.0, 100.0),
    )
    assert float(result.test_r2.mean()) > 0.99
    assert result.alpha.shape == (3,)
    assert all(
        any(np.isclose(alpha, candidate) for candidate in (1e-4, 1e-2, 1.0, 100.0))
        for alpha in result.alpha
    )


def test_r2_marks_constant_test_semantics_as_not_estimable():
    """常数未来语义的 R² 应为 NA，不能被极小分母放大。"""
    target = np.column_stack([np.ones(32), np.arange(32, dtype=np.float64)])
    prediction = np.column_stack([target[:, 0] + 0.5, target[:, 1]])
    result = r2_per_target(target, prediction)
    assert np.isnan(result[0])
    assert np.isclose(result[1], 1.0)


def test_target_semantics_use_registered_analysis_normalization() -> None:
    values = np.stack(
        [np.arange(80, dtype=np.float32), np.arange(80, dtype=np.float32) ** 2],
        axis=1,
    )
    batch = make_windows(
        values,
        np.array([32, 48]),
        seq_len=16,
        pred_len=8,
        columns=("a", "OT"),
    )
    windows = {"train": batch, "validation": batch, "test": batch}
    analysis_futures = {
        split: np.zeros_like(batch.future_normalized) for split in windows
    }
    target, _ = _semantic_splits(
        windows,
        ("a", "OT"),
        1,
        "target",
        analysis_futures,
    )
    expected = build_semantic_targets(analysis_futures["train"][:, :, 1])
    np.testing.assert_allclose(target["train"].values, expected.values)
