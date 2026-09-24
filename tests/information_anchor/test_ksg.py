from __future__ import annotations

import numpy as np
import torch

from experiments.information_anchor.estimators.gcmi import gcmi

from experiments.information_anchor.estimators.ksg import add_deterministic_jitter, ksg_mi
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    all_temporal_circular_shift_offsets,
    benjamini_hochberg,
    circular_shift_offsets,
    temporal_circular_shift_offsets,
)


def test_ksg_detects_correlated_gaussian_signal() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(800, 2))
    y = x + 0.25 * rng.normal(size=(800, 2))
    independent = rng.permutation(y)
    correlated_mi = ksg_mi(x, y, k=5)
    independent_mi = ksg_mi(x, independent, k=5)
    assert correlated_mi > independent_mi + 0.5


def test_null_offsets_and_bh_are_deterministic() -> None:
    first = circular_shift_offsets(100, 10, min_shift=20, seed=3)
    second = circular_shift_offsets(100, 10, min_shift=20, seed=3)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 20)
    assert np.all(first <= 80)
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.9]))
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))


def test_temporal_offsets_respect_every_origin_pair() -> None:
    origins = np.array([0, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85, 99])
    offsets = temporal_circular_shift_offsets(
        origins, n_permutations=6, min_temporal_separation=20, seed=7
    )
    for offset in offsets:
        separation = np.min(np.abs(origins - np.roll(origins, int(offset))))
        assert separation >= 20


def test_all_temporal_offsets_are_exhaustive_and_deterministic() -> None:
    origins = np.array([0, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85, 99])
    expected = np.asarray(
        [
            shift
            for shift in range(1, len(origins))
            if np.min(np.abs(origins - np.roll(origins, shift))) >= 20
        ],
        dtype=np.int64,
    )
    first = all_temporal_circular_shift_offsets(origins, min_temporal_separation=20)
    second = all_temporal_circular_shift_offsets(origins, min_temporal_separation=20)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_gcmi_detects_monotonic_signal() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(600, 3))
    y = np.tanh(x[:, :2]) + 0.2 * rng.normal(size=(600, 2))
    assert gcmi(x, y) > gcmi(x, rng.permutation(y)) + 0.3


def test_torch_ksg_matches_cpu_exactly_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(17)
    x = rng.normal(size=(96, 3))
    y = x[:, :2] + 0.3 * rng.normal(size=(96, 2))
    shifts = np.array([13, 27, 41])
    expected_observed = ksg_mi(x, y, k=5)
    expected_null = np.asarray(
        [ksg_mi(x, np.roll(y, int(shift), axis=0), k=5) for shift in shifts]
    )
    estimator = TorchKSG(y, k=5, device="cuda:0", shift_batch_size=2)
    observed, null_values = estimator.estimate_observed_and_shifts(x, shifts)
    np.testing.assert_allclose(observed, expected_observed, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(null_values, expected_null, atol=1e-12, rtol=0.0)


def test_jitter_is_repeatable() -> None:
    values = np.zeros((10, 2), dtype=np.float32)
    np.testing.assert_allclose(
        add_deterministic_jitter(values, 1e-8, 9),
        add_deterministic_jitter(values, 1e-8, 9),
    )
