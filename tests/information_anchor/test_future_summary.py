from __future__ import annotations

import numpy as np

from experiments.information_anchor.targets.future_summary import build_future_summary


def test_future_summary_dimension_is_horizon_invariant() -> None:
    rng = np.random.default_rng(7)
    short = build_future_summary(rng.normal(size=(8, 96)), bins=16, spectral_bands=4)
    long = build_future_summary(rng.normal(size=(8, 336)), bins=16, spectral_bands=4)
    assert short.values.shape == (8, 26)
    assert long.values.shape == (8, 26)
    assert short.feature_names == long.feature_names
    assert np.isfinite(short.values).all()
    assert np.isfinite(long.values).all()


def test_multivariate_future_summary_concatenates_channel_features() -> None:
    rng = np.random.default_rng(11)
    summary = build_future_summary(
        rng.normal(size=(8, 96, 3)),
        bins=16,
        spectral_bands=4,
        channel_names=("a", "b", "OT"),
    )
    assert summary.values.shape == (8, 78)
    assert len(summary.feature_names) == 78
    assert summary.feature_names[0].startswith("a:")
    assert summary.feature_names[26].startswith("b:")
    assert summary.feature_names[52].startswith("OT:")
    assert np.isfinite(summary.values).all()
