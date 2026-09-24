from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.information_anchor.adapters.timesfm25 import (
    aggregate_timesfm25_channel_hidden,
    replace_timesfm25_history_patches,
)
from experiments.information_anchor.interventions.common import (
    downstream_mixing_layer_indices,
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
    select_functional_anchor,
    select_progressive_patch_sets,
)
from experiments.information_anchor.interventions.multivariate_activation import (
    _moirai_intervention_module,
    _toto2_patch_set_replace,
    _toto2_intervention_module,
)


def test_exact_sign_flip_detects_consistent_positive_effect() -> None:
    two_sided, greater = exact_sign_flip_p(np.ones(8, dtype=np.float64))
    assert two_sided == 2.0 / 256.0
    assert greater == 1.0 / 256.0


def test_hierarchical_bootstrap_is_reproducible_and_contains_constant() -> None:
    values = np.full((4, 20), 3.5, dtype=np.float64)
    first = hierarchical_bootstrap_ci(values, repetitions=100, seed=7)
    second = hierarchical_bootstrap_ci(values, repetitions=100, seed=7)
    assert first == second == (3.5, 3.5)


def test_progressive_patch_sets_use_highest_aggregate_layer_and_are_nested() -> None:
    mi_z = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [5.0, 1.0, 4.0, 2.0, 3.0],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ],
        dtype=np.float64,
    )
    layer_anchor_score = np.asarray([1.0, 8.0, 3.0], dtype=np.float64)
    selections = select_progressive_patch_sets(
        mi_z,
        layer_anchor_score,
        (0.2, 0.4, 0.6),
        random_repetitions=2,
        seed=17,
    )

    assert {selection.layer for selection in selections} == {1}
    top = [selection for selection in selections if selection.strategy == "top"]
    assert [selection.patches for selection in top] == [(0,), (0, 2), (0, 2, 4)]
    assert set(top[0].patches) < set(top[1].patches) < set(top[2].patches)

    for repetition in (0, 1):
        random_sets = [
            selection
            for selection in selections
            if selection.strategy == "random" and selection.repetition == repetition
        ]
        assert set(random_sets[0].patches) < set(random_sets[1].patches)
        assert set(random_sets[1].patches) < set(random_sets[2].patches)


def test_progressive_patch_sets_reject_non_increasing_fractions() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        select_progressive_patch_sets(
            np.ones((2, 4), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            (0.5, 0.25),
        )


def test_chronos2_final_history_layer_is_not_functionally_reachable() -> None:
    assert downstream_mixing_layer_indices("Chronos2", 12) == tuple(range(11))
    assert downstream_mixing_layer_indices("Toto2", 12) == tuple(range(12))
    assert downstream_mixing_layer_indices("TimesFM2.5", 20) == tuple(range(20))


def test_timesfm25_channel_hidden_concatenates_same_time_patch() -> None:
    hidden = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    aggregated = aggregate_timesfm25_channel_hidden(hidden)

    assert aggregated.shape == (2, 4, 15)
    assert torch.equal(aggregated[1, 2], hidden[1, :, 2].reshape(-1))


def test_timesfm25_channel_hidden_supports_mean_std_stress_view() -> None:
    hidden = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    aggregated = aggregate_timesfm25_channel_hidden(hidden, "mean_std")

    assert aggregated.shape == (2, 4, 10)
    assert torch.allclose(aggregated[..., :5], hidden.mean(dim=1))
    assert torch.allclose(aggregated[..., 5:], hidden.std(dim=1, unbiased=False))


def test_timesfm25_replacement_preserves_cache_and_unselected_patches() -> None:
    hidden = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    donor = hidden + 100.0
    cache = object()

    replaced, rms = replace_timesfm25_history_patches(
        (hidden, cache), donor, (1, 3)
    )

    changed, returned_cache = replaced
    assert returned_cache is cache
    assert torch.equal(changed[:, (1, 3)], donor[:, (1, 3)])
    assert torch.equal(changed[:, (0, 2)], hidden[:, (0, 2)])
    assert torch.allclose(rms, torch.full((3,), 100.0))


def test_functional_anchor_excludes_unreachable_global_anchor_layer() -> None:
    mi_z = np.asarray(
        [
            [0.0, 2.0, 1.0],
            [1.0, 5.0, -1.0],
            [4.0, 8.0, 3.0],
        ],
        dtype=np.float64,
    )
    selection = select_functional_anchor(
        mi_z,
        np.asarray([1.0, 3.0, 10.0], dtype=np.float64),
        "Chronos2",
    )

    assert selection.layer == 1
    assert selection.top_patch == 1
    assert selection.low_patch == 2
    assert selection.eligible_layers == (0, 1)
    assert selection.excluded_layers == (2,)


def test_progressive_patch_sets_can_lock_the_shared_functional_layer() -> None:
    mi_z = np.asarray([[9.0, 8.0], [1.0, 3.0]], dtype=np.float64)
    selections = select_progressive_patch_sets(
        mi_z,
        np.asarray([20.0, 2.0], dtype=np.float64),
        (0.5,),
        layer=1,
        random_repetitions=0,
    )

    assert {selection.layer for selection in selections} == {1}
    top = next(selection for selection in selections if selection.strategy == "top")
    assert top.patches == (1,)


def test_progressive_patch_sets_can_select_v6_main_controls_only() -> None:
    selections = select_progressive_patch_sets(
        np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        (0.25, 0.5),
        strategies=("top", "bottom", "random"),
        random_repetitions=2,
        seed=3,
    )

    assert {selection.strategy for selection in selections} == {"top", "bottom", "random"}
    assert len(selections) == 8


def test_progressive_patch_sets_require_random_repetitions_for_random_strategy() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        select_progressive_patch_sets(
            np.ones((1, 4), dtype=np.float64),
            np.ones(1, dtype=np.float64),
            (0.25,),
            strategies=("top", "random"),
            random_repetitions=0,
        )


class _DummyTotoTransformer:
    @staticmethod
    def _if_variate_layer(layer: int) -> bool:
        return layer == 1


class _DummyTotoModel:
    transformer = _DummyTotoTransformer()


@pytest.mark.parametrize("layer", [0, 1])
def test_toto2_patch_set_replaces_only_selected_time_patches(layer: int) -> None:
    batch_size, n_channels, num_time, hidden_size = 2, 3, 4, 2
    if layer == 1:
        shape = (batch_size * num_time, n_channels, hidden_size)
    else:
        shape = (batch_size * n_channels, num_time, hidden_size)
    output = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    donor = output + 1000.0

    changed, rms = _toto2_patch_set_replace(
        output,
        donor,
        _DummyTotoModel(),
        batch_size,
        n_channels,
        (1, 3),
        layer,
    )

    if layer == 1:
        changed_view = changed.reshape(batch_size, num_time, n_channels, hidden_size)
        output_view = output.reshape(batch_size, num_time, n_channels, hidden_size)
        donor_view = donor.reshape(batch_size, num_time, n_channels, hidden_size)
        assert torch.equal(changed_view[:, (1, 3)], donor_view[:, (1, 3)])
        assert torch.equal(changed_view[:, (0, 2)], output_view[:, (0, 2)])
    else:
        changed_view = changed.reshape(batch_size, n_channels, num_time, hidden_size)
        output_view = output.reshape(batch_size, n_channels, num_time, hidden_size)
        donor_view = donor.reshape(batch_size, n_channels, num_time, hidden_size)
        assert torch.equal(changed_view[:, :, (1, 3)], donor_view[:, :, (1, 3)])
        assert torch.equal(changed_view[:, :, (0, 2)], output_view[:, :, (0, 2)])
    assert torch.allclose(rms, torch.full((batch_size,), 1000.0))


def test_toto2_final_layer_intervention_uses_post_norm_state() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    out_norm = torch.nn.Identity()
    transformer = type(
        "Transformer",
        (),
        {"layers": layers, "out_norm": out_norm},
    )()
    model = type("Model", (), {"transformer": transformer})()

    assert _toto2_intervention_module(model, 0) is layers[0]
    assert _toto2_intervention_module(model, 1) is out_norm


def test_toto2_post_norm_patch_set_uses_explicit_channel_time_axes() -> None:
    batch_size, n_channels, num_time, hidden_size = 2, 3, 4, 2
    shape = (batch_size, n_channels, num_time, hidden_size)
    output = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    donor = output + 1000.0

    changed, rms = _toto2_patch_set_replace(
        output,
        donor,
        _DummyTotoModel(),
        batch_size,
        n_channels,
        (1, 3),
        layer=1,
    )

    assert torch.equal(changed[:, :, (1, 3)], donor[:, :, (1, 3)])
    assert torch.equal(changed[:, :, (0, 2)], output[:, :, (0, 2)])
    assert torch.allclose(rms, torch.full((batch_size,), 1000.0))


def test_moirai2_final_layer_intervention_uses_encoder_final_state() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    encoder = torch.nn.Module()
    encoder.layers = layers
    module = type("Module", (), {"encoder": encoder})()
    forecaster = type("Forecaster", (), {"module": module})()

    assert _moirai_intervention_module(forecaster, 0) is layers[0]
    assert _moirai_intervention_module(forecaster, 1) is encoder
