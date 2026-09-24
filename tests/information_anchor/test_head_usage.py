from __future__ import annotations

import numpy as np

from experiments.information_anchor.head_usage import select_head_aware_units


def test_head_aware_selection_returns_four_quadrants() -> None:
    mi = np.asarray(
        [
            [4.0, 3.0, -2.0, -3.0],
            [2.5, 1.0, -1.0, -4.0],
        ],
        dtype=np.float32,
    )
    head = np.asarray([3.0, -2.0, 2.0, -3.0], dtype=np.float32)
    selections = select_head_aware_units(mi, head)
    assert {item.quadrant for item in selections} == {
        "mi_high_head_high",
        "mi_high_head_low",
        "mi_low_head_high",
        "mi_low_head_low",
    }
    assert all(np.isfinite(item.head_z) for item in selections)


def test_head_aware_selection_is_deterministic_when_an_empty_bin_occurs() -> None:
    mi = np.ones((2, 3), dtype=np.float32)
    head = np.ones(3, dtype=np.float32)
    first = select_head_aware_units(mi, head)
    second = select_head_aware_units(mi, head)
    assert [(x.unit.layer, x.unit.patch) for x in first] == [
        (x.unit.layer, x.unit.patch) for x in second
    ]
