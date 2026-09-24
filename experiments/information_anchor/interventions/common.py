from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UnitSelection:
    label: str
    layer: int
    patch: int
    mi_z: float


@dataclass(frozen=True)
class PatchSetSelection:
    label: str
    strategy: str
    layer: int
    patches: tuple[int, ...]
    fraction: float
    repetition: int
    mi_z_mean: float
    mi_z_sum: float


@dataclass(frozen=True)
class FunctionalAnchorSelection:
    """Discovery-only anchor shared by semantic probes and interventions."""

    layer: int
    top_patch: int
    low_patch: int
    eligible_layers: tuple[int, ...]
    excluded_layers: tuple[int, ...]
    layer_anchor_score: float


def downstream_mixing_layer_indices(model_name: str, num_layers: int) -> tuple[int, ...]:
    """返回 post-layer history state 仍可影响预测读出的位置。"""
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    if model_name == "Chronos2":
        if num_layers < 2:
            raise ValueError("Chronos2 needs a non-final layer for history-to-query intervention.")
        # Chronos2 的 head 只读取 final-layer future query；post-final history 无下游 mixing 路径。
        return tuple(range(num_layers - 1))
    return tuple(range(num_layers))


def select_functional_anchor(
    mi_z: np.ndarray,
    layer_anchor_score: np.ndarray,
    model_name: str,
) -> FunctionalAnchorSelection:
    """Select one reachable layer, then matched high/low-MI patches within it.

    The rule uses only the discovery atlas and an architecture-level reachability
    declaration. Probe scores, interventions, and test labels never enter selection.
    """
    mi_z = np.asarray(mi_z, dtype=np.float64)
    layer_anchor_score = np.asarray(layer_anchor_score, dtype=np.float64).reshape(-1)
    if mi_z.ndim != 2 or not np.isfinite(mi_z).all():
        raise ValueError(f"mi_z must be finite [layer, patch], got {mi_z.shape}.")
    if layer_anchor_score.shape != (mi_z.shape[0],) or not np.isfinite(
        layer_anchor_score
    ).all():
        raise ValueError(
            "layer_anchor_score must have shape "
            f"{(mi_z.shape[0],)}, got {layer_anchor_score.shape}."
        )

    eligible_layers = downstream_mixing_layer_indices(model_name, mi_z.shape[0])
    eligible_array = np.asarray(eligible_layers, dtype=np.int64)
    reachable_scores = np.full(mi_z.shape[0], -np.inf, dtype=np.float64)
    reachable_scores[eligible_array] = layer_anchor_score[eligible_array]
    layer = int(np.argmax(reachable_scores))
    excluded_layers = tuple(
        sorted(set(range(mi_z.shape[0])).difference(eligible_layers))
    )
    return FunctionalAnchorSelection(
        layer=layer,
        top_patch=int(np.argmax(mi_z[layer])),
        low_patch=int(np.argmin(mi_z[layer])),
        eligible_layers=eligible_layers,
        excluded_layers=excluded_layers,
        layer_anchor_score=float(layer_anchor_score[layer]),
    )


def _uniform_patch_order(num_patches: int) -> np.ndarray:
    """返回从全历史均匀铺开的确定性 patch 顺序。"""
    if num_patches < 1:
        raise ValueError("num_patches must be positive.")
    # 使用逐轮最远点选择，使任意前缀都尽量覆盖完整历史，而非只保证单个预算均匀。
    selected: list[int] = []
    remaining = set(range(num_patches))
    while remaining:
        if not selected:
            candidate = num_patches - 1
        else:
            candidate = max(
                remaining,
                key=lambda patch: (
                    min(abs(patch - chosen) for chosen in selected),
                    patch,
                ),
            )
        selected.append(int(candidate))
        remaining.remove(candidate)
    return np.asarray(selected, dtype=np.int64)


def select_progressive_patch_sets(
    mi_z: np.ndarray,
    layer_anchor_score: np.ndarray,
    fractions: tuple[float, ...],
    *,
    layer: int | None = None,
    strategies: tuple[str, ...] = ("top", "bottom", "recent", "uniform", "random"),
    random_repetitions: int = 5,
    seed: int = 2021,
) -> list[PatchSetSelection]:
    """在最高聚合 MI 层构造嵌套的递增 patch 集合和同层控制。"""
    mi_z = np.asarray(mi_z, dtype=np.float64)
    layer_anchor_score = np.asarray(layer_anchor_score, dtype=np.float64).reshape(-1)
    if mi_z.ndim != 2 or not np.isfinite(mi_z).all():
        raise ValueError(f"mi_z must be finite [layer, patch], got {mi_z.shape}.")
    if layer_anchor_score.shape != (mi_z.shape[0],) or not np.isfinite(layer_anchor_score).all():
        raise ValueError(
            "layer_anchor_score must have shape "
            f"{(mi_z.shape[0],)}, got {layer_anchor_score.shape}."
        )
    if not fractions:
        raise ValueError("At least one progressive fraction is required.")
    normalized_fractions = tuple(float(value) for value in fractions)
    if any(not 0.0 < value <= 1.0 for value in normalized_fractions):
        raise ValueError("Progressive fractions must be in (0, 1].")
    if tuple(sorted(set(normalized_fractions))) != normalized_fractions:
        raise ValueError("Progressive fractions must be unique and increasing.")
    if random_repetitions < 0:
        raise ValueError("random_repetitions must be non-negative.")
    allowed_strategies = ("top", "bottom", "recent", "uniform", "random")
    normalized_strategies = tuple(str(value).strip().lower() for value in strategies)
    if not normalized_strategies:
        raise ValueError("At least one progressive strategy is required.")
    if len(set(normalized_strategies)) != len(normalized_strategies):
        raise ValueError("Progressive strategies must be unique.")
    unknown = sorted(set(normalized_strategies).difference(allowed_strategies))
    if unknown:
        raise ValueError(
            f"Unknown progressive strategies={unknown}; allowed={allowed_strategies}."
        )
    if "random" in normalized_strategies and random_repetitions < 1:
        raise ValueError("random_repetitions must be positive when random is selected.")

    layer = int(np.argmax(layer_anchor_score)) if layer is None else int(layer)
    if not 0 <= layer < mi_z.shape[0]:
        raise ValueError(f"layer must be in [0, {mi_z.shape[0]}), got {layer}.")
    layer_scores = mi_z[layer]
    num_patches = int(layer_scores.shape[0])
    deterministic_orders = {
        "top": np.argsort(-layer_scores, kind="stable"),
        "bottom": np.argsort(layer_scores, kind="stable"),
        "recent": np.arange(num_patches - 1, -1, -1, dtype=np.int64),
        "uniform": _uniform_patch_order(num_patches),
    }
    orders: list[tuple[str, int, np.ndarray]] = [
        (strategy, 0, deterministic_orders[strategy])
        for strategy in normalized_strategies
        if strategy != "random"
    ]
    rng = np.random.default_rng(seed)
    if "random" in normalized_strategies:
        for repetition in range(random_repetitions):
            orders.append(("random", repetition, rng.permutation(num_patches)))

    selections: list[PatchSetSelection] = []
    for strategy, repetition, order in orders:
        for fraction in normalized_fractions:
            count = min(num_patches, max(1, int(np.ceil(fraction * num_patches))))
            patches = tuple(sorted(int(value) for value in order[:count]))
            scores = layer_scores[np.asarray(patches, dtype=np.int64)]
            suffix = f"_r{repetition:02d}" if strategy == "random" else ""
            label = f"{strategy}{suffix}_f{int(round(fraction * 100)):03d}"
            selections.append(
                PatchSetSelection(
                    label=label,
                    strategy=strategy,
                    layer=layer,
                    patches=patches,
                    fraction=fraction,
                    repetition=repetition,
                    mi_z_mean=float(scores.mean()),
                    mi_z_sum=float(scores.sum()),
                )
            )
    return selections


def hierarchical_bootstrap_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Bootstrap values must have shape [donor_shift, sample].")
    rng = np.random.default_rng(seed)
    shift_count, sample_count = values.shape
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_shifts = rng.integers(0, shift_count, size=shift_count)
        shift_estimates = []
        for shift in sampled_shifts:
            sampled_examples = rng.integers(0, sample_count, size=sample_count)
            shift_estimates.append(values[shift, sampled_examples].mean())
        estimates[index] = np.mean(shift_estimates)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def exact_sign_flip_p(shift_means: np.ndarray) -> tuple[float, float]:
    values = np.asarray(shift_means, dtype=np.float64).reshape(-1)
    if len(values) > 16:
        raise ValueError("Exact sign-flip inference is limited to at most 16 donor shifts.")
    bit_patterns = np.arange(1 << len(values), dtype=np.uint32)[:, None]
    signs = 2.0 * ((bit_patterns >> np.arange(len(values))) & 1) - 1.0
    null = np.mean(signs * values[None, :], axis=1)
    observed = float(values.mean())
    two_sided = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    greater = float(np.mean(null >= observed - 1e-15))
    return two_sided, greater


def select_units(
    mi_z: np.ndarray,
    layer_mi_z: np.ndarray,
    patch_mi_z: np.ndarray,
    *,
    include_controls: bool = False,
    random_repetitions: int = 3,
    seed: int = 2021,
) -> list[UnitSelection]:
    """选择主干预单元，并可选加入位置/随机边界控制。

    控制单元固定在 MI top cell 所在层，避免把层位置差异混入 patch 选择效应；
    random 使用 discovery-independent deterministic seed，只用于附录边界分析。
    """
    mi_z = np.asarray(mi_z, dtype=np.float32)
    if mi_z.ndim != 2 or not np.isfinite(mi_z).all():
        raise ValueError(f"mi_z must be finite [layer, patch], got {mi_z.shape}.")
    top_layer, top_patch = np.unravel_index(int(np.argmax(mi_z)), mi_z.shape)
    aggregate_layer = int(np.argmax(layer_mi_z))
    aggregate_patch = int(np.argmax(patch_mi_z))
    candidates = [
        ("mi_top_cell", int(top_layer), int(top_patch)),
        ("low_mi_same_layer", int(top_layer), int(np.argmin(mi_z[top_layer]))),
        ("mi_aggregate_intersection", aggregate_layer, aggregate_patch),
        ("final_last", mi_z.shape[0] - 1, mi_z.shape[1] - 1),
    ]
    selected: list[UnitSelection] = []
    seen: set[tuple[int, int]] = set()
    for label, layer, patch in candidates:
        if (layer, patch) in seen:
            continue
        seen.add((layer, patch))
        selected.append(UnitSelection(label, layer, patch, float(mi_z[layer, patch])))

    if include_controls:
        # 最近 patch 是位置基线；random patch 只在同一层抽取，以匹配 hidden layer。
        control_candidates = [
            ("recent_same_layer", int(top_layer), int(mi_z.shape[1] - 1)),
        ]
        rng = np.random.default_rng(seed)
        available = np.arange(mi_z.shape[1], dtype=np.int64)
        forbidden = {int(top_patch), int(np.argmin(mi_z[top_layer]))}
        random_pool = np.asarray([item for item in available if int(item) not in forbidden])
        if len(random_pool) == 0:
            random_pool = available
        random_count = max(0, int(random_repetitions))
        if random_count:
            replace = random_count > len(random_pool)
            draws = rng.choice(random_pool, size=random_count, replace=replace)
            control_candidates.extend(
                (f"random_same_layer_{index}", int(top_layer), int(patch))
                for index, patch in enumerate(np.asarray(draws).reshape(-1))
            )
        for label, layer, patch in control_candidates:
            if patch < 0 or patch >= mi_z.shape[1] or (layer, patch) in seen:
                continue
            seen.add((layer, patch))
            selected.append(UnitSelection(label, layer, patch, float(mi_z[layer, patch])))
    return selected


def condition_summary(
    label: str,
    selection: UnitSelection,
    delta_mse: np.ndarray,
    delta_mae: np.ndarray,
    forecast_change_mse: np.ndarray,
    forecast_change_mae: np.ndarray,
    perturbation_rms: np.ndarray,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    mse_ci = hierarchical_bootstrap_ci(delta_mse, bootstrap_repetitions, seed)
    mae_ci = hierarchical_bootstrap_ci(delta_mae, bootstrap_repetitions, seed + 1)
    mse_p_two, mse_p_greater = exact_sign_flip_p(delta_mse.mean(axis=1))
    return {
        "condition": label,
        "layer_zero_based": selection.layer,
        "patch_zero_based": selection.patch,
        "mi_z": selection.mi_z,
        "mean_delta_mse": float(delta_mse.mean()),
        "delta_mse_ci95_lower": mse_ci[0],
        "delta_mse_ci95_upper": mse_ci[1],
        "mean_delta_mae": float(delta_mae.mean()),
        "delta_mae_ci95_lower": mae_ci[0],
        "delta_mae_ci95_upper": mae_ci[1],
        "mean_forecast_change_mse": float(forecast_change_mse.mean()),
        "mean_forecast_change_mae": float(forecast_change_mae.mean()),
        "mean_activation_perturbation_rms": float(perturbation_rms.mean()),
        "delta_mse_per_activation_rms": float(
            delta_mse.mean() / max(perturbation_rms.mean(), 1e-12)
        ),
        "positive_delta_mse_shift_fraction": float(np.mean(delta_mse.mean(axis=1) > 0)),
        "delta_mse_sign_flip_p_two_sided": mse_p_two,
        "delta_mse_sign_flip_p_greater": mse_p_greater,
    }
