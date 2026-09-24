"""Validation-only residual shrinkage and alpha-grid utilities for TS-RAG.

The residual correction is deliberately bounded and has an exact ``alpha=0``
fallback to the frozen TS-RAG prediction.  Keeping this logic in a small,
framework-light module makes the selection protocol reusable by both the
runner and offline validation scripts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
import torch


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha!r}")
    return value


def parse_alpha_grid(spec: str | Sequence[float]) -> tuple[float, ...]:
    """Parse a finite, sorted, duplicate-free grid in the closed unit interval."""

    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError("alpha grid cannot be empty")
        raw_values = [item.strip() for item in spec.split(",")]
        if any(not item for item in raw_values):
            raise ValueError(f"invalid alpha grid: {spec!r}")
        try:
            values = [_validate_alpha(float(item)) for item in raw_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid alpha grid: {spec!r}") from exc
    else:
        if len(spec) == 0:
            raise ValueError("alpha grid cannot be empty")
        try:
            values = [_validate_alpha(value) for value in spec]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid alpha grid") from exc

    unique = sorted(set(values))
    if len(unique) != len(values):
        raise ValueError("alpha grid contains duplicate values")
    return tuple(unique)


def build_detailed_alpha_grid(step: float = 0.0025) -> tuple[float, ...]:
    """Build a reproducible dense grid from zero through one, inclusive."""

    step = float(step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step must be finite and positive, got {step!r}")
    count = round(1.0 / step)
    if not math.isclose(count * step, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("step must divide the [0, 1] interval exactly")
    return parse_alpha_grid(tuple(round(index * step, 10) for index in range(count + 1)))


def select_validation_alpha(
    validation_mse: Mapping[float, float],
    tie_tolerance: float = 0.0,
) -> float:
    """Select the smallest alpha among validation-MSE ties."""

    if not validation_mse:
        raise ValueError("validation_mse cannot be empty")
    tie_tolerance = float(tie_tolerance)
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")

    checked: dict[float, float] = {}
    for alpha, mse in validation_mse.items():
        checked[_validate_alpha(alpha)] = float(mse)
        if not math.isfinite(float(mse)):
            raise ValueError(f"validation MSE must be finite, got {mse!r}")
    best_mse = min(checked.values())
    candidates = [alpha for alpha, mse in checked.items() if mse <= best_mse + tie_tolerance]
    return min(candidates)


def select_validation_alpha_by_metric(
    validation_metrics: Mapping[float, Mapping[str, float]],
    metric: str = "mse",
    tie_tolerance: float = 0.0,
    mae_tolerance: float = 0.0,
) -> float:
    """Select alpha from validation metrics using one declared objective."""

    if metric not in {"mse", "mae", "balanced", "mse_mae_guard"}:
        raise ValueError(
            "metric must be 'mse', 'mae', 'balanced', or 'mse_mae_guard', "
            f"got {metric!r}"
        )
    if not validation_metrics:
        raise ValueError("validation_metrics cannot be empty")
    mae_tolerance = float(mae_tolerance)
    if not math.isfinite(mae_tolerance) or mae_tolerance < 0.0:
        raise ValueError("mae_tolerance must be finite and non-negative")
    if metric == "mse_mae_guard":
        if 0.0 not in validation_metrics:
            raise ValueError("mse_mae_guard requires alpha=0.0 as the official fallback")
        baseline_mae = float(validation_metrics[0.0].get("mae", float("nan")))
        if not math.isfinite(baseline_mae):
            raise ValueError("mse_mae_guard requires a finite alpha=0 MAE")
        feasible = {
            alpha: values
            for alpha, values in validation_metrics.items()
            if "mse" in values
            and "mae" in values
            and math.isfinite(float(values["mse"]))
            and math.isfinite(float(values["mae"]))
            and float(values["mae"]) <= baseline_mae + mae_tolerance
        }
        if not feasible:
            return 0.0
        return select_validation_alpha(
            {alpha: float(values["mse"]) for alpha, values in feasible.items()},
            tie_tolerance=tie_tolerance,
        )
    if metric == "balanced":
        for alpha, values in validation_metrics.items():
            if "mse" not in values or "mae" not in values:
                raise ValueError("balanced selection requires both mse and mae")
        min_mse = min(float(values["mse"]) for values in validation_metrics.values())
        min_mae = min(float(values["mae"]) for values in validation_metrics.values())
        if min_mse <= 0.0 or min_mae <= 0.0:
            raise ValueError("balanced selection requires positive validation metrics")
        objective = {
            alpha: 0.5 * (float(values["mse"]) / min_mse + float(values["mae"]) / min_mae)
            for alpha, values in validation_metrics.items()
        }
        return select_validation_alpha(objective, tie_tolerance=tie_tolerance)

    objective = {}
    for alpha, values in validation_metrics.items():
        if metric not in values:
            raise ValueError(f"validation metrics are missing {metric!r}")
        objective[alpha] = float(values[metric])
    return select_validation_alpha(objective, tie_tolerance=tie_tolerance)


def _checked_metric_candidates(
    candidates: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    if not candidates:
        raise ValueError("candidates cannot be empty")

    checked: list[dict[str, float]] = []
    for candidate in candidates:
        if "mse" not in candidate or "mae" not in candidate:
            raise ValueError("each candidate must contain mse and mae")
        row = dict(candidate)
        mse = float(row["mse"])
        mae = float(row["mae"])
        if not math.isfinite(mse) or mse < 0.0:
            raise ValueError(f"candidate MSE must be finite and non-negative, got {mse!r}")
        if not math.isfinite(mae) or mae < 0.0:
            raise ValueError(f"candidate MAE must be finite and non-negative, got {mae!r}")
        checked.append(row)
    return checked


def select_lexicographic_candidate(
    candidates: Sequence[Mapping[str, float]],
    mse_tolerance: float = 0.005,
) -> dict[str, float]:
    """Select by validation MSE first, then MAE within a relative MSE tie.

    The tie set is defined relative to the best validation MSE:
    ``mse <= best_mse * (1 + mse_tolerance)``.  The input order is used as a
    final deterministic tie-break after MAE and MSE, so callers can provide a
    stable parameter ordering without introducing another selection criterion.
    """

    tolerance = float(mse_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")

    checked = _checked_metric_candidates(candidates)

    best_mse = min(float(row["mse"]) for row in checked)
    threshold = best_mse * (1.0 + tolerance)
    tie_set = [row for row in checked if float(row["mse"]) <= threshold]
    return min(enumerate(tie_set), key=lambda item: (
        float(item[1]["mae"]),
        float(item[1]["mse"]),
        item[0],
    ))[1]


def select_lexicographic_block_candidate(
    gamma_block_metrics: Mapping[
        float, Sequence[Mapping[float, Mapping[str, float]]]
    ],
    block_sizes: Sequence[int],
    mse_tolerance: float = 0.005,
) -> dict[str, object]:
    """Exhaustively select one gamma and one alpha per horizon block.

    ``gamma_block_metrics`` contains validation MSE/MAE for every alpha in
    every block, separately for each gamma.  Because the forecast metrics are
    averages over horizon entries, the full-horizon metric for an alpha vector
    is the block-size-weighted average of its block metrics.  This permits an
    exact ``gamma x alpha_1 x ... x alpha_B`` search without materializing all
    predictions or all Cartesian-product candidates.

    Selection follows the registered lexicographic protocol: first find the
    smallest full-horizon validation MSE, then minimize MAE among candidates
    whose MSE is within the relative tolerance.  Gamma and alpha values are
    sorted before enumeration, giving a deterministic final tie-break.
    """

    tolerance = float(mse_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mse_tolerance must be finite and non-negative")
    if not gamma_block_metrics:
        raise ValueError("gamma_block_metrics cannot be empty")
    if not block_sizes:
        raise ValueError("block_sizes cannot be empty")

    checked_sizes: tuple[int, ...] = tuple(int(size) for size in block_sizes)
    if any(int(size) != size or size <= 0 for size in block_sizes):
        raise ValueError("block_sizes must contain positive integers")
    total_size = sum(checked_sizes)

    checked: list[
        tuple[float, tuple[dict[float, dict[str, float]], ...]]
    ] = []
    for gamma, raw_blocks in sorted(gamma_block_metrics.items(), key=lambda item: float(item[0])):
        gamma_value = float(gamma)
        if not math.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
            raise ValueError(f"gamma must be finite and in [0, 1], got {gamma!r}")
        if len(raw_blocks) != len(checked_sizes):
            raise ValueError("each gamma must provide metrics for every block")
        blocks: list[dict[float, dict[str, float]]] = []
        for raw_metrics in raw_blocks:
            if not raw_metrics:
                raise ValueError("each horizon block must contain alpha metrics")
            block: dict[float, dict[str, float]] = {}
            for alpha, values in raw_metrics.items():
                alpha_value = _validate_alpha(alpha)
                if alpha_value in block:
                    raise ValueError("a horizon block contains duplicate alpha values")
                if "mse" not in values or "mae" not in values:
                    raise ValueError("each block metric must contain mse and mae")
                mse = float(values["mse"])
                mae = float(values["mae"])
                if not math.isfinite(mse) or mse < 0.0:
                    raise ValueError(f"block MSE must be finite and non-negative, got {mse!r}")
                if not math.isfinite(mae) or mae < 0.0:
                    raise ValueError(f"block MAE must be finite and non-negative, got {mae!r}")
                block[alpha_value] = {"mse": mse, "mae": mae}
            blocks.append(block)
        checked.append((gamma_value, tuple(blocks)))

    alpha_grids = tuple(
        tuple(sorted(block.keys()))
        for block in checked[0][1]
    )
    for _gamma, blocks in checked:
        if tuple(tuple(sorted(block.keys())) for block in blocks) != alpha_grids:
            raise ValueError("all gammas must use identical alpha grids per block")

    candidate_count = len(checked)
    for grid in alpha_grids:
        candidate_count *= len(grid)

    def _iter_candidates():
        for gamma, blocks in checked:
            for alphas in product(*alpha_grids):
                mse = sum(
                    size * blocks[index][alpha]["mse"]
                    for index, (size, alpha) in enumerate(zip(checked_sizes, alphas))
                ) / total_size
                mae = sum(
                    size * blocks[index][alpha]["mae"]
                    for index, (size, alpha) in enumerate(zip(checked_sizes, alphas))
                ) / total_size
                yield gamma, tuple(float(alpha) for alpha in alphas), float(mse), float(mae)

    best_mse = min(candidate[2] for candidate in _iter_candidates())
    threshold = best_mse * (1.0 + tolerance)
    selected: tuple[float, tuple[float, ...], float, float] | None = None
    tie_count = 0
    selected_key: tuple[float, float, int] | None = None
    for order, candidate in enumerate(_iter_candidates()):
        gamma, alphas, mse, mae = candidate
        if mse > threshold:
            continue
        tie_count += 1
        key = (mae, mse, order)
        if selected_key is None or key < selected_key:
            selected = candidate
            selected_key = key

    if selected is None:
        raise RuntimeError("lexicographic block selection produced no candidate")
    gamma, alphas, mse, mae = selected
    return {
        "gamma": gamma,
        "alphas": alphas,
        "mse": mse,
        "mae": mae,
        "best_mse": best_mse,
        "mse_threshold": threshold,
        "mse_tie_pair_count": tie_count,
        "candidate_count": candidate_count,
        "alpha_grid_by_block": alpha_grids,
        "block_sizes": checked_sizes,
    }


def select_minimax_regret_candidate(
    candidates: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Select the candidate minimizing the larger relative MSE/MAE regret.

    For each candidate, regret is measured against the best validation value
    of that metric over the complete candidate set.  The selected objective is
    ``max(mse / min_mse - 1, mae / min_mae - 1)``.  Input order is the final
    deterministic tie-break.
    """

    checked = _checked_metric_candidates(candidates)
    best_mse = min(float(row["mse"]) for row in checked)
    best_mae = min(float(row["mae"]) for row in checked)
    if best_mse <= 0.0 or best_mae <= 0.0:
        raise ValueError("minimax regret selection requires positive validation metrics")

    return min(enumerate(checked), key=lambda item: (
        max(
            float(item[1]["mse"]) / best_mse - 1.0,
            float(item[1]["mae"]) / best_mae - 1.0,
        ),
        float(item[1]["mse"]) / best_mse - 1.0,
        float(item[1]["mae"]) / best_mae - 1.0,
        item[0],
    ))[1]


def select_validation_alpha_blocks(
    official_prediction: np.ndarray,
    consensus: np.ndarray,
    confidence: np.ndarray,
    truth: np.ndarray,
    alpha_grid: Sequence[float],
    num_blocks: int,
    metric: str = "balanced",
    tie_tolerance: float = 0.0,
    mae_tolerance: float = 0.0,
) -> tuple[float, ...]:
    """Select one frozen alpha per contiguous forecast-horizon block.

    The block objectives are evaluated only on the supplied validation arrays.
    A small number of contiguous blocks gives the residual correction enough
    horizon flexibility without fitting one independent parameter per step.
    """

    if int(num_blocks) != num_blocks or int(num_blocks) <= 0:
        raise ValueError("num_blocks must be a positive integer")
    official_array = np.asarray(official_prediction)
    consensus_array = np.asarray(consensus)
    truth_array = np.asarray(truth)
    if official_array.ndim != 2:
        raise ValueError("forecast arrays must be rank-2")
    if int(num_blocks) > official_array.shape[1]:
        raise ValueError("num_blocks cannot exceed the forecast horizon")
    if official_array.shape != consensus_array.shape or official_array.shape != truth_array.shape:
        raise ValueError("forecast arrays must have identical shapes")

    selected: list[float] = []
    for horizon_block in np.array_split(np.arange(official_array.shape[1]), int(num_blocks)):
        block_metrics = evaluate_alpha_grid(
            official_array[:, horizon_block],
            consensus_array[:, horizon_block],
            confidence,
            truth_array[:, horizon_block],
            alpha_grid,
        )
        selected.append(
            select_validation_alpha_by_metric(
                block_metrics,
                metric=metric,
                tie_tolerance=tie_tolerance,
                mae_tolerance=mae_tolerance,
            )
        )
    return tuple(selected)


def select_validation_safe_alpha_blocks(
    official_prediction: np.ndarray,
    consensus: np.ndarray,
    confidence: np.ndarray,
    truth: np.ndarray,
    alpha_grid: Sequence[float],
    num_blocks: int,
    mae_tolerance: float = 0.0,
    min_mse_improvement: float = 0.0,
) -> tuple[float, ...]:
    """Select horizon-block strengths with an exact official fallback.

    Each block is allowed to move toward the MI consensus only when its
    validation MAE is no larger than the alpha-zero official forecast and its
    validation MSE is *strictly* smaller.  If no grid point satisfies both
    conditions, that block receives alpha zero.  Because the blocks partition
    the horizon, the resulting vector has the same MAE guard and a strict
    aggregate MSE gain whenever at least one block is selected.

    The selection is intentionally validation-only.  In particular, this
    helper never searches over test predictions and never uses a cross-block
    combination that could hide a bad horizon block behind a good one.
    """

    if int(num_blocks) != num_blocks or int(num_blocks) <= 0:
        raise ValueError("num_blocks must be a positive integer")
    mae_tolerance = float(mae_tolerance)
    min_mse_improvement = float(min_mse_improvement)
    if not math.isfinite(mae_tolerance) or mae_tolerance < 0.0:
        raise ValueError("mae_tolerance must be finite and non-negative")
    if not math.isfinite(min_mse_improvement) or min_mse_improvement < 0.0:
        raise ValueError("min_mse_improvement must be finite and non-negative")

    official_array = np.asarray(official_prediction)
    consensus_array = np.asarray(consensus)
    truth_array = np.asarray(truth)
    if official_array.ndim != 2:
        raise ValueError("forecast arrays must be rank-2")
    if int(num_blocks) > official_array.shape[1]:
        raise ValueError("num_blocks cannot exceed the forecast horizon")
    if official_array.shape != consensus_array.shape or official_array.shape != truth_array.shape:
        raise ValueError("forecast arrays must have identical shapes")

    parsed_grid = parse_alpha_grid(alpha_grid)
    if 0.0 not in parsed_grid:
        raise ValueError("safe validation selection requires alpha=0.0 as official fallback")

    selected: list[float] = []
    for horizon_block in np.array_split(np.arange(official_array.shape[1]), int(num_blocks)):
        block_metrics = evaluate_alpha_grid(
            official_array[:, horizon_block],
            consensus_array[:, horizon_block],
            confidence,
            truth_array[:, horizon_block],
            parsed_grid,
        )
        baseline = block_metrics[0.0]
        feasible = {
            alpha: values
            for alpha, values in block_metrics.items()
            if float(values["mae"]) <= float(baseline["mae"]) + mae_tolerance
            and float(values["mse"]) < float(baseline["mse"]) - min_mse_improvement
        }
        selected.append(
            select_validation_alpha(
                {alpha: values["mse"] for alpha, values in feasible.items()}
            )
            if feasible
            else 0.0
        )
    return tuple(selected)


def evaluate_alpha_grid(
    official_prediction: np.ndarray,
    consensus: np.ndarray,
    confidence: np.ndarray,
    truth: np.ndarray,
    alpha_grid: Sequence[float],
) -> dict[float, dict[str, float]]:
    """Evaluate a frozen residual correction on validation arrays.

    MSE is evaluated from the residual quadratic, avoiding a full prediction
    allocation for every candidate.  MAE is evaluated directly so the output
    remains useful for reporting and tie inspection.
    """

    official = np.asarray(official_prediction, dtype=np.float64)
    retrieved = np.asarray(consensus, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if official.ndim != 2 or retrieved.ndim != 2 or target.ndim != 2:
        raise ValueError("forecast arrays must be rank-2")
    if official.shape != retrieved.shape or official.shape != target.shape:
        raise ValueError("forecast arrays must have identical shapes")
    if confidence_array.ndim != 1 or confidence_array.shape[0] != official.shape[0]:
        raise ValueError("confidence must be rank-1 with one value per query")
    if not all(np.isfinite(value).all() for value in (official, retrieved, confidence_array, target)):
        raise ValueError("alpha-grid inputs must be finite")

    alphas = parse_alpha_grid(alpha_grid)
    base_error = official - target
    delta = confidence_array[:, None] * (retrieved - official)
    base_mse = float(np.mean(base_error * base_error))
    cross_mse = float(np.mean(base_error * delta))
    delta_mse = float(np.mean(delta * delta))
    result: dict[float, dict[str, float]] = {}
    for alpha in alphas:
        mse = base_mse + (2.0 * alpha * cross_mse) + (alpha * alpha * delta_mse)
        prediction_error = base_error + alpha * delta
        result[alpha] = {"mse": float(mse), "mae": float(np.mean(np.abs(prediction_error)))}
    return result


def apply_residual_shrink(
    official_prediction: torch.Tensor,
    consensus: torch.Tensor,
    confidence: torch.Tensor,
    alpha: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Move each forecast toward a history-only retrieved-future consensus.

    ``alpha`` may be a scalar or a frozen horizon-wise vector.  Vector values
    are broadcast over queries and are validated against the forecast horizon.
    """

    if official_prediction.ndim != 2 or consensus.ndim != 2:
        raise ValueError("official_prediction and consensus must be rank-2 tensors")
    if official_prediction.shape != consensus.shape:
        raise ValueError("official_prediction and consensus must have the same shape")
    if confidence.ndim != 1 or confidence.shape[0] != official_prediction.shape[0]:
        raise ValueError("confidence must be rank-1 with one value per query")
    if not all(torch.isfinite(value).all() for value in (official_prediction, consensus, confidence)):
        raise ValueError("residual shrink inputs must be finite")
    alpha_tensor = torch.as_tensor(alpha, device=official_prediction.device, dtype=official_prediction.dtype)
    if alpha_tensor.ndim == 0:
        if not torch.isfinite(alpha_tensor) or float(alpha_tensor) < 0.0:
            raise ValueError(f"alpha must be finite and non-negative, got {alpha!r}")
    elif alpha_tensor.ndim == 1:
        if alpha_tensor.shape[0] != official_prediction.shape[1]:
            raise ValueError("alpha vector must match the forecast horizon")
        if not torch.isfinite(alpha_tensor).all() or torch.any(alpha_tensor < 0.0):
            raise ValueError("alpha vector must be finite and non-negative")
        alpha_tensor = alpha_tensor.unsqueeze(0)
    else:
        raise ValueError("alpha must be a scalar or a horizon vector")
    return official_prediction + alpha_tensor * confidence[:, None] * (consensus - official_prediction)
