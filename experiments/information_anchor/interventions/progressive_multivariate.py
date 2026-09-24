from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.information_anchor.artifacts import (
    capture_run_context,
    create_run_dir,
    write_json,
)
from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.nulls import temporal_circular_shift_offsets
from experiments.information_anchor.interventions.common import (
    PatchSetSelection,
    downstream_mixing_layer_indices,
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
    select_functional_anchor,
    select_progressive_patch_sets,
)
from experiments.information_anchor.interventions.multivariate_activation import ModelRunner
from experiments.information_anchor.metrics import (
    fit_train_standard_scaler,
    forecast_change_metrics,
    per_origin_metrics,
)


@dataclass(frozen=True)
class ProgressiveCondition:
    label: str
    mode: str
    selection: PatchSetSelection
    replaced_patches: tuple[int, ...]


def _parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not fractions:
        raise argparse.ArgumentTypeError("At least one fraction is required.")
    if tuple(sorted(set(fractions))) != fractions or any(not 0.0 < item < 1.0 for item in fractions):
        raise argparse.ArgumentTypeError("Fractions must be unique, increasing, and in (0, 1).")
    return fractions


def _parse_shift_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Donor-shift indices must be integers.") from exc
    if not indices:
        raise argparse.ArgumentTypeError("At least one donor-shift index is required.")
    if tuple(sorted(set(indices))) != indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "Donor-shift indices must be unique, increasing, and non-negative."
        )
    return indices


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"top", "bottom", "recent", "uniform", "random"}
    if not strategies:
        raise argparse.ArgumentTypeError("At least one strategy is required.")
    if len(set(strategies)) != len(strategies):
        raise argparse.ArgumentTypeError("Strategies must be unique.")
    unknown = sorted(set(strategies).difference(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown strategies={unknown}; allowed={sorted(allowed)}."
        )
    return strategies


def _parse_manifest_methods(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    if not methods or len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("Manifest methods must be non-empty and unique.")
    return methods


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive same-layer multivariate activation replacement."
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=(0.125, 0.25, 0.375, 0.5),
    )
    parser.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=("top", "bottom", "random"),
        help=(
            "Comma-separated patch-set strategies. V6 main analysis uses "
            "top,bottom,random; recent/uniform are optional boundary controls."
        ),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "Discovery-only selector manifest CSV. When set, rows for this reference "
            "cell replace the built-in strategies. Use --selection-methods to choose rows."
        ),
    )
    parser.add_argument(
        "--selection-methods",
        help=(
            "Comma-separated manifest methods, e.g. marginal_ksg,marginal_gcmi,"
            "sequential_gcmi,recent,low_mi,random_r00,..."
        ),
    )
    parser.add_argument(
        "--allow-manifest-layer-override",
        action="store_true",
        help=(
            "Allow discovery-frozen manifest rows to select a reachable layer other "
            "than the registered functional layer. This is reserved for explicit "
            "same-temporal-position cross-layer controls."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--donor-shifts", type=int, default=8)
    parser.add_argument(
        "--donor-bank",
        type=Path,
        help=(
            "Optional prospective donor-bank NPZ containing fixed donor_histories "
            "[donor, sample, history, channel] and sampled_origins."
        ),
    )
    parser.add_argument(
        "--donor-shift-indices",
        type=_parse_shift_indices,
        help=(
            "Optional zero-based subset of the deterministic donor-shift list. "
            "Use disjoint subsets for lossless multi-GPU sharding."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-repetitions", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--include-sufficiency", action="store_true")
    parser.add_argument(
        "--keep-only",
        action="store_true",
        help="Run only latent-retention conditions, without removal conditions.",
    )
    parser.add_argument(
        "--include-corrupt-all",
        action="store_true",
        help="Add a zero-retention condition that donor-replaces every history patch.",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print a compact completion record instead of the full JSON payload.",
    )
    parser.add_argument(
        "--output-root",
        default="results/information_anchor_progressive_interventions",
    )
    return parser.parse_args()


def _build_conditions(
    selections: list[PatchSetSelection],
    num_patches: int,
    *,
    include_sufficiency: bool,
    keep_only: bool = False,
    include_corrupt_all: bool = False,
) -> list[ProgressiveCondition]:
    all_patches = set(range(num_patches))
    conditions: list[ProgressiveCondition] = []
    for selection in selections:
        if not keep_only:
            conditions.append(
                ProgressiveCondition(
                    label=f"remove__{selection.label}",
                    mode="remove",
                    selection=selection,
                    replaced_patches=selection.patches,
                )
            )
        if include_sufficiency or keep_only:
            complement = tuple(sorted(all_patches.difference(selection.patches)))
            if complement:
                conditions.append(
                    ProgressiveCondition(
                        label=f"keep__{selection.label}",
                        mode="keep",
                        selection=selection,
                        replaced_patches=complement,
                    )
                )
    if include_corrupt_all:
        if not selections:
            raise ValueError("Corrupt-all requires at least one selection to define the layer.")
        none = PatchSetSelection(
            label="none_f000",
            strategy="none",
            layer=selections[0].layer,
            patches=(),
            fraction=0.0,
            repetition=0,
            mi_z_mean=float("nan"),
            mi_z_sum=0.0,
        )
        conditions.append(
            ProgressiveCondition(
                label="keep__none_f000",
                mode="keep",
                selection=none,
                replaced_patches=tuple(range(num_patches)),
            )
        )
    return conditions


def _load_manifest_selections(
    path: Path,
    *,
    model: str,
    dataset_key: str,
    fractions: tuple[float, ...],
    mi_z: np.ndarray,
    default_layer: int,
    methods: tuple[str, ...] | None,
    allow_layer_override: bool = False,
) -> list[PatchSetSelection]:
    """Load fixed discovery-only patch sets for one reference cell."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed = set(methods) if methods is not None else None
    selected: list[PatchSetSelection] = []
    for row in rows:
        if row.get("model") != model or row.get("dataset_key") != dataset_key:
            continue
        method = str(row["method"])
        if allowed is not None and method not in allowed:
            continue
        fraction = float(row["fraction"])
        if fraction not in fractions:
            continue
        layer = int(row["functional_layer"])
        if layer != default_layer and not allow_layer_override:
            raise ValueError(
                f"Manifest layer {layer} disagrees with functional anchor {default_layer} "
                f"for {model}/{dataset_key}."
            )
        patches = tuple(int(item) for item in json.loads(row["selected_patches"]))
        if any(item < 0 or item >= mi_z.shape[1] for item in patches):
            raise ValueError(f"Manifest patch index out of range for {model}/{dataset_key}.")
        score = mi_z[layer, list(patches)] if patches else np.asarray([], dtype=np.float32)
        selected.append(
            PatchSetSelection(
                label=f"{method}_f{fraction:.3f}",
                strategy=method,
                layer=layer,
                patches=patches,
                fraction=fraction,
                repetition=0,
                mi_z_mean=float(np.mean(score)) if len(score) else 0.0,
                mi_z_sum=float(np.sum(score)),
            )
        )
    expected_methods = tuple(methods) if methods is not None else None
    if not selected:
        raise ValueError(f"No manifest rows found for {model}/{dataset_key}.")
    expected = len(expected_methods) * len(fractions) if expected_methods is not None else None
    if expected is not None and len(selected) != expected:
        raise ValueError(
            f"Manifest rows for {model}/{dataset_key}: expected {expected}, found {len(selected)}."
        )
    selected.sort(key=lambda item: (item.strategy, item.fraction))
    return selected


def _summarize_metric(
    values: np.ndarray,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, float]:
    lower, upper = hierarchical_bootstrap_ci(values, bootstrap_repetitions, seed)
    p_two, p_greater = exact_sign_flip_p(values.mean(axis=1))
    return {
        "mean": float(values.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "positive_shift_fraction": float(np.mean(values.mean(axis=1) > 0)),
        "sign_flip_p_two_sided": p_two,
        "sign_flip_p_greater": p_greater,
    }


def _condition_record(
    condition: ProgressiveCondition,
    metrics: dict[str, np.ndarray],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "condition": condition.label,
        "mode": condition.mode,
        "strategy": condition.selection.strategy,
        "random_repetition": condition.selection.repetition,
        "fraction": condition.selection.fraction,
        "layer_zero_based": condition.selection.layer,
        "selected_patch_count": len(condition.selection.patches),
        "replaced_patch_count": len(condition.replaced_patches),
        "selected_patches": list(condition.selection.patches),
        "replaced_patches": list(condition.replaced_patches),
        "selected_mi_z_mean": condition.selection.mi_z_mean,
        "selected_mi_z_sum": condition.selection.mi_z_sum,
    }
    for offset, name in enumerate(
        (
            "target_delta_mse",
            "target_delta_mae",
            "target_forecast_change_mse",
            "target_forecast_change_mae",
            "all_delta_mse",
            "all_delta_mae",
            "all_forecast_change_mse",
            "all_forecast_change_mae",
        )
    ):
        summary = _summarize_metric(
            metrics[name],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + offset,
        )
        for key, value in summary.items():
            record[f"{name}_{key}"] = value
    perturbation_rms = float(metrics["perturbation_rms"].mean())
    record["mean_activation_perturbation_rms"] = perturbation_rms
    record["all_delta_mse_per_activation_rms"] = float(
        metrics["all_delta_mse"].mean() / max(perturbation_rms, 1e-12)
    )
    return record


def _curve_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    modes = sorted({str(record["mode"]) for record in records})
    strategies = sorted({str(record["strategy"]) for record in records})
    for mode in modes:
        for strategy in strategies:
            subset = [
                record
                for record in records
                if record["mode"] == mode and record["strategy"] == strategy
            ]
            if not subset:
                continue
            fractions = sorted({float(record["fraction"]) for record in subset})
            metric_name = "all_forecast_change_mse_mean"
            values = np.asarray(
                [
                    np.mean(
                        [
                            float(record[metric_name])
                            for record in subset
                            if float(record["fraction"]) == fraction
                        ]
                    )
                    for fraction in fractions
                ],
                dtype=np.float64,
            )
            x = np.asarray(fractions, dtype=np.float64)
            auc = (
                float(np.trapz(values, x) / max(x[-1] - x[0], 1e-12))
                if len(x) > 1
                else float(values[0])
            )
            initial_slope = (
                float((values[1] - values[0]) / (x[1] - x[0]))
                if len(x) > 1
                else float("nan")
            )
            summaries.append(
                {
                    "mode": mode,
                    "strategy": strategy,
                    "metric": metric_name,
                    "fractions": fractions,
                    "values": values.tolist(),
                    "aopc": auc,
                    "initial_slope": initial_slope,
                }
            )
    return summaries


def _paired_contrasts(
    conditions: list[ProgressiveCondition],
    stacked: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """计算同 donor、同 origin 下 top-MI 相对同层控制的配对效应。"""
    if not any(
        condition.selection.strategy == "top" for condition in conditions
    ):
        return []
    condition_by_key = {
        (
            condition.mode,
            condition.selection.strategy,
            condition.selection.fraction,
            condition.selection.repetition,
        ): condition
        for condition in conditions
    }
    contrasts: list[dict[str, Any]] = []
    modes = sorted({condition.mode for condition in conditions})
    fractions = sorted(
        {
            condition.selection.fraction
            for condition in conditions
            if condition.selection.strategy == "top"
        }
    )
    contrast_metrics = (
        "target_forecast_change_mse",
        "all_forecast_change_mse",
        "target_delta_mse",
        "all_delta_mse",
    )
    for mode_index, mode in enumerate(modes):
        for fraction_index, fraction in enumerate(fractions):
            top_condition = condition_by_key[(mode, "top", fraction, 0)]
            controls: list[tuple[str, list[ProgressiveCondition]]] = []
            for strategy in ("bottom", "recent", "uniform"):
                control = condition_by_key.get((mode, strategy, fraction, 0))
                if control is not None:
                    controls.append((strategy, [control]))
            random_conditions = [
                condition
                for condition in conditions
                if condition.mode == mode
                and condition.selection.strategy == "random"
                and condition.selection.fraction == fraction
            ]
            if random_conditions:
                controls.append(("random_mean", random_conditions))

            for control_index, (control_name, control_conditions) in enumerate(controls):
                record: dict[str, Any] = {
                    "mode": mode,
                    "fraction": fraction,
                    "contrast": f"top_minus_{control_name}",
                    "control_repetitions": len(control_conditions),
                }
                for metric_index, metric in enumerate(contrast_metrics):
                    top_values = stacked[top_condition.label][metric]
                    control_values = np.mean(
                        np.stack(
                            [stacked[condition.label][metric] for condition in control_conditions],
                            axis=0,
                        ),
                        axis=0,
                    )
                    difference = top_values - control_values
                    summary = _summarize_metric(
                        difference,
                        bootstrap_repetitions=bootstrap_repetitions,
                        seed=(
                            seed
                            + mode_index * 10_000
                            + fraction_index * 1_000
                            + control_index * 100
                            + metric_index
                        ),
                    )
                    for key, value in summary.items():
                        record[f"{metric}_{key}"] = value
                    record[f"{metric}_top_mean"] = float(top_values.mean())
                    record[f"{metric}_control_mean"] = float(control_values.mean())
                contrasts.append(record)
    return contrasts


def main() -> None:
    args = _parse_args()
    if not 1 <= args.donor_shifts <= 16:
        raise ValueError("donor_shifts must be in [1, 16] for exact sign-flip inference.")
    if args.max_samples < 1 or args.batch_size < 1 or args.random_repetitions < 0:
        raise ValueError("Sample, batch, and random-repetition counts are invalid.")

    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if config.future_target.scope != "all":
        raise ValueError("Progressive main analysis requires future_target.scope='all'.")
    if args.device:
        config = replace(config, model=replace(config.model, device=args.device))

    reference = np.load(reference_dir / "mi_results.npz")
    mi_z = np.asarray(reference["mi_z"], dtype=np.float32)
    layer_mi_z = np.asarray(reference["layer_mi_z"], dtype=np.float32)
    layer_anchor_score = (
        np.asarray(reference["layer_anchor_score"], dtype=np.float32)
        if "layer_anchor_score" in reference.files
        else mi_z.mean(axis=1).astype(np.float32)
    )
    functional_anchor = select_functional_anchor(
        mi_z, layer_anchor_score, config.model.name
    )
    manifest_methods = _parse_manifest_methods(args.selection_methods)
    if args.selection_manifest is not None:
        selections = _load_manifest_selections(
            args.selection_manifest,
            model=config.model.name,
            dataset_key=config.data.dataset.lower(),
            fractions=args.fractions,
            mi_z=mi_z,
            default_layer=functional_anchor.layer,
            methods=manifest_methods,
            allow_layer_override=args.allow_manifest_layer_override,
        )
        reachable_layers = set(downstream_mixing_layer_indices(config.model.name, mi_z.shape[0]))
        invalid_layers = sorted({selection.layer for selection in selections}.difference(reachable_layers))
        if invalid_layers:
            raise ValueError(
                f"Manifest selects unreachable layers {invalid_layers} for {config.model.name}; "
                f"reachable layers are {sorted(reachable_layers)}."
            )
        selected_strategy_names = tuple(dict.fromkeys(item.strategy for item in selections))
    else:
        selections = select_progressive_patch_sets(
            mi_z,
            layer_anchor_score,
            args.fractions,
            layer=functional_anchor.layer,
            strategies=args.strategies,
            random_repetitions=args.random_repetitions,
            seed=config.mi.seed + 97_000,
        )
        selected_strategy_names = args.strategies
    conditions = _build_conditions(
        selections,
        mi_z.shape[1],
        include_sufficiency=args.include_sufficiency,
        keep_only=args.keep_only,
        include_corrupt_all=args.include_corrupt_all,
    )
    top_layer = functional_anchor.layer
    selected_layers = tuple(sorted({condition.selection.layer for condition in conditions}))
    global_anchor_layer = int(np.argmax(layer_anchor_score))

    test_config = replace(config, data=replace(config.data, split="test"))
    test_origins = build_forecast_origins(test_config.data)
    sample_indices = np.linspace(0, len(test_origins) - 1, min(args.max_samples, len(test_origins)))
    sample_indices = np.unique(np.rint(sample_indices).astype(np.int64))
    sampled_origins = test_origins[sample_indices]
    if args.donor_bank is None:
        all_shifts = temporal_circular_shift_offsets(
            sampled_origins,
            args.donor_shifts,
            min_temporal_separation=config.data.seq_len + config.data.pred_len,
            seed=config.mi.seed + 80_000,
        )
        if args.donor_shift_indices is None:
            shift_indices = np.arange(len(all_shifts), dtype=np.int64)
        else:
            shift_indices = np.asarray(args.donor_shift_indices, dtype=np.int64)
            if int(shift_indices[-1]) >= len(all_shifts):
                raise ValueError(
                    f"donor_shift_indices={shift_indices.tolist()} exceed the "
                    f"{len(all_shifts)} deterministic shifts."
                )
        shifts = all_shifts[shift_indices]

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    if len(columns) <= 1:
        raise ValueError("Progressive multivariate intervention requires more than one channel.")
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    target_index = columns.index(config.data.target)
    scaler = fit_train_standard_scaler(values, config.data.train_end)
    windows = make_windows(
        values,
        sampled_origins,
        config.data.seq_len,
        config.data.pred_len,
        columns=columns,
    )
    histories = np.asarray(windows.history_raw, dtype=np.float32)
    futures = np.asarray(windows.future_raw, dtype=np.float32)
    donor_bank: np.ndarray | None = None
    if args.donor_bank is not None:
        with np.load(args.donor_bank) as bank:
            if "donor_histories" not in bank.files or "sampled_origins" not in bank.files:
                raise ValueError("Donor bank must contain donor_histories and sampled_origins.")
            donor_bank = np.asarray(bank["donor_histories"], dtype=np.float32)
            bank_origins = np.asarray(bank["sampled_origins"], dtype=np.int64)
        if donor_bank.ndim != 4 or donor_bank.shape[1:] != histories.shape:
            raise ValueError(
                f"Donor bank shape {donor_bank.shape} does not match target histories {histories.shape}."
            )
        if not np.array_equal(bank_origins, sampled_origins):
            raise ValueError("Donor bank sampled origins do not match this intervention run.")
        if donor_bank.shape[0] > 16:
            raise ValueError("At most 16 donor-bank replicates are supported for exact sign-flip inference.")
        all_shifts = np.arange(donor_bank.shape[0], dtype=np.int64)
        if args.donor_shift_indices is None:
            shift_indices = all_shifts.copy()
        else:
            shift_indices = np.asarray(args.donor_shift_indices, dtype=np.int64)
            if int(shift_indices[-1]) >= len(all_shifts):
                raise ValueError(
                    f"donor_shift_indices={shift_indices.tolist()} exceed donor-bank shifts."
                )
        shifts = all_shifts[shift_indices]
    runner = ModelRunner(config, histories.shape[-1])

    baseline_parts = []
    with torch.no_grad():
        for start in range(0, len(histories), args.batch_size):
            stop = min(len(histories), start + args.batch_size)
            baseline_parts.append(runner.forecast(histories[start:stop]))
    baseline = np.concatenate(baseline_parts, axis=0).astype(np.float32)
    target_baseline_mse, target_baseline_mae = per_origin_metrics(
        baseline, futures, scaler, target_index=target_index
    )
    all_baseline_mse, all_baseline_mae = per_origin_metrics(baseline, futures, scaler)

    check_stop = min(args.batch_size, len(histories))
    noop_differences: dict[int, float] = {}
    with torch.no_grad():
        for layer in selected_layers:
            noop = runner.noop(histories[:check_stop], layer)
            noop_differences[layer] = float(
                np.max(np.abs(noop - baseline[:check_stop]))
            )
    noop_max_abs_difference = max(noop_differences.values())
    if noop_max_abs_difference > 1e-4:
        raise RuntimeError(f"Clone-only hook changed forecasts by {noop_max_abs_difference:.6g}.")

    metric_names = (
        "target_delta_mse",
        "target_delta_mae",
        "target_forecast_change_mse",
        "target_forecast_change_mae",
        "all_delta_mse",
        "all_delta_mae",
        "all_forecast_change_mse",
        "all_forecast_change_mae",
        "perturbation_rms",
        "channel_forecast_change_mse",
    )
    collected = {
        condition.label: {name: [] for name in metric_names}
        for condition in conditions
    }

    with torch.no_grad():
        for shift_index, shift in enumerate(shifts):
            donor_histories = (
                donor_bank[int(shift)]
                if donor_bank is not None
                else np.roll(histories, int(shift), axis=0)
            )
            predictions = {condition.label: [] for condition in conditions}
            rms_values = {condition.label: [] for condition in conditions}
            for start in range(0, len(histories), args.batch_size):
                stop = min(len(histories), start + args.batch_size)
                donor_states = {
                    layer: runner.capture_donor_state(
                        donor_histories[start:stop], layer
                    )
                    for layer in selected_layers
                }
                for condition in conditions:
                    condition_layer = condition.selection.layer
                    prediction, rms = runner.patched_forecast_set(
                        histories[start:stop],
                        donor_histories[start:stop],
                        layer=condition_layer,
                        patches=condition.replaced_patches,
                        donor_state=donor_states[condition_layer],
                    )
                    predictions[condition.label].append(prediction)
                    rms_values[condition.label].append(rms)
            for condition in conditions:
                prediction = np.concatenate(predictions[condition.label], axis=0).astype(np.float32)
                rms = np.concatenate(rms_values[condition.label], axis=0).astype(np.float32)
                target_mse, target_mae = per_origin_metrics(
                    prediction, futures, scaler, target_index=target_index
                )
                target_change_mse, target_change_mae = forecast_change_metrics(
                    prediction, baseline, scaler, target_index=target_index
                )
                all_mse, all_mae = per_origin_metrics(prediction, futures, scaler)
                all_change_mse, all_change_mae = forecast_change_metrics(
                    prediction, baseline, scaler
                )
                channel_change_mse = np.stack(
                    [
                        forecast_change_metrics(
                            prediction,
                            baseline,
                            scaler,
                            target_index=channel_index,
                        )[0]
                        for channel_index in range(len(columns))
                    ],
                    axis=-1,
                )
                destination = collected[condition.label]
                destination["target_delta_mse"].append(target_mse - target_baseline_mse)
                destination["target_delta_mae"].append(target_mae - target_baseline_mae)
                destination["target_forecast_change_mse"].append(target_change_mse)
                destination["target_forecast_change_mae"].append(target_change_mae)
                destination["all_delta_mse"].append(all_mse - all_baseline_mse)
                destination["all_delta_mae"].append(all_mae - all_baseline_mae)
                destination["all_forecast_change_mse"].append(all_change_mse)
                destination["all_forecast_change_mae"].append(all_change_mae)
                destination["perturbation_rms"].append(rms)
                destination["channel_forecast_change_mse"].append(channel_change_mse)
            print(
                f"model={config.model.name} donor_shift={shift_index + 1}/{len(shifts)} "
                f"global_index={int(shift_indices[shift_index])} "
                f"offset={int(shift)} complete",
                flush=True,
            )

    stacked = {
        label: {name: np.stack(values, axis=0) for name, values in metrics.items()}
        for label, metrics in collected.items()
    }
    records = [
        _condition_record(
            condition,
            stacked[condition.label],
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=config.mi.seed + 90_000 + index * 20,
        )
        for index, condition in enumerate(conditions)
    ]
    for condition, record in zip(conditions, records, strict=True):
        footprint = stacked[condition.label]["channel_forecast_change_mse"]
        per_channel = footprint.mean(axis=(0, 1))
        total = float(per_channel.sum())
        normalized = per_channel / max(total, 1e-12)
        record["output_footprint_channel_mse"] = per_channel.tolist()
        record["output_footprint_channel_share"] = normalized.tolist()
        record["output_footprint_effective_channels"] = float(
            1.0 / max(float(np.sum(normalized**2)), 1e-12)
        )
    for record in records:
        record["target_patched_mse_mean"] = float(
            target_baseline_mse.mean() + record["target_delta_mse_mean"]
        )
        record["target_patched_mae_mean"] = float(
            target_baseline_mae.mean() + record["target_delta_mae_mean"]
        )
        record["all_patched_mse_mean"] = float(
            all_baseline_mse.mean() + record["all_delta_mse_mean"]
        )
        record["all_patched_mae_mean"] = float(
            all_baseline_mae.mean() + record["all_delta_mae_mean"]
        )
    max_functional_change = max(
        float(metrics["all_forecast_change_mse"].max()) for metrics in stacked.values()
    )
    mean_perturbation = max(
        float(metrics["perturbation_rms"].mean()) for metrics in stacked.values()
    )
    if max_functional_change <= 1e-12 and mean_perturbation > 1e-6:
        raise RuntimeError(
            "Activation replacement was non-zero but no forecast changed; the selected post-layer "
            "history state is not consumed by the prediction path."
        )
    curves = _curve_summaries(records)
    contrasts = _paired_contrasts(
        conditions,
        stacked,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=config.mi.seed + 120_000,
    )

    output_dir = create_run_dir(
        args.output_root,
        f"{config.experiment_name}_progressive",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "config.json", config.to_dict())
    npz_payload: dict[str, np.ndarray] = {
        "origins": sampled_origins,
        "donor_shift_offsets": shifts,
        "columns": np.asarray(columns),
        "baseline_forecast_raw": baseline,
        "future_raw": futures,
        "target_baseline_mse": target_baseline_mse,
        "target_baseline_mae": target_baseline_mae,
        "all_baseline_mse": all_baseline_mse,
        "all_baseline_mae": all_baseline_mae,
    }
    for label, metrics in stacked.items():
        for name, array in metrics.items():
            npz_payload[f"{label}__{name}"] = array
    np.savez_compressed(output_dir / "progressive_results.npz", **npz_payload)

    payload = {
        "status": "complete",
        "model": config.model.name,
        "model_id": config.model.model_id,
        "dataset": config.data.dataset,
        "reference_run": str(reference_dir),
        "future_scope": config.future_target.scope,
        "split": "test",
        "input_columns": list(columns),
        "target_metric_column": config.data.target,
        "most_informative_layer_zero_based": top_layer,
        "selected_intervention_layers_zero_based": list(selected_layers),
        "manifest_layer_override_applied": bool(
            args.selection_manifest is not None
            and any(layer != top_layer for layer in selected_layers)
        ),
        "global_anchor_layer_zero_based": global_anchor_layer,
        "eligible_functional_layers_zero_based": list(functional_anchor.eligible_layers),
        "excluded_post_layer_history_states_zero_based": list(
            functional_anchor.excluded_layers
        ),
        "layer_selection_rule": (
            "argmax over post-layer history states with a downstream history-to-forecast "
            "mixing path of mean_patch(cell_null_calibrated_z)"
        ),
        "most_informative_layer_anchor_score": float(layer_anchor_score[top_layer]),
        "most_informative_layer_shared_null_z": float(layer_mi_z[top_layer]),
        "functional_anchor_selection": {
            "protocol_version": "functional-anchor-v1",
            "layer_zero_based": functional_anchor.layer,
            "top_patch_zero_based": functional_anchor.top_patch,
            "low_patch_zero_based": functional_anchor.low_patch,
            "uses_probe_or_intervention_outcomes": False,
        },
        "fractions": list(args.fractions),
        "intervention_protocol": "progressive-multivariate-v6.1",
        "strategies": list(selected_strategy_names),
        "selection_manifest": str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        "selection_methods": list(manifest_methods) if manifest_methods else None,
        "random_repetitions": args.random_repetitions,
        "include_sufficiency": bool(args.include_sufficiency),
        "sample_count": int(len(sampled_origins)),
        "origins_hash": origins_hash(sampled_origins),
        "donor_shift_offsets": shifts.tolist(),
        "donor_shift_indices": shift_indices.tolist(),
        "donor_shift_total": int(len(all_shifts)),
        "donor_bank": str(args.donor_bank.resolve()) if args.donor_bank else None,
        "donor_protocol": (
            "Prospectively fixed context-matched donor histories selected from history-only "
            "features before intervention outcomes; identical donor bank is used for every "
            "strategy, fraction, and intervention mode."
            if args.donor_bank
            else "Real temporally shifted origins separated by at least L+P; identical shifts are "
            "used for every strategy, fraction, and intervention mode."
        ),
        "primary_functional_metric": (
            "Train-standardized all-variable macro forecast-change MSE between intervened and "
            "unperturbed predictions; signed delta MSE/MAE are reported separately as performance consequences."
        ),
        "output_footprint_metric": (
            "Per-channel train-standardized forecast-change MSE between intervened and "
            "unperturbed predictions, averaged over paired donor shifts and test origins."
        ),
        "noop_clone_max_abs_forecast_difference": noop_max_abs_difference,
        "noop_clone_max_abs_forecast_difference_by_layer": {
            str(layer): difference for layer, difference in noop_differences.items()
        },
        "baseline_target_mse": float(target_baseline_mse.mean()),
        "baseline_target_mae": float(target_baseline_mae.mean()),
        "baseline_all_mse": float(all_baseline_mse.mean()),
        "baseline_all_mae": float(all_baseline_mae.mean()),
        "records": records,
        "curves": curves,
        "paired_contrasts": contrasts,
    }
    write_json(output_dir / "summary.json", payload)
    with (output_dir / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        flat_records = []
        for record in records:
            flat = dict(record)
            flat["selected_patches"] = json.dumps(flat["selected_patches"])
            flat["replaced_patches"] = json.dumps(flat["replaced_patches"])
            flat_records.append(flat)
        writer = csv.DictWriter(handle, fieldnames=list(flat_records[0]))
        writer.writeheader()
        writer.writerows(flat_records)
    if args.quiet:
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "model": payload["model"],
                    "dataset": payload["dataset"],
                    "sample_count": payload["sample_count"],
                    "donor_shift_count": len(payload["donor_shift_offsets"]),
                    "condition_count": len(payload["records"]),
                }
            ),
            flush=True,
        )
    else:
        print(json.dumps(payload, indent=2), flush=True)
    print(f"intervention_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
