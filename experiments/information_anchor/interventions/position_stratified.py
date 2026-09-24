from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.information_anchor.artifacts import capture_run_context, create_run_dir, write_json
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
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
    select_functional_anchor,
)
from experiments.information_anchor.interventions.multivariate_activation import ModelRunner
from experiments.information_anchor.metrics import (
    fit_train_standard_scaler,
    forecast_change_metrics,
    per_origin_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Position-stratified single-patch functional analysis."
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument(
        "--score-run",
        help=(
            "Optional frozen MI-sensitivity run used only for high/low patch selection. "
            "Its discovery origins must exactly match --reference-run."
        ),
    )
    parser.add_argument("--position-bins", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--donor-shifts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--device")
    parser.add_argument(
        "--output-root",
        default="results/information_anchor_position_stratified",
    )
    return parser.parse_args()


def summarize(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
    lower, upper = hierarchical_bootstrap_ci(values, repetitions, seed)
    p_two, p_greater = exact_sign_flip_p(values.mean(axis=1))
    return {
        "mean": float(values.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "sign_flip_p_two_sided": p_two,
        "sign_flip_p_greater": p_greater,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.position_bins < 2:
        raise ValueError("position-bins must be at least two.")
    if not 1 <= args.donor_shifts <= 16:
        raise ValueError("donor-shifts must be in [1,16].")
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if args.device:
        config = replace(config, model=replace(config.model, device=args.device))
    reference = np.load(reference_dir / "mi_results.npz")
    if args.score_run:
        score_dir = Path(args.score_run).resolve()
        score_path = score_dir / "mi_sensitivity_results.npz"
        if not score_path.exists():
            raise FileNotFoundError(
                f"--score-run must contain mi_sensitivity_results.npz, missing {score_path}."
            )
        score_source = np.load(score_path)
        if not np.array_equal(reference["origins"], score_source["origins"]):
            raise ValueError("--score-run discovery origins do not match --reference-run.")
        mi_z = np.asarray(score_source["mi_z"], dtype=np.float64)
        layer_scores = mi_z.mean(axis=1)
    else:
        score_dir = reference_dir
        mi_z = np.asarray(reference["mi_z"], dtype=np.float64)
        layer_scores = np.asarray(reference["layer_anchor_score"], dtype=np.float64)
    functional = select_functional_anchor(mi_z, layer_scores, config.model.name)
    scores = mi_z[functional.layer]
    if len(scores) < 2 * args.position_bins:
        raise ValueError(
            f"Need at least two patches per position bin, got {len(scores)} patches "
            f"for {args.position_bins} bins."
        )
    conditions: list[dict[str, object]] = []
    for bin_index, patch_indices in enumerate(
        np.array_split(np.arange(len(scores), dtype=np.int64), args.position_bins)
    ):
        high_patch = int(patch_indices[np.argmax(scores[patch_indices])])
        low_patch = int(patch_indices[np.argmin(scores[patch_indices])])
        if high_patch == low_patch:
            raise RuntimeError(f"Position bin {bin_index} has no distinct high/low pair.")
        for rank, patch in (("high", high_patch), ("low", low_patch)):
            conditions.append(
                {
                    "label": f"bin{bin_index}_{rank}_p{patch:03d}",
                    "position_bin": bin_index,
                    "rank": rank,
                    "patch": patch,
                    "relative_position": float((patch + 0.5) / len(scores)),
                    "mi_z": float(scores[patch]),
                }
            )

    test_config = replace(config, data=replace(config.data, split="test"))
    all_origins = build_forecast_origins(test_config.data)
    indices = np.unique(
        np.rint(
            np.linspace(0, len(all_origins) - 1, min(args.max_samples, len(all_origins)))
        ).astype(np.int64)
    )
    origins = all_origins[indices]
    shifts = temporal_circular_shift_offsets(
        origins,
        args.donor_shifts,
        min_temporal_separation=config.data.seq_len + config.data.pred_len,
        seed=config.mi.seed + 80_000,
    )
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    target_index = columns.index(config.data.target)
    scaler = fit_train_standard_scaler(values, config.data.train_end)
    windows = make_windows(
        values,
        origins,
        config.data.seq_len,
        config.data.pred_len,
        columns=columns,
    )
    histories = np.asarray(windows.history_raw, dtype=np.float32)
    futures = np.asarray(windows.future_raw, dtype=np.float32)
    runner = ModelRunner(config, histories.shape[-1])

    baseline_parts = []
    with torch.no_grad():
        for start in range(0, len(histories), args.batch_size):
            baseline_parts.append(runner.forecast(histories[start : start + args.batch_size]))
    baseline = np.concatenate(baseline_parts, axis=0).astype(np.float32)
    all_baseline_mse, _ = per_origin_metrics(baseline, futures, scaler)
    target_baseline_mse, _ = per_origin_metrics(
        baseline, futures, scaler, target_index=target_index
    )
    with torch.no_grad():
        noop = runner.noop(
            histories[: min(args.batch_size, len(histories))], functional.layer
        )
    noop_difference = float(
        np.max(np.abs(noop - baseline[: min(args.batch_size, len(histories))]))
    )
    if noop_difference > 1e-4:
        raise RuntimeError(f"Clone-only hook changed forecasts by {noop_difference}.")

    metric_names = (
        "all_forecast_change_mse",
        "target_forecast_change_mse",
        "all_delta_mse",
        "target_delta_mse",
        "perturbation_rms",
        "all_propagation_gain",
        "target_propagation_gain",
    )
    collected = {
        str(condition["label"]): {name: [] for name in metric_names}
        for condition in conditions
    }
    with torch.no_grad():
        for shift_index, shift in enumerate(shifts):
            donor_histories = np.roll(histories, int(shift), axis=0)
            predictions = {str(condition["label"]): [] for condition in conditions}
            rms_parts = {str(condition["label"]): [] for condition in conditions}
            for start in range(0, len(histories), args.batch_size):
                stop = min(len(histories), start + args.batch_size)
                donor_state = runner.capture_donor_state(
                    donor_histories[start:stop], functional.layer
                )
                for condition in conditions:
                    label = str(condition["label"])
                    prediction, rms = runner.patched_forecast_set(
                        histories[start:stop],
                        donor_histories[start:stop],
                        layer=functional.layer,
                        patches=(int(condition["patch"]),),
                        donor_state=donor_state,
                    )
                    predictions[label].append(prediction)
                    rms_parts[label].append(rms)
            for condition in conditions:
                label = str(condition["label"])
                prediction = np.concatenate(predictions[label], axis=0).astype(np.float32)
                rms = np.concatenate(rms_parts[label], axis=0).astype(np.float64)
                all_mse, _ = per_origin_metrics(prediction, futures, scaler)
                target_mse, _ = per_origin_metrics(
                    prediction, futures, scaler, target_index=target_index
                )
                all_change, _ = forecast_change_metrics(prediction, baseline, scaler)
                target_change, _ = forecast_change_metrics(
                    prediction, baseline, scaler, target_index=target_index
                )
                rms_square = np.maximum(np.square(rms), 1e-12)
                destination = collected[label]
                destination["all_forecast_change_mse"].append(all_change)
                destination["target_forecast_change_mse"].append(target_change)
                destination["all_delta_mse"].append(all_mse - all_baseline_mse)
                destination["target_delta_mse"].append(target_mse - target_baseline_mse)
                destination["perturbation_rms"].append(rms)
                destination["all_propagation_gain"].append(all_change / rms_square)
                destination["target_propagation_gain"].append(target_change / rms_square)
            print(
                f"model={config.model.name} position-stratified donor "
                f"{shift_index + 1}/{len(shifts)} complete",
                flush=True,
            )

    stacked = {
        label: {name: np.stack(rows, axis=0) for name, rows in metrics.items()}
        for label, metrics in collected.items()
    }
    condition_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        label = str(condition["label"])
        record = dict(condition)
        for metric_index, metric in enumerate(metric_names):
            details = summarize(
                stacked[label][metric],
                args.bootstrap_repetitions,
                config.mi.seed + 90_000 + condition_index * 20 + metric_index,
            )
            for key, value in details.items():
                record[f"{metric}_{key}"] = value
        condition_rows.append(record)

    contrasts: list[dict[str, object]] = []
    by_key = {
        (int(condition["position_bin"]), str(condition["rank"])): condition
        for condition in conditions
    }
    contrast_metrics = (
        "all_forecast_change_mse",
        "target_forecast_change_mse",
        "all_propagation_gain",
        "target_propagation_gain",
        "perturbation_rms",
    )
    for bin_index in range(args.position_bins):
        high = by_key[(bin_index, "high")]
        low = by_key[(bin_index, "low")]
        record = {
            "position_bin": bin_index,
            "high_patch": high["patch"],
            "low_patch": low["patch"],
            "patch_distance": abs(int(high["patch"]) - int(low["patch"])),
            "high_mi_z": high["mi_z"],
            "low_mi_z": low["mi_z"],
            "mi_z_difference": float(high["mi_z"]) - float(low["mi_z"]),
        }
        for metric_index, metric in enumerate(contrast_metrics):
            difference = (
                stacked[str(high["label"])][metric]
                - stacked[str(low["label"])][metric]
            )
            details = summarize(
                difference,
                args.bootstrap_repetitions,
                config.mi.seed + 120_000 + bin_index * 20 + metric_index,
            )
            for key, value in details.items():
                record[f"{metric}_high_minus_low_{key}"] = value
        contrasts.append(record)

    output_dir = create_run_dir(
        args.output_root,
        f"{config.model.name.lower()}_{config.data.dataset.lower()}_position_stratified",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "config.json", config.to_dict())
    write_csv(output_dir / "conditions.csv", condition_rows)
    write_csv(output_dir / "contrasts.csv", contrasts)
    arrays: dict[str, np.ndarray] = {
        "origins": origins,
        "donor_shift_offsets": shifts,
        "baseline_prediction": baseline,
        "future": futures,
    }
    for label, metrics in stacked.items():
        for metric, values_array in metrics.items():
            arrays[f"{label}__{metric}"] = values_array
    np.savez_compressed(output_dir / "position_stratified_results.npz", **arrays)
    payload = {
        "status": "complete",
        "reference_run": str(reference_dir),
        "score_run": str(score_dir),
        "score_selection": (
            "independent MI sensitivity atlas" if args.score_run else "reference MI atlas"
        ),
        "model": config.model.name,
        "dataset": config.data.dataset,
        "functional_layer_zero_based": functional.layer,
        "position_bins": args.position_bins,
        "sample_count": len(origins),
        "origins_hash": origins_hash(origins),
        "donor_shift_offsets": shifts.tolist(),
        "noop_max_abs_difference": noop_difference,
        "conditions": condition_rows,
        "contrasts": contrasts,
        "protocol": (
            "Within each contiguous temporal stratum, replace one high-MI and one "
            "low-MI patch at the same functional layer using identical distant donors."
        ),
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps({"status": "complete", "run_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
