from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

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
    parser = argparse.ArgumentParser(description="Hidden-donor interpolation dose response.")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--donor-shifts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--device")
    parser.add_argument(
        "--output-root", default="results/information_anchor_interpolation_dose"
    )
    return parser.parse_args()


def parse_alphas(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if tuple(sorted(set(values))) != values or any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError("alphas must be unique, increasing, and in (0,1].")
    return values


def blend_state(target: Any, donor: Any, alpha: float) -> Any:
    if isinstance(target, torch.Tensor) and isinstance(donor, torch.Tensor):
        if target.shape != donor.shape:
            raise ValueError(f"Target/donor state shape mismatch: {target.shape} != {donor.shape}")
        return target + float(alpha) * (donor - target)
    if isinstance(target, list) and isinstance(donor, list):
        if len(target) != len(donor):
            raise ValueError("Target/donor state list lengths differ.")
        return [blend_state(left, right, alpha) for left, right in zip(target, donor, strict=True)]
    if isinstance(target, tuple) and isinstance(donor, tuple):
        if len(target) != len(donor):
            raise ValueError("Target/donor state tuple lengths differ.")
        return tuple(blend_state(left, right, alpha) for left, right in zip(target, donor, strict=True))
    raise TypeError(f"Unsupported donor-state containers: {type(target)} and {type(donor)}")


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
    alphas = parse_alphas(args.alphas)
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if args.device:
        config = replace(config, model=replace(config.model, device=args.device))
    reference = np.load(reference_dir / "mi_results.npz")
    mi_z = np.asarray(reference["mi_z"], dtype=np.float64)
    layer_scores = np.asarray(reference["layer_anchor_score"], dtype=np.float64)
    functional = select_functional_anchor(mi_z, layer_scores, config.model.name)
    patches = {"high": functional.top_patch, "low": functional.low_patch}

    test_config = replace(config, data=replace(config.data, split="test"))
    all_origins = build_forecast_origins(test_config.data)
    selected = np.unique(
        np.rint(
            np.linspace(0, len(all_origins) - 1, min(args.max_samples, len(all_origins)))
        ).astype(np.int64)
    )
    origins = all_origins[selected]
    shifts = temporal_circular_shift_offsets(
        origins,
        args.donor_shifts,
        min_temporal_separation=config.data.seq_len + config.data.pred_len,
        seed=config.mi.seed + 80_000,
    )
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    scaler = fit_train_standard_scaler(values, config.data.train_end)
    target_index = columns.index(config.data.target)
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
    baseline = np.concatenate(baseline_parts).astype(np.float32)
    all_baseline_mse, _ = per_origin_metrics(baseline, futures, scaler)
    target_baseline_mse, _ = per_origin_metrics(
        baseline, futures, scaler, target_index=target_index
    )

    labels = [f"{rank}_a{int(round(alpha * 100)):03d}" for rank in patches for alpha in alphas]
    metric_names = (
        "all_forecast_change_mse",
        "target_forecast_change_mse",
        "all_delta_mse",
        "target_delta_mse",
        "perturbation_rms",
        "all_propagation_gain",
        "target_propagation_gain",
    )
    collected = {label: {metric: [] for metric in metric_names} for label in labels}
    with torch.no_grad():
        for shift_index, shift in enumerate(shifts):
            donor_histories = np.roll(histories, int(shift), axis=0)
            predictions = {label: [] for label in labels}
            rms_parts = {label: [] for label in labels}
            for start in range(0, len(histories), args.batch_size):
                stop = min(len(histories), start + args.batch_size)
                target_state = runner.capture_donor_state(
                    histories[start:stop], functional.layer
                )
                donor_state = runner.capture_donor_state(
                    donor_histories[start:stop], functional.layer
                )
                blended = {
                    alpha: blend_state(target_state, donor_state, alpha) for alpha in alphas
                }
                for rank, patch in patches.items():
                    for alpha in alphas:
                        label = f"{rank}_a{int(round(alpha * 100)):03d}"
                        prediction, rms = runner.patched_forecast_set(
                            histories[start:stop],
                            donor_histories[start:stop],
                            layer=functional.layer,
                            patches=(patch,),
                            donor_state=blended[alpha],
                        )
                        predictions[label].append(prediction)
                        rms_parts[label].append(rms)
            for rank in patches:
                for alpha in alphas:
                    label = f"{rank}_a{int(round(alpha * 100)):03d}"
                    prediction = np.concatenate(predictions[label]).astype(np.float32)
                    rms = np.concatenate(rms_parts[label]).astype(np.float64)
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
                f"model={config.model.name} interpolation donor {shift_index + 1}/{len(shifts)}",
                flush=True,
            )

    stacked = {
        label: {metric: np.stack(rows) for metric, rows in metrics.items()}
        for label, metrics in collected.items()
    }
    condition_rows: list[dict[str, object]] = []
    for condition_index, (rank, patch) in enumerate(patches.items()):
        for alpha_index, alpha in enumerate(alphas):
            label = f"{rank}_a{int(round(alpha * 100)):03d}"
            record: dict[str, object] = {
                "condition": label,
                "rank": rank,
                "patch": patch,
                "alpha": alpha,
                "mi_z": float(mi_z[functional.layer, patch]),
            }
            for metric_index, metric in enumerate(metric_names):
                details = summarize(
                    stacked[label][metric],
                    args.bootstrap_repetitions,
                    config.mi.seed
                    + 90_000
                    + condition_index * 100
                    + alpha_index * 10
                    + metric_index,
                )
                for key, value in details.items():
                    record[f"{metric}_{key}"] = value
            condition_rows.append(record)

    contrast_rows: list[dict[str, object]] = []
    for alpha_index, alpha in enumerate(alphas):
        high_label = f"high_a{int(round(alpha * 100)):03d}"
        low_label = f"low_a{int(round(alpha * 100)):03d}"
        record = {"alpha": alpha, "contrast": "high_minus_low"}
        for metric_index, metric in enumerate(
            (
                "all_forecast_change_mse",
                "target_forecast_change_mse",
                "all_propagation_gain",
                "target_propagation_gain",
                "perturbation_rms",
            )
        ):
            difference = stacked[high_label][metric] - stacked[low_label][metric]
            details = summarize(
                difference,
                args.bootstrap_repetitions,
                config.mi.seed + 120_000 + alpha_index * 20 + metric_index,
            )
            for key, value in details.items():
                record[f"{metric}_high_minus_low_{key}"] = value
        contrast_rows.append(record)

    dose_rows: list[dict[str, object]] = []
    for rank in patches:
        effect = [
            stacked[f"{rank}_a{int(round(alpha * 100)):03d}"][
                "all_forecast_change_mse"
            ]
            for alpha in alphas
        ]
        stacked_effect = np.stack(effect, axis=0)
        monotonic = np.all(np.diff(stacked_effect, axis=0) >= -1e-12, axis=0)
        scaled_means = np.asarray(
            [values.mean() / (alpha**2) for values, alpha in zip(effect, alphas, strict=True)]
        )
        dose_rows.append(
            {
                "rank": rank,
                "monotonic_donor_origin_fraction": float(monotonic.mean()),
                "effect_over_alpha_squared_mean": float(scaled_means.mean()),
                "effect_over_alpha_squared_cv": float(
                    scaled_means.std() / max(abs(float(scaled_means.mean())), 1e-12)
                ),
                "alpha_effect_means": json.dumps(
                    {str(alpha): float(values.mean()) for alpha, values in zip(alphas, effect, strict=True)}
                ),
            }
        )

    output_dir = create_run_dir(
        args.output_root,
        f"{config.model.name.lower()}_{config.data.dataset.lower()}_interpolation_dose",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "config.json", config.to_dict())
    write_csv(output_dir / "conditions.csv", condition_rows)
    write_csv(output_dir / "contrasts.csv", contrast_rows)
    write_csv(output_dir / "dose_response.csv", dose_rows)
    arrays: dict[str, np.ndarray] = {
        "origins": origins,
        "donor_shift_offsets": shifts,
        "baseline_prediction": baseline,
        "future": futures,
    }
    for label, metrics in stacked.items():
        for metric, values_array in metrics.items():
            arrays[f"{label}__{metric}"] = values_array
    np.savez_compressed(output_dir / "interpolation_results.npz", **arrays)
    payload = {
        "status": "complete",
        "reference_run": str(reference_dir),
        "model": config.model.name,
        "dataset": config.data.dataset,
        "functional_layer_zero_based": functional.layer,
        "high_patch": functional.top_patch,
        "low_patch": functional.low_patch,
        "alphas": list(alphas),
        "sample_count": len(origins),
        "origins_hash": origins_hash(origins),
        "donor_shift_offsets": shifts.tolist(),
        "conditions": condition_rows,
        "contrasts": contrast_rows,
        "dose_response": dose_rows,
        "protocol": "Convex hidden-state interpolation target + alpha*(distant donor-target) at one functional-layer patch.",
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps({"status": "complete", "run_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
