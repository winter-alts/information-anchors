#!/usr/bin/env python3
"""Outcome-blind matched-pair audit of saved donor interventions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import load_benchmark_frame, make_windows, select_value_columns
from experiments.information_anchor.interventions.common import (
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
)


REGISTRY = ROOT / "results/information_anchor_v6_evidence/progressive_conditions.csv"
OUTPUT = ROOT / "results/information_anchor_recency_target_audit"
DATASETS = {"ETTh1": 24, "ETTm1": 96, "weather": 144}
FRACTIONS = (0.125, 0.25, 0.375, 0.5)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return ranks


def temporal_descriptor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    chunks = np.array_split(np.arange(values.shape[1]), 4)
    quarter = np.concatenate([values[:, chunk].mean(axis=1) for chunk in chunks], axis=1)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    time = np.linspace(-1, 1, values.shape[1], dtype=np.float64)
    slope = np.einsum("ntc,t->nc", values - mean[:, None], time) / np.dot(time, time)
    diff_std = np.diff(values, axis=1).std(axis=1)
    descriptor = np.concatenate((quarter, mean, std, slope, diff_std), axis=1)
    return StandardScaler().fit_transform(descriptor)


def paired_distances(descriptor: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.sqrt(np.mean((descriptor - np.roll(descriptor, int(shift), axis=0)) ** 2, axis=1))
            for shift in shifts
        ],
        axis=0,
    )


def two_way_bootstrap(matrix: np.ndarray, repetitions: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(91_007)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        models = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        datasets = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[index] = matrix[np.ix_(models, datasets)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def main() -> None:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    run_lookup: dict[tuple[str, str], Path] = {}
    for row in registry_rows:
        if row["dataset"] in DATASETS:
            run_lookup.setdefault((row["model"], row["dataset"]), Path(row["run_dir"]))
    if len(run_lookup) != 18:
        raise RuntimeError(f"Expected 18 model-dataset runs, found {len(run_lookup)}")

    records: list[dict[str, object]] = []
    for (model, dataset), run_dir in sorted(run_lookup.items()):
        arrays = np.load(run_dir / "progressive_results.npz")
        config = load_config(run_dir / "config.json")
        origins = np.asarray(arrays["origins"], dtype=np.int64)
        shifts = np.asarray(arrays["donor_shift_offsets"], dtype=np.int64)
        frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
        columns = select_value_columns(frame, config.data)
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
        windows = make_windows(
            values, origins, config.data.seq_len, config.data.pred_len, columns=columns
        )
        recent = windows.history_normalized[:, -min(config.data.pred_len, config.data.seq_len):]
        history_descriptor = temporal_descriptor(recent)
        forecast_descriptor = temporal_descriptor(
            np.asarray(arrays["baseline_forecast_raw"], dtype=np.float32)
        )
        history_distance = paired_distances(history_descriptor, shifts)
        forecast_distance = paired_distances(forecast_descriptor, shifts)
        donor_origins = np.stack(
            [np.roll(origins, int(shift)) for shift in shifts], axis=0
        )
        period = DATASETS[dataset]
        phase_delta = np.abs((origins[None, :] - donor_origins) % period)
        phase_distance = np.minimum(phase_delta, period - phase_delta) / max(period / 2, 1)

        for fraction in FRACTIONS:
            code = f"{int(round(fraction * 100)):03d}"
            top_rms = np.asarray(arrays[f"remove__top_f{code}__perturbation_rms"], dtype=np.float64)
            low_rms = np.asarray(arrays[f"remove__bottom_f{code}__perturbation_rms"], dtype=np.float64)
            top_loss = np.asarray(arrays[f"remove__top_f{code}__all_delta_mse"], dtype=np.float64)
            low_loss = np.asarray(arrays[f"remove__bottom_f{code}__all_delta_mse"], dtype=np.float64)
            top_drift = np.asarray(
                arrays[f"remove__top_f{code}__all_forecast_change_mse"], dtype=np.float64
            )
            low_drift = np.asarray(
                arrays[f"remove__bottom_f{code}__all_forecast_change_mse"], dtype=np.float64
            )
            selected_indices: list[np.ndarray] = []
            for donor in range(len(shifts)):
                rms_magnitude = np.maximum(top_rms[donor], low_rms[donor])
                rms_balance = np.abs(
                    np.log(np.maximum(top_rms[donor], 1e-8))
                    - np.log(np.maximum(low_rms[donor], 1e-8))
                )
                score = (
                    rank01(history_distance[donor])
                    + rank01(forecast_distance[donor])
                    + phase_distance[donor]
                    + rank01(rms_magnitude)
                    + rank01(rms_balance)
                )
                count = max(16, int(np.ceil(0.25 * len(origins))))
                selected_indices.append(np.argsort(score, kind="mergesort")[:count])
            matched_loss = np.stack(
                [
                    (top_loss[donor] - low_loss[donor])[indices]
                    for donor, indices in enumerate(selected_indices)
                ]
            )
            matched_drift = np.stack(
                [
                    (top_drift[donor] - low_drift[donor])[indices]
                    for donor, indices in enumerate(selected_indices)
                ]
            )
            loss_ci = hierarchical_bootstrap_ci(
                matched_loss, repetitions=10_000, seed=51_000 + int(fraction * 1000)
            )
            drift_ci = hierarchical_bootstrap_ci(
                matched_drift, repetitions=10_000, seed=61_000 + int(fraction * 1000)
            )
            loss_p = exact_sign_flip_p(matched_loss.mean(axis=1))
            records.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "fraction": fraction,
                    "matched_pairs_per_donor": matched_loss.shape[1],
                    "donor_count": matched_loss.shape[0],
                    "matched_all_delta_mse_top_minus_low": float(matched_loss.mean()),
                    "matched_all_delta_mse_ci95_lower": loss_ci[0],
                    "matched_all_delta_mse_ci95_upper": loss_ci[1],
                    "matched_all_delta_mse_sign_flip_p_two_sided": loss_p[0],
                    "matched_all_forecast_change_mse_top_minus_low": float(matched_drift.mean()),
                    "matched_all_forecast_change_mse_ci95_lower": drift_ci[0],
                    "matched_all_forecast_change_mse_ci95_upper": drift_ci[1],
                    "unmatched_all_delta_mse_top_minus_low": float((top_loss - low_loss).mean()),
                    "unmatched_all_forecast_change_mse_top_minus_low": float(
                        (top_drift - low_drift).mean()
                    ),
                    "matched_exact_phase_fraction": float(
                        np.mean(
                            np.concatenate(
                                [
                                    phase_distance[donor, indices] == 0
                                    for donor, indices in enumerate(selected_indices)
                                ]
                            )
                        )
                    ),
                    "matched_history_distance": float(
                        np.mean(
                            np.concatenate(
                                [history_distance[d, idx] for d, idx in enumerate(selected_indices)]
                            )
                        )
                    ),
                    "unmatched_history_distance": float(history_distance.mean()),
                    "matched_forecast_distance": float(
                        np.mean(
                            np.concatenate(
                                [forecast_distance[d, idx] for d, idx in enumerate(selected_indices)]
                            )
                        )
                    ),
                    "unmatched_forecast_distance": float(forecast_distance.mean()),
                    "run_dir": str(run_dir.resolve()),
                }
            )
        print(f"matched donor audit model={model} dataset={dataset}", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    condition_path = OUTPUT / "matched_donor_conditions.csv"
    with condition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    curves = []
    for model, dataset in sorted({(str(row["model"]), str(row["dataset"])) for row in records}):
        subset = [row for row in records if row["model"] == model and row["dataset"] == dataset]
        curves.append(
            {
                "model": model,
                "dataset": dataset,
                "matched_all_delta_mse_curve_effect": float(
                    np.mean([row["matched_all_delta_mse_top_minus_low"] for row in subset])
                ),
                "unmatched_all_delta_mse_curve_effect": float(
                    np.mean([row["unmatched_all_delta_mse_top_minus_low"] for row in subset])
                ),
                "matched_all_forecast_change_mse_curve_effect": float(
                    np.mean(
                        [row["matched_all_forecast_change_mse_top_minus_low"] for row in subset]
                    )
                ),
            }
        )
    curve_path = OUTPUT / "matched_donor_curves.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    models = sorted({str(row["model"]) for row in curves})
    datasets = sorted({str(row["dataset"]) for row in curves})
    matrix = np.asarray(
        [
            [
                next(
                    row["matched_all_delta_mse_curve_effect"]
                    for row in curves
                    if row["model"] == model and row["dataset"] == dataset
                )
                for dataset in datasets
            ]
            for model in models
        ],
        dtype=np.float64,
    )
    summary = {
        "curve_count": len(curves),
        "positive_matched_ground_truth_curve_count": int(np.sum(matrix > 0)),
        "negative_matched_ground_truth_curve_count": int(np.sum(matrix < 0)),
        "mean_matched_ground_truth_curve_effect": float(matrix.mean()),
        "two_way_model_dataset_bootstrap_ci95": list(two_way_bootstrap(matrix)),
        "mean_exact_phase_fraction": float(
            np.mean([row["matched_exact_phase_fraction"] for row in records])
        ),
        "mean_history_distance_ratio_matched_to_unmatched": float(
            np.mean(
                [row["matched_history_distance"] / row["unmatched_history_distance"] for row in records]
            )
        ),
        "mean_forecast_distance_ratio_matched_to_unmatched": float(
            np.mean(
                [row["matched_forecast_distance"] / row["unmatched_forecast_distance"] for row in records]
            )
        ),
        "protocol_note": (
            "Outcome-blind retrospective subset: within each donor shift and fraction, retain the "
            "nearest 25% of origins under seasonal phase, recent-history trajectory, unperturbed "
            "forecast state, perturbation RMS magnitude, and high/low RMS balance. Saved RMS is a "
            "donor-recipient perturbation magnitude, not absolute activation-norm matching."
        ),
    }
    (OUTPUT / "matched_donor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
