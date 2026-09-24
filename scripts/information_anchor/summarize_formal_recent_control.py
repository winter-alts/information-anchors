#!/usr/bin/env python3
"""Pair formal V6 top-MI interventions with newly computed recent controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.interventions.common import (
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
)


FORMAL = ROOT / "results/information_anchor_v6_evidence/progressive_conditions.csv"
RECENT_ROOTS = (
    ROOT / "results/information_anchor_recent_control_formal_etth1",
    ROOT / "results/information_anchor_recent_control_all_regimes",
)
OUTPUT = ROOT / "results/information_anchor_recency_target_audit"
METRICS = (
    "all_forecast_change_mse",
    "target_forecast_change_mse",
    "all_delta_mse",
    "target_delta_mse",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def two_way_bootstrap(
    matrix: np.ndarray, *, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        rows = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        columns = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[index] = matrix[np.ix_(rows, columns)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def dataset_family(dataset_key: str) -> str:
    if dataset_key in {"etth1", "etth2", "ettm1", "ettm2"}:
        return "ETT"
    if dataset_key == "weather":
        return "Weather"
    if dataset_key == "electricity16":
        return "Electricity"
    if dataset_key == "traffic16":
        return "Traffic"
    raise ValueError(f"Unregistered dataset family for {dataset_key!r}.")


def completed_recent_runs() -> list[Path]:
    output: dict[tuple[str, str], Path] = {}
    for root in RECENT_ROOTS:
        if not root.exists():
            continue
        for summary_path in sorted(root.glob("*/summary.json")):
            summary = load_json(summary_path)
            if summary.get("status") == "complete" and summary.get("strategies") == ["recent"]:
                key = (str(summary["model"]), str(summary["dataset"]))
                previous = output.get(key)
                if previous is None or summary_path.stat().st_mtime > (previous / "summary.json").stat().st_mtime:
                    output[key] = summary_path.parent
    return [output[key] for key in sorted(output)]


def assert_paired(
    top_summary: dict,
    recent_summary: dict,
    top_archive: np.lib.npyio.NpzFile,
    recent_archive: np.lib.npyio.NpzFile,
) -> dict[str, float]:
    scalar_fields = (
        "model",
        "dataset",
        "sample_count",
        "origins_hash",
        "donor_shift_offsets",
        "fractions",
        "functional_anchor_selection",
    )
    for field in scalar_fields:
        if top_summary.get(field) != recent_summary.get(field):
            raise ValueError(
                f"Unpaired formal runs disagree on {field}: "
                f"{top_summary.get(field)!r} != {recent_summary.get(field)!r}"
            )
    exact_keys = (
        "origins",
        "donor_shift_offsets",
        "columns",
        "future_raw",
    )
    baseline_tolerances = {
        "baseline_forecast_raw": 2e-2,
        # GPU re-forwarding can differ by a few parts per million even with
        # identical origins/futures. The complete 42-pair audit has a worst
        # target-MSE absolute difference of 1.21e-5 (2.20e-6 of its scale).
        "target_baseline_mse": 2e-5,
        "target_baseline_mae": 1e-5,
        "all_baseline_mse": 1e-5,
        "all_baseline_mae": 1e-5,
    }
    diagnostics: dict[str, float] = {}
    for key in exact_keys + tuple(baseline_tolerances):
        if key not in top_archive.files or key not in recent_archive.files:
            raise KeyError(f"Paired archive is missing {key!r}.")
        left, right = top_archive[key], recent_archive[key]
        if left.dtype.kind in {"U", "S", "O"} or right.dtype.kind in {"U", "S", "O"}:
            equal = np.array_equal(left, right)
            max_difference = 0.0 if equal else float("inf")
        else:
            max_difference = float(np.max(np.abs(left - right)))
            tolerance = baseline_tolerances.get(key, 0.0)
            equal = bool(np.allclose(left, right, rtol=0.0, atol=tolerance, equal_nan=True))
        diagnostics[f"{key}_max_abs_difference"] = max_difference
        if key in baseline_tolerances and left.dtype.kind not in {"U", "S", "O"}:
            scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-12)
            diagnostics[f"{key}_max_relative_to_absolute_scale"] = max_difference / scale
        if not equal:
            tolerance = baseline_tolerances.get(key, 0.0)
            raise ValueError(
                f"Paired formal archives disagree on {key!r}: "
                f"max_abs_difference={max_difference}, tolerance={tolerance}."
            )
    return diagnostics


def condition_map(summary: dict) -> dict[tuple[str, float], dict]:
    return {
        (str(record["mode"]), float(record["fraction"])): record
        for record in summary["records"]
        if record["strategy"] in {"top", "recent"}
    }


def main() -> None:
    formal_rows = read_csv(FORMAL)
    top_dirs: dict[tuple[str, str], Path] = {}
    dataset_keys: dict[tuple[str, str], str] = {}
    for row in formal_rows:
        if row["strategy"] != "top":
            continue
        key = (row["model"], row["dataset"])
        path = Path(row["run_dir"])
        if key in top_dirs and top_dirs[key] != path:
            raise ValueError(f"Multiple formal top directories for {key}.")
        top_dirs[key] = path
        dataset_keys[key] = row["dataset_key"]

    output: list[dict[str, object]] = []
    for recent_dir in completed_recent_runs():
        recent_summary = load_json(recent_dir / "summary.json")
        key = (recent_summary["model"], recent_summary["dataset"])
        if key not in top_dirs:
            raise KeyError(f"No formal top-MI run for recent control {key}.")
        top_dir = top_dirs[key]
        top_summary = load_json(top_dir / "summary.json")
        top_archive = np.load(top_dir / "progressive_results.npz")
        recent_archive = np.load(recent_dir / "progressive_results.npz")
        pairing_diagnostics = assert_paired(
            top_summary, recent_summary, top_archive, recent_archive
        )
        top_records = condition_map(top_summary)
        recent_records = condition_map(recent_summary)
        if set(top_records) != set(recent_records):
            raise ValueError(
                f"Top/recent condition grid mismatch for {key}: "
                f"{sorted(top_records)} != {sorted(recent_records)}"
            )
        for condition_index, condition_key in enumerate(sorted(top_records)):
            mode, fraction = condition_key
            top_record = top_records[condition_key]
            recent_record = recent_records[condition_key]
            top_patches = set(int(value) for value in top_record["selected_patches"])
            recent_patches = set(int(value) for value in recent_record["selected_patches"])
            union = top_patches | recent_patches
            exact_patch_set_tie = top_patches == recent_patches
            record: dict[str, object] = {
                "model": key[0],
                "dataset_key": dataset_keys[key],
                "dataset": key[1],
                "mode": mode,
                "fraction": fraction,
                "functional_layer": int(top_record["layer_zero_based"]),
                "selected_patch_count": int(top_record["selected_patch_count"]),
                "top_recent_jaccard": len(top_patches & recent_patches) / len(union),
                "exact_patch_set_tie": exact_patch_set_tie,
                "top_selected_patches": json.dumps(sorted(top_patches)),
                "recent_selected_patches": json.dumps(sorted(recent_patches)),
                "top_run_dir": str(top_dir),
                "recent_run_dir": str(recent_dir),
                **pairing_diagnostics,
            }
            for metric_index, metric in enumerate(METRICS):
                top_values = np.asarray(
                    top_archive[f"{top_record['condition']}__{metric}"], dtype=np.float64
                )
                recent_values = np.asarray(
                    recent_archive[f"{recent_record['condition']}__{metric}"], dtype=np.float64
                )
                comparison_top_values = top_values
                comparison_recent_values = recent_values
                if metric in {"all_delta_mse", "target_delta_mse"}:
                    baseline_key = (
                        "all_baseline_mse" if metric == "all_delta_mse" else "target_baseline_mse"
                    )
                    comparison_top_values = top_values + np.asarray(
                        top_archive[baseline_key], dtype=np.float64
                    )[None, :]
                    comparison_recent_values = recent_values + np.asarray(
                        recent_archive[baseline_key], dtype=np.float64
                    )[None, :]
                    record[f"{metric}_top_patched_mse_mean"] = float(
                        comparison_top_values.mean()
                    )
                    record[f"{metric}_recent_patched_mse_mean"] = float(
                        comparison_recent_values.mean()
                    )
                difference = comparison_top_values - comparison_recent_values
                if exact_patch_set_tie:
                    difference = np.zeros_like(difference)
                lower, upper = hierarchical_bootstrap_ci(
                    difference,
                    repetitions=1000,
                    seed=2021 + condition_index * 100 + metric_index,
                )
                p_two, p_greater = exact_sign_flip_p(difference.mean(axis=1))
                record[f"{metric}_top_mean"] = float(top_values.mean())
                record[f"{metric}_recent_mean"] = float(recent_values.mean())
                record[f"{metric}_top_minus_recent"] = float(difference.mean())
                record[f"{metric}_ci95_lower"] = lower
                record[f"{metric}_ci95_upper"] = upper
                record[f"{metric}_sign_flip_p_two_sided"] = p_two
                record[f"{metric}_sign_flip_p_greater"] = p_greater
            output.append(record)

    output.sort(key=lambda row: (str(row["model"]), str(row["dataset"]), str(row["mode"]), float(row["fraction"])))
    write_csv(OUTPUT / "intervention_same_layer_recent_formal.csv", output)

    summary: dict[str, object] = {
        "registered_recent_runs": len(completed_recent_runs()),
        "paired_condition_count": len(output),
        "protocol": (
            "paired donor-shift x origin comparison; raw top-minus-recent is reported; "
            "Remove uses +1 orientation because greater damage supports necessity, whereas "
            "Keep uses -1 because lower post-intervention loss/drift supports sufficiency"
        ),
    }
    for mode in ("remove", "keep"):
        orientation = 1.0 if mode == "remove" else -1.0
        for metric in METRICS:
            subset = [row for row in output if row["mode"] == mode]
            deltas = np.asarray(
                [float(row[f"{metric}_top_minus_recent"]) for row in subset], dtype=np.float64
            )
            exact_ties = np.asarray(
                [bool(row["exact_patch_set_tie"]) for row in subset], dtype=bool
            )
            numeric_ties = np.isclose(deltas, 0.0, rtol=1e-5, atol=1e-7)
            ties = exact_ties | numeric_ties
            oriented = orientation * deltas
            summary[f"{mode}_{metric}"] = {
                "count": len(subset),
                "raw_top_minus_recent_positive": int(np.sum((deltas > 0.0) & ~ties)),
                "ties": int(np.sum(ties)),
                "exact_patch_set_ties": int(np.sum(exact_ties)),
                "raw_top_minus_recent_negative": int(np.sum((deltas < 0.0) & ~ties)),
                "mean_top_minus_recent": float(deltas.mean()) if len(deltas) else float("nan"),
                "median_top_minus_recent": float(np.median(deltas)) if len(deltas) else float("nan"),
                "orientation": (
                    "positive raw difference favors High-MI necessity"
                    if mode == "remove"
                    else "negative raw difference favors High-MI sufficiency"
                ),
                "high_mi_wins_oriented": int(np.sum((oriented > 0.0) & ~ties)),
                "recent_wins_oriented": int(np.sum((oriented < 0.0) & ~ties)),
                "mean_oriented_high_mi_advantage": (
                    float(oriented.mean()) if len(oriented) else float("nan")
                ),
            }
    curve_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in output:
        grouped.setdefault(
            (str(row["model"]), str(row["dataset_key"]), str(row["mode"])), []
        ).append(row)
    for (model, dataset_key, mode), rows in grouped.items():
        rows.sort(key=lambda row: float(row["fraction"]))
        orientation = 1.0 if mode == "remove" else -1.0
        curve: dict[str, object] = {
            "model": model,
            "dataset_key": dataset_key,
            "dataset": rows[0]["dataset"],
            "dataset_family": dataset_family(dataset_key),
            "mode": mode,
            "fraction_count": len(rows),
            "mean_top_recent_jaccard": float(
                np.mean([float(row["top_recent_jaccard"]) for row in rows])
            ),
        }
        for metric in METRICS:
            raw = float(
                np.mean([float(row[f"{metric}_top_minus_recent"]) for row in rows])
            )
            curve[f"{metric}_raw_top_minus_recent_curve_effect"] = raw
            curve[f"{metric}_oriented_high_mi_advantage"] = orientation * raw
        curve_rows.append(curve)
    curve_rows.sort(
        key=lambda row: (str(row["mode"]), str(row["model"]), str(row["dataset_key"]))
    )
    write_csv(OUTPUT / "formal_recent_model_dataset_curves.csv", curve_rows)

    models = sorted({str(row["model"]) for row in curve_rows})
    datasets = sorted({str(row["dataset_key"]) for row in curve_rows})
    families = ["ETT", "Weather", "Electricity", "Traffic"]
    if len(curve_rows) == 84 and len(models) == 6 and len(datasets) == 7:
        hierarchical: dict[str, object] = {
            "registered_models": models,
            "registered_datasets": datasets,
            "dataset_families": families,
            "curve_unit": "one model-dataset effect averaged over four nested fractions",
            "family_rule": "ETTh1, ETTh2, ETTm1, and ETTm2 are averaged into one ETT family before family-level resampling",
        }
        for mode_index, mode in enumerate(("remove", "keep")):
            subset = [row for row in curve_rows if row["mode"] == mode]
            for metric_index, metric in enumerate(METRICS):
                field = f"{metric}_oriented_high_mi_advantage"
                lookup = {
                    (str(row["model"]), str(row["dataset_key"])): float(row[field])
                    for row in subset
                }
                matrix = np.asarray(
                    [[lookup[(model, dataset)] for dataset in datasets] for model in models],
                    dtype=np.float64,
                )
                family_matrix = np.empty((len(models), len(families)), dtype=np.float64)
                for family_index, family in enumerate(families):
                    family_datasets = [
                        dataset for dataset in datasets if dataset_family(dataset) == family
                    ]
                    family_matrix[:, family_index] = np.mean(
                        [matrix[:, datasets.index(dataset)] for dataset in family_datasets], axis=0
                    )
                dataset_ci = two_way_bootstrap(
                    matrix,
                    repetitions=10_000,
                    seed=2021 + mode_index * 100 + metric_index,
                )
                family_ci = two_way_bootstrap(
                    family_matrix,
                    repetitions=10_000,
                    seed=4021 + mode_index * 100 + metric_index,
                )
                hierarchical[f"{mode}_{metric}"] = {
                    "mean_oriented_effect_over_42_curves": float(matrix.mean()),
                    "median_oriented_effect_over_42_curves": float(np.median(matrix)),
                    "high_mi_wins": int(np.sum(matrix > 1e-7)),
                    "ties": int(np.sum(np.isclose(matrix, 0.0, rtol=1e-5, atol=1e-7))),
                    "recent_wins": int(np.sum(matrix < -1e-7)),
                    "two_way_model_dataset_bootstrap_ci95": list(dataset_ci),
                    "mean_over_four_equal_weight_families": float(family_matrix.mean()),
                    "two_way_model_family_bootstrap_ci95": list(family_ci),
                    "family_means": {
                        family: float(family_matrix[:, index].mean())
                        for index, family in enumerate(families)
                    },
                    "leave_one_family_out_means": {
                        family: float(np.delete(family_matrix, index, axis=1).mean())
                        for index, family in enumerate(families)
                    },
                }
        summary["hierarchical_curve_analysis"] = hierarchical
    (OUTPUT / "formal_recent_intervention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
