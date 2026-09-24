from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SOURCE_TOPOLOGY = RESULTS / "information_anchor_v6_topology" / "topology_descriptors.csv"
SOURCE_CONTRASTS = RESULTS / "information_anchor_v6_evidence" / "progressive_contrasts.csv"
SOURCE_CONDITIONS = RESULTS / "information_anchor_v6_evidence" / "progressive_conditions.csv"
OUTPUT = RESULTS / "information_anchor_iclr_reanalysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def holm(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * np.arange(len(p), 0, -1)
    adjusted = np.maximum.accumulate(adjusted)
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def dataset_family(dataset: str) -> str:
    normalized = dataset.lower()
    if normalized.startswith("ett"):
        return "ETT"
    if normalized == "weather":
        return "Weather"
    if normalized == "electricity":
        return "Electricity"
    if normalized == "traffic":
        return "Traffic"
    raise ValueError(f"Unknown dataset family for {dataset!r}.")


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean())


def _family_topology(topology: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = (
        "top_quarter_concentration",
        "anchor_ratio",
        "mean_boundary_distance",
        "expected_layer_depth",
        "significant_cell_fraction",
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in topology:
        grouped.setdefault((row["model"], dataset_family(row["dataset"])), []).append(row)

    output: list[dict[str, object]] = []
    for (model, family), rows in sorted(grouped.items()):
        record: dict[str, object] = {
            "model": model,
            "dataset_family": family,
            "regime_count": len(rows),
        }
        for metric in metrics:
            record[metric] = mean(float(row[metric]) for row in rows)
        output.append(record)
    return output


def _atlas_significance(topology: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    for model in sorted({row["model"] for row in topology}):
        values = np.asarray(
            [float(row["significant_cell_fraction"]) for row in topology if row["model"] == model],
            dtype=np.float64,
        )
        output.append(
            {
                "model": model,
                "atlas_count": len(values),
                "mean_significant_cell_fraction": float(values.mean()),
                "median_significant_cell_fraction": float(np.median(values)),
                "min_significant_cell_fraction": float(values.min()),
                "max_significant_cell_fraction": float(values.max()),
            }
        )

    values = np.asarray(
        [float(row["significant_cell_fraction"]) for row in topology], dtype=np.float64
    )
    elapsed = []
    for row in topology:
        summary = Path(row["run_dir"]) / "summary.json"
        payload = json.loads(summary.read_text(encoding="utf-8"))
        elapsed.append(float(payload["analysis_elapsed_seconds"]))
    global_summary = {
        "atlas_count": len(values),
        "mean_significant_cell_fraction": float(values.mean()),
        "median_significant_cell_fraction": float(np.median(values)),
        "min_significant_cell_fraction": float(values.min()),
        "max_significant_cell_fraction": float(values.max()),
        "atlas_total_elapsed_seconds": float(np.sum(elapsed)),
        "atlas_total_elapsed_hours": float(np.sum(elapsed) / 3600.0),
        "atlas_mean_elapsed_seconds": float(np.mean(elapsed)),
        "atlas_median_elapsed_seconds": float(np.median(elapsed)),
        "atlas_min_elapsed_seconds": float(np.min(elapsed)),
        "atlas_max_elapsed_seconds": float(np.max(elapsed)),
    }
    return output, global_summary


def _intervention_multiplicity(
    contrasts: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    primary = [
        row
        for row in contrasts
        if row["mode"] == "remove"
        and row["contrast"] in {"top_minus_bottom", "top_minus_random_mean"}
    ]
    p_key = "all_forecast_change_mse_sign_flip_p_greater"
    effect_key = "all_forecast_change_mse_mean"

    for contrast in sorted({row["contrast"] for row in primary}):
        subset = [row for row in primary if row["contrast"] == contrast]
        q = benjamini_hochberg(float(row[p_key]) for row in subset)
        h = holm(float(row[p_key]) for row in subset)
        for row, q_value, holm_value in zip(subset, q, h, strict=True):
            row["bh_q_global_contrast"] = str(float(q_value))
            row["holm_p_global_contrast"] = str(float(holm_value))

    for contrast in sorted({row["contrast"] for row in primary}):
        for fraction in sorted({float(row["fraction"]) for row in primary}):
            subset = [
                row
                for row in primary
                if row["contrast"] == contrast and float(row["fraction"]) == fraction
            ]
            q = benjamini_hochberg(float(row[p_key]) for row in subset)
            h = holm(float(row[p_key]) for row in subset)
            for row, q_value, holm_value in zip(subset, q, h, strict=True):
                row["bh_q_within_fraction"] = str(float(q_value))
                row["holm_p_within_fraction"] = str(float(holm_value))

    row_output: list[dict[str, object]] = []
    for row in primary:
        row_output.append(
            {
                "model": row["model"],
                "dataset": row["dataset"],
                "dataset_family": dataset_family(row["dataset"]),
                "contrast": row["contrast"],
                "fraction": float(row["fraction"]),
                "effect": float(row[effect_key]),
                "p_greater": float(row[p_key]),
                "bh_q_within_fraction": float(row["bh_q_within_fraction"]),
                "holm_p_within_fraction": float(row["holm_p_within_fraction"]),
                "bh_q_global_contrast": float(row["bh_q_global_contrast"]),
                "holm_p_global_contrast": float(row["holm_p_global_contrast"]),
            }
        )

    summaries: list[dict[str, object]] = []
    for contrast in sorted({row["contrast"] for row in row_output}):
        for fraction in sorted({float(row["fraction"]) for row in row_output}):
            subset = [
                row
                for row in row_output
                if row["contrast"] == contrast and float(row["fraction"]) == fraction
            ]
            summaries.append(
                {
                    "contrast": contrast,
                    "fraction": fraction,
                    "configuration_count": len(subset),
                    "positive_effect_count": sum(float(row["effect"]) > 0 for row in subset),
                    "raw_p_below_0_05_count": sum(float(row["p_greater"]) < 0.05 for row in subset),
                    "bh_q_within_fraction_below_0_05_count": sum(
                        float(row["bh_q_within_fraction"]) < 0.05 for row in subset
                    ),
                    "holm_within_fraction_below_0_05_count": sum(
                        float(row["holm_p_within_fraction"]) < 0.05 for row in subset
                    ),
                    "bh_q_global_contrast_below_0_05_count": sum(
                        float(row["bh_q_global_contrast"]) < 0.05 for row in subset
                    ),
                    "holm_global_contrast_below_0_05_count": sum(
                        float(row["holm_p_global_contrast"]) < 0.05 for row in subset
                    ),
                }
            )
    return row_output, summaries


def _block_indices(rng: np.random.Generator, sample_count: int, block_length: int) -> np.ndarray:
    block_count = math.ceil(sample_count / block_length)
    starts = rng.integers(0, sample_count, size=block_count)
    blocks = [
        (start + np.arange(block_length, dtype=np.int64)) % sample_count for start in starts
    ]
    return np.concatenate(blocks)[:sample_count]


def _ratio_contrast_bootstrap(
    top_change: np.ndarray,
    top_rms: np.ndarray,
    control_change: np.ndarray,
    control_rms: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    donor_count, sample_count = top_change.shape
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        donor_indices = rng.integers(0, donor_count, size=donor_count)
        top_change_parts = []
        top_rms_parts = []
        control_change_parts = []
        control_rms_parts = []
        for donor in donor_indices:
            samples = _block_indices(rng, sample_count, block_length)
            top_change_parts.append(top_change[donor, samples])
            top_rms_parts.append(top_rms[donor, samples])
            control_change_parts.append(control_change[donor, samples])
            control_rms_parts.append(control_rms[donor, samples])
        top_ratio = np.concatenate(top_change_parts).mean() / max(
            float(np.concatenate(top_rms_parts).mean()), 1e-12
        )
        control_ratio = np.concatenate(control_change_parts).mean() / max(
            float(np.concatenate(control_rms_parts).mean()), 1e-12
        )
        estimates[repetition] = top_ratio - control_ratio
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def _intervention_rms_audit(conditions: list[dict[str, str]]) -> list[dict[str, object]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in conditions:
        unique[(row["model"], row["dataset"], row["run_dir"])] = row

    fraction_codes = {0.125: "012", 0.25: "025", 0.375: "038", 0.5: "050"}
    output: list[dict[str, object]] = []
    for run_index, ((model, dataset, run_dir), _row) in enumerate(sorted(unique.items())):
        result_path = Path(run_dir) / "progressive_results.npz"
        arrays = np.load(result_path)
        origins = np.asarray(arrays["origins"], dtype=np.int64)
        config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
        step = max(float(np.median(np.diff(origins))), 1.0)
        block_length = min(
            len(origins) // 2,
            max(2, int(math.ceil((config["data"]["seq_len"] + config["data"]["pred_len"]) / step))),
        )
        for fraction, code in fraction_codes.items():
            top_change = np.asarray(
                arrays[f"remove__top_f{code}__all_forecast_change_mse"], dtype=np.float64
            )
            top_rms = np.asarray(
                arrays[f"remove__top_f{code}__perturbation_rms"], dtype=np.float64
            )
            controls: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "bottom": (
                    np.asarray(
                        arrays[f"remove__bottom_f{code}__all_forecast_change_mse"],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        arrays[f"remove__bottom_f{code}__perturbation_rms"], dtype=np.float64
                    ),
                )
            }
            random_change = np.mean(
                np.stack(
                    [
                        np.asarray(
                            arrays[
                                f"remove__random_r{repetition:02d}_f{code}__all_forecast_change_mse"
                            ],
                            dtype=np.float64,
                        )
                        for repetition in range(5)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            random_rms = np.mean(
                np.stack(
                    [
                        np.asarray(
                            arrays[
                                f"remove__random_r{repetition:02d}_f{code}__perturbation_rms"
                            ],
                            dtype=np.float64,
                        )
                        for repetition in range(5)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            controls["random_mean"] = (random_change, random_rms)

            top_ratio = float(top_change.mean() / max(float(top_rms.mean()), 1e-12))
            for control_index, (control, (control_change, control_rms)) in enumerate(controls.items()):
                control_ratio = float(
                    control_change.mean() / max(float(control_rms.mean()), 1e-12)
                )
                lower, upper = _ratio_contrast_bootstrap(
                    top_change,
                    top_rms,
                    control_change,
                    control_rms,
                    block_length=block_length,
                    repetitions=2000,
                    seed=2021 + run_index * 100 + int(fraction * 100) + control_index,
                )
                output.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset_family": dataset_family(dataset),
                        "fraction": fraction,
                        "control": control,
                        "block_length_in_sampled_origins": block_length,
                        "top_effect_per_activation_rms": top_ratio,
                        "control_effect_per_activation_rms": control_ratio,
                        "contrast": top_ratio - control_ratio,
                        "block_bootstrap_ci95_lower": lower,
                        "block_bootstrap_ci95_upper": upper,
                        "block_bootstrap_positive": lower > 0,
                    }
                )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    topology = read_csv(SOURCE_TOPOLOGY)
    contrasts = read_csv(SOURCE_CONTRASTS)
    conditions = read_csv(SOURCE_CONDITIONS)

    family_topology = _family_topology(topology)
    atlas_by_model, atlas_global = _atlas_significance(topology)
    intervention_rows, intervention_summary = _intervention_multiplicity(contrasts)
    rms_rows = _intervention_rms_audit(conditions)

    write_csv(OUTPUT / "topology_by_dataset_family.csv", family_topology)
    write_csv(OUTPUT / "atlas_significance_by_model.csv", atlas_by_model)
    write_csv(OUTPUT / "intervention_multiplicity_cells.csv", intervention_rows)
    write_csv(OUTPUT / "intervention_multiplicity_summary.csv", intervention_summary)
    write_csv(OUTPUT / "intervention_rms_block_bootstrap.csv", rms_rows)

    rms_summary = []
    for control in ("bottom", "random_mean"):
        for fraction in sorted({float(row["fraction"]) for row in rms_rows}):
            subset = [
                row
                for row in rms_rows
                if row["control"] == control and float(row["fraction"]) == fraction
            ]
            rms_summary.append(
                {
                    "control": control,
                    "fraction": fraction,
                    "configuration_count": len(subset),
                    "positive_contrast_count": sum(float(row["contrast"]) > 0 for row in subset),
                    "block_bootstrap_positive_count": sum(
                        bool(row["block_bootstrap_positive"]) for row in subset
                    ),
                    "mean_contrast": mean(float(row["contrast"]) for row in subset),
                    "median_contrast": float(
                        np.median([float(row["contrast"]) for row in subset])
                    ),
                }
            )
    write_csv(OUTPUT / "intervention_rms_summary.csv", rms_summary)

    payload = {
        "status": "complete",
        "sources": {
            "topology": str(SOURCE_TOPOLOGY.relative_to(ROOT)),
            "contrasts": str(SOURCE_CONTRASTS.relative_to(ROOT)),
            "conditions": str(SOURCE_CONDITIONS.relative_to(ROOT)),
        },
        "atlas": atlas_global,
        "intervention_multiplicity": intervention_summary,
        "intervention_rms": rms_summary,
        "notes": {
            "intervention_family": (
                "Primary removal-mode top-minus-bottom and top-minus-random contrasts. "
                "BH and Holm corrections are reported both within each fraction and across all "
                "four fractions for a contrast."
            ),
            "block_bootstrap": (
                "Paired donor resampling plus circular moving blocks over ordered test origins. "
                "Block length is ceil((L+P)/median origin spacing), capped at half the sample."
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
