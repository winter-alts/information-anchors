"""Regenerate reviewer-facing summaries from the bundled source tables.

This script does not rerun a model. It fixes the reporting unit to one
four-budget model--regime curve, extracts the held-out roughness audit, and
records the saved perturbation-scale and matched-donor controls.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[3]
TABLES = ARTIFACT / "tables"
OUT = TABLES / "information_anchor_reviewer_revision"

FUNCTIONAL_KEYS = (
    ("remove", "high-low", "remove_top_minus_bottom_all_delta_mse"),
    ("remove", "high-random", "remove_top_minus_random_mean_all_delta_mse"),
    ("keep", "high-low", "keep_top_minus_bottom_all_delta_mse"),
    ("keep", "high-random", "keep_top_minus_random_mean_all_delta_mse"),
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def functional_summary() -> list[dict]:
    source = read_json(
        TABLES
        / "information_anchor_recency_target_audit"
        / "hierarchical_progressive_summary.json"
    )
    rows = []
    for mode, contrast, key in FUNCTIONAL_KEYS:
        item = source[key]
        ci_low, ci_high = item["two_way_model_dataset_bootstrap_ci95"]
        rows.append(
            {
                "mode": mode,
                "contrast": contrast,
                "positive_curves": item["positive_model_dataset_curves"],
                "curve_count": (
                    item["positive_model_dataset_curves"]
                    + item["negative_model_dataset_curves"]
                ),
                "mean_effect": item["mean_oriented_effect"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "reporting_unit": "mean of four registered budgets per model-regime curve",
                "interval": "two-way model x dataset bootstrap",
            }
        )
    return rows


def roughness_summary() -> list[dict]:
    path = TABLES / "information_anchor_v6_evidence" / "probe_semantics.csv"
    grouped: dict[str, list[dict]] = {"global": [], "target": []}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            semantic = row["semantic"]
            is_roughness = semantic == "roughness" or semantic.endswith(":roughness")
            if is_roughness and row["scope"] in grouped:
                grouped[row["scope"]].append(row)

    rows = []
    for scope, values in grouped.items():
        top = [float(row["functional_top_r2"]) for row in values]
        low = [float(row["functional_low_r2"]) for row in values]
        difference = [float(row["functional_top_minus_low_r2"]) for row in values]
        rows.append(
            {
                "scope": scope,
                "property": "second-difference roughness",
                "curve_count": len(values),
                "mean_top_r2": sum(top) / len(top),
                "mean_low_r2": sum(low) / len(low),
                "mean_top_minus_low_r2": sum(difference) / len(difference),
                "positive_top_minus_low": sum(value > 0 for value in difference),
                "selection_status": "not included in the MI future summary",
            }
        )
    return rows


def scale_summary() -> list[dict]:
    path = TABLES / "information_anchor_iclr_reanalysis" / "intervention_rms_summary.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "control": row["control"],
                    "fraction": float(row["fraction"]),
                    "configuration_count": int(row["configuration_count"]),
                    "block_bootstrap_positive_count": int(row["block_bootstrap_positive_count"]),
                    "mean_rms_normalized_contrast": float(row["mean_contrast"]),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    functional = functional_summary()
    roughness = roughness_summary()
    scale = scale_summary()
    matched = read_json(
        TABLES
        / "information_anchor_recency_target_audit"
        / "matched_donor_summary.json"
    )

    write_csv(OUT / "review_resolution_functional.csv", list(functional[0]), functional)
    write_csv(OUT / "review_resolution_probe.csv", list(roughness[0]), roughness)
    write_csv(OUT / "review_resolution_scale.csv", list(scale[0]), scale)
    with (OUT / "review_resolution_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "functional": functional,
                "target_independent_probe": roughness,
                "rms_normalized_control": scale,
                "matched_donor_control": matched,
                "interpretation": (
                    "The three axes are non-equivalent. MI nominates candidates; probes "
                    "measure held-out linear accessibility; donor replacement measures "
                    "prediction response under the saved donor distribution."
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
