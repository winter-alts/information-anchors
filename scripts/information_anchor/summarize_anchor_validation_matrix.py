#!/usr/bin/env python3
"""Build transparent per-atlas information-anchor validation definitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/information_anchor_v6_evidence"
AUDIT = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"
OUTPUT = ROOT / "results/information_anchor_reviewer_revision"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    selected = read_csv(EVIDENCE / "probe_selected_units.csv")
    groups = read_csv(EVIDENCE / "probe_semantic_groups.csv")
    contrasts = read_csv(EVIDENCE / "progressive_contrasts.csv")

    selected_lookup = {
        (row["model"], row["dataset_key"], row["scope"]): row for row in selected
    }
    group_lookup = {
        (row["model"], row["dataset_key"], row["scope"], row["semantic_group"]): row
        for row in groups
    }
    functional: dict[tuple[str, str, str, str], list[float]] = {}
    for row in contrasts:
        if row["contrast"] != "top_minus_bottom":
            continue
        key = (row["model"], row["dataset_key"], row["mode"], "all_delta_mse")
        functional.setdefault(key, []).append(float(row["all_delta_mse_mean"]))

    rows: list[dict[str, object]] = []
    for record in audit["records"]:
        model, dataset_key = str(record["model"]), str(record["dataset_key"])
        run_dir = Path(record["reuse_run_dir"])
        arrays = np.load(run_dir / "mi_results.npz")
        global_probe = selected_lookup[(model, dataset_key, "global")]
        target_probe = selected_lookup[(model, dataset_key, "target")]
        layer = int(global_probe["functional_layer"])
        patch = int(global_probe["functional_top_patch"])
        if layer != int(target_probe["functional_layer"]) or patch != int(target_probe["functional_top_patch"]):
            raise ValueError(f"Probe scopes disagree on selected unit for {(model, dataset_key)}")
        coarse: dict[str, tuple[float, float]] = {}
        for scope in ("global", "target"):
            entries = [
                group_lookup[(model, dataset_key, scope, group)]
                for group in ("level", "trajectory")
            ]
            weights = np.asarray([float(item["semantic_count"]) for item in entries])
            top = float(
                np.average([float(item["functional_top_r2"]) for item in entries], weights=weights)
            )
            gap = float(
                np.average(
                    [float(item["functional_top_minus_low_r2"]) for item in entries],
                    weights=weights,
                )
            )
            coarse[scope] = (top, gap)

        remove_effect = float(
            np.mean(functional[(model, dataset_key, "remove", "all_delta_mse")])
        )
        keep_raw = float(np.mean(functional[(model, dataset_key, "keep", "all_delta_mse")]))
        keep_oriented = -keep_raw
        global_all_top = float(global_probe["functional_top_mean_r2"])
        global_all_gap = float(global_probe["functional_top_minus_low_r2"])
        target_all_top = float(target_probe["functional_top_mean_r2"])
        target_all_gap = float(target_probe["functional_top_minus_low_r2"])
        coarse_global_pass = coarse["global"][0] > 0 and coarse["global"][1] > 0
        coarse_target_pass = coarse["target"][0] > 0 and coarse["target"][1] > 0
        mi_q = float(arrays["q_values"][layer, patch])
        mi_pass = mi_q < 0.05
        row: dict[str, object] = {
            "model": model,
            "dataset_key": dataset_key,
            "dataset": record["dataset"],
            "candidate_layer": layer,
            "candidate_patch": patch,
            "candidate_cell_q": mi_q,
            "mi_null_calibrated_pass": mi_pass,
            "global_all12_top_r2": global_all_top,
            "global_all12_top_minus_low_r2": global_all_gap,
            "target_all12_top_r2": target_all_top,
            "target_all12_top_minus_low_r2": target_all_gap,
            "global_coarse_top_r2": coarse["global"][0],
            "global_coarse_top_minus_low_r2": coarse["global"][1],
            "target_coarse_top_r2": coarse["target"][0],
            "target_coarse_top_minus_low_r2": coarse["target"][1],
            "remove_ground_truth_top_minus_low": remove_effect,
            "keep_ground_truth_raw_top_minus_low": keep_raw,
            "keep_ground_truth_oriented_top_advantage": keep_oriented,
            "pass_manuscript_relative_global": mi_pass and global_all_gap > 0 and remove_effect > 0,
            "pass_absolute_all12_global": mi_pass and global_all_top > 0 and global_all_gap > 0 and remove_effect > 0,
            "pass_absolute_all12_target": mi_pass and target_all_top > 0 and target_all_gap > 0 and remove_effect > 0,
            "pass_coarse_global": mi_pass and coarse_global_pass and remove_effect > 0,
            "pass_coarse_target": mi_pass and coarse_target_pass and remove_effect > 0,
            "pass_coarse_either_scope": mi_pass and (coarse_global_pass or coarse_target_pass) and remove_effect > 0,
            "pass_coarse_both_scopes": mi_pass and coarse_global_pass and coarse_target_pass and remove_effect > 0,
            "pass_coarse_either_and_both_functional_modes": (
                mi_pass
                and (coarse_global_pass or coarse_target_pass)
                and remove_effect > 0
                and keep_oriented > 0
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row["model"]), str(row["dataset_key"])))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "anchor_validation_by_atlas.csv", rows)

    definition_fields = [key for key in rows[0] if key.startswith("pass_")]
    summary = {
        "atlas_count": len(rows),
        "candidate_definition": (
            "Null-calibrated MI top patch at the preregistered reachable functional layer; "
            "all accessibility and intervention outcomes are held out from selection."
        ),
        "functional_definition": (
            "Mean top-minus-low train-standardized all-variable ground-truth delta MSE over "
            "the four nested fractions; keep mode is multiplied by -1 so positive is favorable."
        ),
        "definition_sensitivity": {
            field: {
                "passes": int(sum(bool(row[field]) for row in rows)),
                "fails": int(sum(not bool(row[field]) for row in rows)),
            }
            for field in definition_fields
        },
        "by_model": {
            model: {
                field: int(sum(bool(row[field]) for row in rows if row["model"] == model))
                for field in definition_fields
            }
            for model in sorted({str(row["model"]) for row in rows})
        },
    }
    (OUTPUT / "anchor_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
