#!/usr/bin/env python3
"""Summarize the paired linear recency-residual MI atlas audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/information_anchor_recency_residual"
OUTPUT = ROOT / "results/information_anchor_reviewer_revision"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(INPUT.glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete" or item.get("variant") != "recency_residual":
            continue
        residual = item["recency_residual_protocol"]
        rows.append(
            {
                "model": item["model"],
                "dataset": item["dataset"],
                "reference_run": item["reference_run"],
                "recent_cross_fitted_r2": residual[
                    "cross_fitted_r2_mean_over_target_dimensions"
                ],
                "residual_variance_fraction": residual[
                    "target_residual_variance_fraction"
                ],
                "cell_z_spearman": item["cell_z_spearman_with_primary"],
                "patch_z_spearman": item["patch_z_spearman_with_primary"],
                "layer_profile_spearman": item["layer_profile_spearman_with_primary"],
                "top_quarter_cell_jaccard": item[
                    "top_quarter_cell_jaccard_with_primary"
                ],
                "primary_top_patch": item["primary_top_patch"],
                "residual_top_patch": item["top_patch"],
                "same_top_patch": item["primary_top_patch"] == item["top_patch"],
                "primary_top_cell": json.dumps(item["primary_top_cell"]),
                "residual_top_cell": json.dumps(item["top_cell"]),
                "same_top_cell": item["primary_top_cell"] == item["top_cell"],
            }
        )
    if not rows:
        raise ValueError("No complete recency-residual runs found.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "recency_residual_by_run.csv", rows)
    numeric = (
        "recent_cross_fitted_r2",
        "residual_variance_fraction",
        "cell_z_spearman",
        "patch_z_spearman",
        "layer_profile_spearman",
        "top_quarter_cell_jaccard",
    )
    summary: dict[str, object] = {
        "run_count": len(rows),
        "models": sorted({str(row["model"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "same_top_patch": int(sum(bool(row["same_top_patch"]) for row in rows)),
        "same_top_cell": int(sum(bool(row["same_top_cell"]) for row in rows)),
        "recent_cross_fitted_r2_nonnegative": int(
            sum(float(row["recent_cross_fitted_r2"]) >= 0 for row in rows)
        ),
        "scope_note": "Covers ETT plus Weather for all six models; Electricity and Traffic are not yet included.",
        "interpretation_limit": "Cross-fitted linear residualization is a recency audit, not exact conditional mutual information.",
    }
    for field in numeric:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    summary["by_dataset"] = {
        dataset: {
            "run_count": sum(row["dataset"] == dataset for row in rows),
            "recent_cross_fitted_r2": float(
                np.mean(
                    [
                        float(row["recent_cross_fitted_r2"])
                        for row in rows
                        if row["dataset"] == dataset
                    ]
                )
            ),
            "same_top_patch": int(
                sum(
                    bool(row["same_top_patch"])
                    for row in rows
                    if row["dataset"] == dataset
                )
            ),
        }
        for dataset in sorted({str(row["dataset"]) for row in rows})
    }
    (OUTPUT / "recency_residual_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
