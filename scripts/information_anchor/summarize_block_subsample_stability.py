#!/usr/bin/env python3
"""Consolidate reviewer-facing block-subsample selection stability results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "results/information_anchor_reviewer_revision/stability_block_subsample"
)
OUTPUT = ROOT / "results/information_anchor_reviewer_revision"


FIELDS = (
    "primary_selected_layer_probability",
    "primary_anchor_coordinate_probability",
    "primary_top_patch_index_probability_ignoring_layer",
    "primary_global_top_patch_probability",
    "primary_top_quarter_exact_probability",
    "primary_top_quarter_jaccard_mean",
)


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(INPUT.glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            continue
        rows.append(
            {
                "model": item["model"],
                "dataset": item["dataset"],
                "replicates": item["replicates"],
                "sample_count_per_replicate": item["sample_count_per_replicate"],
                "primary_selected_layer": item["primary_selected_layer"],
                "primary_top_patch_at_selected_layer": item[
                    "primary_top_patch_at_selected_layer"
                ],
                "primary_global_top_patch": item["primary_global_top_patch"],
                **{field: item[field] for field in FIELDS},
                "elapsed_seconds": item["elapsed_seconds"],
                "reference_run": item["reference_run"],
            }
        )
    if len(rows) != 6:
        raise ValueError(f"Expected six model summaries, found {len(rows)}.")
    with (OUTPUT / "stability_block_subsample_by_model.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "status": "complete",
        "model_count": len(rows),
        "dataset": "ETTm1",
        "replicates_per_model": int(rows[0]["replicates"]),
        "total_replicates": int(sum(int(row["replicates"]) for row in rows)),
        "protocol": (
            "Fixed-projection repeated chronological block subsampling: split 1,024 "
            "discovery origins into 16 contiguous blocks, retain 12 without replacement, "
            "and recalibrate all cells with 49 legal temporal circular shifts."
        ),
        "scope_limit": (
            "This avoids duplicate zero-distance KSG neighbours and measures origin-selection "
            "stability; it does not refit hidden/future PCA and covers ETTm1 only."
        ),
    }
    for field in FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary[field] = {
            "mean_across_models": float(values.mean()),
            "median_across_models": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    summary["by_model"] = {
        str(row["model"]): {field: float(row[field]) for field in FIELDS}
        for row in rows
    }
    (OUTPUT / "stability_block_subsample_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
