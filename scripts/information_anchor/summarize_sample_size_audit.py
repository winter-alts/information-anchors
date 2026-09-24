#!/usr/bin/env python3
"""Consolidate the ETTm1 nested-sample KSG audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results/information_anchor_sample_size"
OUTPUT = ROOT / "results/information_anchor_recency_target_audit"


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(RUN_ROOT.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({**payload, "run_dir": str(path.parent.resolve())})
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 sample-size runs, found {len(rows)}")
    fields = [
        "model", "dataset", "sample_count", "primary_sample_count",
        "patch_z_spearman_with_primary", "layer_z_spearman_with_primary",
        "cell_z_spearman_with_primary", "top_quarter_cell_jaccard_with_primary",
        "top_patch", "primary_top_patch", "top_layer", "primary_top_layer",
        "fraction_cell_q_below_0_05", "analysis_elapsed_seconds", "run_dir",
    ]
    path = OUTPUT / "sample_size_stability.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{field: row[field] for field in fields} for row in rows])
    summary: dict[str, object] = {
        "run_count": len(rows),
        "top_patch_same_count": sum(row["top_patch"] == row["primary_top_patch"] for row in rows),
        "protocol": (
            "Nested evenly spaced subsets of the registered 1024 ETTm1 discovery origins; "
            "primary future and layer-shared hidden PCA projections are locked, with KSG and "
            "199 legal temporal shifts recomputed on each subset."
        ),
    }
    for sample_count in (512, 768):
        subset = [row for row in rows if row["sample_count"] == sample_count]
        summary[str(sample_count)] = {
            field: float(np.mean([row[field] for row in subset]))
            for field in (
                "patch_z_spearman_with_primary", "layer_z_spearman_with_primary",
                "cell_z_spearman_with_primary", "top_quarter_cell_jaccard_with_primary",
            )
        }
        summary[str(sample_count)]["top_patch_same_count"] = sum(
            row["top_patch"] == row["primary_top_patch"] for row in subset
        )
        summary[str(sample_count)]["top_layer_same_count"] = sum(
            row["top_layer"] == row["primary_top_layer"] for row in subset
        )
    (OUTPUT / "sample_size_stability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
