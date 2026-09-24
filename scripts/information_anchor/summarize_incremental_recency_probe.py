#!/usr/bin/env python3
"""Summarize semantic accessibility incremental to raw recent history."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/information_anchor_recency_target_audit/incremental_recency_probe_semantics.csv"
OUTPUT = ROOT / "results/information_anchor_recency_target_audit"


def read() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read()
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["model"], row["dataset_key"], row["dataset"], row["scope"])
        grouped.setdefault(key, {}).setdefault(row["condition"], []).append(
            float(row["incremental_test_r2"])
        )
    records: list[dict[str, object]] = []
    for (model, dataset_key, dataset, scope), conditions in sorted(grouped.items()):
        high = np.asarray(conditions["high"], dtype=np.float64)
        low = np.asarray(conditions["low"], dtype=np.float64)
        records.append(
            {
                "model": model,
                "dataset_key": dataset_key,
                "dataset": dataset,
                "scope": scope,
                "semantic_count": len(high),
                "high_incremental_mean_r2": float(high.mean()),
                "low_incremental_mean_r2": float(low.mean()),
                "high_minus_low_incremental_mean_r2": float((high - low).mean()),
                "high_positive_semantic_count": int(np.sum(high > 0)),
                "low_positive_semantic_count": int(np.sum(low > 0)),
                "high_greater_than_low_semantic_count": int(np.sum(high > low)),
            }
        )
    path = OUTPUT / "incremental_recency_probe_conditions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    summary: dict[str, object] = {"condition_count": len(records)}
    for scope in ("global", "target"):
        subset = [row for row in records if row["scope"] == scope]
        high = np.asarray([row["high_incremental_mean_r2"] for row in subset], dtype=np.float64)
        low = np.asarray([row["low_incremental_mean_r2"] for row in subset], dtype=np.float64)
        summary[scope] = {
            "model_dataset_count": len(subset),
            "high_incremental_mean_r2": float(high.mean()),
            "high_incremental_median_r2": float(np.median(high)),
            "high_positive_count": int(np.sum(high > 0)),
            "low_incremental_mean_r2": float(low.mean()),
            "low_incremental_median_r2": float(np.median(low)),
            "high_minus_low_mean_r2": float(np.mean(high - low)),
            "high_greater_than_low_count": int(np.sum(high > low)),
            "by_dataset": {
                dataset: {
                    "high_incremental_mean_r2": float(
                        np.mean([row["high_incremental_mean_r2"] for row in subset if row["dataset"] == dataset])
                    ),
                    "high_minus_low_mean_r2": float(
                        np.mean([row["high_minus_low_incremental_mean_r2"] for row in subset if row["dataset"] == dataset])
                    ),
                }
                for dataset in sorted({str(row["dataset"]) for row in subset})
            },
        }
    (OUTPUT / "incremental_recency_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
