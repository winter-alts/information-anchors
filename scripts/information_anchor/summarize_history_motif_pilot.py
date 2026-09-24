#!/usr/bin/env python3
"""Aggregate direction checks from multiple history-motif enrichment runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="results/information_anchor_history_motif_pilot",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--critic-accuracy-threshold", type=float, default=0.60)
    return parser.parse_args()


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for a flat family of tests."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    count = len(values)
    adjusted_ranked = np.minimum.accumulate((ranked * count / np.arange(1, count + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else input_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        directory for directory in input_root.iterdir()
        if directory.is_dir() and (directory / "summary.json").exists() and (directory / "motif_enrichment.csv").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No motif enrichment runs found below {input_root}")

    run_rows: list[dict[str, object]] = []
    motif_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        pointwise_summary_path = Path(summary["pointwise_run"]) / "summary.json"
        with pointwise_summary_path.open("r", encoding="utf-8") as handle:
            pointwise_summary = json.load(handle)
        critic_accuracy = float(pointwise_summary["mean_critic_heldout_accuracy"])
        critic_valid = critic_accuracy >= args.critic_accuracy_threshold
        csv_rows = list(csv.DictReader((run_dir / "motif_enrichment.csv").open("r", encoding="utf-8")))
        layer = int(summary["layers"][0])
        by_key = {(row["selector"], row["motif"]): row for row in csv_rows if int(row["layer"]) == layer}
        motifs = list(summary["motif_names"])
        high_deltas = []
        for motif in motifs:
            high = by_key[("high_mi", motif)]
            low = by_key[("low_mi", motif)]
            recent = by_key[("recent", motif)]
            high_delta = float(high["delta"])
            low_delta = float(low["delta"])
            recent_delta = float(recent["delta"])
            high_deltas.append(high_delta)
            motif_rows.append(
                {
                    "run": run_dir.name,
                    "model": summary["model"],
                    "dataset": summary["dataset"],
                    "layer": layer,
                    "num_patches": int(summary["num_patches"]),
                    "patch_len": int(summary["patch_len"]),
                    "motif": motif,
                    "high_delta": high_delta,
                    "high_permutation_p": float(high["permutation_p"]),
                    "low_delta": low_delta,
                    "recent_delta": recent_delta,
                    "high_minus_low": high_delta - low_delta,
                    "high_minus_recent": high_delta - recent_delta,
                    "critic_heldout_accuracy": critic_accuracy,
                    "critic_valid": critic_valid,
                }
            )
        strongest_index = int(np.argmax(high_deltas))
        strongest_motif = motifs[strongest_index]
        strongest = by_key[("high_mi", strongest_motif)]
        recent = by_key[("recent", strongest_motif)]
        run_rows.append(
            {
                "run": run_dir.name,
                "model": summary["model"],
                "dataset": summary["dataset"],
                "layer": layer,
                "num_patches": int(summary["num_patches"]),
                "patch_len": int(summary["patch_len"]),
                "heldout_pointwise_critic_accuracy": critic_accuracy,
                "critic_valid": critic_valid,
                "top_high_mi_motif": strongest_motif,
                "top_high_mi_delta": float(strongest["delta"]),
                "top_high_mi_p": float(strongest["permutation_p"]),
                "top_high_minus_recent": float(strongest["delta"]) - float(recent["delta"]),
            }
        )

    valid_indices = [index for index, row in enumerate(motif_rows) if row["critic_valid"]]
    high_p_values = np.asarray(
        [float(motif_rows[index]["high_permutation_p"]) for index in valid_indices]
    )
    high_q_values = _bh_adjust(high_p_values) if len(high_p_values) else np.asarray([])
    q_by_index = dict(zip(valid_indices, high_q_values))
    for index, row in enumerate(motif_rows):
        row["high_fdr_q"] = float(q_by_index[index]) if index in q_by_index else ""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in motif_rows:
        grouped[str(row["motif"])].append(row)
    summary_rows = []
    for motif, rows in sorted(grouped.items()):
        high_delta = np.asarray([float(row["high_delta"]) for row in rows])
        valid_rows = [row for row in rows if row["critic_valid"]]
        valid_high_delta = np.asarray([float(row["high_delta"]) for row in valid_rows])
        valid_high_minus_low = np.asarray([float(row["high_minus_low"]) for row in valid_rows])
        valid_high_minus_recent = np.asarray([float(row["high_minus_recent"]) for row in valid_rows])
        q_values = np.asarray([float(row["high_fdr_q"]) for row in valid_rows]) if valid_rows else np.asarray([])
        summary_rows.append(
            {
                "motif": motif,
                "num_runs": len(rows),
                "valid_runs": len(valid_rows),
                "mean_high_delta_valid": float(valid_high_delta.mean()) if len(valid_rows) else "",
                "median_high_delta_valid": float(np.median(valid_high_delta)) if len(valid_rows) else "",
                "positive_high_valid_runs": int(np.sum(valid_high_delta > 0.0)),
                "fdr_significant_positive_valid_runs": int(
                    np.sum((valid_high_delta > 0.0) & (q_values < 0.05))
                ),
                "mean_high_minus_low_valid": float(valid_high_minus_low.mean()) if len(valid_rows) else "",
                "high_beats_low_valid_runs": int(np.sum(valid_high_minus_low > 0.0)),
                "mean_high_minus_recent_valid": float(valid_high_minus_recent.mean()) if len(valid_rows) else "",
                "high_beats_recent_valid_runs": int(np.sum(valid_high_minus_recent > 0.0)),
            }
        )

    _write_csv(output_root / "motif_direction_by_run.csv", motif_rows)
    _write_csv(output_root / "motif_direction_summary.csv", summary_rows)
    _write_csv(output_root / "run_direction_summary.csv", run_rows)
    aggregate = {
        "status": "complete",
        "input_root": str(input_root),
        "num_runs": len(run_rows),
        "num_valid_runs": len({str(row["run"]) for row in motif_rows if row["critic_valid"]}),
        "models": sorted({str(row["model"]) for row in run_rows}),
        "datasets": sorted({str(row["dataset"]) for row in run_rows}),
        "protocol": {
            "selection": "top 25% patches per origin",
            "mi_layer": "reference global-MI anchor layer; older references use maximum layer_mi_z",
            "critic_accuracy_threshold": args.critic_accuracy_threshold,
            "high_p_family": (
                f"BH-FDR across {len(valid_indices) // 11} valid runs x 11 motifs "
                f"({len(valid_indices)} tests)"
            ),
            "high_minus_recent": "effect-size contrast only; no independent paired p-value is claimed",
        },
        "motif_summary": summary_rows,
        "run_summary": run_rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
