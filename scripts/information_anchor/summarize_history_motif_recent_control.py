#!/usr/bin/env python3
"""Quantify the matched high-MI versus recent history-motif contrast.

This is a read-only reanalysis of saved pointwise MI scores and deterministic
history-only motif labels.  It keeps the per-origin selector budget fixed and
uses circular moving blocks over chronological origins for uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched high-MI versus recent motif contrasts."
    )
    parser.add_argument(
        "--input-root",
        default="results/information_anchor_history_motif_matrix",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--critic-accuracy-threshold", type=float, default=0.60)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--permutation-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def _topk_mask(scores: np.ndarray, keep: int) -> np.ndarray:
    indices = np.argpartition(scores, -keep, axis=1)[:, -keep:]
    mask = np.zeros(scores.shape, dtype=bool)
    mask[np.arange(len(scores))[:, None], indices] = True
    return mask


def _selected_rates(mask: np.ndarray, motifs: np.ndarray) -> np.ndarray:
    """Return [origin, motif] selection rates for an equal-budget mask."""
    selected = (motifs & mask[:, :, None]).sum(axis=1)
    counts = mask.sum(axis=1, keepdims=True)
    if not np.all(counts == counts[0, 0]) or counts[0, 0] < 1:
        raise ValueError("Every origin must select the same positive patch budget.")
    return selected / counts


def _circular_blocks(
    sample_count: int,
    block_size: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_count < 1 or block_size < 1 or replicates < 1:
        raise ValueError("sample_count, block_size, and replicates must be positive.")
    block_count = math.ceil(sample_count / block_size)
    starts = rng.integers(0, sample_count, size=(replicates, block_count))
    offsets = np.arange(block_size, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % sample_count
    return indices.reshape(replicates, -1)[:, :sample_count]


def _paired_block_statistics(
    differences: np.ndarray,
    *,
    block_size: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean effect, circular-block CI, and block-label-permutation p value."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2:
        raise ValueError(f"differences must be [origin, motif], got {differences.shape}")
    sample_count, motif_count = differences.shape
    observed = differences.mean(axis=0)
    rng = np.random.default_rng(seed)
    bootstrap_indices = _circular_blocks(
        sample_count, block_size, bootstrap_replicates, rng
    )
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975], axis=0).T

    # Swapping High and Recent within a chronological block is a paired null:
    # it retains each origin's motif labels and selector budget while removing
    # a systematic selector identity effect.
    block_count = math.ceil(sample_count / block_size)
    padded = np.zeros((block_count * block_size, motif_count), dtype=np.float64)
    padded[:sample_count] = differences
    block_means = padded.reshape(block_count, block_size, motif_count).mean(axis=1)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]), size=(permutation_replicates, block_count, 1)
    )
    null_means = (signs * block_means[None, :, :]).mean(axis=1)
    p_values = (
        1.0
        + np.sum(np.abs(null_means) >= np.abs(observed)[None, :], axis=0)
    ) / (permutation_replicates + 1.0)
    return observed, interval, p_values


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
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
    if args.block_size < 1:
        raise ValueError("block-size must be positive")
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else input_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        directory
        for directory in input_root.iterdir()
        if directory.is_dir()
        and (directory / "summary.json").exists()
        and (directory / "motif_enrichment.npz").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No motif runs found below {input_root}")

    rows: list[dict[str, object]] = []
    for run_index, run_dir in enumerate(run_dirs):
        with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        pointwise_dir = Path(summary["pointwise_run"])
        with (pointwise_dir / "summary.json").open("r", encoding="utf-8") as handle:
            pointwise_summary = json.load(handle)
        pointwise = np.load(pointwise_dir / "pointwise_mi_results.npz")
        motifs_archive = np.load(run_dir / "motif_enrichment.npz")
        scores = np.asarray(pointwise["scores"], dtype=np.float64)
        motifs = np.asarray(motifs_archive["motifs"], dtype=bool)
        motif_names = [str(value) for value in motifs_archive["motif_names"]]
        if scores.shape[1] != 1:
            raise ValueError(
                f"Expected one selected MI layer in {pointwise_dir}, got {scores.shape[1]}"
            )
        layer_scores = scores[:, 0]
        if layer_scores.shape[:2] != motifs.shape[:2]:
            raise ValueError(
                f"Pointwise/motif shape mismatch: {layer_scores.shape} versus {motifs.shape}"
            )
        keep = int(summary["keep"])
        high_mask = _topk_mask(layer_scores, keep)
        recent_mask = np.zeros_like(high_mask)
        recent_mask[:, -keep:] = True
        differences = _selected_rates(high_mask, motifs) - _selected_rates(recent_mask, motifs)
        effect, interval, p_values = _paired_block_statistics(
            differences,
            block_size=min(args.block_size, len(differences)),
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
            seed=args.seed + run_index * 10_003,
        )
        critic_accuracy = float(pointwise_summary["mean_critic_heldout_accuracy"])
        critic_valid = critic_accuracy >= args.critic_accuracy_threshold
        for motif_index, motif in enumerate(motif_names):
            rows.append(
                {
                    "run": run_dir.name,
                    "model": summary["model"],
                    "dataset": summary["dataset"],
                    "layer": int(summary["layers"][0]),
                    "num_origins": int(summary["num_origins"]),
                    "num_patches": int(summary["num_patches"]),
                    "keep": keep,
                    "motif": motif,
                    "high_minus_recent": float(effect[motif_index]),
                    "ci95_low": float(interval[motif_index, 0]),
                    "ci95_high": float(interval[motif_index, 1]),
                    "block_permutation_p": float(p_values[motif_index]),
                    "critic_heldout_accuracy": critic_accuracy,
                    "critic_valid": critic_valid,
                }
            )

    valid_indices = [index for index, row in enumerate(rows) if row["critic_valid"]]
    q_values = _bh_adjust(
        np.asarray([float(rows[index]["block_permutation_p"]) for index in valid_indices])
    )
    q_by_index = dict(zip(valid_indices, q_values))
    for index, row in enumerate(rows):
        row["block_permutation_fdr_q"] = (
            float(q_by_index[index]) if index in q_by_index else ""
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["critic_valid"]:
            grouped[str(row["motif"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for motif, motif_rows in sorted(grouped.items()):
        effects = np.asarray(
            [float(row["high_minus_recent"]) for row in motif_rows], dtype=np.float64
        )
        q = np.asarray(
            [float(row["block_permutation_fdr_q"]) for row in motif_rows],
            dtype=np.float64,
        )
        summary_rows.append(
            {
                "motif": motif,
                "valid_runs": len(motif_rows),
                "mean_high_minus_recent": float(effects.mean()),
                "median_high_minus_recent": float(np.median(effects)),
                "positive_valid_runs": int(np.sum(effects > 0.0)),
                "fdr_significant_positive_valid_runs": int(
                    np.sum((effects > 0.0) & (q < 0.05))
                ),
                "fdr_significant_negative_valid_runs": int(
                    np.sum((effects < 0.0) & (q < 0.05))
                ),
            }
        )

    _write_csv(output_root / "motif_recent_control_by_run.csv", rows)
    _write_csv(output_root / "motif_recent_control_summary.csv", summary_rows)
    payload = {
        "status": "complete",
        "input_root": str(input_root),
        "num_runs": len(run_dirs),
        "num_valid_runs": len({str(row["run"]) for row in rows if row["critic_valid"]}),
        "protocol": {
            "selection": "top 25% pointwise-MI patches versus final 25% recent patches per origin",
            "motifs": "deterministic history-only patch labels",
            "uncertainty": f"95% circular moving-block bootstrap, block_size={args.block_size}",
            "p_value": (
                "paired block-level High/Recent label-swap permutation; "
                f"{args.permutation_replicates} replicates"
            ),
            "multiple_testing": (
                f"BH-FDR across {len(valid_indices)} valid run-motif comparisons"
            ),
            "critic_gate": args.critic_accuracy_threshold,
        },
        "motif_summary": summary_rows,
    }
    with (output_root / "motif_recent_control_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
