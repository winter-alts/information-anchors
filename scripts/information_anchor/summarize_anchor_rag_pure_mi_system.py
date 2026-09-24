#!/usr/bin/env python3
"""Summarize pure-MI Anchor-RAG and locked system ablations by paired origin."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CHANNELS = {
    "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
    "weather": 21, "electricity": 321, "exchange_rate": 8,
}
ARMS = (
    "full", "tsrag", "without_mi_rank", "without_mi_distance",
    "without_mi_gate", "without_bcsa", "without_mi",
    "without_mi_bcsa", "without_residual_correction",
)


def bootstrap_reduction(reference, full, block, reps, rng):
    n = len(reference)
    starts = rng.integers(0, n, size=(reps, math.ceil(n / block)))
    offsets = np.arange(block)
    out = np.empty(reps)
    for start in range(0, reps, 128):
        stop = min(start + 128, reps)
        indices = (starts[start:stop, :, None] + offsets) % n
        indices = indices.reshape(stop - start, -1)[:, :n]
        ref_mean = reference[indices].mean(axis=1)
        full_mean = full[indices].mean(axis=1)
        out[start:stop] = 100.0 * (ref_mean - full_mean) / ref_mean
    return out


def signflip_p(reference, full, block, reps, rng):
    difference = reference - full
    blocks = np.asarray([
        difference[start:start + block].mean()
        for start in range(0, len(difference), block)
    ])
    observed = abs(blocks.mean())
    exceed = 0
    for start in range(0, reps, 2048):
        size = min(2048, reps - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(blocks)))
        exceed += np.count_nonzero(np.abs((signs * blocks).mean(axis=1)) >= observed - 1e-15)
    return (exceed + 1) / (reps + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--signflip-reps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    bootstrap_samples = {}
    input_files = {}

    for dataset in CHANNELS:
        path = args.results_root / dataset / dataset / "system_ablation_losses.npz"
        if not path.exists():
            path = args.results_root / dataset / "system_ablation_losses.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        input_files[dataset] = str(path.resolve())
        channels = CHANNELS[dataset]
        with np.load(path, allow_pickle=False) as values:
            losses = {}
            for arm in ARMS:
                losses[arm] = {}
                for metric in ("mse", "mae"):
                    window_losses = values[f"{arm}_{metric}"].astype(np.float64)
                    if len(window_losses) % channels:
                        raise ValueError(f"{dataset}/{arm}/{metric}: row count not divisible by channels")
                    origin_losses = window_losses.reshape(channels, -1).mean(axis=0)
                    losses[arm][metric] = origin_losses

        for metric in ("mse", "mae"):
            for arm in ARMS[1:]:
                # Positive effects favor Full. Table 1 uses TS-RAG as reference;
                # the ablation appendix uses each ablation as its reference.
                reference = losses[arm][metric]
                full = losses["full"][metric]
                draws = bootstrap_reduction(reference, full, args.block_length, args.bootstrap_reps, rng)
                p = signflip_p(reference, full, args.block_length, args.signflip_reps, rng)
                key = (arm, metric)
                bootstrap_samples.setdefault(key, []).append(draws)
                rows.append({
                    "dataset": dataset,
                    "comparison": f"Full vs {arm}",
                    "metric": metric,
                    "origins": len(full),
                    "reference_mean": float(reference.mean()),
                    "full_mean": float(full.mean()),
                    "relative_reduction_percent": float(100 * (reference.mean() - full.mean()) / reference.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "block_signflip_p_two_sided": float(p),
                })
            print(f"DONE {dataset}", flush=True)

    macro = []
    for (arm, metric), samples in bootstrap_samples.items():
        selected = [r for r in rows if r["comparison"] == f"Full vs {arm}" and r["metric"] == metric]
        draws = np.stack(samples, axis=1).mean(axis=1)
        macro.append({
            "dataset": "Macro mean",
            "comparison": f"Full vs {arm}",
            "metric": metric,
            "origins": sum(r["origins"] for r in selected),
            "reference_mean": "",
            "full_mean": "",
            "relative_reduction_percent": float(np.mean([r["relative_reduction_percent"] for r in selected])),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "block_signflip_p_two_sided": "",
            "dataset_wins": sum(r["relative_reduction_percent"] > 0 for r in selected),
            "significant_datasets_p_lt_0p05": sum(r["block_signflip_p_two_sided"] < 0.05 for r in selected),
        })

    for metric in ("mse", "mae"):
        family = [
            row for row in rows
            if row["metric"] == metric and row["comparison"] != "Full vs tsrag"
        ]
        ordered = sorted(family, key=lambda row: row["block_signflip_p_two_sided"])
        running = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * row["block_signflip_p_two_sided"])
            running = max(running, adjusted)
            row["holm_adjusted_p"] = running
        for row in family:
            row.setdefault("holm_adjusted_p", "")
        for row in macro:
            if row["metric"] == metric and row["comparison"] != "Full vs tsrag":
                row["significant_datasets_holm_p_lt_0p05"] = sum(
                    r["holm_adjusted_p"] < 0.05 for r in family
                    if r["comparison"] == row["comparison"]
                )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = rows + macro
    csv_path = output / "pure_mi_paired_inference.csv"
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    metadata = {
        "protocol": "Per-origin losses average channels after window-level horizon reduction; circular moving-block bootstrap and two-sided non-overlapping block sign-flip tests.",
        "seed": args.seed,
        "block_length_origins": args.block_length,
        "bootstrap_replicates": args.bootstrap_reps,
        "signflip_replicates": args.signflip_reps,
        "input_files": input_files,
        "arms": list(ARMS),
        "metrics": ["mse", "mae"],
    }
    (output / "pure_mi_paired_inference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
