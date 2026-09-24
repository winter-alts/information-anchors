#!/usr/bin/env python3
"""Audit V6 anchor selection under repeated chronological block subsampling.

The fitted hidden/future projections are deliberately held fixed.  This isolates
origin-selection uncertainty without turning duplicate bootstrap observations into
zero-distance KSG neighbours.  Each replicate retains a random 75% of contiguous
discovery blocks, restores chronological order, and recalibrates every atlas cell
against legal temporal circular shifts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    analysis_future_values,
    load_benchmark_frame,
    make_windows,
    select_value_columns,
)
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    robust_null_score,
    temporal_circular_shift_offsets,
)
from experiments.information_anchor.estimators.projection import load_projection
from experiments.information_anchor.interventions.common import select_functional_anchor
from experiments.information_anchor.targets.future_summary import build_future_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--keep-fraction", type=float, default=0.75)
    parser.add_argument("--null-permutations", type=int, default=49)
    parser.add_argument(
        "--output-root",
        default="results/information_anchor_reviewer_revision/stability_block_subsample",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_set(scores: np.ndarray, fraction: float = 0.25) -> tuple[int, ...]:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return tuple(sorted(int(value) for value in order[:count]))


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    reference = np.load(reference_dir / "mi_results.npz")
    origins = np.asarray(reference["origins"], dtype=np.int64)
    primary_cell_z = np.asarray(reference["mi_z"], dtype=np.float64)
    primary_layer_score = np.asarray(reference["layer_anchor_score"], dtype=np.float64)
    primary = select_functional_anchor(
        primary_cell_z, primary_layer_score, config.model.name
    )
    primary_global_top_patch = int(np.argmax(primary_cell_z.mean(axis=0)))
    primary_top_quarter = top_set(primary_cell_z[primary.layer])

    if not (0.5 <= args.keep_fraction < 1.0):
        raise ValueError("keep-fraction must be in [0.5, 1).")
    if args.blocks < 4 or args.blocks > len(origins):
        raise ValueError("blocks must be in [4, number of origins].")
    keep_blocks = int(round(args.blocks * args.keep_fraction))
    if keep_blocks >= args.blocks:
        raise ValueError("keep-fraction leaves no blocks out.")

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    windows = make_windows(
        values,
        origins,
        config.data.seq_len,
        config.data.pred_len,
        columns=columns,
    )
    future_values = analysis_future_values(
        windows,
        values,
        train_end=config.data.train_end,
        normalization=config.future_target.normalization,
    )
    future_summary = build_future_summary(
        future_values,
        bins=config.future_target.bins,
        spectral_bands=config.future_target.spectral_bands,
        channel_names=windows.columns,
    )
    future_projection = load_projection(reference_dir / "future_projection.npz")
    future_projected = future_projection.transform(future_summary.values)

    cache_info = json.loads(
        (reference_dir / "hidden_cache.json").read_text(encoding="utf-8")
    )
    cache_root = Path(config.runtime.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    hidden_path = cache_root / cache_info["cache_key"] / "history_hidden.npy"
    hidden = np.load(hidden_path, mmap_mode="r")
    n_samples, n_layers, n_patches, _ = hidden.shape
    if n_samples != len(origins) or primary_cell_z.shape != (n_layers, n_patches):
        raise ValueError("Reference origins, hidden cache, and MI atlas disagree.")

    projected_layers: list[np.ndarray] = []
    for layer in range(n_layers):
        projection = load_projection(
            reference_dir / f"hidden_projection_layer_{layer:02d}.npz"
        )
        layer_hidden = np.asarray(hidden[:, layer], dtype=np.float32)
        projected_layers.append(
            projection.transform(layer_hidden.reshape(n_samples * n_patches, -1))
            .reshape(n_samples, n_patches, -1)
            .astype(np.float32)
        )

    blocks = [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(n_samples), args.blocks)]
    rng = np.random.default_rng(config.mi.seed + 71_000)
    replicate_rows: list[dict[str, object]] = []
    cell_z_replicates = np.empty(
        (args.replicates, n_layers, n_patches), dtype=np.float32
    )
    required_separation = config.data.seq_len + config.data.pred_len

    for repetition in range(args.replicates):
        chosen_blocks = tuple(
            sorted(
                int(value)
                for value in rng.choice(args.blocks, size=keep_blocks, replace=False)
            )
        )
        indices = np.sort(np.concatenate([blocks[index] for index in chosen_blocks]))
        selected_origins = origins[indices]
        shifts = temporal_circular_shift_offsets(
            selected_origins,
            args.null_permutations,
            min_temporal_separation=required_separation,
            seed=config.mi.seed + 72_000 + repetition,
        )
        future_for_mi = add_deterministic_jitter(
            future_projected[indices],
            config.mi.jitter,
            config.mi.seed + 73_000 + repetition,
        )
        estimator = TorchKSG(
            future_for_mi,
            k=config.mi.k,
            device=args.device,
            shift_batch_size=config.mi.shift_batch_size,
        )
        cell_z = cell_z_replicates[repetition]
        for layer, projected in enumerate(projected_layers):
            selected = projected[indices]
            for patch in range(n_patches):
                x = add_deterministic_jitter(
                    selected[:, patch],
                    config.mi.jitter,
                    config.mi.seed
                    + 74_000
                    + repetition * n_layers * n_patches
                    + layer * n_patches
                    + patch,
                )
                observed, null_values = estimator.estimate_observed_and_shifts(x, shifts)
                cell_z[layer, patch] = robust_null_score(observed, null_values)[0]

        layer_scores = cell_z.mean(axis=1)
        selected_anchor = select_functional_anchor(
            cell_z, layer_scores, config.model.name
        )
        selected_top_quarter = top_set(cell_z[selected_anchor.layer])
        global_top_patch = int(np.argmax(cell_z.mean(axis=0)))
        replicate_rows.append(
            {
                "model": config.model.name,
                "dataset": config.data.dataset,
                "replicate": repetition,
                "kept_blocks": json.dumps(chosen_blocks),
                "sample_count": len(indices),
                "null_shift_count": len(shifts),
                "selected_layer": selected_anchor.layer,
                "top_patch_at_selected_layer": selected_anchor.top_patch,
                "global_top_patch": global_top_patch,
                "top_quarter_at_selected_layer": json.dumps(selected_top_quarter),
                "primary_layer_match": selected_anchor.layer == primary.layer,
                "primary_anchor_coordinate_match": (
                    selected_anchor.layer == primary.layer
                    and selected_anchor.top_patch == primary.top_patch
                ),
                "primary_top_patch_index_match": selected_anchor.top_patch == primary.top_patch,
                "primary_global_top_patch_match": global_top_patch == primary_global_top_patch,
                "primary_top_quarter_exact_match": selected_top_quarter == primary_top_quarter,
                "primary_top_quarter_jaccard": jaccard(
                    selected_top_quarter, primary_top_quarter
                ),
            }
        )
        print(
            f"model={config.model.name} replicate={repetition + 1}/{args.replicates} "
            f"layer={selected_anchor.layer} patch={selected_anchor.top_patch} "
            f"set_jaccard={replicate_rows[-1]['primary_top_quarter_jaccard']:.3f}",
            flush=True,
        )

    output = ROOT / args.output_root / config.model.name.lower().replace(".", "")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "replicates.csv", replicate_rows)
    np.savez_compressed(
        output / "cell_z_replicates.npz",
        cell_z=cell_z_replicates,
        primary_cell_z=primary_cell_z,
        origins=origins,
    )

    selected_layers = np.asarray(
        [int(row["selected_layer"]) for row in replicate_rows], dtype=np.int64
    )
    selected_patches = np.asarray(
        [int(row["top_patch_at_selected_layer"]) for row in replicate_rows],
        dtype=np.int64,
    )
    global_top_patches = np.asarray(
        [int(row["global_top_patch"]) for row in replicate_rows], dtype=np.int64
    )
    top_sets = [
        tuple(json.loads(str(row["top_quarter_at_selected_layer"])))
        for row in replicate_rows
    ]
    inclusion = {
        str(patch): float(np.mean([patch in values for values in top_sets]))
        for patch in range(n_patches)
    }
    summary = {
        "status": "complete",
        "model": config.model.name,
        "dataset": config.data.dataset,
        "reference_run": str(reference_dir),
        "protocol": "fixed-projection repeated chronological block subsampling",
        "reason_not_duplicate_bootstrap": (
            "Sampling blocks without replacement avoids duplicate zero-distance KSG neighbours; "
            "this is a selection-stability audit, not full projection-refit bootstrap."
        ),
        "replicates": args.replicates,
        "blocks": args.blocks,
        "kept_blocks_per_replicate": keep_blocks,
        "sample_count_per_replicate": int(replicate_rows[0]["sample_count"]),
        "null_permutations_per_replicate": args.null_permutations,
        "primary_selected_layer": primary.layer,
        "primary_top_patch_at_selected_layer": primary.top_patch,
        "primary_global_top_patch": primary_global_top_patch,
        "primary_top_quarter_at_selected_layer": list(primary_top_quarter),
        "primary_selected_layer_probability": float(np.mean(selected_layers == primary.layer)),
        "primary_anchor_coordinate_probability": float(
            np.mean(
                (selected_layers == primary.layer)
                & (selected_patches == primary.top_patch)
            )
        ),
        "primary_top_patch_index_probability_ignoring_layer": float(
            np.mean(selected_patches == primary.top_patch)
        ),
        "primary_global_top_patch_probability": float(
            np.mean(global_top_patches == primary_global_top_patch)
        ),
        "primary_top_quarter_exact_probability": float(
            np.mean(
                [values == primary_top_quarter for values in top_sets]
            )
        ),
        "primary_top_quarter_jaccard_mean": float(
            np.mean([jaccard(values, primary_top_quarter) for values in top_sets])
        ),
        "primary_top_quarter_jaccard_median": float(
            np.median([jaccard(values, primary_top_quarter) for values in top_sets])
        ),
        "selected_layer_probabilities": {
            str(layer): float(np.mean(selected_layers == layer))
            for layer in range(n_layers)
        },
        "top_patch_at_selected_layer_probabilities": {
            str(patch): float(np.mean(selected_patches == patch))
            for patch in range(n_patches)
        },
        "top_quarter_patch_inclusion_probabilities": inclusion,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
