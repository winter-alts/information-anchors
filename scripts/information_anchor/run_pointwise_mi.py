#!/usr/bin/env python3
"""Compute cross-fitted pointwise MI scores from an existing hidden cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.artifacts import capture_run_context, create_run_dir, write_json
from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    select_value_columns,
)
from experiments.information_anchor.estimators.pointwise import cross_fitted_pointwise_mi
from experiments.information_anchor.estimators.projection import load_projection
from experiments.information_anchor.targets.future_summary import build_future_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-root", default="results/information_anchor_pointwise_mi")
    parser.add_argument("--layers", default="", help="Comma-separated zero-based layer indices; default global anchor layer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def _local_peak_mask(scores: np.ndarray) -> np.ndarray:
    """Apply the paper's per-origin Q3+1.5 IQR rule across patches."""
    scores = np.asarray(scores, dtype=np.float64)
    q1 = np.quantile(scores, 0.25, axis=1, keepdims=True)
    q3 = np.quantile(scores, 0.75, axis=1, keepdims=True)
    return scores > q3 + 1.5 * (q3 - q1)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    with (reference_dir / "hidden_cache.json").open("r", encoding="utf-8") as handle:
        hidden_info = json.load(handle)

    if args.layers.strip():
        layer_indices = np.asarray(
            [int(value.strip()) for value in args.layers.split(",") if value.strip()],
            dtype=np.int64,
        )
    else:
        mi_reference = np.load(reference_dir / "mi_results.npz")
        # v5 references store the global anchor score explicitly. Older v2
        # references predate that field; their layer-level null-calibrated MI
        # z-score is the equivalent documented global anchor statistic.
        anchor_scores = (
            mi_reference["layer_anchor_score"]
            if "layer_anchor_score" in mi_reference.files
            else mi_reference["layer_mi_z"]
        )
        layer_indices = np.asarray([int(np.argmax(anchor_scores))], dtype=np.int64)
    if len(layer_indices) == 0 or np.any(layer_indices < 0):
        raise ValueError("layers must contain at least one non-negative layer index")

    origins = build_forecast_origins(config.data)
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

    cache_root = Path(config.runtime.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    hidden_path = cache_root / hidden_info["cache_key"] / "history_hidden.npy"
    hidden = np.load(hidden_path, mmap_mode="r")
    if hidden.shape[0] != len(origins):
        raise ValueError(
            f"Hidden cache has {hidden.shape[0]} origins, but config produced {len(origins)} origins"
        )

    layer_outputs = []
    raw_layer_outputs = []
    accuracies = []
    train_accuracies = []
    for layer_index in layer_indices:
        if layer_index >= hidden.shape[1]:
            raise ValueError(f"Layer {layer_index} is outside hidden cache with {hidden.shape[1]} layers")
        projection = load_projection(reference_dir / f"hidden_projection_layer_{layer_index:02d}.npz")
        layer_hidden = np.asarray(hidden[:, layer_index], dtype=np.float32)
        projected = projection.transform(
            layer_hidden.reshape(len(origins) * layer_hidden.shape[1], -1)
        ).reshape(len(origins), layer_hidden.shape[1], -1)
        result = cross_fitted_pointwise_mi(
            projected,
            future_projected,
            folds=args.folds,
            seed=args.seed + int(layer_index),
        )
        layer_outputs.append(result.scores)
        raw_layer_outputs.append(result.raw_scores)
        accuracies.append(result.fold_accuracy)
        train_accuracies.append(result.train_accuracy)
        print(
            f"layer={int(layer_index)} patches={result.scores.shape[1]} "
            f"mean_score={float(result.scores.mean()):.4f} "
            f"critic_heldout_accuracy={float(result.fold_accuracy.mean()):.3f} "
            f"critic_train_accuracy={float(result.train_accuracy.mean()):.3f}",
            flush=True,
        )

    scores = np.stack(layer_outputs, axis=1).astype(np.float32)
    peak_mask = _local_peak_mask(scores.reshape(len(origins) * len(layer_indices), -1)).reshape(
        len(origins), len(layer_indices), -1
    )
    output = create_run_dir(
        args.output_root,
        f"{config.model.name.lower()}_{config.data.dataset.lower()}_pointwise_mi",
    )
    capture_run_context(output, " ".join(sys.argv))
    np.savez_compressed(
        output / "pointwise_mi_results.npz",
        scores=scores,
        raw_scores=np.stack(raw_layer_outputs, axis=1).astype(np.float32),
        local_peak_mask=peak_mask,
        origins=origins,
        layer_indices=layer_indices,
        fold_accuracy=np.stack(accuracies),
        train_accuracy=np.stack(train_accuracies),
    )
    mean_scores = scores.mean(axis=0)
    peak_counts = peak_mask.sum(axis=(0, 1))
    summary = {
        "status": "complete",
        "reference_run": str(reference_dir),
        "model": config.model.name,
        "dataset": config.data.dataset,
        "num_origins": int(len(origins)),
        "num_layers": int(len(layer_indices)),
        "num_patches": int(scores.shape[-1]),
        "layers": layer_indices.tolist(),
        "score_shape": list(scores.shape),
        "mean_score_by_layer_patch": mean_scores.tolist(),
        "top_patch_by_layer": np.argmax(mean_scores, axis=1).astype(int).tolist(),
        "peak_count_by_layer_patch": peak_counts.astype(int).tolist(),
        "mean_peaks_per_origin": float(peak_mask.sum() / len(origins)),
        "fraction_origins_with_peak": float(np.mean(peak_mask.any(axis=(1, 2)))),
        "mean_critic_heldout_accuracy": float(np.mean(np.stack(accuracies))),
        "mean_critic_fit_accuracy": float(np.mean(np.stack(train_accuracies))),
        "analysis_elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": "Held-out density-ratio logits; not a conventional MI estimate from one sample.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"run_dir": str(output), **summary}))


if __name__ == "__main__":
    main()
