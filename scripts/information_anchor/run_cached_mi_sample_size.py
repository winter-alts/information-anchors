#!/usr/bin/env python3
"""Audit KSG atlas stability under nested discovery sample counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    benjamini_hochberg,
    robust_null_score,
    temporal_circular_shift_offsets,
)
from experiments.information_anchor.estimators.projection import load_projection
from experiments.information_anchor.targets.future_summary import build_future_summary
from scripts.information_anchor.run_cached_mi_sensitivity import spearman, top_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--sample-count", type=int, choices=(512, 768), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", default="results/information_anchor_sample_size")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    reference = np.load(reference_dir / "mi_results.npz")
    full_origins = np.asarray(reference["origins"], dtype=np.int64)
    if args.sample_count >= len(full_origins):
        raise ValueError("sample-count must be smaller than the primary discovery sample count")
    indices = np.unique(
        np.rint(np.linspace(0, len(full_origins) - 1, args.sample_count)).astype(np.int64)
    )
    if len(indices) != args.sample_count:
        raise RuntimeError("Nested sample selection did not produce the requested count")
    origins = full_origins[indices]

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    windows = make_windows(
        values, origins, config.data.seq_len, config.data.pred_len, columns=columns
    )
    future_values = analysis_future_values(
        windows,
        values,
        train_end=config.data.train_end,
        normalization=config.future_target.normalization,
    )
    summary = build_future_summary(
        future_values,
        bins=config.future_target.bins,
        spectral_bands=config.future_target.spectral_bands,
        channel_names=windows.columns,
    )
    future_projection = load_projection(reference_dir / "future_projection.npz")
    future_projected = future_projection.transform(summary.values)
    future_for_mi = add_deterministic_jitter(
        future_projected, config.mi.jitter, config.mi.seed + 11
    )
    shifts = temporal_circular_shift_offsets(
        origins,
        config.mi.null_permutations,
        min_temporal_separation=config.data.seq_len + config.data.pred_len,
        seed=config.mi.seed,
    )
    row = np.arange(len(origins), dtype=np.int64)
    permutations = np.stack([(row - int(shift)) % len(origins) for shift in shifts])
    estimator = TorchKSG(
        future_for_mi,
        k=config.mi.k,
        device=args.device,
        shift_batch_size=config.mi.shift_batch_size,
    )
    cache_info = json.loads((reference_dir / "hidden_cache.json").read_text(encoding="utf-8"))
    cache_root = Path(config.runtime.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    hidden = np.load(cache_root / cache_info["cache_key"] / "history_hidden.npy", mmap_mode="r")
    n_samples, n_layers, n_patches, _ = hidden.shape
    if n_samples != len(full_origins):
        raise ValueError("Reference hidden cache and origin count disagree")
    mi_raw = np.empty((n_layers, n_patches), dtype=np.float32)
    mi_z = np.empty_like(mi_raw)
    p_values = np.empty_like(mi_raw)
    null_mi = np.empty((len(shifts), n_layers, n_patches), dtype=np.float32)
    for layer in range(n_layers):
        projection = load_projection(reference_dir / f"hidden_projection_layer_{layer:02d}.npz")
        selected = np.asarray(hidden[indices, layer], dtype=np.float32)
        projected = projection.transform(selected.reshape(args.sample_count * n_patches, -1)).reshape(
            args.sample_count, n_patches, -1
        )
        for patch in range(n_patches):
            x = add_deterministic_jitter(
                projected[:, patch], config.mi.jitter, config.mi.seed + 1000 * layer + patch
            )
            observed, null = estimator.estimate_observed_and_permutations(x, permutations)
            mi_raw[layer, patch] = observed
            null_mi[:, layer, patch] = null
            mi_z[layer, patch], p_values[layer, patch] = robust_null_score(observed, null)
        print(
            f"model={config.model.name} n={args.sample_count} layer={layer + 1}/{n_layers}",
            flush=True,
        )
    q_values = benjamini_hochberg(p_values).astype(np.float32)
    patch_z = np.empty(n_patches, dtype=np.float32)
    layer_z = np.empty(n_layers, dtype=np.float32)
    for patch in range(n_patches):
        patch_z[patch] = robust_null_score(
            float(mi_raw[:, patch].mean()), null_mi[:, :, patch].mean(axis=1)
        )[0]
    for layer in range(n_layers):
        layer_z[layer] = robust_null_score(
            float(mi_raw[layer].mean()), null_mi[:, layer].mean(axis=1)
        )[0]

    output = create_run_dir(
        args.output_root,
        f"{config.model.name.lower()}_{config.data.dataset.lower()}_n{args.sample_count}",
    )
    capture_run_context(output, " ".join(sys.argv))
    np.savez_compressed(
        output / "mi_sample_size_results.npz",
        mi_raw=mi_raw,
        mi_z=mi_z,
        p_values=p_values,
        q_values=q_values,
        null_mi=null_mi,
        patch_mi_z=patch_z,
        layer_mi_z=layer_z,
        origins=origins,
        selected_primary_indices=indices,
        shifts=shifts,
    )
    primary_cell = np.asarray(reference["mi_z"], dtype=np.float64)
    primary_patch = np.asarray(reference["patch_mi_z"], dtype=np.float64)
    primary_layer = np.asarray(reference["layer_anchor_score"], dtype=np.float64)
    result = {
        "status": "complete",
        "model": config.model.name,
        "dataset": config.data.dataset,
        "reference_run": str(reference_dir),
        "primary_sample_count": int(len(full_origins)),
        "sample_count": args.sample_count,
        "null_repetitions": int(len(shifts)),
        "hidden_projection_refit": False,
        "future_projection_refit": False,
        "cell_z_spearman_with_primary": spearman(mi_z, primary_cell),
        "patch_z_spearman_with_primary": spearman(patch_z, primary_patch),
        "layer_z_spearman_with_primary": spearman(layer_z, primary_layer),
        "top_quarter_cell_jaccard_with_primary": top_overlap(mi_z, primary_cell),
        "top_patch": int(np.argmax(patch_z)),
        "primary_top_patch": int(np.argmax(primary_patch)),
        "top_layer": int(np.argmax(layer_z)),
        "primary_top_layer": int(np.argmax(primary_layer)),
        "fraction_cell_q_below_0_05": float(np.mean(q_values < 0.05)),
        "analysis_elapsed_seconds": time.perf_counter() - started,
        "hidden_forward_rerun": False,
    }
    write_json(output / "summary.json", result)
    print(json.dumps({"run_dir": str(output), **result}))


if __name__ == "__main__":
    main()
