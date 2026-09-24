from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import load_benchmark_frame, make_windows
from experiments.information_anchor.estimators.gcmi import (
    gaussian_copula_transform,
    gcmi_from_gaussianized,
)
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter, ksg_mi
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    all_temporal_circular_shift_offsets,
    benjamini_hochberg,
    robust_null_score,
    temporal_circular_shift_offsets,
)
from experiments.information_anchor.estimators.projection import FittedProjection
from experiments.information_anchor.targets.future_summary import build_future_summary


def _load_projection(path: Path) -> FittedProjection:
    values = np.load(path)
    return FittedProjection(
        scaler_mean=values["scaler_mean"],
        scaler_scale=values["scaler_scale"],
        pca_mean=values["pca_mean"],
        pca_components=values["pca_components"],
        explained_variance_ratio=values["explained_variance_ratio"],
        whiten_scale=values["whiten_scale"],
    )


def contiguous_leave_one_out_blocks(n_samples: int, n_blocks: int) -> list[np.ndarray]:
    if n_blocks < 2 or n_blocks > n_samples:
        raise ValueError("n_blocks must be in [2, n_samples].")
    return [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(n_samples), n_blocks)]


def run_fixed_projection_jackknife(
    run_dir: str | Path,
    *,
    n_blocks: int = 8,
    null_permutations: int = 49,
    null_mode: str = "sampled",
    device: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    if device is not None:
        device = device
    elif config.model.device.startswith("cuda"):
        device = config.model.device
    else:
        device = "cpu"
    if config.mi.estimator not in {"ksg_cpu", "ksg_gpu", "gcmi"}:
        raise ValueError(f"Unsupported estimator: {config.mi.estimator}")
    if null_mode not in {"sampled", "all"}:
        raise ValueError("null_mode must be sampled or all.")

    result = np.load(run_dir / "mi_results.npz")
    origins = result["origins"]
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    values = frame[config.data.target].to_numpy(dtype=np.float32)
    windows = make_windows(
        values,
        origins,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
    )
    future_summary = build_future_summary(
        windows.future_normalized,
        bins=config.future_target.bins,
        spectral_bands=config.future_target.spectral_bands,
    )
    future_projection = _load_projection(run_dir / "future_projection.npz")
    future_projected = future_projection.transform(future_summary.values)

    with (run_dir / "hidden_cache.json").open("r", encoding="utf-8") as handle:
        hidden_metadata = json.load(handle)
    hidden_path = Path(config.runtime.cache_root) / hidden_metadata["cache_key"] / "history_hidden.npy"
    hidden = np.asarray(np.load(hidden_path, mmap_mode="r"), dtype=np.float32)
    if hidden.shape[0] != len(origins):
        raise ValueError("Hidden cache and origin count disagree.")
    n_samples, n_layers, n_patches, _ = hidden.shape
    if n_blocks >= n_samples:
        raise ValueError("Too many jackknife blocks for the available origins.")

    projections = [
        _load_projection(run_dir / f"hidden_projection_layer_{layer:02d}.npz")
        for layer in range(n_layers)
    ]
    projected_layers = []
    for layer in range(n_layers):
        flattened = hidden[:, layer].reshape(n_samples * n_patches, -1)
        projected_layers.append(
            projections[layer].transform(flattened).reshape(n_samples, n_patches, -1)
        )

    blocks = contiguous_leave_one_out_blocks(n_samples, n_blocks)
    layer_raw = np.empty((n_blocks, n_layers), dtype=np.float32)
    layer_z = np.empty_like(layer_raw)
    patch_raw = np.empty((n_blocks, n_patches), dtype=np.float32)
    patch_z = np.empty_like(patch_raw)
    required_separation = config.data.seq_len + config.data.pred_len
    null_shift_counts: list[int] = []

    for block_index, held_out in enumerate(blocks):
        keep = np.ones(n_samples, dtype=bool)
        keep[held_out] = False
        keep_indices = np.flatnonzero(keep)
        block_origins = origins[keep_indices]
        shifts = (
            all_temporal_circular_shift_offsets(
                block_origins, min_temporal_separation=required_separation
            )
            if null_mode == "all"
            else temporal_circular_shift_offsets(
                block_origins,
                null_permutations,
                min_temporal_separation=required_separation,
                seed=config.mi.seed + 10_000 + block_index,
            )
        )
        null_shift_counts.append(len(shifts))
        if config.mi.estimator in {"ksg_cpu", "ksg_gpu"}:
            future_for_mi = add_deterministic_jitter(
                future_projected[keep_indices],
                config.mi.jitter,
                config.mi.seed + 20_000 + block_index,
            )
        else:
            future_for_mi = gaussian_copula_transform(future_projected[keep_indices])

        gpu_estimator = None
        if config.mi.estimator == "ksg_gpu":
            gpu_estimator = TorchKSG(
                future_for_mi,
                k=config.mi.k,
                device=device,
                shift_batch_size=config.mi.shift_batch_size,
            )
        observed_cells = np.empty((n_layers, n_patches), dtype=np.float32)
        null_cells = np.empty((len(shifts), n_layers, n_patches), dtype=np.float32)
        for layer_index, projected in enumerate(projected_layers):
            x_all = projected[keep_indices]
            for patch_index in range(n_patches):
                raw_x = x_all[:, patch_index]
                if config.mi.estimator == "ksg_gpu":
                    x = add_deterministic_jitter(
                        raw_x,
                        config.mi.jitter,
                        config.mi.seed + 30_000 * (block_index + 1) + 1000 * layer_index + patch_index,
                    )
                    if gpu_estimator is None:
                        raise RuntimeError("GPU jackknife estimator was not initialized.")
                    observed, null_values = gpu_estimator.estimate_observed_and_shifts(x, shifts)
                elif config.mi.estimator == "ksg_cpu":
                    x = add_deterministic_jitter(
                        raw_x,
                        config.mi.jitter,
                        config.mi.seed + 30_000 * (block_index + 1) + 1000 * layer_index + patch_index,
                    )
                    observed = ksg_mi(x, future_for_mi, k=config.mi.k)
                    null_values = np.asarray(
                        [
                            ksg_mi(x, np.roll(future_for_mi, int(shift), axis=0), k=config.mi.k)
                            for shift in shifts
                        ],
                        dtype=np.float64,
                    )
                else:
                    x = gaussian_copula_transform(raw_x)
                    observed = gcmi_from_gaussianized(x, future_for_mi)
                    null_values = np.asarray(
                        [
                            gcmi_from_gaussianized(x, np.roll(future_for_mi, int(shift), axis=0))
                            for shift in shifts
                        ],
                        dtype=np.float64,
                    )
                observed_cells[layer_index, patch_index] = observed
                null_cells[:, layer_index, patch_index] = null_values

        layer_null = null_cells.mean(axis=2)
        patch_null = null_cells.mean(axis=1)
        layer_raw[block_index] = observed_cells.mean(axis=1)
        patch_raw[block_index] = observed_cells.mean(axis=0)
        for layer_index in range(n_layers):
            layer_z[block_index, layer_index] = robust_null_score(
                layer_raw[block_index, layer_index], layer_null[:, layer_index]
            )[0]
        for patch_index in range(n_patches):
            patch_z[block_index, patch_index] = robust_null_score(
                patch_raw[block_index, patch_index], patch_null[:, patch_index]
            )[0]
        print(
            f"block {block_index + 1}/{n_blocks}: "
            f"top_layer={int(np.argmax(layer_z[block_index]))}, "
            f"top_patch={int(np.argmax(patch_z[block_index]))}",
            flush=True,
        )

    top_layers = np.argmax(layer_z, axis=1)
    top_patches = np.argmax(patch_z, axis=1)
    top3_counts = {
        str(layer): int(np.sum(np.argsort(-layer_z, axis=1)[:, :3] == layer))
        for layer in range(n_layers)
    }
    summary = {
        "status": "complete",
        "estimator": config.mi.estimator,
        "n_samples": int(n_samples),
        "n_blocks": int(n_blocks),
        "null_mode": null_mode,
        "configured_sampled_null_permutations": int(null_permutations),
        "null_shift_counts_by_block": null_shift_counts,
        "required_origin_separation": int(required_separation),
        "top_layer_by_block": top_layers.tolist(),
        "top_patch_by_block": top_patches.tolist(),
        "top3_layer_inclusion_count": top3_counts,
        "layer_z_mean": layer_z.mean(axis=0).tolist(),
        "layer_z_std": layer_z.std(axis=0, ddof=1).tolist(),
        "layer_raw_mean": layer_raw.mean(axis=0).tolist(),
        "layer_raw_std": layer_raw.std(axis=0, ddof=1).tolist(),
        "patch_z_mean": patch_z.mean(axis=0).tolist(),
        "patch_z_std": patch_z.std(axis=0, ddof=1).tolist(),
    }
    np.savez_compressed(
        run_dir / "stability_jackknife.npz",
        layer_raw=layer_raw,
        layer_z=layer_z,
        patch_raw=patch_raw,
        patch_z=patch_z,
        held_out_blocks=np.array([block.tolist() for block in blocks], dtype=object),
    )
    with (run_dir / "stability_layer_profiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["held_out_block", "layer", "mi_raw", "mi_z"])
        for block_index in range(n_blocks):
            for layer_index in range(n_layers):
                writer.writerow([block_index, layer_index, float(layer_raw[block_index, layer_index]), float(layer_z[block_index, layer_index])])
    with (run_dir / "stability_patch_profiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["held_out_block", "patch", "mi_raw", "mi_z"])
        for block_index in range(n_blocks):
            for patch_index in range(n_patches):
                writer.writerow([block_index, patch_index, float(patch_raw[block_index, patch_index]), float(patch_z[block_index, patch_index])])
    output_path = run_dir / "stability_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological block jackknife for an MI run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--null-permutations", type=int, default=49)
    parser.add_argument("--null-mode", choices=["sampled", "all"], default="sampled")
    parser.add_argument("--device")
    args = parser.parse_args()
    output = run_fixed_projection_jackknife(
        args.run_dir,
        n_blocks=args.blocks,
        null_permutations=args.null_permutations,
        null_mode=args.null_mode,
        device=args.device,
    )
    print(f"stability_summary={output.resolve()}")


if __name__ == "__main__":
    main()
