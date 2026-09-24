from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.information_anchor.adapters import (
    adapter_protocol,
    build_adapter,
    representation_depends_on_pred_len,
)
from experiments.information_anchor.artifacts import (
    capture_run_context,
    create_run_dir,
    stable_hash,
    write_json,
)
from experiments.information_anchor.config import load_config, with_overrides
from experiments.information_anchor.data import (
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.gcmi import (
    gaussian_copula_transform,
    gcmi_from_gaussianized,
)
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter, ksg_mi
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    all_temporal_circular_shift_offsets,
    benjamini_hochberg,
    temporal_circular_shift_offsets,
    robust_null_score,
)
from experiments.information_anchor.estimators.projection import fit_projection
from experiments.information_anchor.reporting.plots import save_heatmap, save_profile
from experiments.information_anchor.targets.future_summary import build_future_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Population-level future-information estimation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--null-permutations", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--estimator", choices=["ksg_cpu", "ksg_gpu", "gcmi"])
    parser.add_argument("--k", type=int)
    parser.add_argument("--hidden-pca-dim", type=int)
    parser.add_argument("--future-pca-dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--null-mode", choices=["sampled", "all"])
    parser.add_argument("--future-scope", choices=["target", "all"])
    return parser.parse_args()


def _save_projection(path: Path, projection) -> None:
    np.savez_compressed(
        path,
        scaler_mean=projection.scaler_mean,
        scaler_scale=projection.scaler_scale,
        pca_mean=projection.pca_mean,
        pca_components=projection.pca_components,
        explained_variance_ratio=projection.explained_variance_ratio,
        whiten_scale=projection.whiten_scale,
    )


def _calibrate_profile(
    observed: np.ndarray,
    null_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_scores = np.empty_like(observed, dtype=np.float32)
    p_values = np.empty_like(observed, dtype=np.float32)
    for index in range(len(observed)):
        z_score, p_value = robust_null_score(observed[index], null_matrix[:, index])
        z_scores[index] = z_score
        p_values[index] = p_value
    q_values = benjamini_hochberg(p_values).astype(np.float32)

    return z_scores, p_values, q_values


def _write_profile(
    path: Path,
    unit_name: str,
    observed: np.ndarray,
    z_scores: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    null_matrix: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [unit_name, "mi_raw_mean", "mi_z", "p_value", "q_value", "null_mean", "null_std"]
        )
        for index in range(len(observed)):
            writer.writerow(
                [
                    index,
                    float(observed[index]),
                    float(z_scores[index]),
                    float(p_values[index]),
                    float(q_values[index]),
                    float(null_matrix[:, index].mean()),
                    float(null_matrix[:, index].std(ddof=1)),
                ]
            )


def main() -> None:
    run_started_at = time.perf_counter()
    args = _parse_args()
    config = with_overrides(
        load_config(args.config),
        device=args.device,
        max_origins=args.max_origins,
        null_permutations=args.null_permutations,
        output_root=args.output_root,
        estimator=args.estimator,
        k=args.k,
        hidden_pca_dim=args.hidden_pca_dim,
        future_pca_dim=args.future_pca_dim,
        seed=args.seed,
        null_mode=args.null_mode,
        future_scope=args.future_scope,
    )
    if config.runtime.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    random.seed(config.mi.seed)
    np.random.seed(config.mi.seed)
    torch.manual_seed(config.mi.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.mi.seed)

    run_dir = create_run_dir(config.runtime.output_root, config.experiment_name)
    capture_run_context(run_dir, " ".join(sys.argv))
    write_json(run_dir / "config.json", config.to_dict())

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    origins = build_forecast_origins(config.data)
    value_columns = select_value_columns(frame, config.data)
    if len(value_columns) == 1:
        values = frame.loc[:, value_columns[0]].to_numpy(dtype=np.float32)
    else:
        values = frame.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    windows = make_windows(
        values,
        origins,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        columns=value_columns,
    )
    origin_digest = origins_hash(origins)
    write_json(
        run_dir / "data_manifest.json",
        {
            "rows_available": len(frame),
            "target": config.data.target,
            "future_scope": config.future_target.scope,
            "features": config.data.features,
            "target_columns": list(windows.columns),
            "num_channels": len(windows.columns),
            "window_shapes": {
                "history": list(windows.history_normalized.shape),
                "future": list(windows.future_normalized.shape),
            },
            "split": config.data.split,
            "seq_len": config.data.seq_len,
            "pred_len": config.data.pred_len,
            "origin_seq_len": config.data.origin_seq_len or config.data.seq_len,
            "origin_pred_len": config.data.origin_pred_len or config.data.pred_len,
            "num_origins": len(origins),
            "first_origin": int(origins[0]),
            "last_origin": int(origins[-1]),
            "origins_hash": origin_digest,
            "history_mean_range": [
                float(windows.history_mean.min()),
                float(windows.history_mean.max()),
            ],
            "history_scale_range": [
                float(windows.history_scale.min()),
                float(windows.history_scale.max()),
            ],
        },
    )

    future_values = analysis_future_values(
        windows,
        values,
        train_end=config.data.train_end,
        normalization=config.future_target.normalization,
    )
    future_channel_names = windows.columns
    if config.future_target.scope == "target":
        target_index = windows.columns.index(config.data.target)
        future_values = future_values[:, :, target_index]
        future_channel_names = None
    future_summary = build_future_summary(
        future_values,
        bins=config.future_target.bins,
        spectral_bands=config.future_target.spectral_bands,
        channel_names=future_channel_names,
    )
    future_projection, future_projected = fit_projection(
        future_summary.values,
        config.future_target.pca_dim,
        seed=config.mi.seed,
        standardize_features=True,
    )
    _save_projection(run_dir / "future_projection.npz", future_projection)
    write_json(run_dir / "future_features.json", list(future_summary.feature_names))

    cache_payload = {
        "model_name": config.model.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset": config.data.dataset,
        "target": config.data.target,
        "features": config.data.features,
        "target_columns": list(windows.columns),
        "split": config.data.split,
        "seq_len": config.data.seq_len,
        "origins_hash": origin_digest,
        # Hidden extraction always receives the per-window normalized history.
        # Future-target normalization must not invalidate an identical hidden cache.
        "normalization": "history_window",
        "cache_dtype": config.model.cache_dtype,
    }
    if representation_depends_on_pred_len(config.model.name):
        cache_payload["representation_pred_len"] = config.data.pred_len
    protocol = adapter_protocol(config.model.name)
    if protocol is not None:
        cache_payload["adapter_protocol"] = protocol
    if config.model.channel_aggregation != "concat_same_time_patch":
        cache_payload["channel_aggregation"] = config.model.channel_aggregation
    cache_key = stable_hash(cache_payload)
    cache_dir = Path(config.runtime.cache_root) / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "cache_key.json", cache_payload)

    hidden_path = cache_dir / "history_hidden.npy"
    hidden_metadata_path = cache_dir / "metadata.json"
    hidden_cache_hit = hidden_path.exists() and hidden_metadata_path.exists()
    if hidden_cache_hit:
        with hidden_metadata_path.open("r", encoding="utf-8") as handle:
            hidden_metadata = json.load(handle)
    else:
        adapter = build_adapter(config)
        hidden_metadata = adapter.extract_history_to_cache(
            windows.history_normalized,
            cache_dir,
        )
    write_json(run_dir / "hidden_cache.json", {"cache_key": cache_key, **hidden_metadata})

    hidden = np.load(cache_dir / "history_hidden.npy", mmap_mode="r")
    expected_samples = len(origins)
    if hidden.shape[0] != expected_samples:
        raise ValueError(f"Hidden cache has {hidden.shape[0]} samples, expected {expected_samples}")
    n_samples, n_layers, n_patches, _ = hidden.shape

    required_temporal_separation = config.data.seq_len + config.data.pred_len
    shifts = (
        all_temporal_circular_shift_offsets(
            origins, min_temporal_separation=required_temporal_separation
        )
        if config.mi.null_mode == "all"
        else temporal_circular_shift_offsets(
            origins,
            config.mi.null_permutations,
            min_temporal_separation=required_temporal_separation,
            seed=config.mi.seed,
        )
    )
    realized_shift_separations = np.asarray(
        [
            np.min(np.abs(origins - np.roll(origins, int(shift))))
            for shift in shifts
        ],
        dtype=np.int64,
    )

    mi_raw = np.empty((n_layers, n_patches), dtype=np.float32)
    mi_z = np.empty_like(mi_raw)
    p_values = np.empty_like(mi_raw)
    null_mean = np.empty_like(mi_raw)
    null_std = np.empty_like(mi_raw)
    null_mi = np.empty(
        (len(shifts), n_layers, n_patches),
        dtype=np.float32,
    )
    hidden_variance = []

    if config.mi.estimator in {"ksg_cpu", "ksg_gpu"}:
        future_for_mi = add_deterministic_jitter(
            future_projected,
            config.mi.jitter,
            config.mi.seed + 11,
        )
    else:
        future_for_mi = gaussian_copula_transform(future_projected)

    gpu_ksg = None
    if config.mi.estimator == "ksg_gpu":
        gpu_ksg = TorchKSG(
            future_for_mi,
            k=config.mi.k,
            device=config.model.device,
            shift_batch_size=config.mi.shift_batch_size,
        )
    for layer_index in range(n_layers):
        layer_values = np.asarray(hidden[:, layer_index], dtype=np.float32)
        flattened = layer_values.reshape(n_samples * n_patches, -1)
        hidden_projection, projected_flat = fit_projection(
            flattened,
            config.mi.hidden_pca_dim,
            seed=config.mi.seed + layer_index,
            standardize_features=False,
        )
        _save_projection(
            run_dir / f"hidden_projection_layer_{layer_index:02d}.npz",
            hidden_projection,
        )
        hidden_variance.append(
            {
                "layer": layer_index,
                "explained_variance_ratio": hidden_projection.explained_variance_ratio.tolist(),
                "explained_variance_sum": float(hidden_projection.explained_variance_ratio.sum()),
            }
        )
        projected = projected_flat.reshape(n_samples, n_patches, -1)
        for patch_index in range(n_patches):
            raw_x = projected[:, patch_index]
            if config.mi.estimator == "ksg_cpu":
                x = add_deterministic_jitter(
                    raw_x,
                    config.mi.jitter,
                    config.mi.seed + 1000 * layer_index + patch_index,
                )
                observed = ksg_mi(x, future_for_mi, k=config.mi.k)
                null_values = np.asarray(
                    [
                        ksg_mi(x, np.roll(future_for_mi, int(shift), axis=0), k=config.mi.k)
                        for shift in shifts
                    ],
                    dtype=np.float64,
                )
            elif config.mi.estimator == "ksg_gpu":
                x = add_deterministic_jitter(
                    raw_x,
                    config.mi.jitter,
                    config.mi.seed + 1000 * layer_index + patch_index,
                )
                if gpu_ksg is None:
                    raise RuntimeError("GPU KSG was not initialized.")
                observed, null_values = gpu_ksg.estimate_observed_and_shifts(x, shifts)
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
            null_mi[:, layer_index, patch_index] = null_values
            z_score, p_value = robust_null_score(observed, null_values)
            mi_raw[layer_index, patch_index] = observed
            mi_z[layer_index, patch_index] = z_score
            p_values[layer_index, patch_index] = p_value
            null_mean[layer_index, patch_index] = float(null_values.mean())
            null_std[layer_index, patch_index] = float(null_values.std(ddof=1))
        print(
            f"layer {layer_index + 1:02d}/{n_layers}: "
            f"mean_mi={mi_raw[layer_index].mean():.4f}, "
            f"mean_z={mi_z[layer_index].mean():.3f}",
            flush=True,
        )

    q_values = benjamini_hochberg(p_values).astype(np.float32)

    # Layer means are the primary inferential units for the depth hypothesis.
    # The same temporal shift is shared across patches, preserving their dependence.
    layer_mi_raw = mi_raw.mean(axis=1)
    layer_null = null_mi.mean(axis=2)
    layer_mi_z, layer_p_values, layer_q_values = _calibrate_profile(
        layer_mi_raw, layer_null
    )
    # 整层 shared-null z 用于显著性推断；逐单元 z 的层均值用于定位信息锚点层。
    # 二者不可互换：前者会随层级 null 方差变化，后者直接聚合校准后的 atlas 单元。
    layer_anchor_score = mi_z.mean(axis=1).astype(np.float32)

    # Patch means provide a secondary history-position profile.
    patch_mi_raw = mi_raw.mean(axis=0)
    patch_null = null_mi.mean(axis=1)
    patch_mi_z, patch_p_values, patch_q_values = _calibrate_profile(
        patch_mi_raw, patch_null
    )

    np.savez_compressed(
        run_dir / "mi_results.npz",
        mi_raw=mi_raw,
        mi_z=mi_z,
        p_values=p_values,
        q_values=q_values,
        null_mean=null_mean,
        null_std=null_std,
        null_mi=null_mi,
        layer_mi_raw=layer_mi_raw,
        layer_mi_z=layer_mi_z,
        layer_anchor_score=layer_anchor_score,
        layer_p_values=layer_p_values,
        layer_q_values=layer_q_values,
        layer_null=layer_null,
        patch_mi_raw=patch_mi_raw,
        patch_mi_z=patch_mi_z,
        patch_p_values=patch_p_values,
        patch_q_values=patch_q_values,
        patch_null=patch_null,
        realized_shift_separations=realized_shift_separations,
        shifts=shifts,
        origins=origins,
    )
    write_json(run_dir / "hidden_pca_variance.json", hidden_variance)

    patch_len = int(hidden_metadata["patch_len"])
    patch_stride = int(hidden_metadata.get("patch_stride", patch_len))
    with (run_dir / "mi_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "patch",
                "normalized_depth",
                "normalized_history_position",
                "mi_raw",
                "mi_z",
                "p_value",
                "q_value",
                "null_mean",
                "null_std",
            ]
        )
        for layer_index in range(n_layers):
            for patch_index in range(n_patches):
                center = min(
                    config.data.seq_len,
                    patch_index * patch_stride + 0.5 * patch_len,
                )
                normalized_position = (center - config.data.seq_len) / config.data.seq_len
                writer.writerow(
                    [
                        layer_index,
                        patch_index,
                        (layer_index + 1) / n_layers,
                        normalized_position,
                        float(mi_raw[layer_index, patch_index]),
                        float(mi_z[layer_index, patch_index]),
                        float(p_values[layer_index, patch_index]),
                        float(q_values[layer_index, patch_index]),
                        float(null_mean[layer_index, patch_index]),
                        float(null_std[layer_index, patch_index]),
                    ]
                )

    _write_profile(
        run_dir / "layer_profile.csv",
        "layer",
        layer_mi_raw,
        layer_mi_z,
        layer_p_values,
        layer_q_values,
        layer_null,
    )
    _write_profile(
        run_dir / "patch_profile.csv",
        "patch",
        patch_mi_raw,
        patch_mi_z,
        patch_p_values,
        patch_q_values,
        patch_null,
    )

    best_flat = int(np.nanargmax(mi_z))
    best_layer, best_patch = np.unravel_index(best_flat, mi_z.shape)
    best_aggregate_layer = int(np.nanargmax(layer_mi_z))
    best_anchor_layer = int(np.nanargmax(layer_anchor_score))
    best_aggregate_patch = int(np.nanargmax(patch_mi_z))
    summary = {
        "status": "complete",
        "estimator": config.mi.estimator,
        "hidden_cache_hit": hidden_cache_hit,
        "analysis_elapsed_seconds": float(time.perf_counter() - run_started_at),
        "num_samples": n_samples,
        "num_layers": n_layers,
        "num_patches": n_patches,
        "mi_shape": list(mi_raw.shape),
        "raw_mi_range": [float(mi_raw.min()), float(mi_raw.max())],
        "z_score_range": [float(mi_z.min()), float(mi_z.max())],
        "fraction_cell_q_below_0_05": float(np.mean(q_values < 0.05)),
        "fraction_layer_q_below_0_05": float(np.mean(layer_q_values < 0.05)),
        "fraction_patch_q_below_0_05": float(np.mean(patch_q_values < 0.05)),
        "top_layer": {
            "layer": best_aggregate_layer,
            "mi_raw_mean": float(layer_mi_raw[best_aggregate_layer]),
            "mi_z": float(layer_mi_z[best_aggregate_layer]),
            "q_value": float(layer_q_values[best_aggregate_layer]),
        },
        "top_anchor_layer": {
            "layer": best_anchor_layer,
            "mean_cell_z": float(layer_anchor_score[best_anchor_layer]),
            "shared_null_layer_z": float(layer_mi_z[best_anchor_layer]),
            "shared_null_layer_q": float(layer_q_values[best_anchor_layer]),
            "selection_rule": "argmax_layer mean_patch(cell_null_calibrated_z)",
        },
        "top_history_patch": {
            "patch": best_aggregate_patch,
            "mi_raw_mean": float(patch_mi_raw[best_aggregate_patch]),
            "mi_z": float(patch_mi_z[best_aggregate_patch]),
            "q_value": float(patch_q_values[best_aggregate_patch]),
        },
        "top_cell": {
            "layer": int(best_layer),
            "patch": int(best_patch),
            "mi_raw": float(mi_raw[best_layer, best_patch]),
            "mi_z": float(mi_z[best_layer, best_patch]),
            "q_value": float(q_values[best_layer, best_patch]),
        },
        "null": {
            "mode": config.mi.null_mode,
            "permutations": int(len(shifts)),
            "configured_sampled_permutations": int(config.mi.null_permutations),
            "required_origin_separation": int(required_temporal_separation),
            "minimum_realized_origin_separation": int(realized_shift_separations.min()),
            "maximum_realized_origin_separation": int(realized_shift_separations.max()),
            "offsets_hash": stable_hash(shifts.tolist()),
        },
    }
    write_json(run_dir / "summary.json", summary)

    estimator_label = {
        "ksg_cpu": "KSG",
        "ksg_gpu": "KSG",
        "gcmi": "Gaussian-copula",
    }[config.mi.estimator]
    save_heatmap(
        mi_raw,
        run_dir / "figures" / "mi_raw.png",
        title=f"{config.model.name} {estimator_label} MI: history hidden vs future summary",
        colorbar_label=f"{estimator_label} MI (nats)",
        cmap="viridis",
    )
    save_heatmap(
        mi_z,
        run_dir / "figures" / "mi_null_calibrated_z.png",
        title="Null-calibrated future-information map",
        colorbar_label="Robust null z-score",
        cmap="coolwarm",
    )
    save_profile(
        layer_mi_z,
        run_dir / "figures" / "layer_profile_z.png",
        title="Depth profile of future-relevant information",
        x_label="Transformer block index",
        y_label="Aggregate robust null z-score",
        q_values=layer_q_values,
    )
    save_profile(
        patch_mi_z,
        run_dir / "figures" / "history_position_profile_z.png",
        title="History-position profile of future-relevant information",
        x_label="History patch index (oldest to newest)",
        y_label="Aggregate robust null z-score",
        q_values=patch_q_values,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"run_dir={run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
