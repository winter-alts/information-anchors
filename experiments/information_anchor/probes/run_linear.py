from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.information_anchor.adapters import (
    adapter_protocol,
    build_adapter,
    representation_depends_on_pred_len,
)
from experiments.information_anchor.artifacts import capture_run_context, create_run_dir, stable_hash, write_json
from experiments.information_anchor.config import ExperimentConfig, load_config, validate_config
from experiments.information_anchor.data import (
    WindowBatch,
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.projection import fit_projection
from experiments.information_anchor.interventions.common import select_functional_anchor
from experiments.information_anchor.probes.linear import RidgeProbeResult, fit_ridge_probe
from experiments.information_anchor.probes.semantic import (
    SEMANTIC_PROTOCOL,
    SemanticTargets,
    build_global_semantic_targets,
    build_semantic_targets,
)


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronologically split linear future-semantic probes")
    parser.add_argument("--run-dir", required=True, help="Formal discovery/reference MI run")
    parser.add_argument("--output-root", default="results/information_anchor_probes")
    parser.add_argument("--device")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument(
        "--semantic-scope",
        choices=(
            "target",
            "global",
            "channel",
            "suite",
            "all",
            "aggregate",
        ),
        default="target",
        help="Probe the registered target channel, global system, or channel-resolved semantics.",
    )
    return parser.parse_args()


def _split_config(config: ExperimentConfig, split: str, device: str | None) -> ExperimentConfig:
    model = replace(config.model, device=device or config.model.device)
    data = replace(config.data, split=split)
    derived = replace(config, model=model, data=data)
    validate_config(derived)
    return derived


def _cache_payload(config: ExperimentConfig, origin_digest: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": config.model.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset": config.data.dataset,
        "target": config.data.target,
        "features": config.data.features,
        "target_columns": list(config.data.target_columns),
        "split": config.data.split,
        "seq_len": config.data.seq_len,
        "origins_hash": origin_digest,
        # Adapter inputs remain history-window normalized. Analysis-target
        # normalization therefore must not invalidate the hidden cache.
        "normalization": "history_window",
        "cache_dtype": config.model.cache_dtype,
        "channel_aggregation": config.model.channel_aggregation,
    }
    if representation_depends_on_pred_len(config.model.name):
        payload["representation_pred_len"] = config.data.pred_len
    protocol = adapter_protocol(config.model.name)
    if protocol is not None:
        payload["adapter_protocol"] = protocol
    return payload


def _ensure_hidden_cache(
    config: ExperimentConfig,
    windows: WindowBatch,
    *,
    adapter,
) -> tuple[Path, dict[str, object], object]:
    payload = _cache_payload(config, origins_hash(windows.origins))
    cache_dir = Path(config.runtime.cache_root) / stable_hash(payload)
    hidden_path = cache_dir / "history_hidden.npy"
    metadata_path = cache_dir / "metadata.json"
    if hidden_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return hidden_path, json.load(handle), adapter
    if adapter is None:
        adapter = build_adapter(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "cache_key.json", payload)
    metadata = adapter.extract_history_to_cache(windows.history_normalized, cache_dir)
    return hidden_path, metadata, adapter


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


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have matching length >= 2.")
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _nanmean(values: np.ndarray) -> float:
    """返回有限目标的均值；全为退化目标时保留 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _canonical_semantic_scope(scope: str) -> str:
    return {"all": "channel", "aggregate": "global"}.get(scope, scope)


def _semantic_splits(
    windows: dict[str, WindowBatch],
    columns: tuple[str, ...],
    target_index: int,
    scope: str,
    analysis_futures: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, SemanticTargets], dict[str, tuple[int, int]]]:
    """Build global and target semantics with their preregistered normalizations."""
    canonical_scope = _canonical_semantic_scope(scope)
    output: dict[str, SemanticTargets] = {}
    scope_slices: dict[str, tuple[int, int]] = {}
    for split, batch in windows.items():
        future = (
            batch.future_normalized
            if analysis_futures is None
            else np.asarray(analysis_futures[split], dtype=np.float32)
        )
        targets = {
            "global": build_global_semantic_targets(future, columns),
            "target": build_semantic_targets(
                future[:, :, target_index]
            ),
            "channel": build_semantic_targets(future, columns),
        }
        if canonical_scope == "suite":
            values = []
            names: list[str] = []
            start = 0
            for name in ("global", "target", "channel"):
                item = targets[name]
                stop = start + item.values.shape[1]
                if split == "train":
                    scope_slices[name] = (start, stop)
                values.append(item.values)
                names.extend(f"{name}::{semantic}" for semantic in item.names)
                start = stop
            output[split] = SemanticTargets(
                values=np.concatenate(values, axis=1).astype(np.float32),
                names=tuple(names),
            )
        else:
            output[split] = targets[canonical_scope]
            if split == "train":
                scope_slices[canonical_scope] = (0, output[split].values.shape[1])
    names = output["train"].names
    if any(item.names != names for item in output.values()):
        raise RuntimeError("Semantic target definitions differ across splits.")
    return output, scope_slices


def _probe_baseline(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> RidgeProbeResult:
    """将 raw history 展平成样本特征后拟合统一的 Ridge probe。"""
    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    if train_x.ndim == 3:
        train_x = train_x.reshape(train_x.shape[0], -1)
        validation_x = validation_x.reshape(validation_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
    projection, train_projected = fit_projection(
        train_x,
        pca_dim,
        seed=seed,
        standardize_features=False,
    )
    return fit_ridge_probe(
        train_projected,
        train_y,
        projection.transform(validation_x),
        validation_y,
        projection.transform(test_x),
        test_y,
        alphas=ALPHAS,
    )


def main() -> None:
    args = _parse_args()
    reference_dir = Path(args.run_dir)
    base_config = load_config(reference_dir / "config.json")
    train_config = _split_config(base_config, "discovery", args.device)
    validation_config = _split_config(base_config, "validation", args.device)
    test_config = _split_config(base_config, "test", args.device)

    output_dir = create_run_dir(
        args.output_root,
        f"{base_config.model.name.lower()}_{base_config.data.dataset.lower()}_l{base_config.data.seq_len}_p{base_config.data.pred_len}_semantic_probe",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "reference_config.json", base_config.to_dict())

    frame = load_benchmark_frame(base_config.data, offline=base_config.runtime.offline)
    columns = select_value_columns(frame, base_config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    reference_results = np.load(reference_dir / "mi_results.npz")
    train_origins = np.asarray(reference_results["origins"], dtype=np.int64)
    validation_origins = build_forecast_origins(validation_config.data)
    test_origins = build_forecast_origins(test_config.data)
    windows = {
        "train": make_windows(
            values, train_origins, base_config.data.seq_len, base_config.data.pred_len, columns=columns
        ),
        "validation": make_windows(
            values, validation_origins, base_config.data.seq_len, base_config.data.pred_len, columns=columns
        ),
        "test": make_windows(
            values, test_origins, base_config.data.seq_len, base_config.data.pred_len, columns=columns
        ),
    }
    target_index = columns.index(base_config.data.target)
    analysis_futures = {
        split: analysis_future_values(
            batch,
            values,
            train_end=base_config.data.train_end,
            normalization=base_config.future_target.normalization,
        )
        for split, batch in windows.items()
    }
    semantics, semantic_scope_slices = _semantic_splits(
        windows,
        columns,
        target_index,
        args.semantic_scope,
        analysis_futures,
    )
    canonical_semantic_scope = _canonical_semantic_scope(args.semantic_scope)
    semantic_names = semantics["train"].names

    with (reference_dir / "hidden_cache.json").open("r", encoding="utf-8") as handle:
        train_hidden_metadata = json.load(handle)
    train_hidden_path = (
        Path(base_config.runtime.cache_root) / train_hidden_metadata["cache_key"] / "history_hidden.npy"
    )
    if not train_hidden_path.exists():
        raise FileNotFoundError(train_hidden_path)

    adapter = None
    validation_hidden_path, validation_metadata, adapter = _ensure_hidden_cache(
        validation_config, windows["validation"], adapter=adapter
    )
    test_hidden_path, test_metadata, adapter = _ensure_hidden_cache(
        test_config, windows["test"], adapter=adapter
    )
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hidden = {
        "train": np.load(train_hidden_path, mmap_mode="r"),
        "validation": np.load(validation_hidden_path, mmap_mode="r"),
        "test": np.load(test_hidden_path, mmap_mode="r"),
    }
    n_layers = int(hidden["train"].shape[1])
    n_patches = int(hidden["train"].shape[2])
    if any(array.shape[1:3] != (n_layers, n_patches) for array in hidden.values()):
        raise ValueError("Hidden layer/patch shapes differ across chronological splits.")
    n_targets = len(semantic_names)

    validation_r2 = np.empty((n_layers, n_patches, n_targets), dtype=np.float32)
    test_r2 = np.empty_like(validation_r2)
    test_mae = np.empty_like(validation_r2)
    selected_alpha = np.empty((n_layers, n_patches, n_targets), dtype=np.float32)
    projection_dir = output_dir / "hidden_projections"
    projection_dir.mkdir(parents=True, exist_ok=True)

    for layer in range(n_layers):
        train_layer = np.asarray(hidden["train"][:, layer], dtype=np.float32)
        validation_layer = np.asarray(hidden["validation"][:, layer], dtype=np.float32)
        test_layer = np.asarray(hidden["test"][:, layer], dtype=np.float32)
        projection, train_flat = fit_projection(
            train_layer.reshape(-1, train_layer.shape[-1]),
            args.pca_dim,
            seed=base_config.mi.seed + layer,
            standardize_features=False,
        )
        _save_projection(projection_dir / f"layer_{layer:02d}.npz", projection)
        train_projected = train_flat.reshape(len(train_layer), n_patches, -1)
        validation_projected = projection.transform(
            validation_layer.reshape(-1, validation_layer.shape[-1])
        ).reshape(len(validation_layer), n_patches, -1)
        test_projected = projection.transform(
            test_layer.reshape(-1, test_layer.shape[-1])
        ).reshape(len(test_layer), n_patches, -1)

        for patch in range(n_patches):
            result = fit_ridge_probe(
                train_projected[:, patch],
                semantics["train"].values,
                validation_projected[:, patch],
                semantics["validation"].values,
                test_projected[:, patch],
                semantics["test"].values,
                alphas=ALPHAS,
            )
            validation_r2[layer, patch] = result.validation_r2
            test_r2[layer, patch] = result.test_r2
            test_mae[layer, patch] = result.test_mae_standardized
            selected_alpha[layer, patch] = result.alpha
        print(
            f"probe layer {layer + 1:02d}/{n_layers}: test_mean_r2={_nanmean(test_r2[layer]):.4f}",
            flush=True,
        )

    raw_full = _probe_baseline(
        windows["train"].history_normalized,
        windows["validation"].history_normalized,
        windows["test"].history_normalized,
        semantics["train"].values,
        semantics["validation"].values,
        semantics["test"].values,
        pca_dim=args.pca_dim,
        seed=base_config.mi.seed + 50_000,
    )
    patch_len = int(train_hidden_metadata["patch_len"])
    raw_recent = _probe_baseline(
        windows["train"].history_normalized[:, -patch_len:],
        windows["validation"].history_normalized[:, -patch_len:],
        windows["test"].history_normalized[:, -patch_len:],
        semantics["train"].values,
        semantics["validation"].values,
        semantics["test"].values,
        pca_dim=args.pca_dim,
        seed=base_config.mi.seed + 60_000,
    )

    mi_z = np.asarray(reference_results["mi_z"], dtype=np.float32)
    if mi_z.shape != (n_layers, n_patches):
        raise ValueError("Reference MI map and probe map have different shapes.")
    probe_cell_mean = np.nanmean(test_r2, axis=2)
    semantic_correlations = {
        name: _spearman(mi_z, test_r2[:, :, index]) for index, name in enumerate(semantic_names)
    }
    layer_mean = np.nanmean(test_r2, axis=(1, 2))
    patch_mean = np.nanmean(test_r2, axis=(0, 2))
    top_layer = int(np.argmax(layer_mean))
    top_patch = int(np.argmax(patch_mean))
    mi_top_layer, mi_top_patch = np.unravel_index(int(np.argmax(mi_z)), mi_z.shape)
    low_mi_patch = int(np.argmin(mi_z[mi_top_layer]))
    layer_anchor_score = (
        np.asarray(reference_results["layer_anchor_score"], dtype=np.float32)
        if "layer_anchor_score" in reference_results.files
        else mi_z.mean(axis=1)
    )
    functional_anchor = select_functional_anchor(
        mi_z, layer_anchor_score, base_config.model.name
    )
    functional_layer = functional_anchor.layer
    functional_top_patch = functional_anchor.top_patch
    functional_low_patch = functional_anchor.low_patch
    global_anchor_layer = int(np.argmax(layer_anchor_score))

    def _cell_probe_payload(layer: int, patch: int) -> dict[str, object]:
        """保存指定 layer-patch 的逐语义 probe 结果，避免只保留宏平均。"""
        return {
            "layer": int(layer),
            "patch": int(patch),
            "mi_z": float(mi_z[layer, patch]),
            "mean_test_r2": _nanmean(test_r2[layer, patch]),
            "mean_test_r2_by_scope": {
                scope: _nanmean(test_r2[layer, patch, start:stop])
                for scope, (start, stop) in semantic_scope_slices.items()
            },
            "test_r2_by_semantic": {
                name: float(test_r2[layer, patch, index])
                for index, name in enumerate(semantic_names)
            },
            "test_mae_by_semantic": {
                name: float(test_mae[layer, patch, index])
                for index, name in enumerate(semantic_names)
            },
        }

    split_manifest = {
        split: {
            "num_origins": int(len(batch.origins)),
            "first_origin": int(batch.origins[0]),
            "last_origin": int(batch.origins[-1]),
            "origins_hash": origins_hash(batch.origins),
            "history_start": int(batch.origins[0] - base_config.data.seq_len),
            "future_end": int(batch.origins[-1] + base_config.data.pred_len),
        }
        for split, batch in windows.items()
    }
    summary = {
        "status": "complete",
        "model": base_config.model.name,
        "model_id": base_config.model.model_id,
        "model_revision": base_config.model.revision,
        "dataset": base_config.data.dataset,
        "features": base_config.data.features,
        "target_columns": list(columns),
        "semantic_scope": canonical_semantic_scope,
        "semantic_normalization": {
            scope: base_config.future_target.normalization
            for scope in semantic_scope_slices
        },
        "semantic_scope_slices": {
            scope: [start, stop]
            for scope, (start, stop) in semantic_scope_slices.items()
        },
        "reference_run": str(reference_dir.resolve()),
        "pca_dim": args.pca_dim,
        "alphas": list(ALPHAS),
        "semantic_names": list(semantic_names),
        "semantic_protocol": SEMANTIC_PROTOCOL,
        "split_manifest": split_manifest,
        "num_layers": n_layers,
        "num_patches": n_patches,
        "top_probe_layer": {
            "layer": top_layer,
            "normalized_depth": (top_layer + 1) / n_layers,
            "test_mean_r2": float(layer_mean[top_layer]),
        },
        "top_probe_patch": {
            "patch": top_patch,
            "normalized_center": (top_patch + 0.5) / n_patches,
            "test_mean_r2": float(patch_mean[top_patch]),
        },
        "mi_top_cell_probe": _cell_probe_payload(int(mi_top_layer), int(mi_top_patch)),
        "mi_low_same_layer_probe": _cell_probe_payload(int(mi_top_layer), low_mi_patch),
        "functional_aligned_top_probe": _cell_probe_payload(
            functional_layer, functional_top_patch
        ),
        "functional_aligned_low_probe": _cell_probe_payload(
            functional_layer, functional_low_patch
        ),
        "selection_registry": {
            "protocol_version": "functional-anchor-v1",
            "atlas_top_cell": [int(mi_top_layer), int(mi_top_patch)],
            "global_anchor_layer": global_anchor_layer,
            "functional_anchor_layer": functional_layer,
            "functional_top_patch": functional_top_patch,
            "functional_low_patch": functional_low_patch,
            "functional_eligible_layers": list(functional_anchor.eligible_layers),
            "functional_excluded_layers": list(functional_anchor.excluded_layers),
            "main_probe_selection": "functional_aligned_top_vs_same_layer_low",
            "uses_probe_or_intervention_outcomes": False,
        },
        "last_patch_test_mean_r2": float(patch_mean[-1]),
        "overall_test_mean_r2": _nanmean(test_r2),
        "scope_mean_test_r2": {
            scope: _nanmean(test_r2[:, :, start:stop])
            for scope, (start, stop) in semantic_scope_slices.items()
        },
        "fraction_test_r2_above_zero": float(np.sum(test_r2 > 0.0) / max(1, np.isfinite(test_r2).sum())),
        "valid_r2_fraction": float(np.isfinite(test_r2).mean()),
        "raw_history_pca_baseline_mean_r2": _nanmean(raw_full.test_r2),
        "raw_recent_patch_pca_baseline_mean_r2": _nanmean(raw_recent.test_r2),
        "mi_vs_probe_mean_spearman": _spearman(mi_z, probe_cell_mean),
        "mi_vs_semantic_probe_spearman": semantic_correlations,
        "semantic_mean_test_r2": {
            name: _nanmean(test_r2[:, :, index]) for index, name in enumerate(semantic_names)
        },
        "validation_hidden_metadata": validation_metadata,
        "test_hidden_metadata": test_metadata,
        "protocol_note": (
            "PCA, target scaling, and Ridge weights use train only; validation selects alpha; "
            "test is evaluated once without refitting. Global and target-channel semantics use "
            f"the MI target's {base_config.future_target.normalization} normalization. The V6.1 "
            "protocol runs global and target-channel probes separately; global semantics are "
            "channel-macro means of 12 nonredundant future properties."
        ),
    }
    np.savez_compressed(
        output_dir / "probe_results.npz",
        validation_r2=validation_r2,
        test_r2=test_r2,
        test_mae_standardized=test_mae,
        selected_alpha=selected_alpha,
        mi_z=mi_z,
        raw_history_test_r2=raw_full.test_r2,
        raw_recent_patch_test_r2=raw_recent.test_r2,
        layer_mean_test_r2=layer_mean,
        patch_mean_test_r2=patch_mean,
    )
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "probe_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["layer", "patch", "semantic", "validation_r2", "test_r2", "test_mae_standardized", "alpha", "mi_z"]
        )
        for layer in range(n_layers):
            for patch in range(n_patches):
                for target, name in enumerate(semantic_names):
                    writer.writerow(
                        [
                            layer,
                            patch,
                            name,
                            float(validation_r2[layer, patch, target]),
                            float(test_r2[layer, patch, target]),
                            float(test_mae[layer, patch, target]),
                            float(selected_alpha[layer, patch, target]),
                            float(mi_z[layer, patch]),
                        ]
                    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"probe_run_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
