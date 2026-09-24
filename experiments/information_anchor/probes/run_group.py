#!/usr/bin/env python3
"""固定 25% MI patch group 的语义 probe。

该实验面向固定 token budget：先在 discovery MI profile 的最高层内排序历史
patch，分别选取 top/middle/bottom 25%，再对选中 patch 的 PCA 表示做 mean
pooling。所有投影和 Ridge 权重仍只使用 train，validation 选择 alpha，test
只做一次评估；脚本不使用自适应峰值阈值。
"""

from __future__ import annotations

import argparse
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
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.projection import fit_projection
from experiments.information_anchor.interventions.common import select_functional_anchor
from experiments.information_anchor.probes.linear import fit_ridge_probe
from experiments.information_anchor.probes.run_linear import (
    ALPHAS,
    _canonical_semantic_scope,
    _ensure_hidden_cache,
    _semantic_splits,
)


def _parse_args() -> argparse.Namespace:
    """解析固定预算 group probe 参数。"""
    parser = argparse.ArgumentParser(description="Fixed-budget MI group semantic probes")
    parser.add_argument("--run-dir", required=True, help="Formal discovery/reference MI run")
    parser.add_argument("--output-root", default="results/information_anchor_group_probes")
    parser.add_argument("--device")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.25,
        help="固定 patch budget；默认 25%%，只按 discovery MI 排序。",
    )
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
        help="Probe OT, system-level, channel-resolved, or the complete v5 semantic suite.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "concat"),
        default="mean",
        help="Mean pooling ablates time identity; concat preserves selected patch order.",
    )
    return parser.parse_args()


def _split_config(config: ExperimentConfig, split: str, device: str | None) -> ExperimentConfig:
    """为按时间切分的 probe 构造配置。"""
    model = replace(config.model, device=device or config.model.device)
    data = replace(config.data, split=split)
    derived = replace(config, model=model, data=data)
    validate_config(derived)
    return derived


def _cache_payload(config: ExperimentConfig, origin_digest: str) -> dict[str, object]:
    """构造与单 cell probe 一致的 hidden cache key。"""
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


def _group_indices(scores: np.ndarray, fraction: float) -> dict[str, np.ndarray]:
    """按 discovery patch MI 固定划分 top/middle/bottom group。"""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1].")
    count = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    middle_start = max(0, (len(scores) - count) // 2)
    return {
        "top_mi": np.sort(order[:count]).astype(np.int64),
        "middle_mi": np.sort(order[middle_start : middle_start + count]).astype(np.int64),
        "bottom_mi": np.sort(order[-count:]).astype(np.int64),
    }


def _nanmean(values: np.ndarray) -> float:
    """返回有限语义的均值，避免退化目标污染 group 对比。"""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def main() -> None:
    """运行固定预算 group probe 并保存逐语义结果。"""
    args = _parse_args()
    reference_dir = Path(args.run_dir).resolve()
    base_config = load_config(reference_dir / "config.json")
    train_config = _split_config(base_config, "discovery", args.device)
    validation_config = _split_config(base_config, "validation", args.device)
    test_config = _split_config(base_config, "test", args.device)
    output_dir = create_run_dir(
        args.output_root,
        f"{base_config.model.name.lower()}_{base_config.data.dataset.lower()}_"
        f"l{base_config.data.seq_len}_p{base_config.data.pred_len}_group_probe",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "reference_config.json", base_config.to_dict())

    frame = load_benchmark_frame(base_config.data, offline=base_config.runtime.offline)
    columns = select_value_columns(frame, base_config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    reference_results = np.load(reference_dir / "mi_results.npz")
    train_origins = np.asarray(reference_results["origins"], dtype=np.int64)
    windows = {
        "train": make_windows(values, train_origins, base_config.data.seq_len, base_config.data.pred_len, columns=columns),
        "validation": make_windows(
            values,
            build_forecast_origins(validation_config.data),
            base_config.data.seq_len,
            base_config.data.pred_len,
            columns=columns,
        ),
        "test": make_windows(
            values,
            build_forecast_origins(test_config.data),
            base_config.data.seq_len,
            base_config.data.pred_len,
            columns=columns,
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
    train_hidden_path = Path(base_config.runtime.cache_root) / train_hidden_metadata["cache_key"] / "history_hidden.npy"
    if not train_hidden_path.exists():
        raise FileNotFoundError(train_hidden_path)

    adapter = None
    validation_hidden_path, validation_metadata, adapter = _ensure_hidden_cache(
        validation_config, windows["validation"], adapter=adapter
    )
    test_hidden_path, test_metadata, adapter = _ensure_hidden_cache(test_config, windows["test"], adapter=adapter)
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hidden = {
        "train": np.load(train_hidden_path, mmap_mode="r"),
        "validation": np.load(validation_hidden_path, mmap_mode="r"),
        "test": np.load(test_hidden_path, mmap_mode="r"),
    }
    n_layers, n_patches = int(hidden["train"].shape[1]), int(hidden["train"].shape[2])
    mi_z = np.asarray(reference_results["mi_z"], dtype=np.float32)
    if mi_z.shape != (n_layers, n_patches):
        raise ValueError(f"MI shape {mi_z.shape} != hidden map {(n_layers, n_patches)}")
    layer_anchor_score = (
        np.asarray(reference_results["layer_anchor_score"], dtype=np.float32)
        if "layer_anchor_score" in reference_results.files
        else mi_z.mean(axis=1)
    )
    functional_anchor = select_functional_anchor(
        mi_z, layer_anchor_score, base_config.model.name
    )
    anchor_layer = functional_anchor.layer
    groups = _group_indices(mi_z[anchor_layer], args.budget_fraction)

    train_layer = np.asarray(hidden["train"][:, anchor_layer], dtype=np.float32)
    validation_layer = np.asarray(hidden["validation"][:, anchor_layer], dtype=np.float32)
    test_layer = np.asarray(hidden["test"][:, anchor_layer], dtype=np.float32)
    projection, train_flat = fit_projection(
        train_layer.reshape(-1, train_layer.shape[-1]),
        args.pca_dim,
        seed=base_config.mi.seed + anchor_layer,
        standardize_features=False,
    )
    validation_flat = projection.transform(validation_layer.reshape(-1, validation_layer.shape[-1]))
    test_flat = projection.transform(test_layer.reshape(-1, test_layer.shape[-1]))
    train_projected = train_flat.reshape(len(train_layer), n_patches, -1)
    validation_projected = validation_flat.reshape(len(validation_layer), n_patches, -1)
    test_projected = test_flat.reshape(len(test_layer), n_patches, -1)

    results: dict[str, dict[str, object]] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for label, indices in groups.items():
        if args.pooling == "mean":
            train_group = train_projected[:, indices].mean(axis=1)
            validation_group = validation_projected[:, indices].mean(axis=1)
            test_group = test_projected[:, indices].mean(axis=1)
        else:
            train_group = train_projected[:, indices].reshape(len(train_projected), -1)
            validation_group = validation_projected[:, indices].reshape(
                len(validation_projected), -1
            )
            test_group = test_projected[:, indices].reshape(len(test_projected), -1)
        result = fit_ridge_probe(
            train_group,
            semantics["train"].values,
            validation_group,
            semantics["validation"].values,
            test_group,
            semantics["test"].values,
            alphas=ALPHAS,
        )
        results[label] = {
            "patch_indices": indices.tolist(),
            "patch_count": int(len(indices)),
            "mean_test_r2": _nanmean(result.test_r2),
            "mean_test_r2_by_scope": {
                scope: _nanmean(result.test_r2[start:stop])
                for scope, (start, stop) in semantic_scope_slices.items()
            },
            "mean_validation_r2": _nanmean(result.validation_r2),
            "fraction_test_r2_above_zero": float(
                np.sum(result.test_r2 > 0.0) / max(1, np.isfinite(result.test_r2).sum())
            ),
            "valid_r2_fraction": float(np.isfinite(result.test_r2).mean()),
            "test_r2_by_semantic": {
                name: float(result.test_r2[i]) for i, name in enumerate(semantic_names)
            },
            "test_mae_by_semantic": {
                name: float(result.test_mae_standardized[i]) for i, name in enumerate(semantic_names)
            },
            "selected_alpha_by_semantic": {
                name: float(result.alpha[i]) for i, name in enumerate(semantic_names)
            },
        }
        npz_payload[f"{label}__test_r2"] = result.test_r2
        npz_payload[f"{label}__test_mae"] = result.test_mae_standardized
        npz_payload[f"{label}__validation_r2"] = result.validation_r2

    top_r2 = results["top_mi"]["mean_test_r2"]
    bottom_r2 = results["bottom_mi"]["mean_test_r2"]
    summary = {
        "status": "complete",
        "model": base_config.model.name,
        "model_id": base_config.model.model_id,
        "dataset": base_config.data.dataset,
        "features": base_config.data.features,
        "target_columns": list(columns),
        "semantic_scope": canonical_semantic_scope,
        "semantic_scope_slices": {
            scope: [start, stop]
            for scope, (start, stop) in semantic_scope_slices.items()
        },
        "reference_run": str(reference_dir),
        "pca_dim": args.pca_dim,
        "budget_fraction": args.budget_fraction,
        "pooling": args.pooling,
        "group_feature_dimension": int(train_group.shape[1]),
        "anchor_layer": int(anchor_layer),
        "anchor_layer_selection": (
            "argmax over downstream-reachable layers of mean_patch(cell null-calibrated z); "
            "identical to the functional intervention layer"
        ),
        "eligible_layers": list(functional_anchor.eligible_layers),
        "num_layers": n_layers,
        "num_patches": n_patches,
        "semantic_names": list(semantic_names),
        "groups": results,
        "top_minus_bottom_mean_test_r2": float(top_r2 - bottom_r2),
        "protocol_note": (
            "Patch groups are selected from discovery MI only; PCA/Ridge use train, validation selects alpha, "
            f"test is evaluated once. Every semantic scope uses the same {base_config.future_target.normalization} normalization as the "
            "MI target. Concat pooling preserves chronological patch identity at a matched feature dimension; "
            "mean pooling is retained only as a position-destroying ablation. No Q3/IQR peak threshold is used."
        ),
        "validation_hidden_metadata": validation_metadata,
        "test_hidden_metadata": test_metadata,
    }
    np.savez_compressed(output_dir / "group_probe_results.npz", **npz_payload)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"group_probe_run_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
