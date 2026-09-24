from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.information_anchor.artifacts import stable_hash, write_json
from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import build_forecast_origins, load_benchmark_frame, make_windows, origins_hash
from experiments.information_anchor.estimators.nulls import (
    benjamini_hochberg,
    robust_null_score,
    temporal_circular_shift_offsets,
)
from experiments.information_anchor.estimators.projection import FittedProjection, fit_projection
from experiments.information_anchor.probes.semantic import build_semantic_targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out semantic geometry with temporal nulls")
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--null-permutations", type=int, default=49)
    parser.add_argument("--max-samples", type=int, default=256)
    return parser.parse_args()


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


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_rank(np.asarray(x)), _rank(np.asarray(y)))[0, 1])


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    return kernel - kernel.mean(axis=0, keepdims=True) - kernel.mean(axis=1, keepdims=True) + kernel.mean()


def _squared_distances(values: np.ndarray) -> np.ndarray:
    norms = np.square(values).sum(axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * values @ values.T, 0.0)


def _rbf_kernel(values: np.ndarray) -> np.ndarray:
    distances = _squared_distances(values)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-distances / max(2.0 * bandwidth, 1e-12))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    numerator = np.square(cross).sum()
    denominator = np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())
    return float(numerator / max(denominator, 1e-12))


def rbf_alignment(x: np.ndarray, y: np.ndarray) -> float:
    x_kernel = _center_kernel(_rbf_kernel(x))
    y_kernel = _center_kernel(_rbf_kernel(y))
    numerator = np.sum(x_kernel * y_kernel)
    denominator = np.sqrt(np.square(x_kernel).sum() * np.square(y_kernel).sum())
    return float(numerator / max(denominator, 1e-12))


def rsa_spearman(x: np.ndarray, y: np.ndarray) -> float:
    upper = np.triu_indices(len(x), k=1)
    return _spearman(_squared_distances(x)[upper], _squared_distances(y)[upper])


METRICS = {
    "linear_cka": linear_cka,
    "rbf_hsic_alignment": rbf_alignment,
    "rsa_spearman": rsa_spearman,
}


def _test_hidden_path(config, test_origins: np.ndarray) -> Path:
    payload: dict[str, object] = {
        "model_name": config.model.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset": config.data.dataset,
        "target": config.data.target,
        "split": "test",
        "seq_len": config.data.seq_len,
        "origins_hash": origins_hash(test_origins),
        "normalization": config.future_target.normalization,
        "cache_dtype": config.model.cache_dtype,
    }
    if config.model.name == "Chronos2":
        payload["representation_pred_len"] = config.data.pred_len
    return Path(config.runtime.cache_root) / stable_hash(payload) / "history_hidden.npy"


def main() -> None:
    args = _parse_args()
    probe_dir = Path(args.probe_dir)
    with (probe_dir / "summary.json").open("r", encoding="utf-8") as handle:
        probe_summary = json.load(handle)
    reference_dir = Path(probe_summary["reference_run"])
    config = load_config(reference_dir / "config.json")
    reference = np.load(reference_dir / "mi_results.npz")
    mi_z = np.asarray(reference["mi_z"], dtype=np.float32)
    train_origins = np.asarray(reference["origins"], dtype=np.int64)
    test_config = replace(config, data=replace(config.data, split="test"))
    test_origins = build_forecast_origins(test_config.data)

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    values = frame[config.data.target].to_numpy(dtype=np.float32)
    train_windows = make_windows(values, train_origins, config.data.seq_len, config.data.pred_len)
    test_windows = make_windows(values, test_origins, config.data.seq_len, config.data.pred_len)
    train_semantics = build_semantic_targets(train_windows.future_normalized)
    test_semantics = build_semantic_targets(test_windows.future_normalized)
    semantic_mean = train_semantics.values.mean(axis=0, keepdims=True, dtype=np.float64)
    semantic_scale = train_semantics.values.std(axis=0, keepdims=True, dtype=np.float64)
    semantic_scale = np.maximum(semantic_scale, 1e-6)
    test_y = ((test_semantics.values - semantic_mean) / semantic_scale).astype(np.float32)

    sample_indices = np.linspace(0, len(test_origins) - 1, min(args.max_samples, len(test_origins)))
    sample_indices = np.unique(np.rint(sample_indices).astype(np.int64))
    sampled_origins = test_origins[sample_indices]
    test_y = test_y[sample_indices]
    shifts = temporal_circular_shift_offsets(
        sampled_origins,
        args.null_permutations,
        min_temporal_separation=config.data.seq_len + config.data.pred_len,
        seed=config.mi.seed + 70_000,
    )

    hidden_path = _test_hidden_path(config, test_origins)
    if not hidden_path.exists():
        raise FileNotFoundError(hidden_path)
    hidden = np.load(hidden_path, mmap_mode="r")
    top_flat = int(np.argmax(mi_z))
    top_layer, top_patch = np.unravel_index(top_flat, mi_z.shape)
    top_layer, top_patch = int(top_layer), int(top_patch)
    aggregate_layer = int(np.argmax(np.asarray(reference["layer_mi_z"])))
    aggregate_patch = int(np.argmax(np.asarray(reference["patch_mi_z"])))
    selections = [
        ("mi_top_cell", top_layer, top_patch),
        ("mi_aggregate_intersection", aggregate_layer, aggregate_patch),
        ("low_mi_same_layer", top_layer, int(np.argmin(mi_z[top_layer]))),
        ("low_mi_same_patch", int(np.argmin(mi_z[:, top_patch])), top_patch),
        ("final_last", mi_z.shape[0] - 1, mi_z.shape[1] - 1),
    ]
    unique = []
    seen = set()
    for label, layer, patch in selections:
        key = (layer, patch)
        if key not in seen:
            unique.append((label, layer, patch))
            seen.add(key)

    units: list[tuple[str, int | None, int | None, np.ndarray]] = []
    for label, layer, patch in unique:
        projection = _load_projection(probe_dir / "hidden_projections" / f"layer_{layer:02d}.npz")
        projected = projection.transform(np.asarray(hidden[:, layer, patch], dtype=np.float32))
        units.append((label, layer, patch, projected[sample_indices]))

    patch_len = int(probe_summary["test_hidden_metadata"]["patch_len"])
    raw_projection, _ = fit_projection(
        train_windows.history_normalized[:, -patch_len:],
        probe_summary["pca_dim"],
        seed=config.mi.seed + 60_000,
        standardize_features=False,
    )
    raw_test = raw_projection.transform(test_windows.history_normalized[:, -patch_len:])[sample_indices]
    units.append(("raw_recent_patch_pca", None, None, raw_test))

    records = []
    p_values = []
    for label, layer, patch, x in units:
        per_semantic_linear = {
            name: linear_cka(x, test_y[:, index : index + 1])
            for index, name in enumerate(test_semantics.names)
        }
        for metric_name, metric in METRICS.items():
            observed = metric(x, test_y)
            null = np.asarray([metric(x, np.roll(test_y, int(shift), axis=0)) for shift in shifts])
            z_score, p_value = robust_null_score(observed, null)
            p_values.append(p_value)
            records.append(
                {
                    "unit": label,
                    "layer": layer,
                    "patch": patch,
                    "mi_z": None if layer is None else float(mi_z[layer, patch]),
                    "metric": metric_name,
                    "observed": observed,
                    "null_mean": float(null.mean()),
                    "null_std": float(null.std(ddof=1)),
                    "robust_null_z": z_score,
                    "p_value": p_value,
                    "per_semantic_linear_cka": per_semantic_linear if metric_name == "linear_cka" else None,
                }
            )
    q_values = benjamini_hochberg(np.asarray(p_values, dtype=np.float64))
    for record, q_value in zip(records, q_values):
        record["q_value"] = float(q_value)

    output_dir = probe_dir / "semantic_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "model": config.model.name,
        "reference_run": str(reference_dir.resolve()),
        "probe_run": str(probe_dir.resolve()),
        "sample_count": int(len(sample_indices)),
        "sampled_first_origin": int(sampled_origins[0]),
        "sampled_last_origin": int(sampled_origins[-1]),
        "null_permutations": args.null_permutations,
        "required_temporal_separation": config.data.seq_len + config.data.pred_len,
        "semantic_names": list(test_semantics.names),
        "selection_note": "All hidden units are selected from discovery MI before test geometry evaluation.",
        "records": records,
    }
    write_json(output_dir / "summary.json", payload)
    with (output_dir / "geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "unit",
            "layer",
            "patch",
            "mi_z",
            "metric",
            "observed",
            "null_mean",
            "null_std",
            "robust_null_z",
            "p_value",
            "q_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fieldnames})
    print(json.dumps(payload, indent=2), flush=True)
    print(f"geometry_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
