from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from experiments.information_anchor.artifacts import (
    capture_run_context,
    create_run_dir,
    stable_hash,
    write_json,
)
from experiments.information_anchor.config import ExperimentConfig, load_config
from experiments.information_anchor.data import (
    WindowBatch,
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.nulls import temporal_circular_shift_offsets
from experiments.information_anchor.estimators.projection import fit_projection, load_projection
from experiments.information_anchor.probes.linear import RidgeProbeResult, fit_ridge_probe
from experiments.information_anchor.probes.run_linear import (
    ALPHAS,
    _cache_payload,
    _save_projection,
    _split_config,
)
from experiments.information_anchor.probes.target_exclusive import (
    TARGET_EXCLUSIVE_PROTOCOL,
    BlockBootstrapResult,
    ResidualizedTargets,
    build_target_exclusive_targets,
    cross_fitted_residualize,
    paired_r2_block_bootstrap,
)
from experiments.information_anchor.targets.future_summary import build_future_summary


SCOPES = ("global", "target")
TARGET_KINDS = ("raw", "residual")
SOURCE_NAMES = (
    "hidden_high",
    "hidden_low",
    "raw_full_history",
    "raw_high_patch",
    "raw_low_patch",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache-only cross-fitted target-exclusive residual probes"
    )
    parser.add_argument("--probe-dir", required=True, help="Completed registered V6 probe run")
    parser.add_argument(
        "--output-root", default="results/information_anchor_target_exclusive_residual"
    )
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--null-permutations", type=int, default=199)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _required_cache_path(
    config: ExperimentConfig,
    split: str,
    origins: np.ndarray,
    reference_dir: Path,
) -> Path:
    if split == "train":
        cache_key = str(_read_json(reference_dir / "hidden_cache.json")["cache_key"])
    else:
        split_config = _split_config(config, split, None)
        cache_key = stable_hash(_cache_payload(split_config, origins_hash(origins)))
    path = Path(config.runtime.cache_root) / cache_key / "history_hidden.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache-only protocol refuses model extraction; missing {split} hidden cache: {path}"
        )
    return path


def _project_raw_splits(
    values: dict[str, np.ndarray],
    *,
    pca_dim: int,
    seed: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    flattened = {
        split: np.asarray(item, dtype=np.float32).reshape(len(item), -1)
        for split, item in values.items()
    }
    projection, train_projected = fit_projection(
        flattened["train"], pca_dim, seed=seed, standardize_features=False
    )
    _save_projection(output_path, projection)
    return {
        "train": train_projected,
        "validation": projection.transform(flattened["validation"]),
        "test": projection.transform(flattened["test"]),
    }


def _raw_patch_splits(
    windows: dict[str, WindowBatch],
    patch: int,
    *,
    patch_len: int,
    patch_stride: int,
) -> dict[str, np.ndarray]:
    start = patch * patch_stride
    stop = start + patch_len
    output = {
        split: batch.history_normalized[:, start:stop]
        for split, batch in windows.items()
    }
    if any(item.shape[1] != patch_len for item in output.values()):
        raise ValueError(
            f"Raw patch {patch} spans [{start}, {stop}), outside the registered history grid."
        )
    return output


def _mi_covariates(
    futures: dict[str, np.ndarray],
    config: ExperimentConfig,
    columns: tuple[str, ...],
    target_index: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    expected_names: tuple[str, ...] | None = None
    for split, future in futures.items():
        values = future
        channel_names: tuple[str, ...] | None = columns
        if config.future_target.scope == "target":
            values = future[:, :, target_index]
            channel_names = None
        summary = build_future_summary(
            values,
            bins=config.future_target.bins,
            spectral_bands=config.future_target.spectral_bands,
            channel_names=channel_names,
        )
        if expected_names is None:
            expected_names = summary.feature_names
        elif summary.feature_names != expected_names:
            raise RuntimeError("MI nuisance feature definitions differ across splits.")
        output[split] = summary.values
    return output


def _target_splits(
    futures: dict[str, np.ndarray], target_index: int, scope: str
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    targets = {
        split: build_target_exclusive_targets(
            future, scope=scope, target_index=target_index
        )
        for split, future in futures.items()
    }
    names = targets["train"].names
    if any(item.names != names for item in targets.values()):
        raise RuntimeError("Target-exclusive property definitions differ across splits.")
    return {split: item.values for split, item in targets.items()}, names


def _probe_sources(
    features: dict[str, dict[str, np.ndarray]], targets: dict[str, np.ndarray]
) -> dict[str, RidgeProbeResult]:
    return {
        source: fit_ridge_probe(
            splits["train"],
            targets["train"],
            splits["validation"],
            targets["validation"],
            splits["test"],
            targets["test"],
            alphas=ALPHAS,
        )
        for source, splits in features.items()
    }


def _circular_null(
    high: dict[str, np.ndarray],
    low: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    origins: dict[str, np.ndarray],
    *,
    permutations: int,
    separation: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    shifts = {
        split: temporal_circular_shift_offsets(
            split_origins,
            permutations,
            min_temporal_separation=separation,
            seed=seed + index,
        )
        for index, (split, split_origins) in enumerate(origins.items())
    }
    null_delta = np.empty((permutations, targets["train"].shape[1]), dtype=np.float32)
    for permutation in range(permutations):
        shifted = {
            split: np.roll(values, int(shifts[split][permutation]), axis=0)
            for split, values in targets.items()
        }
        high_result = fit_ridge_probe(
            high["train"],
            shifted["train"],
            high["validation"],
            shifted["validation"],
            high["test"],
            shifted["test"],
            alphas=ALPHAS,
        )
        low_result = fit_ridge_probe(
            low["train"],
            shifted["train"],
            low["validation"],
            shifted["validation"],
            low["test"],
            shifted["test"],
            alphas=ALPHAS,
        )
        null_delta[permutation] = high_result.test_r2 - low_result.test_r2
    return null_delta, shifts


def _float(value: float | np.floating) -> float:
    return float(value)


def _bootstrap_fields(prefix: str, result: BlockBootstrapResult, index: int) -> dict[str, object]:
    return {
        f"{prefix}_high_minus_low_r2": _float(result.observed[index]),
        f"{prefix}_block_bootstrap_ci95_lower": _float(result.ci95_lower[index]),
        f"{prefix}_block_bootstrap_ci95_upper": _float(result.ci95_upper[index]),
        f"{prefix}_block_bootstrap_p_greater": _float(result.p_greater[index]),
    }


def main() -> None:
    args = _parse_args()
    if args.nuisance_folds < 2:
        raise ValueError("--nuisance-folds must be at least 2.")
    if args.bootstrap_repetitions < 1 or args.null_permutations < 1:
        raise ValueError("Bootstrap repetitions and null permutations must be positive.")

    source_probe_dir = Path(args.probe_dir).resolve()
    source_probe = _read_json(source_probe_dir / "summary.json")
    if source_probe.get("status") != "complete":
        raise ValueError(f"Source probe is not complete: {source_probe_dir}")
    reference_dir = Path(source_probe["reference_run"]).resolve()
    config = load_config(source_probe_dir / "reference_config.json")
    selection = source_probe["selection_registry"]
    layer = int(selection["functional_anchor_layer"])
    high_patch = int(selection["functional_top_patch"])
    low_patch = int(selection["functional_low_patch"])
    pca_dim = int(source_probe["pca_dim"])

    output_dir = create_run_dir(
        args.output_root,
        (
            f"{config.model.name.lower()}_{config.data.dataset.lower()}_"
            f"l{config.data.seq_len}_p{config.data.pred_len}_target_exclusive"
        ),
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    write_json(output_dir / "reference_config.json", config.to_dict())

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    target_index = columns.index(config.data.target)
    with np.load(reference_dir / "mi_results.npz") as reference_results:
        train_origins = np.asarray(reference_results["origins"], dtype=np.int64)
    origins = {
        "train": train_origins,
        "validation": build_forecast_origins(_split_config(config, "validation", None).data),
        "test": build_forecast_origins(_split_config(config, "test", None).data),
    }
    windows = {
        split: make_windows(
            values,
            split_origins,
            config.data.seq_len,
            config.data.pred_len,
            columns=columns,
        )
        for split, split_origins in origins.items()
    }
    futures = {
        split: analysis_future_values(
            batch,
            values,
            train_end=config.data.train_end,
            normalization=config.future_target.normalization,
        )
        for split, batch in windows.items()
    }
    nuisance_covariates = _mi_covariates(futures, config, columns, target_index)

    hidden_paths = {
        split: _required_cache_path(config, split, split_origins, reference_dir)
        for split, split_origins in origins.items()
    }
    hidden_projection_path = source_probe_dir / "hidden_projections" / f"layer_{layer:02d}.npz"
    if not hidden_projection_path.exists():
        raise FileNotFoundError(
            f"Frozen source-probe projection is missing: {hidden_projection_path}"
        )
    hidden_projection = load_projection(hidden_projection_path)
    hidden_features: dict[str, dict[str, np.ndarray]] = {
        "hidden_high": {},
        "hidden_low": {},
    }
    for split, path in hidden_paths.items():
        hidden = np.load(path, mmap_mode="r")
        if hidden.shape[0] != len(origins[split]):
            raise ValueError(f"{split} hidden cache has {hidden.shape[0]} rows, expected {len(origins[split])}.")
        hidden_features["hidden_high"][split] = hidden_projection.transform(
            np.asarray(hidden[:, layer, high_patch], dtype=np.float32)
        )
        hidden_features["hidden_low"][split] = hidden_projection.transform(
            np.asarray(hidden[:, layer, low_patch], dtype=np.float32)
        )
        del hidden

    metadata = source_probe["validation_hidden_metadata"]
    patch_len = int(metadata["patch_len"])
    patch_stride = int(metadata.get("patch_stride", patch_len))
    raw_projection_dir = output_dir / "raw_projections"
    raw_projection_dir.mkdir(parents=True, exist_ok=True)
    raw_features = {
        "raw_full_history": _project_raw_splits(
            {split: batch.history_normalized for split, batch in windows.items()},
            pca_dim=pca_dim,
            seed=config.mi.seed + 50_000,
            output_path=raw_projection_dir / "full_history.npz",
        ),
        "raw_high_patch": _project_raw_splits(
            _raw_patch_splits(
                windows,
                high_patch,
                patch_len=patch_len,
                patch_stride=patch_stride,
            ),
            pca_dim=pca_dim,
            seed=config.mi.seed + 60_000 + high_patch,
            output_path=raw_projection_dir / "high_patch.npz",
        ),
        "raw_low_patch": _project_raw_splits(
            _raw_patch_splits(
                windows,
                low_patch,
                patch_len=patch_len,
                patch_stride=patch_stride,
            ),
            pca_dim=pca_dim,
            seed=config.mi.seed + 70_000 + low_patch,
            output_path=raw_projection_dir / "low_patch.npz",
        ),
    }
    features = {**hidden_features, **raw_features}

    output_arrays: dict[str, np.ndarray] = {"test_origins": origins["test"]}
    rows: list[dict[str, object]] = []
    scope_summaries: dict[str, object] = {}
    for scope_index, scope in enumerate(SCOPES):
        raw_targets, property_names = _target_splits(futures, target_index, scope)
        nuisance = cross_fitted_residualize(
            raw_targets["train"],
            raw_targets["validation"],
            raw_targets["test"],
            nuisance_covariates["train"],
            nuisance_covariates["validation"],
            nuisance_covariates["test"],
            origins["train"],
            alphas=ALPHAS,
            folds=args.nuisance_folds,
            purge_gap=config.data.pred_len,
        )
        residual_targets = {
            "train": nuisance.train,
            "validation": nuisance.validation,
            "test": nuisance.test,
        }
        target_sets = {"raw": raw_targets, "residual": residual_targets}
        probe_results = {
            kind: _probe_sources(features, target_values)
            for kind, target_values in target_sets.items()
        }
        bootstraps = {
            kind: paired_r2_block_bootstrap(
                probe_results[kind]["hidden_high"].test_target_standardized,
                probe_results[kind]["hidden_high"].test_prediction_standardized,
                probe_results[kind]["hidden_low"].test_prediction_standardized,
                origins["test"],
                dependence_span=config.data.seq_len + config.data.pred_len,
                repetitions=args.bootstrap_repetitions,
                seed=config.mi.seed + 80_000 + 1000 * scope_index + kind_index,
            )
            for kind_index, kind in enumerate(TARGET_KINDS)
        }
        null_delta, null_shifts = _circular_null(
            features["hidden_high"],
            features["hidden_low"],
            residual_targets,
            origins,
            permutations=args.null_permutations,
            separation=config.data.pred_len,
            seed=config.mi.seed + 90_000 + 1000 * scope_index,
        )
        null_p = (
            1.0
            + np.sum(null_delta >= bootstraps["residual"].observed[None, :], axis=0)
        ) / (args.null_permutations + 1.0)

        output_arrays[f"{scope}_property_names"] = np.asarray(property_names)
        output_arrays[f"{scope}_raw_bootstrap_delta"] = bootstraps["raw"].estimates
        output_arrays[f"{scope}_residual_bootstrap_delta"] = bootstraps["residual"].estimates
        output_arrays[f"{scope}_residual_circular_null_delta"] = null_delta
        output_arrays[f"{scope}_nuisance_alpha"] = nuisance.alpha
        output_arrays[f"{scope}_nuisance_validation_r2"] = nuisance.validation_r2
        output_arrays[f"{scope}_nuisance_test_r2"] = nuisance.test_r2
        for kind in TARGET_KINDS:
            output_arrays[f"{scope}_{kind}_test_target_standardized"] = probe_results[kind][
                "hidden_high"
            ].test_target_standardized
            for source in SOURCE_NAMES:
                output_arrays[f"{scope}_{kind}_{source}_test_r2"] = probe_results[kind][
                    source
                ].test_r2
            output_arrays[f"{scope}_{kind}_hidden_high_test_prediction"] = probe_results[kind][
                "hidden_high"
            ].test_prediction_standardized
            output_arrays[f"{scope}_{kind}_hidden_low_test_prediction"] = probe_results[kind][
                "hidden_low"
            ].test_prediction_standardized

        for property_index, property_name in enumerate(property_names):
            row: dict[str, object] = {
                "model": config.model.name,
                "dataset": config.data.dataset,
                "scope": scope,
                "property": property_name,
                "functional_layer": layer,
                "high_patch": high_patch,
                "low_patch": low_patch,
                "nuisance_alpha": _float(nuisance.alpha[property_index]),
                "nuisance_validation_r2": _float(nuisance.validation_r2[property_index]),
                "nuisance_test_r2": _float(nuisance.test_r2[property_index]),
                "discovery_residual_variance_ratio": _float(
                    nuisance.train_residual_variance_ratio[property_index]
                ),
                "residual_circular_null_p_greater": _float(null_p[property_index]),
                "residual_circular_null_delta_mean": _float(
                    np.mean(null_delta[:, property_index])
                ),
            }
            for kind in TARGET_KINDS:
                result = probe_results[kind]
                row.update(_bootstrap_fields(kind, bootstraps[kind], property_index))
                for source in SOURCE_NAMES:
                    row[f"{kind}_{source}_r2"] = _float(result[source].test_r2[property_index])
            rows.append(row)

        scope_summaries[scope] = {
            "property_names": list(property_names),
            "nuisance_alpha": nuisance.alpha.tolist(),
            "nuisance_validation_r2": nuisance.validation_r2.tolist(),
            "nuisance_test_r2": nuisance.test_r2.tolist(),
            "discovery_residual_variance_ratio": nuisance.train_residual_variance_ratio.tolist(),
            "fold_train_sizes": list(nuisance.fold_train_sizes),
            "raw_mean_high_minus_low_r2": _float(np.nanmean(bootstraps["raw"].observed)),
            "residual_mean_high_minus_low_r2": _float(
                np.nanmean(bootstraps["residual"].observed)
            ),
            "residual_circular_null_p_greater": null_p.tolist(),
            "circular_shift_offsets": {
                split: shifts.tolist() for split, shifts in null_shifts.items()
            },
        }
        print(
            f"scope={scope} residual_high_minus_low_mean_r2="
            f"{np.nanmean(bootstraps['residual'].observed):.4f}",
            flush=True,
        )

    split_manifest = {
        split: {
            "num_origins": int(len(split_origins)),
            "first_origin": int(split_origins[0]),
            "last_origin": int(split_origins[-1]),
            "origins_hash": origins_hash(split_origins),
            "hidden_cache": str(hidden_paths[split].resolve()),
        }
        for split, split_origins in origins.items()
    }
    summary = {
        "status": "complete",
        "protocol": TARGET_EXCLUSIVE_PROTOCOL,
        "model": config.model.name,
        "dataset": config.data.dataset,
        "target_channel": config.data.target,
        "target_columns": list(columns),
        "source_probe_dir": str(source_probe_dir),
        "reference_run": str(reference_dir),
        "cache_only": True,
        "selection": {
            "layer": layer,
            "high_patch": high_patch,
            "low_patch": low_patch,
            "source_protocol": selection["protocol_version"],
            "uses_probe_outcomes": False,
        },
        "hidden_pca_dim": pca_dim,
        "mi_nuisance_dimension": int(nuisance_covariates["train"].shape[1]),
        "nuisance": {
            "model": "ridge",
            "alpha_selection": "minimum purged blocked cross-fitted discovery MSE per property",
            "alphas": list(ALPHAS),
            "folds": args.nuisance_folds,
            "purge_gap_in_observations": config.data.pred_len,
            "covariates": "complete pre-PCA structured MI target G_tau",
        },
        "test_inference": {
            "paired_moving_block_bootstrap_repetitions": args.bootstrap_repetitions,
            "dependence_span_in_observations": config.data.seq_len + config.data.pred_len,
            "circular_shift_target_null_permutations": args.null_permutations,
            "circular_shift_target_minimum_separation": config.data.pred_len,
        },
        "raw_input_baselines": [
            "full_history_pca",
            "selected_high_source_patch_pca",
            "same_layer_low_source_patch_pca",
        ],
        "split_manifest": split_manifest,
        "scopes": scope_summaries,
        "protocol_note": (
            "MI coordinates and the registered layer-shared hidden PCA are frozen. The nuisance "
            "ridge and its alpha use discovery origins only through purged blocked cross-fitting; "
            "validation selects probe ridge alpha; test is evaluated once. Missing hidden caches "
            "are fatal and never trigger model extraction."
        ),
    }
    np.savez_compressed(output_dir / "target_exclusive_results.npz", **output_arrays)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "protocol": summary["protocol"],
                "model": summary["model"],
                "dataset": summary["dataset"],
                "cache_only": summary["cache_only"],
                "selection": summary["selection"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"target_exclusive_run_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
