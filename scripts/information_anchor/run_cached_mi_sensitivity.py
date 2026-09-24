#!/usr/bin/env python3
"""Run target/null MI sensitivities from locked hidden caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


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
    origins_hash,
    select_value_columns,
)
from experiments.information_anchor.estimators.ksg import add_deterministic_jitter
from experiments.information_anchor.estimators.ksg_torch import TorchKSG
from experiments.information_anchor.estimators.nulls import (
    benjamini_hochberg,
    robust_null_score,
)
from experiments.information_anchor.estimators.projection import (
    fit_projection,
    load_projection,
)
from experiments.information_anchor.targets.future_summary import build_future_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument(
        "--variant",
        choices=("dynamics_legal", "full_seasonal", "recency_residual"),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--output-root", default="results/information_anchor_cached_sensitivity")
    return parser.parse_args()


def save_projection(path: Path, projection) -> None:
    np.savez_compressed(
        path,
        scaler_mean=projection.scaler_mean,
        scaler_scale=projection.scaler_scale,
        pca_mean=projection.pca_mean,
        pca_components=projection.pca_components,
        explained_variance_ratio=projection.explained_variance_ratio,
        whiten_scale=projection.whiten_scale,
    )


def seasonal_permutations(
    origins: np.ndarray,
    repetitions: int,
    period: int,
    minimum_separation: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if period < 2:
        raise ValueError("seasonal-period must be at least two.")
    groups = [
        np.flatnonzero(origins % period == phase)
        for phase in range(period)
    ]
    groups = [group for group in groups if len(group)]
    legal_shifts: list[np.ndarray] = []
    for group in groups:
        candidates = []
        group_origins = origins[group]
        for shift in range(1, len(group)):
            paired = np.roll(group_origins, shift)
            if int(np.min(np.abs(group_origins - paired))) >= minimum_separation:
                candidates.append(shift)
        if not candidates:
            raise ValueError(
                f"No legal within-season shift for phase group of size {len(group)}."
            )
        legal_shifts.append(np.asarray(candidates, dtype=np.int64))
    rng = np.random.default_rng(seed)
    permutations = np.empty((repetitions, len(origins)), dtype=np.int64)
    for repetition in range(repetitions):
        permutation = np.empty(len(origins), dtype=np.int64)
        for group, candidates in zip(groups, legal_shifts, strict=True):
            shift = int(rng.choice(candidates))
            permutation[group] = np.roll(group, shift)
        permutations[repetition] = permutation
    separations = np.min(
        np.abs(origins[None, :] - origins[permutations]), axis=1
    ).astype(np.int64)
    if not np.all(origins[None, :] % period == origins[permutations] % period):
        raise RuntimeError("Seasonal permutation did not preserve phase.")
    return permutations, separations


def legal_shift_permutations(reference: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(reference["origins"], dtype=np.int64)
    shifts = np.asarray(reference["shifts"], dtype=np.int64)
    row = np.arange(len(origins), dtype=np.int64)
    permutations = np.stack([(row - int(shift)) % len(origins) for shift in shifts])
    separations = np.min(
        np.abs(origins[None, :] - origins[permutations]), axis=1
    ).astype(np.int64)
    return permutations, separations


def calibrate(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.empty_like(observed, dtype=np.float32)
    p = np.empty_like(observed, dtype=np.float32)
    for index in range(len(observed)):
        z[index], p[index] = robust_null_score(observed[index], null[:, index])
    return z, p, benjamini_hochberg(p).astype(np.float32)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values).reshape(-1), kind="stable")
    result = np.empty(len(order), dtype=np.float64)
    result[order] = np.arange(len(order), dtype=np.float64)
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rank(left)
    right_rank = rank(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    return float(left_rank.dot(right_rank) / denominator)


def top_overlap(left: np.ndarray, right: np.ndarray, fraction: float = 0.25) -> float:
    count = max(1, int(np.ceil(fraction * left.size)))
    left_top = set(np.argsort(-left.reshape(-1), kind="stable")[:count].tolist())
    right_top = set(np.argsort(-right.reshape(-1), kind="stable")[:count].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def recency_residual_target(
    history_normalized: np.ndarray,
    future_projected: np.ndarray,
    origins: np.ndarray,
    *,
    recent_length: int,
    bins: int,
    spectral_bands: int,
    channel_names: tuple[str, ...],
    purge_gap: int,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    """Cross-fit future-target residuals from a model-independent recent-history summary."""
    recent = np.asarray(history_normalized[:, -recent_length:], dtype=np.float32)
    recent_summary = build_future_summary(
        recent,
        bins=min(bins, recent_length),
        spectral_bands=spectral_bands,
        channel_names=channel_names,
    )
    features = np.asarray(recent_summary.values, dtype=np.float64)
    target = np.asarray(future_projected, dtype=np.float64)
    predictions = np.full_like(target, np.nan)
    fold_ids = np.full(len(origins), -1, dtype=np.int16)
    fold_alphas: list[float] = []
    fold_sizes: list[dict[str, int]] = []
    alpha_grid = np.logspace(-4, 4, 17, dtype=np.float64)
    for fold, validation_indices in enumerate(np.array_split(np.arange(len(origins)), 5)):
        validation_origins = origins[validation_indices]
        train_mask = (origins < int(validation_origins.min()) - purge_gap) | (
            origins > int(validation_origins.max()) + purge_gap
        )
        train_indices = np.flatnonzero(train_mask)
        if len(train_indices) < max(64, target.shape[1] * 4):
            raise ValueError(
                f"Recency residual fold {fold} has only {len(train_indices)} purged training origins."
            )
        scaler = StandardScaler().fit(features[train_indices])
        train_x = scaler.transform(features[train_indices])
        validation_x = scaler.transform(features[validation_indices])
        ridge = RidgeCV(alphas=alpha_grid, scoring="neg_mean_squared_error").fit(
            train_x, target[train_indices]
        )
        predictions[validation_indices] = ridge.predict(validation_x)
        fold_ids[validation_indices] = fold
        fold_alphas.append(float(ridge.alpha_))
        fold_sizes.append(
            {
                "fold": fold,
                "train_origins": int(len(train_indices)),
                "validation_origins": int(len(validation_indices)),
            }
        )
    if not np.isfinite(predictions).all() or np.any(fold_ids < 0):
        raise RuntimeError("Recency residual cross-fitting did not cover every discovery origin.")
    residual = target - predictions
    residual_mean = residual.mean(axis=0, keepdims=True)
    residual_scale = np.maximum(residual.std(axis=0, keepdims=True), 1e-8)
    residual_standardized = ((residual - residual_mean) / residual_scale).astype(np.float32)
    target_mean = target.mean(axis=0, keepdims=True)
    ss_res = np.sum((target - predictions) ** 2, axis=0)
    ss_total = np.maximum(np.sum((target - target_mean) ** 2, axis=0), 1e-12)
    per_dimension_r2 = 1.0 - ss_res / ss_total
    audit = {
        "recent_length": recent_length,
        "recent_feature_count": int(features.shape[1]),
        "cross_fit_folds": 5,
        "purge_gap": purge_gap,
        "ridge_alpha_grid": alpha_grid.tolist(),
        "selected_alphas": fold_alphas,
        "fold_sizes": fold_sizes,
        "cross_fitted_r2_mean_over_target_dimensions": float(per_dimension_r2.mean()),
        "cross_fitted_r2_by_target_dimension": per_dimension_r2.tolist(),
        "target_residual_variance_fraction": float(
            np.mean(np.var(residual, axis=0) / np.maximum(np.var(target, axis=0), 1e-12))
        ),
    }
    arrays = {
        "future_target": target.astype(np.float32),
        "recency_prediction": predictions.astype(np.float32),
        "recency_residual": residual_standardized,
        "fold_ids": fold_ids,
    }
    return residual_standardized, audit, arrays


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if config.mi.estimator != "ksg_gpu":
        raise ValueError("Cached sensitivity is registered for the KSG-GPU atlas.")
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    origins = build_forecast_origins(config.data)
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
    recency_audit: dict[str, object] | None = None
    recency_arrays: dict[str, np.ndarray] | None = None
    if args.variant == "dynamics_legal":
        keep = np.asarray(
            [
                index
                for index, name in enumerate(future_summary.feature_names)
                if name.endswith(
                    (
                        "future_std",
                        "future_slope",
                        "future_total_change",
                        "future_diff_std",
                    )
                )
                or ":spectral_band_energy_" in name
            ],
            dtype=np.int64,
        )
        selected_features = tuple(future_summary.feature_names[index] for index in keep)
        future_projection, future_projected = fit_projection(
            future_summary.values[:, keep],
            config.future_target.pca_dim,
            seed=config.mi.seed,
            standardize_features=True,
        )
    else:
        selected_features = tuple(future_summary.feature_names)
        future_projection = load_projection(reference_dir / "future_projection.npz")
        future_projected = future_projection.transform(future_summary.values)
        if args.variant == "recency_residual":
            future_projected, recency_audit, recency_arrays = recency_residual_target(
                windows.history_normalized,
                future_projected,
                origins,
                recent_length=min(config.data.pred_len, config.data.seq_len),
                bins=config.future_target.bins,
                spectral_bands=config.future_target.spectral_bands,
                channel_names=windows.columns,
                purge_gap=config.data.seq_len + config.data.pred_len,
            )

    reference = np.load(reference_dir / "mi_results.npz")
    if not np.array_equal(origins, reference["origins"]):
        raise ValueError("Reference origins changed; cached sensitivity is not paired.")
    if args.variant == "full_seasonal":
        permutations, realized_separations = seasonal_permutations(
            origins,
            config.mi.null_permutations,
            args.seasonal_period,
            config.data.seq_len + config.data.pred_len,
            config.mi.seed + 510_000,
        )
        null_protocol = "within-hour-of-day distant permutation"
    else:
        permutations, realized_separations = legal_shift_permutations(reference)
        null_protocol = "original legal temporal circular shifts"

    future_for_mi = add_deterministic_jitter(
        future_projected, config.mi.jitter, config.mi.seed + 11
    )
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
    hidden = np.load(
        cache_root / cache_info["cache_key"] / "history_hidden.npy", mmap_mode="r"
    )
    n_samples, n_layers, n_patches, _ = hidden.shape
    mi_raw = np.empty((n_layers, n_patches), dtype=np.float32)
    mi_z = np.empty_like(mi_raw)
    p_values = np.empty_like(mi_raw)
    null_mi = np.empty((len(permutations), n_layers, n_patches), dtype=np.float32)
    for layer in range(n_layers):
        hidden_projection = load_projection(
            reference_dir / f"hidden_projection_layer_{layer:02d}.npz"
        )
        flat = np.asarray(hidden[:, layer], dtype=np.float32).reshape(n_samples * n_patches, -1)
        projected = hidden_projection.transform(flat).reshape(n_samples, n_patches, -1)
        for patch in range(n_patches):
            x = add_deterministic_jitter(
                projected[:, patch],
                config.mi.jitter,
                config.mi.seed + 1000 * layer + patch,
            )
            observed, null_values = estimator.estimate_observed_and_permutations(
                x, permutations
            )
            mi_raw[layer, patch] = observed
            null_mi[:, layer, patch] = null_values
            mi_z[layer, patch], p_values[layer, patch] = robust_null_score(
                observed, null_values
            )
        print(
            f"model={config.model.name} variant={args.variant} layer={layer + 1}/{n_layers}",
            flush=True,
        )
    q_values = benjamini_hochberg(p_values).astype(np.float32)
    layer_raw = mi_raw.mean(axis=1)
    layer_null = null_mi.mean(axis=2)
    layer_z, layer_p, layer_q = calibrate(layer_raw, layer_null)
    patch_raw = mi_raw.mean(axis=0)
    patch_null = null_mi.mean(axis=1)
    patch_z, patch_p, patch_q = calibrate(patch_raw, patch_null)

    output_dir = create_run_dir(
        args.output_root,
        f"{config.model.name.lower()}_{config.data.dataset.lower()}_{args.variant}",
    )
    capture_run_context(output_dir, " ".join(sys.argv))
    save_projection(output_dir / "future_projection.npz", future_projection)
    write_json(output_dir / "future_features.json", list(selected_features))
    if recency_audit is not None and recency_arrays is not None:
        write_json(output_dir / "recency_residual_protocol.json", recency_audit)
        np.savez_compressed(output_dir / "recency_residual_target.npz", **recency_arrays)
    np.savez_compressed(
        output_dir / "mi_sensitivity_results.npz",
        mi_raw=mi_raw,
        mi_z=mi_z,
        p_values=p_values,
        q_values=q_values,
        null_mi=null_mi,
        layer_mi_raw=layer_raw,
        layer_mi_z=layer_z,
        layer_p_values=layer_p,
        layer_q_values=layer_q,
        patch_mi_raw=patch_raw,
        patch_mi_z=patch_z,
        patch_p_values=patch_p,
        patch_q_values=patch_q,
        permutations=permutations,
        realized_separations=realized_separations,
        origins=origins,
    )
    original_z = np.asarray(reference["mi_z"], dtype=np.float64)
    original_patch = np.asarray(reference["patch_mi_z"], dtype=np.float64)
    original_layer = np.asarray(reference["layer_anchor_score"], dtype=np.float64)
    summary = {
        "status": "complete",
        "variant": args.variant,
        "reference_run": str(reference_dir),
        "model": config.model.name,
        "dataset": config.data.dataset,
        "origins_hash": origins_hash(origins),
        "sample_count": len(origins),
        "future_feature_count": len(selected_features),
        "future_pca_explained_variance": float(
            future_projection.explained_variance_ratio.sum()
        ),
        "null_protocol": null_protocol,
        "null_repetitions": len(permutations),
        "minimum_realized_separation": int(realized_separations.min()),
        "seasonal_period": args.seasonal_period if args.variant == "full_seasonal" else None,
        "cell_z_spearman_with_primary": spearman(mi_z, original_z),
        "patch_z_spearman_with_primary": spearman(patch_z, original_patch),
        "layer_profile_spearman_with_primary": spearman(mi_z.mean(axis=1), original_layer),
        "top_quarter_cell_jaccard_with_primary": top_overlap(mi_z, original_z),
        "top_patch": int(np.argmax(patch_z)),
        "primary_top_patch": int(np.argmax(original_patch)),
        "top_cell": [
            int(value) for value in np.unravel_index(int(np.argmax(mi_z)), mi_z.shape)
        ],
        "primary_top_cell": [
            int(value)
            for value in np.unravel_index(int(np.argmax(original_z)), original_z.shape)
        ],
        "fraction_cell_q_below_0_05": float(np.mean(q_values < 0.05)),
        "analysis_elapsed_seconds": time.perf_counter() - started,
        "hidden_forward_rerun": False,
    }
    if recency_audit is not None:
        summary["recency_residual_protocol"] = recency_audit
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"status": "complete", "run_dir": str(output_dir), **summary}))


if __name__ == "__main__":
    main()
