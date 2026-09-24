#!/usr/bin/env python3
"""Test whether selected hidden units add semantics beyond raw recent history."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    analysis_future_values,
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    select_value_columns,
    origins_hash,
)
from experiments.information_anchor.artifacts import stable_hash
from experiments.information_anchor.estimators.projection import fit_projection, load_projection
from experiments.information_anchor.probes.linear import fit_ridge_probe
from experiments.information_anchor.probes.run_linear import _cache_payload, _semantic_splits


REGISTRY = ROOT / "results/information_anchor_v6_evidence/probe_selected_units.csv"
OUTPUT = ROOT / "results/information_anchor_recency_target_audit"
DATASETS = {"etth1", "etth2", "ettm1", "ettm2", "weather"}
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def cache_path(config, split: str, split_origins: np.ndarray) -> Path:
    root = Path(config.runtime.cache_root)
    if not root.is_absolute():
        root = ROOT / root
    split_config = replace(config, data=replace(config.data, split=split))
    key = stable_hash(_cache_payload(split_config, origins_hash(split_origins)))
    path = root / key / "history_hidden.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mean_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def evaluate(row: dict[str, str]) -> list[dict[str, object]]:
    probe_dir = Path(row["probe_dir"])
    summary = json.loads((probe_dir / "summary.json").read_text(encoding="utf-8"))
    reference = Path(summary["reference_run"])
    config = load_config(reference / "config.json")
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    reference_results = np.load(reference / "mi_results.npz")
    split_configs = {
        "train": config,
        "validation": replace(config, data=replace(config.data, split="validation")),
        "test": replace(config, data=replace(config.data, split="test")),
    }
    origins = {
        "train": np.asarray(reference_results["origins"], dtype=np.int64),
        "validation": build_forecast_origins(split_configs["validation"].data),
        "test": build_forecast_origins(split_configs["test"].data),
    }
    windows = {
        split: make_windows(
            values,
            origins[split],
            config.data.seq_len,
            config.data.pred_len,
            columns=columns,
        )
        for split in origins
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
    semantics, _ = _semantic_splits(
        windows,
        columns,
        columns.index(config.data.target),
        row["scope"],
        futures,
    )
    hidden = {
        "train": np.load(
            Path(config.runtime.cache_root)
            / json.loads((reference / "hidden_cache.json").read_text(encoding="utf-8"))["cache_key"]
            / "history_hidden.npy",
            mmap_mode="r",
        ),
        "validation": np.load(cache_path(config, "validation", origins["validation"]), mmap_mode="r"),
        "test": np.load(cache_path(config, "test", origins["test"]), mmap_mode="r"),
    }
    layer = int(row["functional_layer"])
    patches = {"high": int(row["functional_top_patch"]), "low": int(row["functional_low_patch"])}
    recent_length = min(config.data.pred_len, config.data.seq_len)
    raw = {
        split: batch.history_normalized[:, -recent_length:].reshape(len(batch.origins), -1)
        for split, batch in windows.items()
    }
    raw_projection, raw_train = fit_projection(
        raw["train"], 32, seed=config.mi.seed + 71_000, standardize_features=False
    )
    raw_projected = {
        "train": raw_train,
        "validation": raw_projection.transform(raw["validation"]),
        "test": raw_projection.transform(raw["test"]),
    }
    recent_result = fit_ridge_probe(
        raw_projected["train"], semantics["train"].values,
        raw_projected["validation"], semantics["validation"].values,
        raw_projected["test"], semantics["test"].values, alphas=ALPHAS,
    )
    hidden_projection = load_projection(probe_dir / "hidden_projections" / f"layer_{layer:02d}.npz")
    results: list[dict[str, object]] = []
    combined_by_condition: dict[str, np.ndarray] = {}
    for condition, patch in patches.items():
        selected = {
            split: hidden_projection.transform(
                np.asarray(array[:, layer, patch], dtype=np.float32)
            )
            for split, array in hidden.items()
        }
        combined = {
            split: np.concatenate((raw_projected[split], selected[split]), axis=1)
            for split in selected
        }
        result = fit_ridge_probe(
            combined["train"], semantics["train"].values,
            combined["validation"], semantics["validation"].values,
            combined["test"], semantics["test"].values, alphas=ALPHAS,
        )
        combined_by_condition[condition] = result.test_r2
        for semantic_index, semantic in enumerate(semantics["train"].names):
            results.append(
                {
                    "model": row["model"], "dataset_key": row["dataset_key"],
                    "dataset": row["dataset"], "scope": row["scope"],
                    "layer": layer, "condition": condition, "patch": patch,
                    "semantic": semantic,
                    "recent_only_test_r2": float(recent_result.test_r2[semantic_index]),
                    "recent_plus_hidden_test_r2": float(result.test_r2[semantic_index]),
                    "incremental_test_r2": float(
                        result.test_r2[semantic_index] - recent_result.test_r2[semantic_index]
                    ),
                    "probe_dir": str(probe_dir.resolve()),
                }
            )
    high = combined_by_condition["high"]
    low = combined_by_condition["low"]
    for record in results:
        index = semantics["train"].names.index(str(record["semantic"]))
        record["high_minus_low_combined_test_r2"] = float(high[index] - low[index])
    print(
        f"model={row['model']} dataset={row['dataset']} scope={row['scope']} "
        f"recent={mean_finite(recent_result.test_r2):.4f} "
        f"high_increment={mean_finite(high - recent_result.test_r2):.4f}",
        flush=True,
    )
    return results


def main() -> None:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        registry = [row for row in csv.DictReader(handle) if row["dataset_key"] in DATASETS]
    if len(registry) != 60:
        raise RuntimeError(f"Expected 60 model-dataset-scope probe rows, found {len(registry)}")
    records: list[dict[str, object]] = []
    for row in registry:
        records.extend(evaluate(row))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "incremental_recency_probe_semantics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f"Wrote {path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
