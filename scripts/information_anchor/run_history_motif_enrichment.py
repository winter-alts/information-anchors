#!/usr/bin/env python3
"""Test whether MI-selected patches are enriched for history-only motifs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import build_forecast_origins, load_benchmark_frame, make_windows, select_value_columns
from experiments.information_anchor.targets.history_motif import build_history_motif_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise-run", required=True)
    parser.add_argument("--results-file", default="pointwise_mi_results.npz")
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def _topk_mask(scores: np.ndarray, keep: int, *, largest: bool) -> np.ndarray:
    indices = np.argpartition(scores, -keep if largest else keep - 1, axis=1)
    indices = indices[:, -keep:] if largest else indices[:, :keep]
    mask = np.zeros(scores.shape, dtype=bool)
    mask[np.arange(len(scores))[:, None], indices] = True
    return mask


def _random_mask(n_samples: int, n_patches: int, keep: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n_samples, n_patches), dtype=bool)
    for row in range(n_samples):
        mask[row, rng.choice(n_patches, size=keep, replace=False)] = True
    return mask


def _rates(mask: np.ndarray, motifs: np.ndarray) -> tuple[np.ndarray, float]:
    selected = int(mask.sum())
    if selected < 1:
        raise ValueError("Selector selected no patches")
    selected_rate = (motifs & mask[:, :, None]).sum(axis=(0, 1)) / selected
    base_rate = motifs.mean(axis=(0, 1))
    return (selected_rate - base_rate).astype(np.float64), float(selected)


def _permutation_p_values(
    mask: np.ndarray,
    motifs: np.ndarray,
    observed_delta: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples, n_patches = mask.shape
    keep = int(mask.sum(axis=1)[0])
    null = np.empty((permutations, motifs.shape[-1]), dtype=np.float64)
    for index in range(permutations):
        random = _random_mask(n_samples, n_patches, keep, rng)
        null[index] = _rates(random, motifs)[0]
    return (
        1.0 + np.sum(np.abs(null) >= np.abs(observed_delta)[None, :], axis=0)
    ) / (permutations + 1.0)


def _selector_seed(selector_name: str) -> int:
    """Return a process-stable offset for a selector's permutation null."""
    offsets = {
        "high_mi": 1,
        "low_mi": 2,
        "recent": 3,
        "random_1": 4,
        "random_2": 5,
        "random_3": 6,
        "random_4": 7,
        "random_5": 8,
    }
    return offsets[selector_name]


def main() -> None:
    args = parse_args()
    pointwise_dir = Path(args.pointwise_run).resolve()
    with (pointwise_dir / "summary.json").open("r", encoding="utf-8") as handle:
        pointwise_summary = json.load(handle)
    reference_dir = Path(pointwise_summary["reference_run"])
    config = load_config(reference_dir / "config.json")
    pointwise = np.load(pointwise_dir / args.results_file)
    scores = np.asarray(pointwise["scores"], dtype=np.float32)
    origins = np.asarray(pointwise["origins"], dtype=np.int64)
    layers = np.asarray(pointwise["layer_indices"], dtype=np.int64)
    n_origins, n_layers, n_patches = scores.shape
    keep = max(1, min(n_patches, int(round(args.keep_fraction * n_patches))))

    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    columns = select_value_columns(frame, config.data)
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    windows = make_windows(values, origins, config.data.seq_len, config.data.pred_len, columns=columns)
    with (reference_dir / "hidden_cache.json").open("r", encoding="utf-8") as handle:
        hidden_info = json.load(handle)
    patch_len = int(hidden_info["patch_len"])
    patch_stride = int(hidden_info.get("patch_stride", patch_len))
    target_channel = windows.columns.index(config.data.target)
    motifs, motif_names = build_history_motif_labels(
        windows.history_normalized,
        patch_len=patch_len,
        patch_stride=patch_stride,
        num_patches=n_patches,
        target_channel=target_channel,
    )

    rows: list[dict[str, object]] = []
    layer_summaries = []
    rng = np.random.default_rng(args.seed)
    for layer_position, layer_index in enumerate(layers):
        layer_scores = scores[:, layer_position]
        selectors: dict[str, np.ndarray] = {
            "high_mi": _topk_mask(layer_scores, keep, largest=True),
            "low_mi": _topk_mask(layer_scores, keep, largest=False),
            "recent": np.zeros((n_origins, n_patches), dtype=bool),
        }
        selectors["recent"][:, -keep:] = True
        for random_index in range(5):
            selectors[f"random_{random_index + 1}"] = _random_mask(
                n_origins, n_patches, keep, rng
            )

        summary_selectors = {}
        for selector_name, mask in selectors.items():
            delta, selected_count = _rates(mask, motifs)
            p_values = _permutation_p_values(
                mask,
                motifs,
                delta,
                permutations=args.permutations,
                seed=args.seed + layer_position * 10_000 + _selector_seed(selector_name),
            )
            selected_rate = motifs[mask].mean(axis=0) if np.any(mask) else np.zeros(len(motif_names))
            base_rate = motifs.mean(axis=(0, 1))
            log2_ratio = np.log2((selected_rate + 1e-6) / (base_rate + 1e-6))
            summary_selectors[selector_name] = {
                "selected_count": int(selected_count),
                "top_motifs_by_delta": [
                    {
                        "motif": motif_names[int(index)],
                        "delta": float(delta[index]),
                        "log2_enrichment": float(log2_ratio[index]),
                        "p_value": float(p_values[index]),
                    }
                    for index in np.argsort(delta)[::-1][:3]
                ],
            }
            for motif_index, motif_name in enumerate(motif_names):
                rows.append(
                    {
                        "layer": int(layer_index),
                        "selector": selector_name,
                        "motif": motif_name,
                        "relative_start": "",
                        "relative_stop": "",
                        "selected_rate": float(selected_rate[motif_index]),
                        "base_rate": float(base_rate[motif_index]),
                        "delta": float(delta[motif_index]),
                        "log2_enrichment": float(log2_ratio[motif_index]),
                        "permutation_p": float(p_values[motif_index]),
                    }
                )
        layer_summaries.append({"layer": int(layer_index), "selectors": summary_selectors})

    output_root = Path(args.output_root) if args.output_root else pointwise_dir / "history_motif_enrichment"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "motif_enrichment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_root / "motif_enrichment.npz",
        origins=origins,
        layers=layers,
        motifs=motifs,
        motif_names=np.asarray(motif_names),
    )
    summary = {
        "status": "complete",
        "pointwise_run": str(pointwise_dir),
        "reference_run": str(reference_dir),
        "model": config.model.name,
        "dataset": config.data.dataset,
        "layers": layers.tolist(),
        "num_origins": n_origins,
        "num_patches": n_patches,
        "keep": keep,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "motif_names": list(motif_names),
        "motif_base_rates": motifs.mean(axis=(0, 1)).tolist(),
        "layer_summaries": layer_summaries,
        "protocol": "Motifs are deterministic history-only labels; MI selectors use target-conditioned pointwise scores and are evaluated for descriptive enrichment.",
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
