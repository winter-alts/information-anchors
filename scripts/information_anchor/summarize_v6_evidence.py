#!/usr/bin/env python3
"""Build protocol-locked V6 evidence tables from the matrix audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"
REGISTRY_PATH = ROOT / "configs/information_anchor/v6_registered/registry.json"
TOPOLOGY_PATH = ROOT / "results/information_anchor_v6_topology/topology_descriptors.csv"
OUTPUT_ROOT = ROOT / "results/information_anchor_v6_evidence"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _probe_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for record in records:
        for scope in ("global", "target"):
            probe_dir_value = record.get(f"probe_{scope}_dir", "")
            if not probe_dir_value:
                continue
            probe_dir = Path(probe_dir_value)
            summary = _read_json(probe_dir / "summary.json")
            arrays = np.load(probe_dir / "probe_results.npz")
            top = summary["functional_aligned_top_probe"]
            low = summary["functional_aligned_low_probe"]
            mi_top = summary["mi_top_cell_probe"]
            row = {
                "model": record["model"],
                "dataset_key": record["dataset_key"],
                "dataset": record["dataset"],
                "scope": scope,
                "semantic_protocol": summary["semantic_protocol"],
                "functional_layer": int(top["layer"]),
                "functional_top_patch": int(top["patch"]),
                "functional_low_patch": int(low["patch"]),
                "functional_top_mean_r2": float(top["mean_test_r2"]),
                "functional_low_mean_r2": float(low["mean_test_r2"]),
                "functional_top_minus_low_r2": float(
                    top["mean_test_r2"] - low["mean_test_r2"]
                ),
                "mi_top_cell_layer": int(mi_top["layer"]),
                "mi_top_cell_patch": int(mi_top["patch"]),
                "mi_top_cell_mean_r2": float(mi_top["mean_test_r2"]),
                "mi_vs_probe_spearman": float(summary["mi_vs_probe_mean_spearman"]),
                "raw_history_pca_mean_r2": float(
                    summary["raw_history_pca_baseline_mean_r2"]
                ),
                "raw_recent_patch_pca_mean_r2": float(
                    summary["raw_recent_patch_pca_baseline_mean_r2"]
                ),
                "valid_r2_fraction": float(summary["valid_r2_fraction"]),
                "probe_dir": str(probe_dir),
            }
            selected_rows.append(row)

            test_r2 = np.asarray(arrays["test_r2"], dtype=np.float64)
            layer_mean = np.asarray(arrays["layer_mean_test_r2"], dtype=np.float64)
            semantic_names = list(summary["semantic_names"])
            for semantic_index, semantic_name in enumerate(semantic_names):
                semantic_rows.append(
                    {
                        "model": record["model"],
                        "dataset_key": record["dataset_key"],
                        "dataset": record["dataset"],
                        "scope": scope,
                        "semantic": semantic_name,
                        "functional_top_r2": float(
                            top["test_r2_by_semantic"][semantic_name]
                        ),
                        "functional_low_r2": float(
                            low["test_r2_by_semantic"][semantic_name]
                        ),
                        "functional_top_minus_low_r2": float(
                            top["test_r2_by_semantic"][semantic_name]
                            - low["test_r2_by_semantic"][semantic_name]
                        ),
                        "mi_top_cell_r2": float(
                            mi_top["test_r2_by_semantic"][semantic_name]
                        ),
                        "best_cell_r2": float(np.max(test_r2[:, :, semantic_index])),
                        "best_layer": int(
                            np.unravel_index(
                                int(np.argmax(test_r2[:, :, semantic_index])),
                                test_r2[:, :, semantic_index].shape,
                            )[0]
                        ),
                        "best_patch": int(
                            np.unravel_index(
                                int(np.argmax(test_r2[:, :, semantic_index])),
                                test_r2[:, :, semantic_index].shape,
                            )[1]
                        ),
                        "top_mean_layer": int(np.argmax(layer_mean)),
                    }
                )
    return selected_rows, semantic_rows


def _progressive_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for record in records:
        run_dir_value = record.get("progressive_dir", "")
        if not run_dir_value:
            continue
        run_dir = Path(run_dir_value)
        summary = _read_json(run_dir / "summary.json")
        common = {
            "model": record["model"],
            "dataset_key": record["dataset_key"],
            "dataset": record["dataset"],
            "functional_layer": summary["most_informative_layer_zero_based"],
            "sample_count": summary["sample_count"],
            "donor_shift_count": len(summary["donor_shift_offsets"]),
            "run_dir": str(run_dir),
        }
        baseline_rows.append(
            {
                **common,
                "target_channel": summary["target_metric_column"],
                "baseline_target_mse": summary["baseline_target_mse"],
                "baseline_target_mae": summary["baseline_target_mae"],
                "baseline_all_mse": summary["baseline_all_mse"],
                "baseline_all_mae": summary["baseline_all_mae"],
            }
        )
        for item in summary["records"]:
            condition_rows.append(
                {
                    **common,
                    **{
                        key: value
                        for key, value in item.items()
                        if key
                        in {
                            "mode",
                            "strategy",
                            "random_repetition",
                            "fraction",
                            "selected_patch_count",
                            "replaced_patch_count",
                            "selected_mi_z_mean",
                            "target_delta_mse_mean",
                            "target_forecast_change_mse_mean",
                            "target_forecast_change_mse_ci95_lower",
                            "target_forecast_change_mse_ci95_upper",
                            "all_delta_mse_mean",
                            "all_forecast_change_mse_mean",
                            "all_forecast_change_mse_ci95_lower",
                            "all_forecast_change_mse_ci95_upper",
                            "output_footprint_effective_channels",
                        }
                    },
                }
            )
        for item in summary["paired_contrasts"]:
            contrast_rows.append({**common, **item})
    return condition_rows, contrast_rows, baseline_rows


def _semantic_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "level": {"level_q1", "level_q2", "level_q3", "level_q4"},
        "trajectory": {"global_change", "linear_trend"},
        "local_dynamics": {"mean_absolute_change", "diff_std", "roughness"},
        "frequency": {
            "low_frequency_energy",
            "mid_frequency_energy",
            "high_frequency_energy",
        },
    }
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        semantic = str(row["semantic"]).split(":")[-1]
        group = next((name for name, members in groups.items() if semantic in members), None)
        if group is None:
            raise ValueError(f"Unregistered semantic={semantic!r}.")
        key = (
            str(row["model"]),
            str(row["dataset_key"]),
            str(row["dataset"]),
            str(row["scope"]),
            group,
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    metric_names = (
        "functional_top_r2",
        "functional_low_r2",
        "functional_top_minus_low_r2",
        "mi_top_cell_r2",
        "best_cell_r2",
    )
    for (model, dataset_key, dataset, scope, group), items in grouped.items():
        output.append(
            {
                "model": model,
                "dataset_key": dataset_key,
                "dataset": dataset,
                "scope": scope,
                "semantic_group": group,
                "semantic_count": len(items),
                **{
                    metric: float(np.mean([float(item[metric]) for item in items]))
                    for metric in metric_names
                },
            }
        )
    return output


def _architecture_rows(
    topology_rows: list[dict[str, str]],
    registry: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    descriptor_names = (
        "top_quarter_concentration",
        "anchor_ratio",
        "mean_boundary_distance",
        "recent_quarter_mass",
        "expected_layer_depth",
        "layer_entropy",
        "effective_layer_count",
    )
    for model, architecture in registry["models"].items():
        model_topology = [row for row in topology_rows if row["model"] == model]
        model_probes = [row for row in probe_rows if row["model"] == model]
        model_interventions = [
            row
            for row in intervention_rows
            if row["model"] == model
            and row.get("mode") == "remove"
            and row.get("strategy") == "top"
        ]
        row: dict[str, Any] = {
            "model": model,
            **architecture,
            "atlas_dataset_count": len(model_topology),
            "probe_scope_dataset_count": len(model_probes),
            "progressive_dataset_count": len(
                {item["dataset_key"] for item in model_interventions}
            ),
        }
        for descriptor in descriptor_names:
            values = np.asarray(
                [float(item[descriptor]) for item in model_topology], dtype=np.float64
            )
            row[f"mean_{descriptor}"] = float(values.mean()) if len(values) else float("nan")
            row[f"sd_{descriptor}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        probe_differences = np.asarray(
            [item["functional_top_minus_low_r2"] for item in model_probes],
            dtype=np.float64,
        )
        row["mean_probe_top_minus_low_r2"] = (
            float(probe_differences.mean()) if len(probe_differences) else float("nan")
        )
        row["probe_top_above_low_fraction"] = (
            float(np.mean(probe_differences > 0)) if len(probe_differences) else float("nan")
        )
        output.append(row)
    return output


def main() -> None:
    audit = _read_json(AUDIT_PATH)
    registry = _read_json(REGISTRY_PATH)
    topology_rows = _read_csv(TOPOLOGY_PATH) if TOPOLOGY_PATH.exists() else []
    records = audit["records"]
    probe_selected, probe_semantics = _probe_rows(records)
    probe_semantic_groups = _semantic_group_rows(probe_semantics)
    intervention_conditions, intervention_contrasts, intervention_baselines = (
        _progressive_rows(records)
    )
    architecture = _architecture_rows(
        topology_rows,
        registry,
        probe_selected,
        intervention_conditions,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_ROOT / "probe_selected_units.csv", probe_selected)
    _write_csv(OUTPUT_ROOT / "probe_semantics.csv", probe_semantics)
    _write_csv(OUTPUT_ROOT / "probe_semantic_groups.csv", probe_semantic_groups)
    _write_csv(OUTPUT_ROOT / "progressive_conditions.csv", intervention_conditions)
    _write_csv(OUTPUT_ROOT / "progressive_contrasts.csv", intervention_contrasts)
    _write_csv(OUTPUT_ROOT / "forecast_baselines.csv", intervention_baselines)
    _write_csv(OUTPUT_ROOT / "architecture_synthesis.csv", architecture)

    payload = {
        "status": "complete",
        "protocol_version": audit["protocol_version"],
        "semantic_protocol": audit["semantic_protocol"],
        "atlas_rows": audit["reusable_atlas_rows"],
        "complete_probe_rows": audit["complete_probe_rows"],
        "complete_progressive_rows": audit["complete_progressive_rows"],
        "probe_selected_rows": len(probe_selected),
        "probe_semantic_rows": len(probe_semantics),
        "probe_semantic_group_rows": len(probe_semantic_groups),
        "progressive_condition_rows": len(intervention_conditions),
        "progressive_contrast_rows": len(intervention_contrasts),
        "outputs": {
            "probe_selected_units": "probe_selected_units.csv",
            "probe_semantics": "probe_semantics.csv",
            "probe_semantic_groups": "probe_semantic_groups.csv",
            "progressive_conditions": "progressive_conditions.csv",
            "progressive_contrasts": "progressive_contrasts.csv",
            "forecast_baselines": "forecast_baselines.csv",
            "architecture_synthesis": "architecture_synthesis.csv",
        },
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
