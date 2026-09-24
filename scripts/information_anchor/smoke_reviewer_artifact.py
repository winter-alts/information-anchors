#!/usr/bin/env python3
"""Cache-free integrity and protocol smoke test for the anonymous artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact directory. Defaults to repository root when run in the repo.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(args: argparse.Namespace) -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    if args.artifact_root:
        artifact = Path(args.artifact_root).resolve()
        code = (
            artifact / "code"
            if (artifact / "code/experiments").is_dir()
            else artifact
        )
    elif "artifact" in script.parts and (script.parents[2] / "experiments").is_dir():
        code = script.parents[2]
        artifact = code.parent
    else:
        code = script.parents[2]
        artifact = code
    return artifact, code


def load_modules(code: Path):
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from experiments.information_anchor.estimators.ksg import (
        add_deterministic_jitter,
        ksg_mi,
    )
    from experiments.information_anchor.estimators.nulls import (
        benjamini_hochberg,
        robust_null_score,
        temporal_circular_shift_offsets,
    )
    from experiments.information_anchor.interventions.common import (
        downstream_mixing_layer_indices,
        select_functional_anchor,
    )
    from experiments.information_anchor.signatures import experiment_signature

    return {
        "add_deterministic_jitter": add_deterministic_jitter,
        "ksg_mi": ksg_mi,
        "benjamini_hochberg": benjamini_hochberg,
        "robust_null_score": robust_null_score,
        "temporal_circular_shift_offsets": temporal_circular_shift_offsets,
        "downstream_mixing_layer_indices": downstream_mixing_layer_indices,
        "select_functional_anchor": select_functional_anchor,
        "experiment_signature": experiment_signature,
    }


def verify_manifest(artifact: Path) -> int:
    manifest = artifact / "SHA256SUMS.txt"
    if not manifest.is_file():
        return 0
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        if relative == "SHA256SUMS.txt":
            continue
        path = artifact / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Integrity check failed: {relative}")
        checked += 1
    return checked


def verify_registry(artifact: Path, modules: dict) -> dict[str, object]:
    config_root = artifact / "configs"
    if not (config_root / "registry.json").is_file():
        config_root = artifact / "configs/information_anchor/v6_registered"
    registry_path = config_root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matrix = registry["matrix"]
    if len(matrix) != 42:
        raise RuntimeError(f"Expected 42 registry rows, found {len(matrix)}")
    signatures = []
    models, datasets = set(), set()
    for row in matrix:
        source = Path(str(row["config"]))
        path = config_root / source.name
        config = json.loads(path.read_text(encoding="utf-8"))
        signatures.append(modules["experiment_signature"](config))
        models.add(config["model"]["name"])
        datasets.add(config["data"]["dataset"])
    if len(set(signatures)) != 42 or len(models) != 6 or len(datasets) != 7:
        raise RuntimeError("Registry signatures are not a unique 6x7 matrix")
    return {
        "config_count": len(matrix),
        "model_count": len(models),
        "dataset_count": len(datasets),
    }


def verify_selection_and_ksg(modules: dict) -> dict[str, object]:
    mi_z = np.asarray(
        [
            [0.1, 0.4, 0.2, 0.3],
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 2.0, 4.0, -1.0],
        ],
        dtype=np.float64,
    )
    layer_score = mi_z.mean(axis=1)
    selection = modules["select_functional_anchor"](
        mi_z, layer_score, "Chronos2"
    )
    if selection.layer == 2:
        raise RuntimeError("Chronos2 reachability guard selected the no-op final layer")
    if modules["downstream_mixing_layer_indices"]("Chronos2", 3) != (0, 1):
        raise RuntimeError("Chronos2 reachability declaration changed")

    rng = np.random.default_rng(2021)
    x = rng.normal(size=(256, 2))
    y = x + 0.2 * rng.normal(size=(256, 2))
    x = modules["add_deterministic_jitter"](x, 1e-8, 7)
    signal = modules["ksg_mi"](x, y, k=5)
    null = modules["ksg_mi"](x, rng.permutation(y), k=5)
    if signal <= null + 0.5:
        raise RuntimeError("KSG smoke signal did not exceed the independent control")
    origins = np.arange(256, dtype=np.int64) * 16
    shifts = modules["temporal_circular_shift_offsets"](
        origins,
        n_permutations=19,
        min_temporal_separation=608,
        seed=2021,
    )
    null_values = np.asarray(
        [modules["ksg_mi"](x, np.roll(y, int(shift), axis=0), k=5) for shift in shifts]
    )
    z_score, p_value = modules["robust_null_score"](signal, null_values)
    q_values = modules["benjamini_hochberg"](
        np.asarray([p_value, 0.3, 0.8], dtype=np.float64)
    )
    if not np.isfinite(z_score) or not np.all((0 <= q_values) & (q_values <= 1)):
        raise RuntimeError("Null calibration smoke check failed")
    return {
        "chronos2_noop_final_layer_excluded": True,
        "ksg_signal_minus_permuted": float(signal - null),
        "legal_null_shift_count": len(shifts),
    }


def verify_tables(artifact: Path) -> dict[str, int]:
    tables = artifact / "tables"
    if not tables.is_dir():
        return {"csv_count": 0, "json_count": 0}
    csv_count = 0
    for path in tables.rglob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                raise RuntimeError(f"Empty CSV: {path.relative_to(artifact)}")
        csv_count += 1
    json_paths = list(tables.rglob("*.json"))
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    return {"csv_count": csv_count, "json_count": len(json_paths)}


def verify_history_motif(artifact: Path) -> dict[str, object]:
    root = artifact / "tables/information_anchor_history_motif_matrix"
    direction = list(csv.DictReader(
        (root / "motif_direction_summary.csv").open(encoding="utf-8", newline="")
    ))
    matched = list(csv.DictReader(
        (root / "motif_recent_control_summary.csv").open(encoding="utf-8", newline="")
    ))
    if len(direction) != 11 or len(matched) != 11:
        raise RuntimeError("History-motif summaries must contain all 11 motifs")
    if {int(row["valid_runs"]) for row in direction} != {36}:
        raise RuntimeError("Motif direction summary does not preserve the 36-run gate")
    if {int(row["valid_runs"]) for row in matched} != {36}:
        raise RuntimeError("Matched motif summary does not preserve the 36-run gate")
    by_name = {row["motif"]: row for row in matched}
    if float(by_name["turning_down"]["mean_high_minus_recent"]) <= 0.05:
        raise RuntimeError("Turning-down matched motif effect lost its registered direction")
    if float(by_name["smooth_segment"]["mean_high_minus_recent"]) >= -0.02:
        raise RuntimeError("Smooth-segment depletion lost its registered direction")
    return {
        "included": True,
        "motif_count": len(matched),
        "valid_atlas_count": 36,
        "critic_invalid_atlas_count": 6,
        "turning_down_high_minus_final_quarter": float(
            by_name["turning_down"]["mean_high_minus_recent"]
        ),
        "smooth_segment_high_minus_final_quarter": float(
            by_name["smooth_segment"]["mean_high_minus_recent"]
        ),
    }


def main() -> None:
    args = parse_args()
    artifact, code = locate(args)
    modules = load_modules(code)
    result = {
        "status": "complete",
        "artifact_root": str(artifact),
        "integrity_files_checked": verify_manifest(artifact),
        "registry": verify_registry(artifact, modules),
        "protocol": verify_selection_and_ksg(modules),
        "tables": verify_tables(artifact),
        "history_motif": verify_history_motif(artifact),
        "requires_checkpoint_or_hidden_cache": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
