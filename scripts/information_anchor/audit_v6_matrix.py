#!/usr/bin/env python3
"""Audit V6 matrix coverage without conflating V5 semantic results."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.signatures import experiment_signature


REGISTRY_PATH = ROOT / "configs/information_anchor/v6_registered/registry.json"
SEARCH_ROOTS = (
    ROOT / "results/information_anchor_v5_global_reference",
    ROOT / "results/information_anchor_v6_reference",
)
PROBE_ROOTS = {
    "global": ROOT / "results/information_anchor_v6_probe_global",
    "target": ROOT / "results/information_anchor_v6_probe_target",
}
PROGRESSIVE_ROOT = ROOT / "results/information_anchor_v6_progressive_formal"
OUTPUT_ROOT = ROOT / "results/information_anchor_v6_matrix_audit"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    """中文说明：通过原子替换发布审计快照，避免并发 launcher 读到半写入 JSON。"""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _completed_runs() -> dict[tuple, list[Path]]:
    runs: dict[tuple, list[Path]] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for config_path in root.glob("*/config.json"):
            run_dir = config_path.parent
            summary_path = run_dir / "summary.json"
            if not summary_path.exists() or not (run_dir / "mi_results.npz").exists():
                continue
            summary = _read(summary_path)
            if summary.get("status") != "complete":
                continue
            runs.setdefault(experiment_signature(_read(config_path)), []).append(run_dir)
    return runs


def _completed_probes(scope: str, semantic_protocol: str) -> dict[tuple, list[Path]]:
    runs: dict[tuple, list[Path]] = {}
    root = PROBE_ROOTS[scope]
    if not root.exists():
        return runs
    for summary_path in root.glob("*/summary.json"):
        run_dir = summary_path.parent
        config_path = run_dir / "reference_config.json"
        if not config_path.exists():
            continue
        summary = _read(summary_path)
        if (
            summary.get("status") != "complete"
            or summary.get("semantic_scope") != scope
            or summary.get("semantic_protocol") != semantic_protocol
        ):
            continue
        runs.setdefault(experiment_signature(_read(config_path)), []).append(run_dir)
    return runs


def _completed_progressive() -> dict[tuple, list[Path]]:
    runs: dict[tuple, list[Path]] = {}
    if not PROGRESSIVE_ROOT.exists():
        return runs
    for summary_path in PROGRESSIVE_ROOT.glob("*/summary.json"):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        summary = _read(summary_path)
        if (
            summary.get("status") != "complete"
            or summary.get("intervention_protocol") != "progressive-multivariate-v6.1"
            or summary.get("fractions") != [0.125, 0.25, 0.375, 0.5]
            or summary.get("strategies") != ["top", "bottom", "random"]
            or int(summary.get("random_repetitions", 0)) < 5
            or not summary.get("include_sufficiency", False)
            or int(summary.get("sample_count", 0)) < 128
            or len(summary.get("donor_shift_offsets", [])) < 8
        ):
            continue
        runs.setdefault(experiment_signature(_read(config_path)), []).append(run_dir)
    return runs


def main() -> None:
    registry = _read(REGISTRY_PATH)
    completed = _completed_runs()
    semantic_protocol = registry["semantic_protocol"]
    completed_probes = {
        scope: _completed_probes(scope, semantic_protocol) for scope in PROBE_ROOTS
    }
    completed_progressive = _completed_progressive()
    records = []
    for row in registry["matrix"]:
        config_path = ROOT / row["config"]
        config = _read(config_path)
        signature = experiment_signature(config)
        candidates = completed.get(signature, [])
        latest = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
        probe_candidates = {
            scope: completed_probes[scope].get(signature, [])
            for scope in PROBE_ROOTS
        }
        latest_probes = {
            scope: max(paths, key=lambda path: path.stat().st_mtime) if paths else None
            for scope, paths in probe_candidates.items()
        }
        progressive_candidates = completed_progressive.get(signature, [])
        latest_progressive = (
            max(progressive_candidates, key=lambda path: path.stat().st_mtime)
            if progressive_candidates
            else None
        )
        records.append(
            {
                **row,
                "atlas_status": "reusable" if latest else "missing",
                "reuse_run_dir": str(latest.resolve()) if latest else "",
                "probe_global_status": "complete" if latest_probes["global"] else "missing",
                "probe_global_dir": (
                    str(latest_probes["global"].resolve()) if latest_probes["global"] else ""
                ),
                "probe_target_status": "complete" if latest_probes["target"] else "missing",
                "probe_target_dir": (
                    str(latest_probes["target"].resolve()) if latest_probes["target"] else ""
                ),
                "semantic_probe_status": (
                    "complete"
                    if all(latest_probes.values())
                    else "partial"
                    if any(latest_probes.values())
                    else "missing"
                ),
                "progressive_status": "complete" if latest_progressive else "missing",
                "progressive_dir": (
                    str(latest_progressive.resolve()) if latest_progressive else ""
                ),
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with (OUTPUT_ROOT / "matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "protocol_version": registry["protocol_version"],
        "semantic_protocol": registry["semantic_protocol"],
        "total_rows": len(records),
        "reusable_atlas_rows": sum(row["atlas_status"] == "reusable" for row in records),
        "missing_atlas_rows": sum(row["atlas_status"] == "missing" for row in records),
        "complete_probe_rows": sum(
            row["semantic_probe_status"] == "complete" for row in records
        ),
        "complete_progressive_rows": sum(
            row["progressive_status"] == "complete" for row in records
        ),
        "records": records,
        "lineage_note": (
            "A reusable atlas matches the model, data, normalization, projection, estimator, "
            "origin-count and null protocol. V5 semantic probes are never reused as V6 probes."
        ),
    }
    _atomic_write_json(OUTPUT_ROOT / "matrix.json", payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "total_rows",
                    "reusable_atlas_rows",
                    "missing_atlas_rows",
                    "complete_probe_rows",
                    "complete_progressive_rows",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
