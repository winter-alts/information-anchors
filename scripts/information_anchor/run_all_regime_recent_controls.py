#!/usr/bin/env python3
"""Run paired same-layer Recent controls for the registered V6 matrix.

The launcher assigns one model to one GPU and processes that model's missing
regimes sequentially.  Completed runs are detected by model/dataset identity,
so the command is safe to resume without overwriting existing artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"
OUTPUT_ROOT = ROOT / "results/information_anchor_recent_control_all_regimes"
LEGACY_ROOT = ROOT / "results/information_anchor_recent_control_formal_etth1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5",
        help="Comma-separated devices; assigned to models in registry order.",
    )
    parser.add_argument(
        "--models",
        help="Optional comma-separated model names.",
    )
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated registered dataset keys.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for root in (LEGACY_ROOT, OUTPUT_ROOT):
        if not root.exists():
            continue
        for path in root.glob("*/summary.json"):
            summary = read_json(path)
            if summary.get("status") == "complete" and summary.get("strategies") == ["recent"]:
                keys.add((str(summary["model"]), str(summary["dataset"])))
    return keys


def run_model(model: str, device: str, rows: list[dict], dry_run: bool) -> dict:
    batch_sizes = {
        "Chronos2": 4,
        "ChronosBolt": 8,
        "Moirai2": 4,
        "Toto2": 4,
        "TimesFM2.5": 4,
        "TTM": 8,
    }
    commands: list[list[str]] = []
    for row in rows:
        commands.append(
            [
                sys.executable,
                "-m",
                "experiments.information_anchor.interventions.progressive_multivariate",
                "--reference-run",
                row["reuse_run_dir"],
                "--fractions",
                "0.125,0.25,0.375,0.5",
                "--strategies",
                "recent",
                "--max-samples",
                "128",
                "--donor-shifts",
                "8",
                "--batch-size",
                str(batch_sizes[model]),
                "--random-repetitions",
                "0",
                "--bootstrap-repetitions",
                "1000",
                "--include-sufficiency",
                "--device",
                device,
                "--quiet",
                "--output-root",
                str(OUTPUT_ROOT.relative_to(ROOT)),
            ]
        )
    if dry_run:
        return {"model": model, "device": device, "commands": commands, "status": "dry_run"}
    completed = 0
    for row, command in zip(rows, commands, strict=True):
        print(
            f"recent-control model={model} dataset={row['dataset_key']} device={device}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        completed += 1
    return {
        "model": model,
        "device": device,
        "requested": len(rows),
        "completed": completed,
        "status": "complete",
    }


def main() -> None:
    args = parse_args()
    audit = read_json(AUDIT)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    requested_models = (
        {item.strip() for item in args.models.split(",") if item.strip()}
        if args.models
        else None
    )
    requested_datasets = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    done = completed_keys()
    rows_by_model: dict[str, list[dict]] = {}
    for row in audit["records"]:
        key = (str(row["model"]), str(row["dataset"]))
        if row["atlas_status"] != "reusable" or key in done:
            continue
        if requested_models is not None and row["model"] not in requested_models:
            continue
        if requested_datasets is not None and row["dataset_key"] not in requested_datasets:
            continue
        rows_by_model.setdefault(str(row["model"]), []).append(row)
    if not rows_by_model:
        print(json.dumps({"status": "complete", "missing_runs": 0}, indent=2))
        return
    models = list(rows_by_model)
    if len(devices) < len(models):
        raise ValueError(f"Need at least {len(models)} devices for model-isolated parallelism.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = [
            run_model(model, devices[index], rows_by_model[model], True)
            for index, model in enumerate(models)
        ]
        print(json.dumps(payload, indent=2))
        return
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                run_model, model, devices[index], rows_by_model[model], False
            ): model
            for index, model in enumerate(models)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["model"])
    print(json.dumps({"status": "complete", "models": results}, indent=2))


if __name__ == "__main__":
    main()
