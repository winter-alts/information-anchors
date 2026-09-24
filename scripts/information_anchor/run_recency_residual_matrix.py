#!/usr/bin/env python3
"""Run one model's five-dataset recency-residual MI audit queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = ROOT / "results/information_anchor_v6_topology/topology_descriptors.csv"
OUTPUT_ROOT = ROOT / "results/information_anchor_recency_residual"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
MODELS = ("Chronos2", "Moirai2", "Toto2", "TimesFM2.5", "ChronosBolt", "TTM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def references(model: str) -> list[tuple[str, Path]]:
    with TOPOLOGY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        row["dataset"]: Path(row["run_dir"])
        for row in rows
        if row["model"] == model and row["dataset"] in DATASETS
    }
    missing = [dataset for dataset in DATASETS if dataset not in lookup]
    if missing:
        raise RuntimeError(f"Missing topology references for {model}: {missing}")
    return [(dataset, lookup[dataset]) for dataset in DATASETS]


def completed(reference: Path) -> Path | None:
    if not OUTPUT_ROOT.exists():
        return None
    for summary_path in OUTPUT_ROOT.glob("*/summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "complete"
            and payload.get("variant") == "recency_residual"
            and Path(str(payload.get("reference_run", ""))).resolve() == reference.resolve()
        ):
            return summary_path.parent
    return None


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset, reference in references(args.model):
        existing = completed(reference)
        if existing is not None:
            print(f"skip model={args.model} dataset={dataset} run_dir={existing}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts/information_anchor/run_cached_mi_sensitivity.py"),
            "--reference-run",
            str(reference),
            "--variant",
            "recency_residual",
            "--device",
            args.device,
            "--output-root",
            str(OUTPUT_ROOT),
        ]
        print(f"start model={args.model} dataset={dataset}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"complete model={args.model} dataset={dataset}", flush=True)


if __name__ == "__main__":
    main()
