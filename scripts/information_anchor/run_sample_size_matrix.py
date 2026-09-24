#!/usr/bin/env python3
"""Run one model's nested-sample ETTm1 MI audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = ROOT / "results/information_anchor_v6_topology/topology_descriptors.csv"
MODELS = ("Chronos2", "Moirai2", "Toto2", "TimesFM2.5", "ChronosBolt", "TTM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with TOPOLOGY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["model"] == args.model and row["dataset"] == "ETTm1")
    for sample_count in (512, 768):
        command = [
            sys.executable,
            str(ROOT / "scripts/information_anchor/run_cached_mi_sample_size.py"),
            "--reference-run", row["run_dir"],
            "--sample-count", str(sample_count),
            "--device", args.device,
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
