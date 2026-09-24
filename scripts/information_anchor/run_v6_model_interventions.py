#!/usr/bin/env python3
"""Run missing protocol-locked V6 progressive interventions for one model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = ROOT / "scripts/information_anchor/audit_v6_matrix.py"
AUDIT_PATH = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the model-aware evaluation batch size.",
    )
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated dataset keys; default runs every missing registered row.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    default_batch_sizes = {
        "Chronos2": 32,
        "ChronosBolt": 64,
        "Moirai2": 32,
        "Toto2": 32,
        "TimesFM2.5": 16,
        "TTM": 32,
    }
    batch_size = args.batch_size or default_batch_sizes.get(args.model, 16)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    subprocess.run([sys.executable, str(AUDIT_SCRIPT)], cwd=ROOT, check=True)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    requested = (
        {item.strip() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    records = [
        record
        for record in audit["records"]
        if record["model"] == args.model
        and record["atlas_status"] == "reusable"
        and record["progressive_status"] != "complete"
        and (requested is None or record["dataset_key"] in requested)
    ]
    for record in records:
        command = [
            sys.executable,
            "-m",
            "experiments.information_anchor.interventions.progressive_multivariate",
            "--reference-run",
            record["reuse_run_dir"],
            "--fractions",
            "0.125,0.25,0.375,0.5",
            "--strategies",
            "top,bottom,random",
            "--max-samples",
            "128",
            "--donor-shifts",
            "8",
            "--batch-size",
            str(batch_size),
            "--random-repetitions",
            "5",
            "--bootstrap-repetitions",
            "1000",
            "--include-sufficiency",
            "--device",
            args.device,
            "--output-root",
            "results/information_anchor_v6_progressive_formal",
            "--quiet",
        ]
        print(f"run model={record['model']} dataset={record['dataset_key']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(
        json.dumps(
            {
                "status": "complete",
                "model": args.model,
                "requested_rows": len(records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
