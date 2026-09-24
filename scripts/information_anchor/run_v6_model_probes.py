#!/usr/bin/env python3
"""Run idempotent V6 global/target probes for one registered model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"
SEMANTIC_PROTOCOL = "future-semantics-v6.1-12-nonredundant"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _already_complete(output_root: Path, reference_run: str, scope: str) -> bool:
    if not output_root.exists():
        return False
    reference = str(Path(reference_run).resolve())
    for summary_path in output_root.glob("*/summary.json"):
        summary = _read(summary_path)
        if (
            summary.get("status") == "complete"
            and str(Path(summary.get("reference_run", "")).resolve()) == reference
            and summary.get("semantic_scope") == scope
            and summary.get("semantic_protocol") == SEMANTIC_PROTOCOL
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--scopes", nargs="+", default=("global", "target"))
    args = parser.parse_args()

    audit = _read(AUDIT_PATH)
    rows = [row for row in audit["records"] if row["model"] == args.model]
    if len(rows) != 7:
        raise RuntimeError(f"Expected seven registered rows for {args.model}, found {len(rows)}.")
    for row in rows:
        if row["atlas_status"] != "reusable" or not row["reuse_run_dir"]:
            raise RuntimeError(f"Atlas is not complete for {args.model}/{row['dataset_key']}.")
        for scope in args.scopes:
            output_root = ROOT / f"results/information_anchor_v6_probe_{scope}"
            if _already_complete(output_root, row["reuse_run_dir"], scope):
                print(f"skip {args.model}/{row['dataset_key']}/{scope}", flush=True)
                continue
            command = [
                sys.executable,
                "-m",
                "experiments.information_anchor.probes.run_linear",
                "--run-dir",
                row["reuse_run_dir"],
                "--output-root",
                str(output_root.relative_to(ROOT)),
                "--device",
                args.device,
                "--semantic-scope",
                scope,
            ]
            print("run " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
