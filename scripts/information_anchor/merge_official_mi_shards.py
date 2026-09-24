#!/usr/bin/env python3
"""Merge disjoint official MI score-shard runs with coverage checks.

Parallel electricity jobs write separate roots to avoid manifest races.  This
utility only merges compact ``scores/*.npz`` sidecars; hidden/input
intermediates are never copied.  Every shard range must be contiguous and
non-overlapping before the merged manifest is marked ready.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--part-root", action="append", required=True,
                        help="part root containing manifest.json and scores/; repeat")
    parser.add_argument("--total-query-windows", type=int, required=True)
    parser.add_argument("--replace", action="store_true",
                        help="replace existing output score files after validation")
    return parser.parse_args()


def _range_from_name(path: Path) -> tuple[int, int]:
    stem = path.stem
    if not stem.startswith("shard_"):
        raise ValueError(f"unexpected score filename: {path.name}")
    try:
        lo, hi = stem.split("_")[1:]
        return int(lo), int(hi)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot parse shard range: {path.name}") from exc


def _validate_part(path: Path, dataset: str) -> list[tuple[int, int, Path, Path]]:
    manifest_path = path / "manifest.json"
    scores = path / "scores"
    if not manifest_path.exists() or not scores.is_dir():
        raise FileNotFoundError(f"part lacks manifest/scores: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch in {manifest_path}")
    rows = []
    for npz in sorted(scores.glob("shard_*.npz")):
        lo, hi = _range_from_name(npz)
        if hi <= lo:
            raise ValueError(f"invalid range in {npz.name}")
        sidecar = npz.with_suffix(".json")
        if not sidecar.exists():
            raise FileNotFoundError(f"missing score metadata: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("dataset") != dataset:
            raise ValueError(f"dataset mismatch in {sidecar}")
        # Older score builders recorded the range only in the filename.  Make
        # that legacy artifact explicit during merge, while still rejecting a
        # contradictory range when one is present.
        if metadata.get("query_start") is None and metadata.get("query_end") is None:
            metadata["query_start"] = lo
            metadata["query_end"] = hi
            sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif (metadata.get("query_start"), metadata.get("query_end")) != (lo, hi):
            raise ValueError(f"metadata range mismatch in {sidecar}")
        with np.load(npz, allow_pickle=False) as arrays:
            if "official_distances" not in arrays:
                raise ValueError(f"score sidecar lacks official_distances: {npz}")
            if arrays["official_distances"].shape[0] != hi - lo:
                raise ValueError(
                    f"score row count mismatch in {npz}: "
                    f"{arrays['official_distances'].shape[0]} != {hi - lo}"
                )
            for method in metadata.get("methods", []):
                key = f"selected_ranks_{method}"
                if key in arrays and arrays[key].shape[0] != hi - lo:
                    raise ValueError(f"score row count mismatch in {key}: {npz}")
        rows.append((lo, hi, npz, sidecar))
    return rows


def main() -> None:
    args = _args()
    if args.total_query_windows < 1:
        raise ValueError("total-query-windows must be positive")
    parts = [Path(value).resolve() for value in args.part_root]
    shards = []
    for part in parts:
        shards.extend(_validate_part(part, args.dataset))
    shards.sort(key=lambda row: (row[0], row[1], str(row[2])))
    expected = 0
    seen: set[tuple[int, int]] = set()
    for lo, hi, _npz, _json in shards:
        if (lo, hi) in seen:
            raise ValueError(f"duplicate shard range: {lo}:{hi}")
        seen.add((lo, hi))
        if lo != expected:
            raise ValueError(f"coverage gap/overlap before {lo}; expected {expected}")
        expected = hi
    if expected != args.total_query_windows:
        raise ValueError(f"coverage ends at {expected}, expected {args.total_query_windows}")

    output = Path(args.output_root).resolve()
    score_dir = output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for lo, hi, npz, metadata in shards:
        target_npz = score_dir / npz.name
        target_json = score_dir / metadata.name
        for source, target in ((npz, target_npz), (metadata, target_json)):
            if target.exists():
                if not args.replace:
                    # Existing files are accepted only when their size and
                    # bytes are identical; this makes reruns idempotent.
                    if target.stat().st_size != source.stat().st_size:
                        raise FileExistsError(f"conflicting output: {target}")
                    continue
                target.unlink()
            try:
                os.link(source, target)
                mode = "hardlink"
            except OSError:
                shutil.copy2(source, target)
                mode = "copy"
            copied.append({"source": str(source), "target": str(target), "mode": mode})

    first_meta = json.loads(shards[0][3].read_text(encoding="utf-8"))
    manifest = {
        "dataset": args.dataset,
        "seq_len": int(first_meta.get("seq_len", 512)),
        "pred_len": int(first_meta.get("pred_len", 64)),
        "pool_k": int(first_meta.get("candidate_count", 20)),
        "query_start": 0,
        "query_end": int(args.total_query_windows),
        "query_windows": int(args.total_query_windows),
        "scores": str(score_dir),
        "score_directory_ready": True,
        "runner": {"status": "not_run"},
        "merged_from": [str(path) for path in parts],
        "shard_count": len(shards),
        "coverage_checked": True,
        "leakage_ok": all(
            json.loads(metadata.read_text(encoding="utf-8")).get("leakage_ok") is True
            for _lo, _hi, _npz, metadata in shards
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output / "manifest.json"),
                      "shards": len(shards), "query_windows": args.total_query_windows,
                      "coverage_checked": True, "leakage_ok": manifest["leakage_ok"],
                      "copied": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
