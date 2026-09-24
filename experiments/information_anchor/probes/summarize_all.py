"""汇总全变量语义 probe，避免把异质语义压成单一主指标。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _macro_by_prefix(values: Dict[str, float]) -> Dict[str, float]:
    """按变量前缀计算语义均值。"""
    groups: Dict[str, List[float]] = {}
    for name, value in values.items():
        prefix = name.split(":", 1)[0]
        groups.setdefault(prefix, []).append(float(value))
    return {
        key: float(np.nanmean(vals))
        for key, vals in groups.items()
        if np.isfinite(vals).any()
    }


def _macro_by_semantic(values: Dict[str, float]) -> Dict[str, float]:
    """按语义名称计算跨变量均值。"""
    groups: Dict[str, List[float]] = {}
    for name, value in values.items():
        suffix = name.split(":", 1)[-1]
        groups.setdefault(suffix, []).append(float(value))
    return {
        key: float(np.nanmean(vals))
        for key, vals in groups.items()
        if np.isfinite(vals).any()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 all-variable semantic probe")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        values = data.get("semantic_mean_test_r2", {})
        if not values:
            continue
        variable_macro = _macro_by_prefix(values)
        semantic_macro = _macro_by_semantic(values)
        row = {
            "model": data.get("model"),
            "dataset": data.get("dataset"),
            "semantic_scope": data.get("semantic_scope"),
            "overall_test_mean_r2": data.get("overall_test_mean_r2"),
            "fraction_test_r2_above_zero": data.get("fraction_test_r2_above_zero"),
            "valid_r2_fraction": data.get("valid_r2_fraction"),
            "mi_vs_probe_mean_spearman": data.get("mi_vs_probe_mean_spearman"),
            "raw_history_pca_baseline_mean_r2": data.get("raw_history_pca_baseline_mean_r2"),
            "raw_recent_patch_pca_baseline_mean_r2": data.get("raw_recent_patch_pca_baseline_mean_r2"),
            "ot_macro_r2": variable_macro.get("OT"),
            "macro_variable_r2": sum(variable_macro.values()) / len(variable_macro),
            "macro_semantic_r2": sum(semantic_macro.values()) / len(semantic_macro),
            "summary_path": str(path),
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    csv_path = args.output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
