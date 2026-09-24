#!/usr/bin/env python3
"""Generate the preregistered six-model by seven-dataset V6 matrix."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "configs/information_anchor/v6_registered"
DATA_ROOT = Path(os.environ.get("INFORMATION_ANCHOR_DATA_ROOT", "."))

MODEL_BASES = {
    "chronos2": ("chronos2_ettm1_m_l512_p96_global_full.json", "cuda:2"),
    "moirai2": ("moirai2_ettm1_m_l512_p96_global_full.json", "cuda:3"),
    "toto2": ("toto2_ettm1_m_l512_p96_global_full.json", "cuda:4"),
    "timesfm25": ("timesfm25_ettm1_m_l512_p96_global_full.json", "cuda:5"),
    "chronos_bolt": ("chronos_bolt_ettm1_m_l512_p96_global_full.json", "cuda:6"),
    "ttm": ("ttm_ettm1_m_l512_p96_global_full.json", "cuda:7"),
}

DATASETS = {
    "etth1": {
        "dataset": "ETTh1", "target": "OT", "seq_len": 512,
        "train_end": 8640, "validation_end": 11520, "test_end": 14400,
    },
    "etth2": {
        "dataset": "ETTh2", "target": "OT", "seq_len": 512,
        "train_end": 8640, "validation_end": 11520, "test_end": 14400,
    },
    "ettm1": {
        "dataset": "ETTm1", "target": "OT", "seq_len": 512,
        "train_end": 34560, "validation_end": 46080, "test_end": 57600,
    },
    "ettm2": {
        "dataset": "ETTm2", "target": "OT", "seq_len": 512,
        "train_end": 34560, "validation_end": 46080, "test_end": 57600,
    },
    "weather": {
        "dataset": "weather", "target": "T (degC)", "seq_len": 384,
        "train_end": 36887, "validation_end": 42157, "test_end": 52696,
    },
    "electricity16": {
        "dataset": "electricity", "target": "0", "seq_len": 512,
        "train_end": 18412, "validation_end": 21042, "test_end": 26304,
        "target_columns": [
            "0", "1", "24", "46", "69", "92", "115", "137",
            "160", "183", "205", "228", "251", "274", "296", "319",
        ],
    },
    "traffic16": {
        "dataset": "traffic", "target": "0", "seq_len": 512,
        "train_end": 12280, "validation_end": 14034, "test_end": 17544,
        "target_columns": [
            "0", "1", "62", "124", "185", "246", "308", "369",
            "431", "492", "553", "615", "676", "737", "799", "860",
        ],
    },
}

ARCHITECTURE_REGISTRY = {
    "Chronos2": {
        "compute_path": "encoder-only",
        "cross_channel_interaction": "joint-multivariate",
        "attention_path": "history plus masked future-query encoder",
        "forecast_readout": "direct multi-patch quantile head from future-query states",
        "analyzed_history_unit": "post-encoder history patch",
    },
    "Moirai2": {
        "compute_path": "causal-decoder-style",
        "cross_channel_interaction": "packed-joint-multivariate",
        "attention_path": "packed causal attention",
        "forecast_readout": "multi-token quantile head over prediction tokens",
        "analyzed_history_unit": "post-block packed history patch",
    },
    "Toto2": {
        "compute_path": "decoder-only",
        "cross_channel_interaction": "joint-time-variate",
        "attention_path": "alternating causal time and variate attention",
        "forecast_readout": "quantile head",
        "analyzed_history_unit": "post-decoder history patch",
    },
    "TimesFM2.5": {
        "compute_path": "decoder-only",
        "cross_channel_interaction": "channel-independent",
        "attention_path": "causal prefill followed by autoregressive output patches",
        "forecast_readout": "point and quantile patch head",
        "analyzed_history_unit": "prefill history patch",
    },
    "ChronosBolt": {
        "compute_path": "encoder-decoder",
        "cross_channel_interaction": "channel-independent",
        "attention_path": "T5 encoder self-attention and decoder cross-attention",
        "forecast_readout": "direct multi-patch quantile decoder",
        "analyzed_history_unit": "encoder history patch read by decoder cross-attention",
    },
    "TTM": {
        "compute_path": "mixer",
        "cross_channel_interaction": "shared common-channel mixer",
        "attention_path": "adaptive patch mixer stages",
        "forecast_readout": "direct multi-horizon forecast decoder",
        "analyzed_history_unit": "common-grid encoder and forecast-decoder patch stage",
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _data_payload(dataset_key: str, model_key: str) -> dict:
    registered = DATASETS[dataset_key]
    seq_len = 512 if dataset_key == "weather" and model_key == "ttm" else registered["seq_len"]
    features = "custom" if "target_columns" in registered else "M"
    return {
        "dataset": registered["dataset"],
        "source": "thuml/Time-Series-Library",
        "target": registered["target"],
        "seq_len": seq_len,
        "pred_len": 96,
        "origin_seq_len": seq_len,
        "origin_pred_len": 96,
        "split": "discovery",
        "origin_stride": 4,
        "max_origins": 1024,
        "strict_guard": True,
        "train_end": registered["train_end"],
        "validation_end": registered["validation_end"],
        "test_end": registered["test_end"],
        "discovery_end": registered["train_end"],
        "local_path": str(DATA_ROOT / f"{registered['dataset']}.csv"),
        "features": features,
        "target_columns": registered.get("target_columns", []),
    }


def main() -> None:
    base_root = ROOT / "configs/information_anchor/v5_global"
    generated: list[dict] = []
    for model_key, (filename, device) in MODEL_BASES.items():
        base = _read(base_root / filename)
        for dataset_key in DATASETS:
            config = deepcopy(base)
            stem = f"{model_key}_{dataset_key}_v6"
            config["experiment_name"] = stem
            config["protocol_version"] = "information-anchor-v6-multidataset"
            config["model"]["device"] = device
            config["data"] = _data_payload(dataset_key, model_key)
            config["runtime"]["output_root"] = "results/information_anchor_v6_reference"
            config["runtime"]["cache_root"] = "results/information_anchor_cache_v5_global_reference"
            output = OUTPUT_ROOT / f"{stem}.json"
            _write(output, config)
            generated.append(
                {
                    "config": str(output.relative_to(ROOT)),
                    "model": config["model"]["name"],
                    "dataset_key": dataset_key,
                    "dataset": config["data"]["dataset"],
                    "target_channel": config["data"]["target"],
                    "channels": len(config["data"]["target_columns"]) or "all",
                    "seq_len": config["data"]["seq_len"],
                    "pred_len": config["data"]["pred_len"],
                    "device": device,
                }
            )
    _write(
        OUTPUT_ROOT / "registry.json",
        {
            "protocol_version": "information-anchor-v6-multidataset",
            "semantic_protocol": "future-semantics-v6.1-12-nonredundant",
            "models": ARCHITECTURE_REGISTRY,
            "datasets": DATASETS,
            "matrix": generated,
        },
    )
    print(f"generated={len(generated)}")


if __name__ == "__main__":
    main()
