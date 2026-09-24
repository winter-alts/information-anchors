from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.information_anchor.adapters.moirai2 import Moirai2Adapter
from experiments.information_anchor.artifacts import write_json
from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
)
from experiments.information_anchor.estimators.nulls import temporal_circular_shift_offsets
from experiments.information_anchor.interventions.common import (
    UnitSelection,
    condition_summary,
    exact_sign_flip_p,
    hierarchical_bootstrap_ci,
    select_units,
)
from experiments.information_anchor.metrics import (
    fit_train_standard_scaler,
    forecast_change_metrics,
    per_origin_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moirai2 frozen-backbone activation replacement on held-out forecasts"
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--donor-shifts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--device",
        help="Override config.model.device (for example cuda:0 after CUDA_VISIBLE_DEVICES remapping).",
    )
    parser.add_argument("--output-root", default="results/information_anchor_interventions")
    return parser.parse_args()


def _build_forecaster(adapter: Moirai2Adapter):
    from uni2ts.model.moirai2 import Moirai2Forecast

    config = adapter.config
    forecaster = Moirai2Forecast(
        prediction_length=config.data.pred_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=config.data.seq_len,
        module=adapter.model,
    )
    return forecaster.to(adapter.device).eval()


def _median_index(module: Any) -> int:
    return int(np.argmin(np.abs(np.asarray(module.quantile_levels, dtype=np.float64) - 0.5)))


def _forecast_batch(
    forecaster: Any,
    histories: np.ndarray,
    pred_len: int,
    device: torch.device,
) -> np.ndarray:
    past_target = torch.from_numpy(histories).to(device=device, dtype=torch.float32).unsqueeze(-1)
    prediction = forecaster(
        past_target=past_target,
        past_observed_target=torch.ones_like(past_target, dtype=torch.bool),
        past_is_pad=torch.zeros(past_target.shape[:2], device=device, dtype=torch.bool),
    )
    median = _median_index(forecaster.module)
    if prediction.ndim == 4:
        point = prediction[:, median, :pred_len, 0]
    elif prediction.ndim == 3:
        point = prediction[:, median, :pred_len]
    else:
        raise ValueError(f"Unexpected Moirai2 forecast shape: {tuple(prediction.shape)}")
    return point.detach().float().cpu().numpy()


def _capture_layer_outputs(
    forecaster: Any,
    histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(output.detach().clone())
        return output

    handle = forecaster.module.encoder.layers[layer].register_forward_hook(capture_hook)
    try:
        _forecast_batch(forecaster, histories, pred_len, device)
    finally:
        handle.remove()
    if not captures:
        raise RuntimeError("Moirai2 donor forward produced no layer captures.")
    return captures


def _patched_forecast_batch(
    forecaster: Any,
    histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
    patch: int,
    donor_outputs: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    call_index = 0
    perturbation_rms: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        donor = donor_outputs[call_index].to(device=output.device, dtype=output.dtype)
        if donor.shape != output.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {output.shape}")
        changed = output.clone()
        delta = donor[..., patch, :] - output[..., patch, :]
        changed[..., patch, :] = donor[..., patch, :]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=-1))
        if rms.ndim > 1:
            rms = rms.mean(dim=tuple(range(1, rms.ndim)))
        perturbation_rms.append(rms.detach().cpu().numpy())
        call_index += 1
        return changed

    handle = forecaster.module.encoder.layers[layer].register_forward_hook(replacement_hook)
    try:
        forecast = _forecast_batch(forecaster, histories, pred_len, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(perturbation_rms), axis=0).astype(np.float32)


def _noop_clone_forecast(
    forecaster: Any,
    histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        return output.clone()

    handle = forecaster.module.encoder.layers[layer].register_forward_hook(clone_hook)
    try:
        return _forecast_batch(forecaster, histories, pred_len, device)
    finally:
        handle.remove()


def main() -> None:
    args = _parse_args()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if args.device:
        config = replace(config, model=replace(config.model, device=args.device))
    if config.model.name != "Moirai2":
        raise ValueError("This intervention runner supports Moirai2 only.")
    reference = np.load(reference_dir / "mi_results.npz")
    mi_z = np.asarray(reference["mi_z"], dtype=np.float32)
    selections: list[UnitSelection] = select_units(
        mi_z,
        np.asarray(reference["layer_mi_z"]),
        np.asarray(reference["patch_mi_z"]),
    )

    test_config = replace(config, data=replace(config.data, split="test"))
    test_origins = build_forecast_origins(test_config.data)
    sample_indices = np.linspace(0, len(test_origins) - 1, min(args.max_samples, len(test_origins)))
    sample_indices = np.unique(np.rint(sample_indices).astype(np.int64))
    sampled_origins = test_origins[sample_indices]
    shifts = temporal_circular_shift_offsets(
        sampled_origins,
        args.donor_shifts,
        min_temporal_separation=config.data.seq_len + config.data.pred_len,
        seed=config.mi.seed + 80_000,
    )
    frame = load_benchmark_frame(config.data, offline=config.runtime.offline)
    values = frame[config.data.target].to_numpy(dtype=np.float32)
    scaler = fit_train_standard_scaler(values, config.data.train_end)
    windows = make_windows(values, sampled_origins, config.data.seq_len, config.data.pred_len)
    histories, futures = windows.history_raw, windows.future_raw

    adapter = Moirai2Adapter(config)
    forecaster = _build_forecaster(adapter)
    baseline_parts = []
    with torch.no_grad():
        for start in range(0, len(histories), args.batch_size):
            stop = min(len(histories), start + args.batch_size)
            baseline_parts.append(
                _forecast_batch(forecaster, histories[start:stop], config.data.pred_len, adapter.device)
            )
    baseline = np.concatenate(baseline_parts)
    baseline_mse, baseline_mae = per_origin_metrics(baseline, futures, scaler)

    check_stop = min(args.batch_size, len(histories))
    with torch.no_grad():
        noop = _noop_clone_forecast(
            forecaster,
            histories[:check_stop],
            config.data.pred_len,
            adapter.device,
            selections[0].layer,
        )
    noop_max_abs_difference = float(np.max(np.abs(noop - baseline[:check_stop])))
    if noop_max_abs_difference > 1e-5:
        raise RuntimeError(f"Clone-only hook changed forecasts by {noop_max_abs_difference:.6g}.")

    metric_names = (
        "delta_mse",
        "delta_mae",
        "forecast_change_mse",
        "forecast_change_mae",
        "perturbation_rms",
    )
    metrics = {unit.label: {name: [] for name in metric_names} for unit in selections}
    grouped: dict[int, list[UnitSelection]] = {}
    for unit in selections:
        grouped.setdefault(unit.layer, []).append(unit)

    with torch.no_grad():
        for shift_index, shift in enumerate(shifts):
            donor_histories = np.roll(histories, int(shift), axis=0)
            shift_predictions = {unit.label: [] for unit in selections}
            shift_rms = {unit.label: [] for unit in selections}
            for start in range(0, len(histories), args.batch_size):
                stop = min(len(histories), start + args.batch_size)
                for layer, layer_units in grouped.items():
                    donor_outputs = _capture_layer_outputs(
                        forecaster,
                        donor_histories[start:stop],
                        config.data.pred_len,
                        adapter.device,
                        layer,
                    )
                    for unit in layer_units:
                        predictions, rms = _patched_forecast_batch(
                            forecaster,
                            histories[start:stop],
                            config.data.pred_len,
                            adapter.device,
                            unit.layer,
                            unit.patch,
                            donor_outputs,
                        )
                        shift_predictions[unit.label].append(predictions)
                        shift_rms[unit.label].append(rms)
            for unit in selections:
                prediction = np.concatenate(shift_predictions[unit.label])
                rms = np.concatenate(shift_rms[unit.label])
                patched_mse, patched_mae = per_origin_metrics(prediction, futures, scaler)
                change_mse, change_mae = forecast_change_metrics(prediction, baseline, scaler)
                metrics[unit.label]["delta_mse"].append(patched_mse - baseline_mse)
                metrics[unit.label]["delta_mae"].append(patched_mae - baseline_mae)
                metrics[unit.label]["forecast_change_mse"].append(change_mse)
                metrics[unit.label]["forecast_change_mae"].append(change_mae)
                metrics[unit.label]["perturbation_rms"].append(rms)
            print(f"donor_shift={shift_index + 1}/{len(shifts)} offset={int(shift)} complete", flush=True)

    stacked = {
        label: {name: np.stack(arrays) for name, arrays in condition.items()}
        for label, condition in metrics.items()
    }
    records = []
    for index, unit in enumerate(selections):
        condition = stacked[unit.label]
        records.append(
            condition_summary(
                unit.label,
                unit,
                condition["delta_mse"],
                condition["delta_mae"],
                condition["forecast_change_mse"],
                condition["forecast_change_mae"],
                condition["perturbation_rms"],
                args.bootstrap_repetitions,
                config.mi.seed + 90_000 + index * 10,
            )
        )

    primary_difference = stacked["mi_top_cell"]["delta_mse"] - stacked["low_mi_same_layer"]["delta_mse"]
    contrast_ci = hierarchical_bootstrap_ci(
        primary_difference, args.bootstrap_repetitions, config.mi.seed + 95_000
    )
    contrast_p_two, contrast_p_greater = exact_sign_flip_p(primary_difference.mean(axis=1))
    lookup = {record["condition"]: record for record in records}
    primary_contrast = {
        "contrast": "mi_top_cell_minus_low_mi_same_layer",
        "mean_delta_mse_difference": float(primary_difference.mean()),
        "ci95_lower": contrast_ci[0],
        "ci95_upper": contrast_ci[1],
        "positive_shift_fraction": float(np.mean(primary_difference.mean(axis=1) > 0)),
        "sign_flip_p_two_sided": contrast_p_two,
        "sign_flip_p_greater": contrast_p_greater,
        "top_condition": lookup["mi_top_cell"],
        "matched_low_condition": lookup["low_mi_same_layer"],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"moirai2_l{config.data.seq_len}_p{config.data.pred_len}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {
        "origins": sampled_origins,
        "donor_shift_offsets": shifts,
        "baseline_forecast": baseline,
        "future_raw": futures,
        "tslib_mean": scaler.mean.astype(np.float32),
        "tslib_scale": scaler.scale.astype(np.float32),
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
    }
    for label, condition in stacked.items():
        for name, array in condition.items():
            arrays[f"{label}__{name}"] = array
    np.savez_compressed(output_dir / "intervention_results.npz", **arrays)
    payload = {
        "status": "complete",
        "model": config.model.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "reference_run": str(reference_dir),
        "split": "test",
        "metric_protocol": "TSLib-style train-split StandardScaler; MSE/MAE averaged over the forecast horizon.",
        "selection_protocol": "All intervention units were selected from discovery MI before held-out forecast evaluation.",
        "intervention": "Replace one post-block history token with the same token from a temporally distant donor during every deterministic quantile-recursion call; all weights remain frozen.",
        "point_forecast": "Checkpoint median (0.5) quantile; no stochastic sampling.",
        "donor_protocol": "Every paired origin is separated by at least L+P; identical donor shifts are used for all conditions.",
        "sample_count": int(len(sampled_origins)),
        "origins_hash": origins_hash(sampled_origins),
        "donor_shift_count": int(len(shifts)),
        "donor_shift_offsets": shifts.tolist(),
        "required_temporal_separation": config.data.seq_len + config.data.pred_len,
        "baseline_mse": float(baseline_mse.mean()),
        "baseline_mae": float(baseline_mae.mean()),
        "noop_clone_max_abs_forecast_difference": noop_max_abs_difference,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "records": records,
        "primary_contrast": primary_contrast,
    }
    write_json(output_dir / "summary.json", payload)
    with (output_dir / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"intervention_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
