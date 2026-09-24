from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.information_anchor.adapters.chronos2 import Chronos2Adapter
from experiments.information_anchor.adapters.chronos_bolt import ChronosBoltAdapter
from experiments.information_anchor.adapters.moirai2 import Moirai2Adapter
from experiments.information_anchor.adapters.timesfm import TimesFMAdapter
from experiments.information_anchor.adapters.timesfm25 import (
    TimesFM25Adapter,
    replace_timesfm25_history_patches,
)
from experiments.information_anchor.adapters.timesfm3 import (
    TimesFM3Adapter,
    replace_timesfm3_history_patches,
)
from experiments.information_anchor.adapters.toto2 import Toto2Adapter
from experiments.information_anchor.adapters.ttm import TTMAdapter
from experiments.information_anchor.artifacts import write_json
from experiments.information_anchor.config import load_config
from experiments.information_anchor.data import (
    build_forecast_origins,
    load_benchmark_frame,
    make_windows,
    origins_hash,
    select_value_columns,
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
        description="Multivariate frozen-backbone activation replacement with TSLib-style metrics."
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--donor-shifts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="加入同层 recent/random patch 控制，仅用于附录边界分析。",
    )
    parser.add_argument(
        "--random-control-repetitions",
        type=int,
        default=3,
        help="同层 random 控制的确定性重复次数。",
    )
    parser.add_argument("--device", help="Override config.model.device after CUDA_VISIBLE_DEVICES remapping.")
    parser.add_argument("--output-root", default="results/information_anchor_multivariate_interventions")
    return parser.parse_args()


def _as_3d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected [B,T] or [B,T,C], got {values.shape}.")
    return values


def _paired_contrast_summary(
    high: dict[str, np.ndarray],
    low: dict[str, np.ndarray],
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired high-minus-low effects for one explicit metric scope."""
    fields = (
        "delta_mse",
        "delta_mae",
        "forecast_change_mse",
        "forecast_change_mae",
    )
    output: dict[str, Any] = {"contrast": "mi_top_cell_minus_low_mi_same_layer"}
    for offset, field in enumerate(fields):
        difference = high[field] - low[field]
        lower, upper = hierarchical_bootstrap_ci(
            difference,
            bootstrap_repetitions,
            seed + offset,
        )
        p_two, p_greater = exact_sign_flip_p(difference.mean(axis=1))
        output[f"mean_{field}_difference"] = float(difference.mean())
        output[f"{field}_ci95_lower"] = lower
        output[f"{field}_ci95_upper"] = upper
        output[f"{field}_positive_shift_fraction"] = float(
            np.mean(difference.mean(axis=1) > 0)
        )
        output[f"{field}_sign_flip_p_two_sided"] = p_two
        output[f"{field}_sign_flip_p_greater"] = p_greater

    # Backward-compatible aliases always refer to the primary delta-MSE contrast.
    output["ci95_lower"] = output["delta_mse_ci95_lower"]
    output["ci95_upper"] = output["delta_mse_ci95_upper"]
    output["positive_shift_fraction"] = output["delta_mse_positive_shift_fraction"]
    output["sign_flip_p_two_sided"] = output["delta_mse_sign_flip_p_two_sided"]
    output["sign_flip_p_greater"] = output["delta_mse_sign_flip_p_greater"]
    return output


# ---------------- TimesFM ----------------


def _timesfm_hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _timesfm_replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _timesfm_forecast_2d(outer_model: Any, histories: np.ndarray, pred_len: int) -> np.ndarray:
    masks = np.zeros_like(histories, dtype=bool)
    point, _ = outer_model.compiled_decode(pred_len, histories, masks)
    return np.asarray(point, dtype=np.float32)


def _timesfm_forecast(adapter: TimesFMAdapter, histories: np.ndarray, pred_len: int) -> np.ndarray:
    histories = _as_3d(histories)
    outer = adapter.model.model
    parts = []
    for channel in range(histories.shape[-1]):
        parts.append(_timesfm_forecast_2d(outer, histories[:, :, channel], pred_len))
    return np.stack(parts, axis=-1).astype(np.float32)


def _timesfm_capture_channel(
    outer_model: Any,
    histories_2d: np.ndarray,
    pred_len: int,
    layer: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(_timesfm_hidden_from_output(output).detach().clone())
        return output

    handle = outer_model.model.stacked_xf[layer].register_forward_hook(capture_hook)
    try:
        _timesfm_forecast_2d(outer_model, histories_2d, pred_len)
    finally:
        handle.remove()
    if not captures:
        raise RuntimeError("TimesFM layer hook did not capture hidden states.")
    return captures


def _timesfm_patched_channel(
    outer_model: Any,
    histories_2d: np.ndarray,
    pred_len: int,
    layer: int,
    patch: int,
    donor_outputs: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    call_index = 0
    rms_parts: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _timesfm_hidden_from_output(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        if donor.shape != original.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {original.shape}")
        changed = original.clone()
        delta = donor[:, patch] - original[:, patch]
        changed[:, patch] = donor[:, patch]
        rms_parts.append(torch.sqrt(torch.mean(delta.float().square(), dim=-1)).detach().cpu().numpy())
        call_index += 1
        return _timesfm_replace_hidden(output, changed)

    handle = outer_model.model.stacked_xf[layer].register_forward_hook(replacement_hook)
    try:
        forecast = _timesfm_forecast_2d(outer_model, histories_2d, pred_len)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor forwards but captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts, axis=0), axis=0).astype(np.float32)


def _timesfm_patched_channel_set(
    outer_model: Any,
    histories_2d: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """在 TimesFM 同一层一次替换多个历史 patch。"""
    call_index = 0
    rms_parts: list[np.ndarray] = []
    patch_indices = torch.tensor(patches, dtype=torch.long)

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _timesfm_hidden_from_output(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        indices = patch_indices.to(original.device)
        if donor.shape != original.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {original.shape}")
        if not len(indices) or int(indices.min()) < 0 or int(indices.max()) >= original.shape[1]:
            raise ValueError(
                f"patches={patches} outside TimesFM sequence length={original.shape[1]}."
            )
        changed = original.clone()
        delta = donor[:, indices] - original[:, indices]
        changed[:, indices] = donor[:, indices]
        rms_parts.append(
            torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2))).detach().cpu().numpy()
        )
        call_index += 1
        return _timesfm_replace_hidden(output, changed)

    handle = outer_model.model.stacked_xf[layer].register_forward_hook(replacement_hook)
    try:
        forecast = _timesfm_forecast_2d(outer_model, histories_2d, pred_len)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor forwards but captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts, axis=0), axis=0).astype(np.float32)


def _timesfm_patched_forecast(
    adapter: TimesFMAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    outer = adapter.model.model
    pred_parts = []
    rms_parts = []
    for channel in range(histories.shape[-1]):
        donor_outputs = _timesfm_capture_channel(
            outer,
            donor_histories[:, :, channel],
            pred_len,
            selection.layer,
        )
        pred, rms = _timesfm_patched_channel(
            outer,
            histories[:, :, channel],
            pred_len,
            selection.layer,
            selection.patch,
            donor_outputs,
        )
        pred_parts.append(pred)
        rms_parts.append(rms)
    return np.stack(pred_parts, axis=-1).astype(np.float32), np.mean(np.stack(rms_parts, axis=0), axis=0).astype(np.float32)


def _timesfm_patched_forecast_set(
    adapter: TimesFMAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs_by_channel: list[list[torch.Tensor]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    outer = adapter.model.model
    pred_parts = []
    rms_parts = []
    if donor_outputs_by_channel is not None and len(donor_outputs_by_channel) != histories.shape[-1]:
        raise ValueError("TimesFM cached donor states do not match the channel count.")
    for channel in range(histories.shape[-1]):
        donor_outputs = (
            donor_outputs_by_channel[channel]
            if donor_outputs_by_channel is not None
            else _timesfm_capture_channel(
                outer,
                donor_histories[:, :, channel],
                pred_len,
                layer,
            )
        )
        pred, rms = _timesfm_patched_channel_set(
            outer,
            histories[:, :, channel],
            pred_len,
            layer,
            patches,
            donor_outputs,
        )
        pred_parts.append(pred)
        rms_parts.append(rms)
    return (
        np.stack(pred_parts, axis=-1).astype(np.float32),
        np.mean(np.stack(rms_parts, axis=0), axis=0).astype(np.float32),
    )


def _timesfm_noop(adapter: TimesFMAdapter, histories: np.ndarray, pred_len: int, layer: int) -> np.ndarray:
    outer = adapter.model.model

    def clone_hook(_module, _inputs, output):
        return _timesfm_replace_hidden(output, _timesfm_hidden_from_output(output).clone())

    histories = _as_3d(histories)
    outputs = []
    for channel in range(histories.shape[-1]):
        handle = outer.model.stacked_xf[layer].register_forward_hook(clone_hook)
        try:
            outputs.append(_timesfm_forecast_2d(outer, histories[:, :, channel], pred_len))
        finally:
            handle.remove()
    return np.stack(outputs, axis=-1).astype(np.float32)


# ---------------- TimesFM 2.5 ----------------


def _timesfm25_capture(
    adapter: TimesFM25Adapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> torch.Tensor:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        if not captures:
            hidden = output[0] if isinstance(output, tuple) else output
            captures.append(hidden.detach().clone())
        return output

    handle = adapter.model.stacked_xf[layer].register_forward_hook(capture_hook)
    try:
        adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"Expected one TimesFM 2.5 prefill capture, got {len(captures)}.")
    return captures[0]


def _timesfm25_patched_forecast_set(
    adapter: TimesFM25Adapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_state: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    if histories.shape != donor_histories.shape:
        raise ValueError("TimesFM 2.5 recipient and donor histories must have identical shape.")
    if donor_state is None:
        donor_state = _timesfm25_capture(adapter, donor_histories, pred_len, layer)
    replaced_prefill = 0
    rms_parts: list[torch.Tensor] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal replaced_prefill
        # A horizon above 128 triggers later autoregressive calls. Only the
        # first call contains the registered history-patch coordinates.
        if replaced_prefill:
            return output
        replaced, rms = replace_timesfm25_history_patches(output, donor_state, patches)
        rms_parts.append(rms.detach().cpu())
        replaced_prefill += 1
        return replaced

    handle = adapter.model.stacked_xf[layer].register_forward_hook(replacement_hook)
    try:
        forecast = adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if replaced_prefill != 1 or len(rms_parts) != 1:
        raise RuntimeError("TimesFM 2.5 history prefill was not replaced exactly once.")
    batch, _length, channels = histories.shape
    rms = rms_parts[0].numpy().reshape(batch, channels).mean(axis=1).astype(np.float32)
    return forecast, rms


def _timesfm25_patched_forecast(
    adapter: TimesFM25Adapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    return _timesfm25_patched_forecast_set(
        adapter,
        histories,
        donor_histories,
        pred_len,
        selection.layer,
        (selection.patch,),
    )


def _timesfm25_noop(
    adapter: TimesFM25Adapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = adapter.model.stacked_xf[layer].register_forward_hook(clone_hook)
    try:
        return adapter.forecast(histories, pred_len)
    finally:
        handle.remove()


# ---------------- TimesFM 3 ----------------


def _timesfm3_capture(
    adapter: TimesFM3Adapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> torch.Tensor:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        if not captures:
            hidden = output[0] if isinstance(output, tuple) else output
            captures.append(hidden.detach().clone())
        return output

    handle = adapter.model.transformer_stack.layers[layer].register_forward_hook(
        capture_hook
    )
    try:
        adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"Expected one TimesFM 3 prefill capture, got {len(captures)}.")
    return captures[0]


def _timesfm3_patched_forecast_set(
    adapter: TimesFM3Adapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_state: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    if histories.shape != donor_histories.shape:
        raise ValueError("TimesFM 3 recipient and donor histories must have identical shape.")
    if donor_state is None:
        donor_state = _timesfm3_capture(adapter, donor_histories, pred_len, layer)
    replaced_prefill = 0
    rms_parts: list[torch.Tensor] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal replaced_prefill
        if replaced_prefill:
            return output
        replaced, rms = replace_timesfm3_history_patches(output, donor_state, patches)
        rms_parts.append(rms.detach().cpu())
        replaced_prefill += 1
        return replaced

    handle = adapter.model.transformer_stack.layers[layer].register_forward_hook(
        replacement_hook
    )
    try:
        forecast = adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if replaced_prefill != 1 or len(rms_parts) != 1:
        raise RuntimeError("TimesFM 3 history prefill was not replaced exactly once.")
    return forecast, rms_parts[0].numpy().astype(np.float32)


def _timesfm3_patched_forecast(
    adapter: TimesFM3Adapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    return _timesfm3_patched_forecast_set(
        adapter,
        histories,
        donor_histories,
        pred_len,
        selection.layer,
        (selection.patch,),
    )


def _timesfm3_noop(
    adapter: TimesFM3Adapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = adapter.model.transformer_stack.layers[layer].register_forward_hook(
        clone_hook
    )
    try:
        return adapter.forecast(histories, pred_len)
    finally:
        handle.remove()


# ---------------- Chronos2 ----------------


def _chronos_median_index(model: Any) -> int:
    quantiles = model.quantiles.detach().float().cpu().numpy()
    return int(np.argmin(np.abs(quantiles - 0.5)))


def _chronos_context(histories: np.ndarray, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    histories = _as_3d(histories)
    batch_size, seq_len, n_channels = histories.shape
    context = torch.from_numpy(histories).to(device=device, dtype=dtype).permute(0, 2, 1).reshape(batch_size * n_channels, seq_len)
    group_ids = torch.arange(batch_size, dtype=torch.long, device=device).repeat_interleave(n_channels)
    return context, group_ids, batch_size, n_channels


def _chronos_forecast(
    model: Any,
    histories: np.ndarray,
    pred_len: int,
    num_output_patches: int,
    device: torch.device,
) -> np.ndarray:
    context, group_ids, batch_size, n_channels = _chronos_context(histories, device, model.dtype)
    outputs = model(
        context=context,
        context_mask=torch.ones_like(context),
        group_ids=group_ids,
        future_covariates=None,
        future_covariates_mask=None,
        num_output_patches=num_output_patches,
        future_target=None,
        future_target_mask=None,
        output_attentions=False,
    )
    pred = outputs.quantile_preds[:, _chronos_median_index(model), :pred_len]
    return pred.detach().float().cpu().numpy().reshape(batch_size, n_channels, pred_len).transpose(0, 2, 1).astype(np.float32)


def _chronos_block_hidden(output: Any) -> torch.Tensor:
    if hasattr(output, "hidden_states"):
        return output.hidden_states
    return output[0]


def _chronos_replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if hasattr(output, "hidden_states"):
        return type(output)(
            hidden_states=hidden,
            time_self_attn_weights=output.time_self_attn_weights,
            group_self_attn_weights=output.group_self_attn_weights,
        )
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _chronos_capture(
    model: Any,
    histories: np.ndarray,
    pred_len: int,
    num_output_patches: int,
    device: torch.device,
    layer: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(_chronos_block_hidden(output).detach().clone())
        return output

    handle = model.encoder.block[layer].register_forward_hook(capture_hook)
    try:
        _chronos_forecast(model, histories, pred_len, num_output_patches, device)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"Expected one Chronos2 donor forward, captured {len(captures)}.")
    return captures


def _chronos_patched_forecast(
    model: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    num_output_patches: int,
    device: torch.device,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    histories_3d = _as_3d(histories)
    batch_size, _seq_len, n_channels = histories_3d.shape
    donor_outputs = _chronos_capture(model, donor_histories, pred_len, num_output_patches, device, selection.layer)
    call_index = 0
    rms_parts: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _chronos_block_hidden(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        changed = original.clone()
        delta = donor[:, selection.patch] - original[:, selection.patch]
        changed[:, selection.patch] = donor[:, selection.patch]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=-1)).reshape(batch_size, n_channels).mean(dim=1)
        rms_parts.append(rms.detach().cpu().numpy())
        call_index += 1
        return _chronos_replace_hidden(output, changed)

    handle = model.encoder.block[selection.layer].register_forward_hook(replacement_hook)
    try:
        forecast = _chronos_forecast(model, histories, pred_len, num_output_patches, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _chronos_patched_forecast_set(
    model: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    num_output_patches: int,
    device: torch.device,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """替换 Chronos2 encoder 同一层的一组历史 patch。"""
    histories_3d = _as_3d(histories)
    batch_size, _seq_len, n_channels = histories_3d.shape
    if donor_outputs is None:
        donor_outputs = _chronos_capture(
            model, donor_histories, pred_len, num_output_patches, device, layer
        )
    call_index = 0
    rms_parts: list[np.ndarray] = []
    patch_indices = torch.tensor(patches, dtype=torch.long)

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _chronos_block_hidden(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        indices = patch_indices.to(original.device)
        if donor.shape != original.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {original.shape}")
        if not len(indices) or int(indices.min()) < 0 or int(indices.max()) >= original.shape[1]:
            raise ValueError(
                f"patches={patches} outside Chronos2 sequence length={original.shape[1]}."
            )
        changed = original.clone()
        delta = donor[:, indices] - original[:, indices]
        changed[:, indices] = donor[:, indices]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
        rms_parts.append(rms.reshape(batch_size, n_channels).mean(dim=1).detach().cpu().numpy())
        call_index += 1
        return _chronos_replace_hidden(output, changed)

    handle = model.encoder.block[layer].register_forward_hook(replacement_hook)
    try:
        forecast = _chronos_forecast(model, histories, pred_len, num_output_patches, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _chronos_noop(model: Any, histories: np.ndarray, pred_len: int, num_output_patches: int, device: torch.device, layer: int) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        return _chronos_replace_hidden(output, _chronos_block_hidden(output).clone())

    handle = model.encoder.block[layer].register_forward_hook(clone_hook)
    try:
        return _chronos_forecast(model, histories, pred_len, num_output_patches, device)
    finally:
        handle.remove()


# ---------------- Chronos-Bolt ----------------


def _chronos_bolt_intervention_module(adapter: ChronosBoltAdapter, layer: int) -> Any:
    blocks = adapter.model.encoder.block
    if not 0 <= layer < len(blocks):
        raise ValueError(f"Chronos-Bolt layer={layer} is out of range.")
    if layer == len(blocks) - 1:
        return adapter.model.encoder.final_layer_norm
    return blocks[layer]


def _chronos_bolt_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("Chronos-Bolt intervention output has no hidden tensor.")


def _chronos_bolt_replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise ValueError("Chronos-Bolt intervention output cannot be replaced.")


def _chronos_bolt_capture(
    adapter: ChronosBoltAdapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(_chronos_bolt_hidden(output).detach().clone())
        return output

    module = _chronos_bolt_intervention_module(adapter, layer)
    handle = module.register_forward_hook(capture_hook)
    try:
        adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if not captures:
        raise RuntimeError("Chronos-Bolt donor forecast produced no encoder captures.")
    return captures


def _chronos_bolt_patched_forecast_set(
    adapter: ChronosBoltAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_3d(histories)
    batch_size, _seq_len, n_channels = values.shape
    if donor_outputs is None:
        donor_outputs = _chronos_bolt_capture(adapter, donor_histories, pred_len, layer)
    indices = torch.as_tensor(patches, dtype=torch.long)
    call_index = 0
    rms_parts: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _chronos_bolt_hidden(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        selected = indices.to(original.device)
        if donor.shape != original.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {original.shape}.")
        if not len(selected) or int(selected.min()) < 0 or int(selected.max()) >= adapter.num_history_patches:
            raise ValueError(
                f"patches={patches} outside Chronos-Bolt history length={adapter.num_history_patches}."
            )
        changed = original.clone()
        delta = donor[:, selected] - original[:, selected]
        changed[:, selected] = donor[:, selected]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
        base_series = batch_size * n_channels
        if rms.numel() % base_series != 0:
            raise RuntimeError(
                "Chronos-Bolt autoregressive batch is not a multiple of batch*channels."
            )
        rms_parts.append(
            rms.reshape(batch_size, n_channels, -1)
            .mean(dim=(1, 2))
            .detach()
            .cpu()
            .numpy()
        )
        call_index += 1
        return _chronos_bolt_replace_output(output, changed)

    module = _chronos_bolt_intervention_module(adapter, layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(
            f"Used {call_index} Chronos-Bolt donor states, captured {len(donor_outputs)}."
        )
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _chronos_bolt_patched_forecast(
    adapter: ChronosBoltAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    return _chronos_bolt_patched_forecast_set(
        adapter,
        histories,
        donor_histories,
        pred_len,
        selection.layer,
        (selection.patch,),
    )


def _chronos_bolt_noop(
    adapter: ChronosBoltAdapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        return _chronos_bolt_replace_output(output, _chronos_bolt_hidden(output).clone())

    module = _chronos_bolt_intervention_module(adapter, layer)
    handle = module.register_forward_hook(clone_hook)
    try:
        return adapter.forecast(histories, pred_len)
    finally:
        handle.remove()


# ---------------- TTM ----------------


def _ttm_intervention_module(adapter: TTMAdapter, layer: int) -> Any:
    backbone_stages = adapter.model.backbone.encoder.mlp_mixer_encoder.mixers
    decoder_stages = adapter.model.decoder.decoder_block.mixers
    if not 0 <= layer < len(backbone_stages) + len(decoder_stages):
        raise ValueError(f"TTM layer={layer} is out of range.")
    if layer < len(backbone_stages):
        return backbone_stages[layer]
    return decoder_stages[layer - len(backbone_stages)]


def _ttm_stage_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("TTM stage output has no hidden tensor.")


def _ttm_replace_stage_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise ValueError("TTM stage output cannot be replaced.")


def _ttm_capture(
    adapter: TTMAdapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(_ttm_stage_hidden(output).detach().clone())
        return output

    module = _ttm_intervention_module(adapter, layer)
    handle = module.register_forward_hook(capture_hook)
    try:
        adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"Expected one TTM donor state, captured {len(captures)}.")
    return captures


def _ttm_patched_forecast_set(
    adapter: TTMAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_3d(histories)
    batch_size = values.shape[0]
    if donor_outputs is None:
        donor_outputs = _ttm_capture(adapter, donor_histories, pred_len, layer)
    patch_indices = torch.as_tensor(patches, dtype=torch.long)
    call_index = 0
    rms_parts: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        original = _ttm_stage_hidden(output)
        donor = donor_outputs[call_index].to(device=original.device, dtype=original.dtype)
        if donor.shape != original.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {original.shape}.")
        if not len(patch_indices) or int(patch_indices.min()) < 0 or int(patch_indices.max()) >= adapter.num_history_patches:
            raise ValueError(
                f"patches={patches} outside TTM history length={adapter.num_history_patches}."
            )
        # Atlas patch j maps to stage position j+1 because position zero is the
        # excluded frequency-prefix token.
        indices = (patch_indices + 1).to(original.device)
        changed = original.clone()
        delta = donor[:, :, indices] - original[:, :, indices]
        changed[:, :, indices] = donor[:, :, indices]
        rms_parts.append(
            torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2, 3)))
            .reshape(batch_size)
            .detach()
            .cpu()
            .numpy()
        )
        call_index += 1
        return _ttm_replace_stage_output(output, changed)

    module = _ttm_intervention_module(adapter, layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = adapter.forecast(histories, pred_len)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} TTM donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _ttm_patched_forecast(
    adapter: TTMAdapter,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    return _ttm_patched_forecast_set(
        adapter,
        histories,
        donor_histories,
        pred_len,
        selection.layer,
        (selection.patch,),
    )


def _ttm_noop(
    adapter: TTMAdapter,
    histories: np.ndarray,
    pred_len: int,
    layer: int,
) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        return _ttm_replace_stage_output(output, _ttm_stage_hidden(output).clone())

    module = _ttm_intervention_module(adapter, layer)
    handle = module.register_forward_hook(clone_hook)
    try:
        return adapter.forecast(histories, pred_len)
    finally:
        handle.remove()


# ---------------- Toto2 ----------------


def _toto2_inputs(histories: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """将 [B,T,C] 历史转换为 Toto2 decoder 的联合变量输入协议。"""
    histories = _as_3d(histories)
    batch_size, _seq_len, n_channels = histories.shape
    target = torch.from_numpy(histories).to(device=device, dtype=torch.float32).permute(0, 2, 1).contiguous()
    target_mask = torch.ones_like(target, dtype=torch.bool)
    return {
        "target": target,
        "target_mask": target_mask,
        "series_ids": torch.zeros(batch_size, n_channels, dtype=torch.long, device=device),
    }


def _toto2_forecast(model: Any, histories: np.ndarray, pred_len: int, device: torch.device) -> np.ndarray:
    """执行 Toto2 的联合多变量 quantile forecast，并取 0.5 分位点。"""
    inputs = _toto2_inputs(histories, device)
    quantiles = model.forecast(
        inputs,
        pred_len,
        decode_block_size=0,
        has_missing_values=True,
    )
    if quantiles.ndim != 4:
        raise RuntimeError(f"Unexpected Toto2 forecast shape: {tuple(quantiles.shape)}")
    median_index = int(model.output_head.knots.index(0.5))
    point = quantiles[median_index, ..., :pred_len]
    if point.ndim != 3:
        raise RuntimeError(f"Unexpected Toto2 median shape: {tuple(point.shape)}")
    return point.permute(0, 2, 1).detach().float().cpu().numpy().astype(np.float32)


def _toto2_capture(
    model: Any,
    histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
) -> list[torch.Tensor]:
    """捕获与 atlas 对齐的 Toto2 donor hidden。"""
    captures: list[torch.Tensor] = []

    def capture_hook(_module, _inputs, output):
        captures.append(output.detach().clone())
        return output

    module = _toto2_intervention_module(model, layer)
    handle = module.register_forward_hook(capture_hook)
    try:
        _toto2_forecast(model, histories, pred_len, device)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"Expected one Toto2 donor forward, captured {len(captures)}.")
    return captures


def _toto2_intervention_module(model: Any, layer: int) -> Any:
    """最后层使用预测头实际读取的 out_norm，其余层使用 post-block state。"""
    if not 0 <= layer < len(model.transformer.layers):
        raise ValueError(f"Toto2 layer={layer} is out of range.")
    if layer == len(model.transformer.layers) - 1:
        return model.transformer.out_norm
    return model.transformer.layers[layer]


def _toto2_patch_replace(
    output: torch.Tensor,
    donor: torch.Tensor,
    model: Any,
    batch_size: int,
    n_channels: int,
    patch: int,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Toto2 的 time/variate 展平方式替换一个完整时间 patch。"""
    if output.shape != donor.shape:
        raise ValueError(f"Donor shape {donor.shape} != target shape {output.shape}")
    changed = output.clone()
    if output.ndim == 4:
        if output.shape[0] != batch_size or output.shape[1] != n_channels:
            raise ValueError(
                "Toto2 post-norm hidden must have shape [batch, channel, time, hidden]."
            )
        num_time = output.shape[2]
        if patch >= num_time:
            raise ValueError(f"patch={patch} outside Toto2 post-norm time length={num_time}.")
        delta = donor[:, :, patch] - output[:, :, patch]
        changed[:, :, patch] = donor[:, :, patch]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
        return changed, rms
    if model.transformer._if_variate_layer(layer):
        if output.shape[0] % batch_size != 0:
            raise ValueError("Toto2 variate-layer batch flattening is inconsistent.")
        num_time = output.shape[0] // batch_size
        target_view = changed.reshape(batch_size, num_time, n_channels, output.shape[-1])
        donor_view = donor.reshape(batch_size, num_time, n_channels, output.shape[-1])
        if patch >= num_time:
            raise ValueError(f"patch={patch} outside Toto2 variate sequence length={num_time}.")
        delta = donor_view[:, patch] - target_view[:, patch]
        target_view[:, patch] = donor_view[:, patch]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
        return target_view.reshape_as(changed), rms

    if output.shape[0] != batch_size * n_channels:
        raise ValueError("Toto2 time-layer batch/channel flattening is inconsistent.")
    num_time = output.shape[1]
    target_view = changed.reshape(batch_size, n_channels, num_time, output.shape[-1])
    donor_view = donor.reshape(batch_size, n_channels, num_time, output.shape[-1])
    if patch >= num_time:
        raise ValueError(f"patch={patch} outside Toto2 time sequence length={num_time}.")
    delta = donor_view[:, :, patch] - target_view[:, :, patch]
    target_view[:, :, patch] = donor_view[:, :, patch]
    rms = torch.sqrt(torch.mean(delta.float().square(), dim=-1)).mean(dim=1)
    return target_view.reshape_as(changed), rms


def _toto2_patched_forecast(
    model: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    """替换 Toto2 联合 decoder 中指定 layer/time patch 的全部变量 hidden。"""
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    batch_size, _seq_len, n_channels = histories.shape
    donor_outputs = _toto2_capture(model, donor_histories, pred_len, device, selection.layer)
    call_index = 0
    rms_values: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        donor = donor_outputs[call_index].to(device=output.device, dtype=output.dtype)
        changed, rms = _toto2_patch_replace(
            output, donor, model, batch_size, n_channels, selection.patch, selection.layer
        )
        rms_values.append(rms.detach().cpu().numpy())
        call_index += 1
        return changed

    module = _toto2_intervention_module(model, selection.layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = _toto2_forecast(model, histories, pred_len, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} Toto2 donor forwards but captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_values), axis=0).astype(np.float32)


def _toto2_patch_set_replace(
    output: torch.Tensor,
    donor: torch.Tensor,
    model: Any,
    batch_size: int,
    n_channels: int,
    patches: tuple[int, ...],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Toto2 的层内布局替换一组完整多变量时间 patch。"""
    if output.shape != donor.shape:
        raise ValueError(f"Donor shape {donor.shape} != target shape {output.shape}")
    indices = torch.tensor(patches, dtype=torch.long, device=output.device)
    if not len(indices) or int(indices.min()) < 0:
        raise ValueError("Toto2 progressive intervention requires non-empty valid patches.")
    changed = output.clone()
    if output.ndim == 4:
        if output.shape[0] != batch_size or output.shape[1] != n_channels:
            raise ValueError(
                "Toto2 post-norm hidden must have shape [batch, channel, time, hidden]."
            )
        num_time = output.shape[2]
        if int(indices.max()) >= num_time:
            raise ValueError(f"patches={patches} outside Toto2 post-norm time length={num_time}.")
        delta = donor[:, :, indices] - output[:, :, indices]
        changed[:, :, indices] = donor[:, :, indices]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2, 3)))
        return changed, rms
    if model.transformer._if_variate_layer(layer):
        if output.shape[0] % batch_size != 0:
            raise ValueError("Toto2 variate-layer batch flattening is inconsistent.")
        num_time = output.shape[0] // batch_size
        if int(indices.max()) >= num_time:
            raise ValueError(f"patches={patches} outside Toto2 variate sequence length={num_time}.")
        target_view = changed.reshape(batch_size, num_time, n_channels, output.shape[-1])
        donor_view = donor.reshape(batch_size, num_time, n_channels, output.shape[-1])
        delta = donor_view[:, indices] - target_view[:, indices]
        target_view[:, indices] = donor_view[:, indices]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2, 3)))
        return target_view.reshape_as(changed), rms

    if output.shape[0] != batch_size * n_channels:
        raise ValueError("Toto2 time-layer batch/channel flattening is inconsistent.")
    num_time = output.shape[1]
    if int(indices.max()) >= num_time:
        raise ValueError(f"patches={patches} outside Toto2 time sequence length={num_time}.")
    target_view = changed.reshape(batch_size, n_channels, num_time, output.shape[-1])
    donor_view = donor.reshape(batch_size, n_channels, num_time, output.shape[-1])
    delta = donor_view[:, :, indices] - target_view[:, :, indices]
    target_view[:, :, indices] = donor_view[:, :, indices]
    rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2, 3)))
    return target_view.reshape_as(changed), rms


def _toto2_patched_forecast_set(
    model: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    histories = _as_3d(histories)
    donor_histories = _as_3d(donor_histories)
    batch_size, _seq_len, n_channels = histories.shape
    if donor_outputs is None:
        donor_outputs = _toto2_capture(model, donor_histories, pred_len, device, layer)
    call_index = 0
    rms_values: list[np.ndarray] = []

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        donor = donor_outputs[call_index].to(device=output.device, dtype=output.dtype)
        changed, rms = _toto2_patch_set_replace(
            output, donor, model, batch_size, n_channels, patches, layer
        )
        rms_values.append(rms.detach().cpu().numpy())
        call_index += 1
        return changed

    module = _toto2_intervention_module(model, layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = _toto2_forecast(model, histories, pred_len, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} Toto2 donor forwards but captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_values), axis=0).astype(np.float32)


def _toto2_noop(model: Any, histories: np.ndarray, pred_len: int, device: torch.device, layer: int) -> np.ndarray:
    """Toto2 clone-only hook，用于验证 hook 本身不改变 forecast。"""
    def clone_hook(_module, _inputs, output):
        return output.clone()

    module = _toto2_intervention_module(model, layer)
    handle = module.register_forward_hook(clone_hook)
    try:
        return _toto2_forecast(model, histories, pred_len, device)
    finally:
        handle.remove()


# ---------------- Moirai2 ----------------


def _moirai_build_forecaster(adapter: Moirai2Adapter, n_channels: int):
    from uni2ts.model.moirai2 import Moirai2Forecast

    config = adapter.config
    # target_dim 使用真实变量数；Moirai2 在 packed token 中联合建模所有 channel。
    # 当前 uni2ts 递归分支在多变量 P=96 时有 shape bug，因此先构造一个
    # 不触发递归的最大原生 block，外层按 block 做中位数反馈。
    block_length = int(adapter.model.num_predict_token * adapter.model.patch_size)
    forecaster = Moirai2Forecast(
        prediction_length=block_length,
        target_dim=n_channels,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=config.data.seq_len,
        module=adapter.model,
    ).to(adapter.device).eval()
    forecaster._information_anchor_block_length = block_length
    return forecaster
    return Moirai2TokenBudgetForecaster(adapter, native_forecaster)


def _moirai_median_index(module: Any) -> int:
    return int(np.argmin(np.abs(np.asarray(module.quantile_levels, dtype=np.float64) - 0.5)))


def _moirai_forecast(forecaster: Any, histories: np.ndarray, pred_len: int, device: torch.device) -> np.ndarray:
    histories = _as_3d(histories)
    block_length = int(getattr(forecaster, "_information_anchor_block_length", pred_len))
    context = histories.copy()
    predictions: list[np.ndarray] = []
    remaining = int(pred_len)
    median = _moirai_median_index(forecaster.module)
    while remaining > 0:
        past_target = torch.from_numpy(context).to(device=device, dtype=torch.float32)
        prediction = forecaster(
            past_target=past_target,
            past_observed_target=torch.ones_like(past_target, dtype=torch.bool),
            past_is_pad=torch.zeros(past_target.shape[:2], device=device, dtype=torch.bool),
        )
        if prediction.ndim != 4:
            raise ValueError(f"Unexpected joint Moirai2 forecast shape: {tuple(prediction.shape)}")
        take = min(remaining, block_length)
        point = prediction[:, median, :take, :].detach().float().cpu().numpy().astype(np.float32)
        predictions.append(point)
        remaining -= take
        if remaining > 0:
            context = np.concatenate([context, point], axis=1)
            context = context[:, -forecaster.hparams.context_length :, :]
    return np.concatenate(predictions, axis=1).astype(np.float32)


def _moirai_capture(
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

    module = _moirai_intervention_module(forecaster, layer)
    handle = module.register_forward_hook(capture_hook)
    try:
        _moirai_forecast(forecaster, histories, pred_len, device)
    finally:
        handle.remove()
    if not captures:
        raise RuntimeError("Moirai2 donor forward produced no layer captures.")
    return captures


def _moirai_intervention_module(forecaster: Any, layer: int) -> Any:
    """最后层使用 encoder final state，与 adapter 保存的表示严格对齐。"""
    layers = forecaster.module.encoder.layers
    if not 0 <= layer < len(layers):
        raise ValueError(f"Moirai2 layer={layer} is out of range.")
    if layer == len(layers) - 1:
        return forecaster.module.encoder
    return layers[layer]


def _moirai_patch_token_indices(seq_len: int, patch_len: int, n_channels: int, patch: int) -> torch.Tensor:
    num_patches = math.ceil(seq_len / patch_len)
    if patch < 0 or patch >= num_patches:
        raise ValueError(f"patch={patch} outside Moirai2 context num_patches={num_patches}.")
    return torch.tensor([channel * num_patches + patch for channel in range(n_channels)], dtype=torch.long)


def _moirai_patched_forecast(
    forecaster: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    selection: UnitSelection,
) -> tuple[np.ndarray, np.ndarray]:
    histories_3d = _as_3d(histories)
    batch_size, seq_len, n_channels = histories_3d.shape
    donor_outputs = _moirai_capture(forecaster, donor_histories, pred_len, device, selection.layer)
    call_index = 0
    rms_parts: list[np.ndarray] = []
    context_patches = math.ceil(seq_len / forecaster.module.patch_size)
    token_indices = torch.tensor(
        [channel * context_patches + selection.patch for channel in range(n_channels)],
        dtype=torch.long,
    )

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        donor = donor_outputs[call_index].to(device=output.device, dtype=output.dtype)
        if donor.shape != output.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {output.shape}")
        changed = output.clone()
        indices = token_indices.to(output.device)
        if output.shape[0] != batch_size or output.shape[1] <= int(indices.max()):
            raise ValueError(
                f"Moirai2 packed hidden shape {tuple(output.shape)} cannot address context indices {indices.tolist()}"
            )
        delta = donor[:, indices, :] - output[:, indices, :]
        changed[:, indices, :] = donor[:, indices, :]
        rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
        rms_parts.append(rms.detach().cpu().numpy())
        call_index += 1
        return changed

    module = _moirai_intervention_module(forecaster, selection.layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = _moirai_forecast(forecaster, histories, pred_len, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _moirai_patched_forecast_set(
    forecaster: Any,
    histories: np.ndarray,
    donor_histories: np.ndarray,
    pred_len: int,
    device: torch.device,
    layer: int,
    patches: tuple[int, ...],
    donor_outputs: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """替换 Moirai2 packed hidden 中所有变量对应的一组历史 patch。"""
    histories_3d = _as_3d(histories)
    batch_size, seq_len, n_channels = histories_3d.shape
    if donor_outputs is None:
        donor_outputs = _moirai_capture(forecaster, donor_histories, pred_len, device, layer)
    call_index = 0
    rms_parts: list[np.ndarray] = []
    context_patches = math.ceil(seq_len / forecaster.module.patch_size)
    if not patches or min(patches) < 0 or max(patches) >= context_patches:
        raise ValueError(
            f"patches={patches} outside Moirai2 context num_patches={context_patches}."
        )
    token_indices = torch.tensor(
        [
            channel * context_patches + patch
            for channel in range(n_channels)
            for patch in patches
        ],
        dtype=torch.long,
    )

    def replacement_hook(_module, _inputs, output):
        nonlocal call_index
        donor = donor_outputs[call_index].to(device=output.device, dtype=output.dtype)
        if donor.shape != output.shape:
            raise ValueError(f"Donor shape {donor.shape} != target shape {output.shape}")
        indices = token_indices.to(output.device)
        if output.shape[0] != batch_size or output.shape[1] <= int(indices.max()):
            raise ValueError(
                f"Moirai2 packed hidden shape {tuple(output.shape)} cannot address context indices."
            )
        changed = output.clone()
        delta = donor[:, indices] - output[:, indices]
        changed[:, indices] = donor[:, indices]
        rms_parts.append(
            torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2))).detach().cpu().numpy()
        )
        call_index += 1
        return changed

    module = _moirai_intervention_module(forecaster, layer)
    handle = module.register_forward_hook(replacement_hook)
    try:
        forecast = _moirai_forecast(forecaster, histories, pred_len, device)
    finally:
        handle.remove()
    if call_index != len(donor_outputs):
        raise RuntimeError(f"Used {call_index} donor states, captured {len(donor_outputs)}.")
    return forecast, np.mean(np.stack(rms_parts), axis=0).astype(np.float32)


def _moirai_noop(forecaster: Any, histories: np.ndarray, pred_len: int, device: torch.device, layer: int) -> np.ndarray:
    def clone_hook(_module, _inputs, output):
        return output.clone()

    module = _moirai_intervention_module(forecaster, layer)
    handle = module.register_forward_hook(clone_hook)
    try:
        return _moirai_forecast(forecaster, histories, pred_len, device)
    finally:
        handle.remove()


class ModelRunner:
    def __init__(self, config: Any, n_channels: int):
        self.config = config
        self.name = config.model.name
        self.n_channels = n_channels
        if self.name == "TimesFM":
            self.adapter = TimesFMAdapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        elif self.name == "TimesFM2.5":
            self.adapter = TimesFM25Adapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        elif self.name == "TimesFM3":
            self.adapter = TimesFM3Adapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        elif self.name == "Chronos2":
            self.adapter = Chronos2Adapter(config)
            self.device = self.adapter.device
            self.num_output_patches = math.ceil(
                config.data.pred_len / self.adapter.model.chronos_config.output_patch_size
            )
        elif self.name == "ChronosBolt":
            self.adapter = ChronosBoltAdapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        elif self.name == "Moirai2":
            self.adapter = Moirai2Adapter(config)
            self.device = self.adapter.device
            self.forecaster = _moirai_build_forecaster(self.adapter, n_channels)
            self.num_output_patches = None
        elif self.name == "Toto2":
            self.adapter = Toto2Adapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        elif self.name == "TTM":
            self.adapter = TTMAdapter(config)
            self.device = self.adapter.device
            self.num_output_patches = None
        else:
            raise ValueError(f"Unsupported model={self.name}.")

    def forecast(self, histories: np.ndarray) -> np.ndarray:
        if self.name == "TimesFM":
            return _timesfm_forecast(self.adapter, histories, self.config.data.pred_len)
        if self.name == "TimesFM2.5":
            return self.adapter.forecast(histories, self.config.data.pred_len)
        if self.name == "TimesFM3":
            return self.adapter.forecast(histories, self.config.data.pred_len)
        if self.name == "Chronos2":
            return _chronos_forecast(
                self.adapter.model,
                histories,
                self.config.data.pred_len,
                int(self.num_output_patches),
                self.device,
            )
        if self.name == "ChronosBolt":
            return self.adapter.forecast(histories, self.config.data.pred_len)
        if self.name == "TTM":
            return self.adapter.forecast(histories, self.config.data.pred_len)
        if self.name == "Moirai2":
            return _moirai_forecast(self.forecaster, histories, self.config.data.pred_len, self.device)
        return _toto2_forecast(self.adapter.model, histories, self.config.data.pred_len, self.device)

    def patched_forecast(
        self,
        histories: np.ndarray,
        donor_histories: np.ndarray,
        selection: UnitSelection,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.name == "TimesFM":
            return _timesfm_patched_forecast(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                selection,
            )
        if self.name == "TimesFM2.5":
            return _timesfm25_patched_forecast(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                selection,
            )
        if self.name == "TimesFM3":
            return _timesfm3_patched_forecast(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                selection,
            )
        if self.name == "Chronos2":
            return _chronos_patched_forecast(
                self.adapter.model,
                histories,
                donor_histories,
                self.config.data.pred_len,
                int(self.num_output_patches),
                self.device,
                selection,
            )
        if self.name == "ChronosBolt":
            return _chronos_bolt_patched_forecast(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                selection,
            )
        if self.name == "TTM":
            return _ttm_patched_forecast(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                selection,
            )
        if self.name == "Moirai2":
            return _moirai_patched_forecast(
                self.forecaster,
                histories,
                donor_histories,
                self.config.data.pred_len,
                self.device,
                selection,
            )
        return _toto2_patched_forecast(
            self.adapter.model,
            histories,
            donor_histories,
            self.config.data.pred_len,
            self.device,
            selection,
        )

    def patched_forecast_set(
        self,
        histories: np.ndarray,
        donor_histories: np.ndarray,
        *,
        layer: int,
        patches: tuple[int, ...],
        donor_state: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """在一个原生层内替换多个历史 patch，供递增比例干预使用。"""
        if not patches:
            raise ValueError("patched_forecast_set requires at least one patch.")
        if self.name == "TimesFM":
            return _timesfm_patched_forecast_set(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                layer,
                patches,
                donor_state,
            )
        if self.name == "TimesFM2.5":
            return _timesfm25_patched_forecast_set(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                layer,
                patches,
                donor_state,
            )
        if self.name == "TimesFM3":
            return _timesfm3_patched_forecast_set(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                layer,
                patches,
                donor_state,
            )
        if self.name == "Chronos2":
            return _chronos_patched_forecast_set(
                self.adapter.model,
                histories,
                donor_histories,
                self.config.data.pred_len,
                int(self.num_output_patches),
                self.device,
                layer,
                patches,
                donor_state,
            )
        if self.name == "ChronosBolt":
            return _chronos_bolt_patched_forecast_set(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                layer,
                patches,
                donor_state,
            )
        if self.name == "TTM":
            return _ttm_patched_forecast_set(
                self.adapter,
                histories,
                donor_histories,
                self.config.data.pred_len,
                layer,
                patches,
                donor_state,
            )
        if self.name == "Moirai2":
            return _moirai_patched_forecast_set(
                self.forecaster,
                histories,
                donor_histories,
                self.config.data.pred_len,
                self.device,
                layer,
                patches,
                donor_state,
            )
        return _toto2_patched_forecast_set(
            self.adapter.model,
            histories,
            donor_histories,
            self.config.data.pred_len,
            self.device,
            layer,
            patches,
            donor_state,
        )

    def capture_donor_state(self, donor_histories: np.ndarray, layer: int) -> Any:
        """捕获一次同层 donor state，供同 batch 的全部 patch 条件复用。"""
        if self.name == "TimesFM":
            histories = _as_3d(donor_histories)
            outer = self.adapter.model.model
            return [
                _timesfm_capture_channel(
                    outer,
                    histories[:, :, channel],
                    self.config.data.pred_len,
                    layer,
                )
                for channel in range(histories.shape[-1])
            ]
        if self.name == "TimesFM2.5":
            return _timesfm25_capture(
                self.adapter,
                donor_histories,
                self.config.data.pred_len,
                layer,
            )
        if self.name == "TimesFM3":
            return _timesfm3_capture(
                self.adapter,
                donor_histories,
                self.config.data.pred_len,
                layer,
            )
        if self.name == "Chronos2":
            return _chronos_capture(
                self.adapter.model,
                donor_histories,
                self.config.data.pred_len,
                int(self.num_output_patches),
                self.device,
                layer,
            )
        if self.name == "ChronosBolt":
            return _chronos_bolt_capture(
                self.adapter,
                donor_histories,
                self.config.data.pred_len,
                layer,
            )
        if self.name == "TTM":
            return _ttm_capture(
                self.adapter,
                donor_histories,
                self.config.data.pred_len,
                layer,
            )
        if self.name == "Moirai2":
            return _moirai_capture(
                self.forecaster,
                donor_histories,
                self.config.data.pred_len,
                self.device,
                layer,
            )
        return _toto2_capture(
            self.adapter.model,
            donor_histories,
            self.config.data.pred_len,
            self.device,
            layer,
        )

    def noop(self, histories: np.ndarray, layer: int) -> np.ndarray:
        if self.name == "TimesFM":
            return _timesfm_noop(self.adapter, histories, self.config.data.pred_len, layer)
        if self.name == "TimesFM2.5":
            return _timesfm25_noop(self.adapter, histories, self.config.data.pred_len, layer)
        if self.name == "TimesFM3":
            return _timesfm3_noop(self.adapter, histories, self.config.data.pred_len, layer)
        if self.name == "Chronos2":
            return _chronos_noop(
                self.adapter.model,
                histories,
                self.config.data.pred_len,
                int(self.num_output_patches),
                self.device,
                layer,
            )
        if self.name == "ChronosBolt":
            return _chronos_bolt_noop(
                self.adapter, histories, self.config.data.pred_len, layer
            )
        if self.name == "TTM":
            return _ttm_noop(self.adapter, histories, self.config.data.pred_len, layer)
        if self.name == "Moirai2":
            return _moirai_noop(self.forecaster, histories, self.config.data.pred_len, self.device, layer)
        return _toto2_noop(self.adapter.model, histories, self.config.data.pred_len, self.device, layer)


def main() -> None:
    args = _parse_args()
    reference_dir = Path(args.reference_run).resolve()
    config = load_config(reference_dir / "config.json")
    if args.device:
        config = replace(config, model=replace(config.model, device=args.device))

    reference = np.load(reference_dir / "mi_results.npz")
    mi_z = np.asarray(reference["mi_z"], dtype=np.float32)
    selections: list[UnitSelection] = select_units(
        mi_z,
        np.asarray(reference["layer_mi_z"]),
        np.asarray(reference["patch_mi_z"]),
        include_controls=args.include_controls,
        random_repetitions=args.random_control_repetitions,
        seed=config.mi.seed + 97_000,
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
    columns = select_value_columns(frame, config.data)
    if len(columns) <= 1:
        raise ValueError(
            "This runner is for multivariate causal interventions; use features=M/custom with >1 column."
        )
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32)
    target_index = columns.index(config.data.target)
    scaler = fit_train_standard_scaler(values, config.data.train_end)
    windows = make_windows(values, sampled_origins, config.data.seq_len, config.data.pred_len, columns=columns)
    histories, futures = _as_3d(windows.history_raw), _as_3d(windows.future_raw)

    runner = ModelRunner(config, histories.shape[-1])
    baseline_parts = []
    with torch.no_grad():
        for start in range(0, len(histories), args.batch_size):
            stop = min(len(histories), start + args.batch_size)
            baseline_parts.append(runner.forecast(histories[start:stop]))
    baseline = np.concatenate(baseline_parts, axis=0).astype(np.float32)
    baseline_mse, baseline_mae = per_origin_metrics(
        baseline,
        futures,
        scaler,
        target_index=target_index,
    )
    all_channel_baseline_mse, all_channel_baseline_mae = per_origin_metrics(
        baseline,
        futures,
        scaler,
    )

    check_stop = min(args.batch_size, len(histories))
    with torch.no_grad():
        noop = runner.noop(histories[:check_stop], selections[0].layer)
    noop_max_abs_difference = float(np.max(np.abs(noop - baseline[:check_stop])))
    if noop_max_abs_difference > 1e-4:
        raise RuntimeError(f"Clone-only hook changed forecasts by {noop_max_abs_difference:.6g}.")

    metric_names = (
        "delta_mse",
        "delta_mae",
        "forecast_change_mse",
        "forecast_change_mae",
        "all_channel_delta_mse",
        "all_channel_delta_mae",
        "all_channel_forecast_change_mse",
        "all_channel_forecast_change_mae",
        "perturbation_rms",
    )
    metrics = {unit.label: {name: [] for name in metric_names} for unit in selections}

    with torch.no_grad():
        for shift_index, shift in enumerate(shifts):
            donor_histories = np.roll(histories, int(shift), axis=0)
            shift_predictions = {unit.label: [] for unit in selections}
            shift_rms = {unit.label: [] for unit in selections}
            for start in range(0, len(histories), args.batch_size):
                stop = min(len(histories), start + args.batch_size)
                target_batch = histories[start:stop]
                donor_batch = donor_histories[start:stop]
                for unit in selections:
                    prediction, rms = runner.patched_forecast(target_batch, donor_batch, unit)
                    shift_predictions[unit.label].append(prediction)
                    shift_rms[unit.label].append(rms)
            for unit in selections:
                prediction = np.concatenate(shift_predictions[unit.label], axis=0).astype(np.float32)
                rms = np.concatenate(shift_rms[unit.label], axis=0).astype(np.float32)
                patched_mse, patched_mae = per_origin_metrics(
                    prediction,
                    futures,
                    scaler,
                    target_index=target_index,
                )
                change_mse, change_mae = forecast_change_metrics(
                    prediction,
                    baseline,
                    scaler,
                    target_index=target_index,
                )
                all_channel_patched_mse, all_channel_patched_mae = per_origin_metrics(
                    prediction,
                    futures,
                    scaler,
                )
                all_channel_change_mse, all_channel_change_mae = forecast_change_metrics(
                    prediction,
                    baseline,
                    scaler,
                )
                metrics[unit.label]["delta_mse"].append(patched_mse - baseline_mse)
                metrics[unit.label]["delta_mae"].append(patched_mae - baseline_mae)
                metrics[unit.label]["forecast_change_mse"].append(change_mse)
                metrics[unit.label]["forecast_change_mae"].append(change_mae)
                metrics[unit.label]["all_channel_delta_mse"].append(
                    all_channel_patched_mse - all_channel_baseline_mse
                )
                metrics[unit.label]["all_channel_delta_mae"].append(
                    all_channel_patched_mae - all_channel_baseline_mae
                )
                metrics[unit.label]["all_channel_forecast_change_mse"].append(
                    all_channel_change_mse
                )
                metrics[unit.label]["all_channel_forecast_change_mae"].append(
                    all_channel_change_mae
                )
                metrics[unit.label]["perturbation_rms"].append(rms)
            print(
                f"model={config.model.name} donor_shift={shift_index + 1}/{len(shifts)} offset={int(shift)} complete",
                flush=True,
            )

    stacked = {
        label: {name: np.stack(arrays, axis=0) for name, arrays in condition.items()}
        for label, condition in metrics.items()
    }
    records = []
    for index, unit in enumerate(selections):
        condition = stacked[unit.label]
        record = condition_summary(
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
        all_channel_record = condition_summary(
            unit.label,
            unit,
            condition["all_channel_delta_mse"],
            condition["all_channel_delta_mae"],
            condition["all_channel_forecast_change_mse"],
            condition["all_channel_forecast_change_mae"],
            condition["perturbation_rms"],
            args.bootstrap_repetitions,
            config.mi.seed + 190_000 + index * 10,
        )
        for key, value in all_channel_record.items():
            if key not in {"condition", "layer_zero_based", "patch_zero_based", "mi_z"}:
                record[f"all_channel_{key}"] = value
        record["mean_patched_mse"] = float(
            baseline_mse.mean() + record["mean_delta_mse"]
        )
        record["mean_patched_mae"] = float(
            baseline_mae.mean() + record["mean_delta_mae"]
        )
        record["all_channel_mean_patched_mse"] = float(
            all_channel_baseline_mse.mean() + all_channel_record["mean_delta_mse"]
        )
        record["all_channel_mean_patched_mae"] = float(
            all_channel_baseline_mae.mean() + all_channel_record["mean_delta_mae"]
        )
        records.append(record)

    lookup = {record["condition"]: record for record in records}
    primary_contrast = None
    all_channel_primary_contrast = None
    if "mi_top_cell" in stacked and "low_mi_same_layer" in stacked:
        primary_high = stacked["mi_top_cell"]
        primary_low = stacked["low_mi_same_layer"]
        primary_contrast = _paired_contrast_summary(
            primary_high,
            primary_low,
            args.bootstrap_repetitions,
            config.mi.seed + 95_000,
        )
        primary_contrast.update(
            {
                "metric_scope": "target",
                "metric_target": config.data.target,
                "top_condition": lookup.get("mi_top_cell"),
                "matched_low_condition": lookup.get("low_mi_same_layer"),
            }
        )
        all_channel_primary_contrast = _paired_contrast_summary(
            {
                "delta_mse": primary_high["all_channel_delta_mse"],
                "delta_mae": primary_high["all_channel_delta_mae"],
                "forecast_change_mse": primary_high["all_channel_forecast_change_mse"],
                "forecast_change_mae": primary_high["all_channel_forecast_change_mae"],
            },
            {
                "delta_mse": primary_low["all_channel_delta_mse"],
                "delta_mae": primary_low["all_channel_delta_mae"],
                "forecast_change_mse": primary_low["all_channel_forecast_change_mse"],
                "forecast_change_mae": primary_low["all_channel_forecast_change_mae"],
            },
            args.bootstrap_repetitions,
            config.mi.seed + 195_000,
        )
        all_channel_primary_contrast.update(
            {
                "metric_scope": "all_channels_macro",
                "metric_target": None,
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{config.model.name.lower()}_{config.data.dataset.lower()}_m_l{config.data.seq_len}_p{config.data.pred_len}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    # 保存完整配置侧车，保证干预产物可被离线 TSLib 审计器独立重算。
    write_json(output_dir / "config.json", config.to_dict())

    npz_payload: dict[str, np.ndarray] = {
        "origins": sampled_origins,
        "donor_shift_offsets": shifts,
        "columns": np.asarray(columns),
        "tslib_mean": scaler.mean.astype(np.float32),
        "tslib_scale": scaler.scale.astype(np.float32),
        "metric_target_index": np.asarray(target_index, dtype=np.int64),
        "metric_target_name": np.asarray(config.data.target),
        "baseline_forecast_raw": baseline,
        "future_raw": futures,
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "all_channel_baseline_mse": all_channel_baseline_mse,
        "all_channel_baseline_mae": all_channel_baseline_mae,
    }
    for label, condition in stacked.items():
        for name, array in condition.items():
            npz_payload[f"{label}__{name}"] = array
    np.savez_compressed(output_dir / "intervention_results.npz", **npz_payload)

    payload = {
        "status": "complete",
        "model": config.model.name,
        "model_id": config.model.model_id,
        "dataset": config.data.dataset,
        "features": config.data.features,
        "metric_schema_version": 2,
        "input_columns": list(columns),
        "forecast_columns": list(columns),
        "target_columns": list(columns),
        "metric_target_columns": [config.data.target],
        "metric_scope": "target",
        "metric_target": config.data.target,
        "metric_target_index": target_index,
        "reference_run": str(reference_dir),
        "split": "test",
        "metric_protocol": (
            "Primary target-channel metric: train-split per-variable StandardScaler; "
            f"MSE/MAE averaged over prediction horizon for {config.data.target} only, then origins."
        ),
        "all_channel_metric_protocol": (
            "Secondary macro metric: train-split per-variable StandardScaler; "
            "MSE/MAE averaged over prediction horizon and all forecast variables, then origins."
        ),
        "selection_protocol": "All intervention units were selected from discovery MI before held-out forecast evaluation.",
        "controls_enabled": bool(args.include_controls),
        "intervention": "Replace the selected post-layer history patch representation with a temporally distant donor. For multivariate models, all variables at the same time patch are replaced together.",
        "control_protocol": (
            "Controls included: recent_same_layer and deterministic random_same_layer_* are "
            "selected at the MI-top layer and measured with the same test origins and donor shifts."
            if args.include_controls
            else "Recent/random controls were not requested for this run."
        ),
        "donor_protocol": "Every paired origin is separated by at least L+P; identical donor shifts are used for all conditions.",
        "sample_count": int(len(sampled_origins)),
        "origins_hash": origins_hash(sampled_origins),
        "donor_shift_count": int(len(shifts)),
        "donor_shift_offsets": shifts.tolist(),
        "required_temporal_separation": config.data.seq_len + config.data.pred_len,
        "baseline_mse": float(baseline_mse.mean()),
        "baseline_mae": float(baseline_mae.mean()),
        "all_channel_baseline_mse": float(all_channel_baseline_mse.mean()),
        "all_channel_baseline_mae": float(all_channel_baseline_mae.mean()),
        "noop_clone_max_abs_forecast_difference": noop_max_abs_difference,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "records": records,
        "primary_contrast": primary_contrast,
        "all_channel_primary_contrast": all_channel_primary_contrast,
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
