from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.information_anchor.config import ExperimentConfig


_FREQUENCY_TOKEN = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 5,
    "ETTm2": 5,
    "weather": 4,
    "electricity": 7,
    "traffic": 7,
}


def aggregate_ttm_channel_hidden(hidden: torch.Tensor, strategy: str) -> torch.Tensor:
    """Map TTM [B,C,N,D] stages to a shared [B,N,D*] representation."""
    if hidden.ndim != 4:
        raise ValueError(f"Expected TTM hidden [B,C,N,D], got {tuple(hidden.shape)}.")
    if strategy == "concat_same_time_patch":
        batch, channels, patches, width = hidden.shape
        return hidden.permute(0, 2, 1, 3).reshape(batch, patches, channels * width)
    mean = hidden.mean(dim=1)
    if strategy == "mean":
        return mean
    if strategy == "mean_std":
        std = hidden.float().std(dim=1, unbiased=False).to(hidden.dtype)
        return torch.cat((mean, std), dim=-1)
    raise ValueError(f"Unsupported TTM channel aggregation: {strategy!r}.")


class TTMAdapter:
    """Extract common-grid stages from frozen Granite TTM-R2.1."""

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self.device = torch.device(config.model.device)
        if not config.model.revision:
            raise ValueError("TTM requires an explicit checkpoint revision.")
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            local_files_only=config.runtime.offline,
        ).to(self.device).eval()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"TTM config patch_len={config.model.patch_len}, "
                f"checkpoint patch_length={self.patch_len}."
            )
        if config.data.seq_len > int(self.model.config.context_length):
            raise ValueError("TTM seq_len exceeds the checkpoint context length.")
        if config.data.pred_len < 1:
            raise ValueError("TTM pred_len must be positive.")
        if not bool(self.model.config.resolution_prefix_tuning):
            raise ValueError("The registered TTM protocol expects frequency prefix tuning.")
        if config.data.dataset not in _FREQUENCY_TOKEN:
            raise ValueError(f"TTM frequency token is not registered for {config.data.dataset!r}.")

    @property
    def patch_len(self) -> int:
        return int(self.model.config.patch_length)

    @property
    def patch_stride(self) -> int:
        return int(self.model.config.patch_stride)

    @property
    def num_history_patches(self) -> int:
        # Shorter research contexts are left-padded to the frozen checkpoint
        # context.  The padding is causal (the first observed value is
        # repeated) and never contains future observations.
        length = int(self.model.config.context_length)
        return max(1, math.floor((length - self.patch_len) / self.patch_stride) + 1)

    @property
    def native_stage_count(self) -> int:
        backbone = self.model.backbone.encoder.mlp_mixer_encoder.mixers
        decoder = self.model.decoder.decoder_block.mixers
        return len(backbone) + len(decoder)

    @property
    def frequency_token_value(self) -> int:
        return _FREQUENCY_TOKEN[self.config.data.dataset]

    def frequency_tokens(self, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            self.frequency_token_value,
            dtype=torch.long,
            device=self.device,
        )

    @property
    def model_context_length(self) -> int:
        return int(self.model.config.context_length)

    @property
    def native_prediction_length(self) -> int:
        return int(self.model.config.prediction_length)

    def _pad_context(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.model_context_length:
            raise ValueError(
                f"TTM history length {values.shape[1]} exceeds context_length={self.model_context_length}."
            )
        if values.shape[1] == self.model_context_length:
            return values
        pad = self.model_context_length - values.shape[1]
        # Repeat the oldest observed value. This carries no information from
        # after the query origin and avoids injecting an artificial zero trend.
        prefix = values[:, :1].expand(-1, pad, -1)
        return torch.cat([prefix, values], dim=1)

    def forecast(self, histories: np.ndarray, pred_len: int) -> np.ndarray:
        values = np.asarray(histories, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError(f"TTM history must be [B,T,C], got {values.shape}.")
        context = torch.from_numpy(values).to(self.device, dtype=torch.float32)
        predictions: list[torch.Tensor] = []
        remaining = int(pred_len)
        while remaining > 0:
            model_context = self._pad_context(context)
            output = self.model(
                past_values=model_context,
                output_hidden_states=False,
                return_loss=False,
                freq_token=self.frequency_tokens(model_context.shape[0]),
            )
            point = output.prediction_outputs[:, : min(remaining, self.native_prediction_length)]
            predictions.append(point)
            remaining -= int(point.shape[1])
            if remaining > 0:
                context = torch.cat([model_context, point.detach()], dim=1)
                context = context[:, -self.model_context_length :]
        return torch.cat(predictions, dim=1).detach().float().cpu().numpy().astype(np.float32)

    def _extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 2:
            batch = batch[:, :, None]
        if batch.ndim != 3:
            raise ValueError(f"TTM history must be [B,T,C], got {tuple(batch.shape)}.")
        batch = self._pad_context(batch)
        captures: list[torch.Tensor] = []

        def capture_backbone_stage(_module, _inputs, output) -> None:
            captures.append(output[0].detach())

        def capture_decoder_stage(_module, _inputs, output) -> None:
            captures.append(output.detach())

        handles = [
            stage.register_forward_hook(capture_backbone_stage)
            for stage in self.model.backbone.encoder.mlp_mixer_encoder.mixers
        ]
        handles.extend(
            stage.register_forward_hook(capture_decoder_stage)
            for stage in self.model.decoder.decoder_block.mixers
        )
        freq_token = self.frequency_tokens(batch.shape[0])
        try:
            output = self.model(
                past_values=batch,
                output_hidden_states=False,
                return_loss=False,
                freq_token=freq_token,
            )
        finally:
            for handle in handles:
                handle.remove()
        if output.prediction_outputs.shape[1] != self.native_prediction_length:
            raise RuntimeError("TTM native prediction length changed during hidden extraction.")
        if len(captures) != self.native_stage_count:
            raise RuntimeError(
                f"Captured {len(captures)} TTM stages, expected {self.native_stage_count}."
            )

        expected_tokens = self.num_history_patches + 1
        if any(hidden.shape[-2] != expected_tokens for hidden in captures):
            shapes = [tuple(hidden.shape) for hidden in captures]
            raise RuntimeError(f"TTM common-grid stage shapes disagree: {shapes}.")
        history = [hidden[:, :, 1:] for hidden in captures]
        max_width = max(int(hidden.shape[-1]) for hidden in history)
        padded = [F.pad(hidden, (0, max_width - hidden.shape[-1])) for hidden in history]
        aggregated = [
            aggregate_ttm_channel_hidden(hidden, self.config.model.channel_aggregation)
            for hidden in padded
        ]
        return torch.stack(aggregated, dim=1)

    def extract_history_to_cache(
        self,
        history_normalized: np.ndarray,
        cache_dir: str | Path,
    ) -> dict[str, object]:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        hidden_path = cache_dir / "history_hidden.npy"
        metadata_path = cache_dir / "metadata.json"
        if hidden_path.exists() and metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        values = np.asarray(history_normalized, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError(f"TTM history must be [N,T,C], got {values.shape}.")
        n_samples, _length, n_channels = values.shape
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        memmap = None
        metadata: dict[str, object] | None = None

        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                batch = torch.from_numpy(values[start:stop]).to(self.device, dtype=torch.float32)
                hidden = self._extract_batch(batch).float().cpu().numpy()
                if memmap is None:
                    shape = (n_samples,) + hidden.shape[1:]
                    memmap = np.lib.format.open_memmap(
                        hidden_path, mode="w+", dtype=cache_dtype, shape=shape
                    )
                    base_widths = [
                        int(self.model.config.d_model)
                    ] * len(self.model.backbone.encoder.mlp_mixer_encoder.mixers)
                    base_widths.extend(
                        [int(self.model.config.decoder_d_model)]
                        * len(self.model.decoder.decoder_block.mixers)
                    )
                    metadata = {
                        "shape": list(shape),
                        "dtype": str(np.dtype(cache_dtype)),
                        "num_layers": int(shape[1]),
                        "num_patches": int(shape[2]),
                        "hidden_size": int(shape[3]),
                        "base_hidden_size": max(base_widths),
                        "base_hidden_size_by_layer": base_widths,
                        "num_channels": int(n_channels),
                        "channel_aggregation": self.config.model.channel_aggregation,
                        "patch_len": self.patch_len,
                        "patch_stride": self.patch_stride,
                        "seq_len": self.config.data.seq_len,
                        "pred_len_conditioning": None,
                        "model_context_length": self.model_context_length,
                        "history_padding": "causal_repeat_oldest_to_checkpoint_context",
                        "excluded_tokens": ["frequency_prefix"],
                        "native_stage_definition": "post common-grid adaptive block and decoder layer",
                        "frequency_token": self.frequency_token_value,
                        "model_id": self.config.model.model_id,
                        "model_revision": self.config.model.revision,
                        "adapter_protocol": "ttm-r2-common-grid-stages-v1",
                    }
                memmap[start:stop] = hidden.astype(cache_dtype, copy=False)
                memmap.flush()

        if metadata is None:
            raise RuntimeError("No TTM hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(metadata_path)
        return metadata
