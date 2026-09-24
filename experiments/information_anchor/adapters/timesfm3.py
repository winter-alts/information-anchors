from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.information_anchor.config import ExperimentConfig


def aggregate_timesfm3_channel_hidden(
    hidden: torch.Tensor,
    strategy: str = "concat_same_time_patch",
) -> torch.Tensor:
    """Map native TimesFM3 [B,C,N,D] states to temporal MI units."""
    if hidden.ndim != 4:
        raise ValueError(f"Expected [B,C,N,D], got {tuple(hidden.shape)}.")
    batch, channels, patches, width = hidden.shape
    if strategy == "concat_same_time_patch":
        return hidden.permute(0, 2, 1, 3).reshape(batch, patches, channels * width)
    mean = hidden.mean(dim=1)
    if strategy == "mean":
        return mean
    if strategy == "mean_std":
        std = hidden.float().std(dim=1, unbiased=False).to(hidden.dtype)
        return torch.cat((mean, std), dim=-1)
    raise ValueError(f"Unsupported TimesFM 3 channel aggregation: {strategy!r}.")


def _timesfm3_hidden_from_output(output: object) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 4:
        raise ValueError(
            "TimesFM 3 layer output must contain [B,C,N,D] hidden states."
        )
    return hidden


def replace_timesfm3_history_patches(
    output: object,
    donor: torch.Tensor,
    patches: tuple[int, ...],
) -> tuple[object, torch.Tensor]:
    """Replace all variates at selected temporal patches in a TimesFM3 layer."""
    hidden = _timesfm3_hidden_from_output(output)
    if donor.shape != hidden.shape:
        raise ValueError(
            f"Donor shape {tuple(donor.shape)} != hidden shape {tuple(hidden.shape)}."
        )
    indices = torch.as_tensor(patches, dtype=torch.long, device=hidden.device)
    if len(indices) == 0:
        raise ValueError("TimesFM 3 replacement requires at least one patch.")
    if int(indices.min()) < 0 or int(indices.max()) >= hidden.shape[2]:
        raise ValueError(
            f"patches={patches} outside TimesFM 3 length={hidden.shape[2]}."
        )
    donor = donor.to(device=hidden.device, dtype=hidden.dtype)
    changed = hidden.clone()
    delta = donor[:, :, indices, :] - hidden[:, :, indices, :]
    changed[:, :, indices, :] = donor[:, :, indices, :]
    rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2, 3)))
    replaced = (changed, *output[1:]) if isinstance(output, tuple) else changed
    return replaced, rms


class TimesFM3Adapter:
    """Frozen TimesFM3 decoder with native joint variate attention."""

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.device = torch.device(config.model.device)
        self.model = self._load_model()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"TimesFM 3 config patch_len={config.model.patch_len}, "
                f"checkpoint patch_len={self.patch_len}."
            )

    def _load_model(self):
        try:
            from timesfm3 import TimesFM3Torch
        except ImportError as exc:
            raise ImportError(
                "TimesFM3Adapter requires the official TimesFM3 package. "
                "Install the current google-research/timesfm source or add its "
                "src directory to PYTHONPATH."
            ) from exc

        model = TimesFM3Torch.from_pretrained(
            self.config.model.model_id,
            revision=self.config.model.revision or None,
            local_files_only=self.config.runtime.offline,
        )
        return model.to(self.device).eval()

    @property
    def patch_len(self) -> int:
        return int(self.model.input_patch_len)

    @property
    def output_patch_len(self) -> int:
        return int(self.model.output_patch_len)

    @property
    def num_layers(self) -> int:
        return len(self.model.transformer_stack.layers)

    @property
    def median_quantile_index(self) -> int:
        quantiles = np.asarray(self.model.quantiles, dtype=np.float32)
        return int(np.argmin(np.abs(quantiles - 0.5)))

    def _target_tensor(self, histories: np.ndarray) -> torch.Tensor:
        values = np.asarray(histories, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError(
                f"TimesFM 3 history must be [B,T,C], got {values.shape}."
            )
        return torch.from_numpy(values).to(self.device).permute(0, 2, 1)

    def _decode(
        self,
        histories: np.ndarray,
        pred_len: int,
        *,
        return_aux_outputs: bool = False,
    ) -> Any:
        return self.model.decode(
            target=self._target_tensor(histories),
            horizon=pred_len,
            return_aux_outputs=return_aux_outputs,
        )

    def forecast(self, histories: np.ndarray, pred_len: int) -> np.ndarray:
        logits = self._decode(histories, pred_len)
        if isinstance(logits, tuple):
            logits = logits[0]
        point = logits[..., self.median_quantile_index]
        return (
            point.permute(0, 2, 1)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _capture_batch(
        self,
        batch: np.ndarray,
        pred_len: int,
    ) -> torch.Tensor:
        captures: list[torch.Tensor | None] = [None] * self.num_layers
        handles = []
        for layer_index, layer in enumerate(self.model.transformer_stack.layers):

            def capture_hook(_module, _inputs, output, *, index=layer_index):
                if captures[index] is None:
                    captures[index] = _timesfm3_hidden_from_output(output).detach()
                return output

            handles.append(layer.register_forward_hook(capture_hook))
        try:
            self._decode(batch, pred_len)
        finally:
            for handle in handles:
                handle.remove()
        if any(item is None for item in captures):
            raise RuntimeError("TimesFM 3 did not emit one prefill state per layer.")

        context = batch.shape[1]
        context_patches = math.ceil(context / self.patch_len)
        layers = []
        for item in captures:
            if item is None:  # pragma: no cover - guarded above
                raise RuntimeError("Missing TimesFM 3 layer capture.")
            context_hidden = item[:, :, :context_patches, :]
            layers.append(
                aggregate_timesfm3_channel_hidden(
                    context_hidden,
                    self.config.model.channel_aggregation,
                )
            )
        return torch.stack(layers, dim=1)

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
            raise ValueError(
                f"TimesFM 3 history must be [N,T,C], got {values.shape}."
            )
        n_samples, length, n_channels = values.shape
        cache_dtype = (
            np.float16
            if self.config.model.cache_dtype == "float16"
            else np.float32
        )
        hidden_memmap = None
        metadata: dict[str, object] | None = None
        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                hidden = self._capture_batch(
                    values[start:stop],
                    self.config.data.pred_len,
                ).detach().cpu().numpy()
                if hidden_memmap is None:
                    shape = (n_samples,) + hidden.shape[1:]
                    hidden_memmap = np.lib.format.open_memmap(
                        hidden_path,
                        mode="w+",
                        dtype=cache_dtype,
                        shape=shape,
                    )
                    metadata = {
                        "shape": list(shape),
                        "dtype": str(np.dtype(cache_dtype)),
                        "num_layers": int(shape[1]),
                        "num_patches": int(shape[2]),
                        "hidden_size": int(shape[3]),
                        "base_hidden_size": int(self.model.transformer_config.transformer.model_dims),
                        "num_channels": int(n_channels),
                        "channel_aggregation": self.config.model.channel_aggregation,
                        "patch_len": self.patch_len,
                        "patch_stride": self.patch_len,
                        "seq_len": int(length),
                        "output_patch_len": self.output_patch_len,
                        "model_id": self.config.model.model_id,
                        "adapter_protocol": "timesfm3-prefill-v1",
                        "native_representation": "joint_variate_attention_prefill",
                    }
                hidden_memmap[start:stop] = hidden.astype(cache_dtype, copy=False)
                hidden_memmap.flush()
        if metadata is None:
            raise RuntimeError("No TimesFM 3 hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
        return metadata
