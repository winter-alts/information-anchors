from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments.information_anchor.adapters.channel_aggregation import aggregate_channel_hidden
from experiments.information_anchor.config import ExperimentConfig


class ChronosBoltAdapter:
    """Extract encoder history-patch states from frozen Chronos-Bolt."""

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from chronos import BaseChronosPipeline

        self.device = torch.device(config.model.device)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            config.model.model_id,
            device_map={"": str(self.device)},
            local_files_only=config.runtime.offline,
            revision=config.model.revision or None,
        )
        self.model = self.pipeline.model.eval()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"Chronos-Bolt config patch_len={config.model.patch_len}, "
                f"checkpoint input_patch_size={self.patch_len}."
            )
        if config.data.seq_len > int(self.model.chronos_config.context_length):
            raise ValueError("seq_len exceeds the Chronos-Bolt context length.")

    @property
    def patch_len(self) -> int:
        return int(self.model.chronos_config.input_patch_size)

    @property
    def patch_stride(self) -> int:
        return int(self.model.chronos_config.input_patch_stride)

    @property
    def num_history_patches(self) -> int:
        length = self.config.data.seq_len
        return max(1, math.ceil((length - self.patch_len) / self.patch_stride) + 1)

    def _extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 3:
            batch_size, seq_len, n_channels = batch.shape
            context = batch.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len)
        elif batch.ndim == 2:
            batch_size, _seq_len = batch.shape
            n_channels = 1
            context = batch
        else:
            raise ValueError(
                "Chronos-Bolt history must be [B,T] or [B,T,C], "
                f"got {tuple(batch.shape)}."
            )

        captured: list[torch.Tensor] = []

        def capture_post_block(_module, _inputs, output) -> None:
            captured.append(output[0].detach())

        hooks = [block.register_forward_hook(capture_post_block) for block in self.model.encoder.block]
        try:
            final_hidden, _loc_scale, input_embeds, _attention_mask = self.model.encode(context)
        finally:
            for hook in hooks:
                hook.remove()

        if len(captured) != len(self.model.encoder.block):
            raise RuntimeError(
                f"Captured {len(captured)} Chronos-Bolt blocks, "
                f"expected {len(self.model.encoder.block)}."
            )
        expected_tokens = self.num_history_patches + int(self.model.chronos_config.use_reg_token)
        if input_embeds.shape[-2] != expected_tokens:
            raise RuntimeError(
                f"Chronos-Bolt emitted {input_embeds.shape[-2]} tokens, expected {expected_tokens}."
            )

        captured[-1] = final_hidden.detach()
        stacked = torch.stack(
            [hidden[:, : self.num_history_patches] for hidden in captured],
            dim=1,
        )
        return aggregate_channel_hidden(
            stacked,
            batch_size=batch_size,
            n_channels=n_channels,
            strategy=self.config.model.channel_aggregation,
        )

    def forecast(self, histories: np.ndarray, pred_len: int) -> np.ndarray:
        values = np.asarray(histories, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError(f"Chronos-Bolt history must be [B,T,C], got {values.shape}.")
        batch, length, channels = values.shape
        context = (
            torch.from_numpy(values)
            .to(self.device, dtype=self.model.dtype)
            .permute(0, 2, 1)
            .reshape(batch * channels, length)
        )
        quantiles = self.pipeline.predict(
            context,
            prediction_length=pred_len,
            limit_prediction_length=False,
        )
        median = int(torch.argmin(torch.abs(self.model.quantiles.float() - 0.5)).item())
        point = quantiles[:, median, :pred_len]
        return (
            point.reshape(batch, channels, pred_len)
            .permute(0, 2, 1)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

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
        if values.ndim not in {2, 3}:
            raise ValueError(f"Chronos-Bolt history must be [N,T] or [N,T,C], got {values.shape}.")
        n_samples = values.shape[0]
        n_channels = 1 if values.ndim == 2 else int(values.shape[-1])
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        memmap = None
        metadata: dict[str, object] | None = None

        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                batch = torch.from_numpy(values[start:stop]).to(
                    self.device, dtype=self.model.dtype
                )
                hidden = self._extract_batch(batch).float().cpu().numpy()
                if memmap is None:
                    shape = (n_samples,) + hidden.shape[1:]
                    memmap = np.lib.format.open_memmap(
                        hidden_path, mode="w+", dtype=cache_dtype, shape=shape
                    )
                    metadata = {
                        "shape": list(shape),
                        "dtype": str(np.dtype(cache_dtype)),
                        "num_layers": int(shape[1]),
                        "num_patches": int(shape[2]),
                        "hidden_size": int(shape[3]),
                        "base_hidden_size": int(self.model.config.d_model),
                        "num_channels": n_channels,
                        "channel_aggregation": self.config.model.channel_aggregation,
                        "patch_len": self.patch_len,
                        "patch_stride": self.patch_stride,
                        "seq_len": self.config.data.seq_len,
                        "pred_len_conditioning": None,
                        "excluded_tokens": ["reg"],
                        "final_layer_post_norm": True,
                        "model_id": self.config.model.model_id,
                        "model_revision": self.config.model.revision,
                        "adapter_protocol": "chronos-bolt-encoder-history-v1",
                    }
                memmap[start:stop] = hidden.astype(cache_dtype, copy=False)
                memmap.flush()

        if metadata is None:
            raise RuntimeError("No Chronos-Bolt hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(metadata_path)
        return metadata

    def extract_projected_history_to_cache(
        self,
        history_normalized: np.ndarray,
        cache_dir: str | Path,
        projection,
        layer: int,
    ) -> dict[str, object]:
        """Cache only one projected token layer for large MI query shards.

        The regular cache is intentionally lossless for activation audits, but
        an official MI sidecar only consumes one frozen layer after the locked
        discovery PCA.  Projecting before writing avoids multi-gigabyte raw
        hidden intermediates on electricity while preserving the exact
        projection/normalisation used by ``_project_tokens``.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        token_path = cache_dir / "projected_tokens.npy"
        metadata_path = cache_dir / "metadata.json"
        if token_path.exists() and metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.asarray(history_normalized, dtype=np.float32)
        if values.ndim not in {2, 3}:
            raise ValueError(f"Chronos-Bolt history must be [N,T] or [N,T,C], got {values.shape}.")
        n_samples = values.shape[0]
        n_channels = 1 if values.ndim == 2 else int(values.shape[-1])
        if layer < 0:
            raise ValueError("projected cache layer must be non-negative")
        memmap = None
        metadata: dict[str, object] | None = None
        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                batch = torch.from_numpy(values[start:stop]).to(
                    self.device, dtype=self.model.dtype
                )
                hidden = self._extract_batch(batch).float().cpu().numpy()
                if layer >= hidden.shape[1]:
                    raise ValueError(f"layer={layer} exceeds hidden layers={hidden.shape[1]}")
                selected = hidden[:, layer]
                projected = projection.transform(
                    selected.reshape(-1, selected.shape[-1])
                ).reshape(selected.shape[0], selected.shape[1], -1)
                projected /= np.maximum(
                    np.linalg.norm(projected, axis=2, keepdims=True), 1e-8
                )
                projected = projected.astype(np.float32, copy=False)
                if memmap is None:
                    shape = (n_samples,) + projected.shape[1:]
                    memmap = np.lib.format.open_memmap(
                        token_path, mode="w+", dtype=np.float32, shape=shape
                    )
                    metadata = {
                        "shape": list(shape),
                        "dtype": "float32",
                        "layer": int(layer),
                        "num_patches": int(shape[1]),
                        "projected_dim": int(shape[2]),
                        "num_channels": n_channels,
                        "seq_len": self.config.data.seq_len,
                        "pred_len_conditioning": None,
                        "projection_cache": True,
                        "adapter_protocol": "chronos-bolt-encoder-history-projected-v1",
                    }
                memmap[start:stop] = projected
                memmap.flush()
        if metadata is None:
            raise RuntimeError("No projected Chronos-Bolt tokens were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
        return metadata
