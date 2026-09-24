from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments.information_anchor.adapters.channel_aggregation import aggregate_channel_hidden
from experiments.information_anchor.config import ExperimentConfig


class Chronos2Adapter:
    """Extract post-block history-patch states from the frozen Chronos-2 encoder.

    Chronos-2 jointly encodes context patches, an optional [REG] token, and masked
    future query patches.  Only the context-token slice is persisted for the
    primary history-to-future MI analysis.  The requested prediction length must
    therefore be part of the cache key even though future target values are never
    passed to the model.
    """

    representation_depends_on_pred_len = True

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
                f"Chronos2 config patch_len={config.model.patch_len}, "
                f"checkpoint input_patch_size={self.patch_len}"
            )
        if config.data.seq_len > int(self.model.chronos_config.context_length):
            raise ValueError("seq_len exceeds the Chronos2 checkpoint context length.")
        if self.num_output_patches > int(self.model.chronos_config.max_output_patches):
            raise ValueError("pred_len exceeds the checkpoint's native output-patch budget.")

    @property
    def patch_len(self) -> int:
        return int(self.model.chronos_config.input_patch_size)

    @property
    def patch_stride(self) -> int:
        return int(self.model.chronos_config.input_patch_stride)

    @property
    def num_output_patches(self) -> int:
        output_patch_size = int(self.model.chronos_config.output_patch_size)
        return math.ceil(self.config.data.pred_len / output_patch_size)

    def _extract_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, int]:
        if batch.ndim == 3:
            batch_size, seq_len, n_channels = batch.shape
            context = batch.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len)
            context_mask = torch.ones_like(context)
            group_ids = (
                torch.arange(batch_size, dtype=torch.long, device=batch.device)
                .repeat_interleave(n_channels)
            )
        elif batch.ndim == 2:
            batch_size = batch.shape[0]
            n_channels = 1
            context = batch
            context_mask = torch.ones_like(batch)
            group_ids = None
        else:
            raise ValueError(f"Chronos2 history must be [batch, time] or [batch, time, channels], got {batch.shape}")
        captured: list[torch.Tensor] = []

        def capture_post_block(_module, _inputs, output) -> None:
            captured.append(output[0].detach())

        hooks = [block.register_forward_hook(capture_post_block) for block in self.model.encoder.block]
        try:
            encoder_outputs, _loc_scale, _future_mask, num_context_patches = self.model.encode(
                context=context,
                context_mask=context_mask,
                group_ids=group_ids,
                future_covariates=None,
                future_covariates_mask=None,
                num_output_patches=self.num_output_patches,
                future_target=None,
                future_target_mask=None,
                output_attentions=False,
            )
        finally:
            for hook in hooks:
                hook.remove()

        expected_layers = len(self.model.encoder.block)
        if len(captured) != expected_layers:
            raise RuntimeError(f"Captured {len(captured)} Chronos2 blocks, expected {expected_layers}.")

        # Hooks capture raw post-block states.  Apply the checkpoint's final norm
        # only to the last block so that it exactly matches the representation
        # consumed by the forecast head; earlier entries remain true post-blocks.
        final_hidden = encoder_outputs.last_hidden_state
        captured[-1] = final_hidden
        history_layers = [hidden[:, :num_context_patches] for hidden in captured]
        stacked = torch.stack(history_layers, dim=1)
        if n_channels > 1:
            stacked = aggregate_channel_hidden(
                stacked,
                batch_size=batch_size,
                n_channels=n_channels,
                strategy=self.config.model.channel_aggregation,
            )
        return stacked, int(num_context_patches)

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
            with metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        history_normalized = np.asarray(history_normalized, dtype=np.float32)
        if history_normalized.ndim not in {2, 3}:
            raise ValueError(
                f"Chronos2 history must be [samples, time] or [samples, time, channels], got {history_normalized.shape}"
            )
        n_samples = history_normalized.shape[0]
        n_channels = 1 if history_normalized.ndim == 2 else int(history_normalized.shape[-1])
        batch_size = self.config.runtime.batch_size
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        hidden_memmap = None
        metadata: dict[str, object] | None = None

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                stop = min(n_samples, start + batch_size)
                batch = torch.from_numpy(history_normalized[start:stop]).to(
                    device=self.device,
                    dtype=self.model.dtype,
                )
                stacked, num_context_patches = self._extract_batch(batch)
                stacked_np = stacked.float().cpu().numpy()
                if hidden_memmap is None:
                    shape = (n_samples,) + stacked_np.shape[1:]
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
                        "base_hidden_size": int(
                            shape[3]
                            // (
                                n_channels
                                if self.config.model.channel_aggregation
                                == "concat_same_time_patch"
                                else 2
                                if self.config.model.channel_aggregation == "mean_std"
                                else 1
                            )
                        ),
                        "num_channels": int(n_channels),
                        "channel_aggregation": self.config.model.channel_aggregation,
                        "patch_len": self.patch_len,
                        "patch_stride": self.patch_stride,
                        "seq_len": self.config.data.seq_len,
                        "pred_len_conditioning": self.config.data.pred_len,
                        "num_output_query_patches": self.num_output_patches,
                        "num_context_patches": num_context_patches,
                        "excluded_tokens": ["reg", "future_query"],
                        "model_id": self.config.model.model_id,
                        "model_revision": self.config.model.revision,
                    }
                hidden_memmap[start:stop] = stacked_np.astype(cache_dtype, copy=False)
                hidden_memmap.flush()

        if metadata is None:
            raise RuntimeError("No Chronos2 hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        temporary.replace(metadata_path)
        return metadata
