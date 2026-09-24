from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.information_anchor.adapters.channel_aggregation import (
    aggregate_channel_hidden,
)
from experiments.information_anchor.config import ExperimentConfig


class Moirai2Adapter:
    """Extract frozen Moirai-2 post-block history-patch representations.

    Moirai-2 uses packed causal attention. Context tokens cannot attend to
    later prediction tokens, so history-only states are reusable across P.
    """

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from uni2ts.model.moirai2 import Moirai2Module

        self.device = torch.device(config.model.device)
        self.model = Moirai2Module.from_pretrained(
            config.model.model_id,
            revision=config.model.revision or None,
        ).to(self.device).eval()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"Moirai2 config patch_len={config.model.patch_len}, checkpoint patch_size={self.patch_len}"
            )
        num_context_tokens = (config.data.seq_len + self.patch_len - 1) // self.patch_len
        if num_context_tokens > int(self.model.max_seq_len):
            raise ValueError("seq_len exceeds the Moirai2 checkpoint token limit.")

    @property
    def patch_len(self) -> int:
        return int(self.model.patch_size)

    @property
    def patch_stride(self) -> int:
        return self.patch_len

    def _prepare_history_tokens(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.ndim == 2:
            batch = batch[:, :, None]
        if batch.ndim != 3:
            raise ValueError(f"Moirai2 history must be [batch, time] or [batch, time, channels], got {batch.shape}")

        batch_size, seq_len, n_channels = batch.shape
        pad = (-seq_len) % self.patch_len
        if pad:
            padded = F.pad(batch, (0, 0, pad, 0), value=0.0)
            observed_time = torch.cat(
                [
                    torch.zeros(batch_size, pad, dtype=torch.bool, device=batch.device),
                    torch.ones(batch_size, seq_len, dtype=torch.bool, device=batch.device),
                ],
                dim=1,
            )
        else:
            padded = batch
            observed_time = torch.ones(batch_size, seq_len, dtype=torch.bool, device=batch.device)

        num_patches = padded.shape[1] // self.patch_len
        total_tokens = n_channels * num_patches
        if total_tokens > int(self.model.max_seq_len):
            raise ValueError(
                f"Moirai2 context uses {total_tokens} tokens ({n_channels} channels x {num_patches} patches), "
                f"exceeding max_seq_len={int(self.model.max_seq_len)}."
            )

        target = (
            padded.reshape(batch_size, num_patches, self.patch_len, n_channels)
            .permute(0, 3, 1, 2)
            .reshape(batch_size, total_tokens, self.patch_len)
        )
        observed_mask = (
            observed_time.reshape(batch_size, num_patches, self.patch_len)[:, None, :, :]
            .expand(-1, n_channels, -1, -1)
            .reshape(batch_size, total_tokens, self.patch_len)
        )
        valid_patch = observed_mask.any(dim=-1).to(torch.long)
        sample_id = valid_patch
        time_id = (
            torch.arange(num_patches, dtype=torch.long, device=batch.device)
            .repeat(n_channels)
            .expand(batch_size, -1)
        )
        variate_id = (
            torch.arange(n_channels, dtype=torch.long, device=batch.device)
            .repeat_interleave(num_patches)
            .expand(batch_size, -1)
        )
        prediction_mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=batch.device)
        return target, observed_mask, sample_id, time_id, variate_id, prediction_mask

    def _extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        final_state: list[torch.Tensor] = []

        def capture_post_block(_module, _inputs, output) -> None:
            captured.append(output.detach())

        def capture_final_encoder(_module, _inputs, output) -> None:
            final_state.append(output.detach())

        layer_hooks = [layer.register_forward_hook(capture_post_block) for layer in self.model.encoder.layers]
        encoder_hook = self.model.encoder.register_forward_hook(capture_final_encoder)
        try:
            self.model(*self._prepare_history_tokens(batch), training_mode=False)
        finally:
            for hook in layer_hooks:
                hook.remove()
            encoder_hook.remove()

        expected_layers = len(self.model.encoder.layers)
        if len(captured) != expected_layers or len(final_state) != 1:
            raise RuntimeError(
                f"Captured {len(captured)} layer states and {len(final_state)} final states; "
                f"expected {expected_layers} and 1."
            )
        captured[-1] = final_state[0]
        stacked = torch.stack(captured, dim=1)
        n_channels = 1 if batch.ndim == 2 else int(batch.shape[-1])
        if n_channels == 1:
            return stacked
        batch_size, _n_layers, total_tokens, _hidden_size = stacked.shape
        if total_tokens % n_channels != 0:
            raise RuntimeError(
                f"Cannot regroup {total_tokens} Moirai2 tokens into {n_channels} channels."
            )
        num_patches = total_tokens // n_channels
        channel_major = (
            stacked.reshape(batch_size, stacked.shape[1], n_channels, num_patches, stacked.shape[-1])
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size * n_channels, stacked.shape[1], num_patches, stacked.shape[-1])
        )
        return aggregate_channel_hidden(
            channel_major,
            batch_size=batch_size,
            n_channels=n_channels,
            strategy=self.config.model.channel_aggregation,
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
            with metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        history_normalized = np.asarray(history_normalized, dtype=np.float32)
        if history_normalized.ndim not in {2, 3}:
            raise ValueError(
                f"Moirai2 history must be [samples, time] or [samples, time, channels], got {history_normalized.shape}"
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
                    dtype=torch.float32,
                )
                stacked = self._extract_batch(batch)
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
                        "pred_len_conditioning": None,
                        "num_output_query_patches": 0,
                        "excluded_tokens": [],
                        "causal_history_only": True,
                        "model_id": self.config.model.model_id,
                        "model_revision": self.config.model.revision,
                    }
                hidden_memmap[start:stop] = stacked_np.astype(cache_dtype, copy=False)
                hidden_memmap.flush()

        if metadata is None:
            raise RuntimeError("No Moirai2 hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        temporary.replace(metadata_path)
        return metadata
