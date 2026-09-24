from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.information_anchor.config import ExperimentConfig


class TimesFMAdapter:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from models.TimesFM import Model

        model_config = SimpleNamespace(
            task_name="zero_shot_forecast",
            seq_len=config.data.seq_len,
            pred_len=config.data.pred_len,
        )
        self.device = torch.device(config.model.device)
        self.model = Model(model_config).to(self.device).eval()

    @property
    def patch_len(self) -> int:
        return int(self.model.patch_len)

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
        if history_normalized.ndim == 2:
            history_normalized = history_normalized[:, :, None]
        if history_normalized.ndim != 3:
            raise ValueError(
                f"TimesFM history must be [samples, time] or [samples, time, channels], got {history_normalized.shape}"
            )
        n_samples, _seq_len, n_channels = history_normalized.shape
        batch_size = self.config.runtime.batch_size
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        hidden_memmap = None
        metadata: dict[str, object] | None = None

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                stop = min(n_samples, start + batch_size)
                batch_np = history_normalized[start:stop]
                channel_stacks = []
                for channel_index in range(n_channels):
                    batch = torch.from_numpy(batch_np[:, :, channel_index]).to(
                        device=self.device,
                        dtype=torch.float32,
                    )
                    layer_hidden, _ = self.model.extract_layer_tokens(batch)
                    channel_stacks.append(torch.stack(layer_hidden, dim=1))
                stacked = torch.cat(channel_stacks, dim=-1).detach().cpu().numpy()
                if hidden_memmap is None:
                    shape = (n_samples,) + stacked.shape[1:]
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
                        "base_hidden_size": int(shape[3] // n_channels),
                        "num_channels": int(n_channels),
                        "channel_aggregation": "concat_same_time_patch",
                        "patch_len": self.patch_len,
                        "seq_len": self.config.data.seq_len,
                        "model_id": self.config.model.model_id,
                    }
                hidden_memmap[start:stop] = stacked.astype(cache_dtype, copy=False)
                hidden_memmap.flush()

        if metadata is None:
            raise RuntimeError("No hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        temporary.replace(metadata_path)
        return metadata
