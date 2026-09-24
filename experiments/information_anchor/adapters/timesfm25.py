from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.information_anchor.config import ExperimentConfig


def aggregate_timesfm25_channel_hidden(
    hidden: torch.Tensor,
    strategy: str = "concat_same_time_patch",
) -> torch.Tensor:
    """Map channel-independent [B,C,N,D] states to a shared temporal grid."""
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
    raise ValueError(f"Unsupported TimesFM 2.5 channel aggregation: {strategy!r}.")


def replace_timesfm25_history_patches(
    output: object,
    donor: torch.Tensor,
    patches: tuple[int, ...],
) -> tuple[object, torch.Tensor]:
    """Replace prefill patch states and return per-series perturbation RMS."""
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("TimesFM 2.5 layer output must contain [B*C,N,D] hidden states.")
    if donor.shape != hidden.shape:
        raise ValueError(f"Donor shape {tuple(donor.shape)} != hidden shape {tuple(hidden.shape)}.")
    indices = torch.as_tensor(patches, dtype=torch.long, device=hidden.device)
    if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= hidden.shape[1]:
        raise ValueError(f"patches={patches} outside TimesFM 2.5 length={hidden.shape[1]}.")
    donor = donor.to(device=hidden.device, dtype=hidden.dtype)
    changed = hidden.clone()
    delta = donor[:, indices] - hidden[:, indices]
    changed[:, indices] = donor[:, indices]
    rms = torch.sqrt(torch.mean(delta.float().square(), dim=(1, 2)))
    replaced = (changed, *output[1:]) if isinstance(output, tuple) else changed
    return replaced, rms


class TimesFM25Adapter:
    """Frozen TimesFM 2.5 decoder with native channel-independent patch states."""

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.device = torch.device(config.model.device)
        self.wrapper, self.model = self._load_model()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"TimesFM 2.5 config patch_len={config.model.patch_len}, "
                f"checkpoint patch_len={self.patch_len}."
            )

    def _load_model(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from timesfm import TimesFM_2p5_200M_torch

        model_id = self.config.model.model_id
        checkpoint_root = Path(model_id).expanduser()
        if checkpoint_root.is_dir():
            checkpoint = checkpoint_root / "model.safetensors"
        else:
            checkpoint = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=self.config.model.revision or None,
                    local_files_only=self.config.runtime.offline,
                )
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        # The upstream loader assigns cuda:0 internally. Loading the state dict
        # explicitly keeps device placement under the experiment configuration.
        wrapper = TimesFM_2p5_200M_torch(torch_compile=False)
        state_dict = load_file(str(checkpoint), device="cpu")
        wrapper.model.load_state_dict(state_dict, strict=True)
        del state_dict
        model = wrapper.model.to(self.device).eval()
        model.device = self.device
        model.device_count = 1
        return wrapper, model

    @property
    def patch_len(self) -> int:
        return int(self.model.p)

    @property
    def output_patch_len(self) -> int:
        return int(self.model.o)

    @property
    def num_layers(self) -> int:
        return len(self.model.stacked_xf)

    def prepare_histories(
        self, histories: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        values = np.asarray(histories, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError(f"TimesFM 2.5 history must be [B,T,C], got {values.shape}.")
        batch, length, channels = values.shape
        flat = torch.from_numpy(values).to(self.device).permute(0, 2, 1).reshape(
            batch * channels, length
        )
        mask = torch.zeros_like(flat, dtype=torch.bool)
        left_pad = (-length) % self.patch_len
        if left_pad:
            flat = torch.nn.functional.pad(flat, (left_pad, 0), value=0.0)
            mask = torch.nn.functional.pad(mask, (left_pad, 0), value=True)
        return flat, mask, batch, channels

    def forecast(self, histories: np.ndarray, pred_len: int) -> np.ndarray:
        inputs, masks, batch, channels = self.prepare_histories(histories)
        prefill, _quantile_spread, autoregressive = self.model.decode(
            pred_len, inputs, masks
        )
        pieces = [prefill[:, -1]]
        if autoregressive is not None:
            pieces.append(autoregressive.reshape(batch * channels, -1, self.model.q))
        quantiles = torch.cat(pieces, dim=1)[:, :pred_len]
        point = quantiles[..., int(self.model.aridx)]
        return (
            point.reshape(batch, channels, pred_len)
            .permute(0, 2, 1)
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _extract_batch(self, batch: np.ndarray) -> torch.Tensor:
        captures: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        handles = []
        for layer_index, layer in enumerate(self.model.stacked_xf):
            def capture_hook(_module, _inputs, output, *, index=layer_index):
                # Only the first call is the history prefill; later calls are
                # autoregressive output patches and are not atlas history units.
                if not captures[index]:
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[index].append(hidden.detach())
                return output

            handles.append(layer.register_forward_hook(capture_hook))
        try:
            self.forecast(batch, self.config.data.pred_len)
        finally:
            for handle in handles:
                handle.remove()
        if any(len(layer_captures) != 1 for layer_captures in captures):
            raise RuntimeError("TimesFM 2.5 did not emit one prefill state per layer.")

        values = np.asarray(batch)
        batch_size = values.shape[0]
        channels = 1 if values.ndim == 2 else values.shape[-1]
        layers = []
        for layer_captures in captures:
            hidden = layer_captures[0]
            hidden = hidden.reshape(batch_size, channels, hidden.shape[1], hidden.shape[2])
            layers.append(
                aggregate_timesfm25_channel_hidden(
                    hidden, self.config.model.channel_aggregation
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
            raise ValueError(f"TimesFM 2.5 history must be [N,T,C], got {values.shape}.")
        n_samples, _length, n_channels = values.shape
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        hidden_memmap = None
        metadata: dict[str, object] | None = None
        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                hidden = self._extract_batch(values[start:stop]).detach().cpu().numpy()
                if hidden_memmap is None:
                    shape = (n_samples,) + hidden.shape[1:]
                    hidden_memmap = np.lib.format.open_memmap(
                        hidden_path, mode="w+", dtype=cache_dtype, shape=shape
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
                        "seq_len": self.config.data.seq_len,
                        "model_id": self.config.model.model_id,
                        "adapter_protocol": "timesfm25-prefill-v1",
                    }
                hidden_memmap[start:stop] = hidden.astype(cache_dtype, copy=False)
                hidden_memmap.flush()
        if metadata is None:
            raise RuntimeError("No TimesFM 2.5 hidden states were extracted.")
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(metadata_path)
        return metadata
