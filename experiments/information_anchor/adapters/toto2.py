from __future__ import annotations

import importlib
import json
import os
import sys
import typing
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from experiments.information_anchor.config import ExperimentConfig


def _load_toto2_class():
    """加载 Toto2 模型类，并允许通过 TOTO2_PATH 指向本地源码包。"""
    # Toto2 imports NotRequired from typing, which is only available in Python 3.11+.
    # Keep the compatibility shim local to the third-party import path.
    if not hasattr(typing, "NotRequired"):
        from typing_extensions import NotRequired

        typing.NotRequired = NotRequired

    candidates = []
    configured = os.environ.get("TOTO2_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "toto2")

    # Toto2 的源码包和 dd_unit_scaling 位于同一个仓库的两个子目录，
    # 先一次性加入所有可能的父目录，避免导入到模型后才发现依赖路径缺失。
    search_roots: list[Path] = []
    for candidate in candidates:
        search_roots.extend((candidate, candidate / "src", candidate.parent))
        if candidate.name == "toto2":
            search_roots.append(candidate.parent.parent)
            # dd-unit-scaling 项目是独立的源码包，包根目录不是 Toto 仓库根目录。
            search_roots.append(candidate.parent / "dd_unit_scaling")
    for root in search_roots:
        if not root.exists():
            continue
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    try:
        module = importlib.import_module("toto2")
    except ModuleNotFoundError as error:
        if error.name != "toto2":
            raise
        searched = ", ".join(str(path) for path in candidates)
        raise ImportError(f"Toto2 package not found; searched: {searched}") from error
    try:
        return module.Toto2Model
    except AttributeError as error:
        raise ImportError("The Toto2 package does not expose Toto2Model.") from error


def aggregate_toto_channel_hidden(
    hidden: torch.Tensor,
    channel_aggregation: str,
) -> torch.Tensor:
    """把 [B,C,N,D] 的变量 token 映射到 [B,N,C*D] 或统计聚合表示。"""
    if hidden.ndim != 4:
        raise ValueError(f"Expected Toto2 hidden [B,C,N,D], got {tuple(hidden.shape)}")
    if channel_aggregation == "concat_same_time_patch":
        batch, channels, patches, width = hidden.shape
        return hidden.permute(0, 2, 1, 3).reshape(batch, patches, channels * width)
    if channel_aggregation == "mean":
        return hidden.mean(dim=1)
    if channel_aggregation == "mean_std":
        return torch.cat([hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)], dim=-1)
    raise ValueError(f"Unsupported Toto2 channel aggregation: {channel_aggregation}")


class Toto2Adapter:
    """提取冻结 Toto2 decoder 的多变量历史 patch hidden。"""

    representation_depends_on_pred_len = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.runtime.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        Toto2Model = _load_toto2_class()
        self.device = torch.device(config.model.device)
        self.model = Toto2Model.from_pretrained(config.model.model_id, map_location="cpu")
        self.model = self.model.to(self.device).eval()
        if self.patch_len != config.model.patch_len:
            raise ValueError(
                f"Toto2 config patch_len={config.model.patch_len}, checkpoint patch_size={self.patch_len}"
            )

    @property
    def patch_len(self) -> int:
        return int(self.model.config.patch_size)

    @property
    def patch_stride(self) -> int:
        return self.patch_len

    def _embed_history(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """完成 Toto2 的缩放、patch embedding 和缺失 patch 标记。"""
        if batch.ndim == 2:
            batch = batch[:, :, None]
        if batch.ndim != 3:
            raise ValueError(f"Toto2 history must be [B,T,C], got {tuple(batch.shape)}")
        target = batch.permute(0, 2, 1).contiguous()
        pad_len = (-target.shape[-1]) % self.patch_len
        if pad_len:
            target = torch.nn.functional.pad(target, (0, pad_len), value=0.0)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        if pad_len:
            target_mask[..., -pad_len:] = False
        scaled, _, _ = self.model.scaler(target, target_mask)
        state = self.model._embed_patches(scaled.asinh(), target_mask, self.patch_len)
        n_channels = target.shape[1]
        observed = target_mask.unflatten(-1, (-1, self.patch_len)).any(dim=-1)
        group_ids = torch.zeros(
            target.shape[0], n_channels, observed.shape[-1], dtype=torch.long, device=target.device
        )
        group_ids[~observed] = -1
        return state, group_ids, pad_len

    def _extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """运行 Toto2 decoder，并将每层 hidden 恢复成时间 patch 轴。"""
        state, group_ids, _pad_len = self._embed_history(batch)
        n_channels, n_patches = state.shape[-3], state.shape[-2]
        time_ids = torch.arange(n_patches, device=state.device, dtype=torch.int32)
        time_kwargs, var_kwargs = self.model.transformer._sdpa_kwargs(
            state, time_ids, group_ids, has_missing_values=True
        )
        leading = state.shape[:-2]
        state = rearrange(state, "... seq_len dim -> (...) seq_len dim")
        layer_tokens: list[torch.Tensor] = []
        for index, layer in enumerate(self.model.transformer.layers):
            if self.model.transformer._if_variate_layer(index):
                state = rearrange(state, "(b n) s d -> (b s) n d", n=n_channels)
                state = layer(state, **var_kwargs)
                state = rearrange(state, "(b s) n d -> (b n) s d", s=n_patches)
            else:
                state = layer(state, seq_ids=time_ids, **time_kwargs)
            state_view = state.unflatten(0, leading)
            layer_tokens.append(
                aggregate_toto_channel_hidden(
                    state_view,
                    self.config.model.channel_aggregation,
                ).detach()
            )
        # 预测头接收 transformer.out_norm 的输出；最后一层采用同一后归一化表示，
        # 这样与 Chronos2/Moirai2 的“最后层对齐预测头”协议一致。
        normalized_state = self.model.transformer.out_norm(state)
        normalized_view = normalized_state.unflatten(0, leading)
        layer_tokens[-1] = aggregate_toto_channel_hidden(
            normalized_view,
            self.config.model.channel_aggregation,
        ).detach()
        return torch.stack(layer_tokens, dim=1)

    def extract_history_to_cache(
        self,
        history_normalized: np.ndarray,
        cache_dir: str | Path,
    ) -> dict[str, object]:
        """批量抽取并写入 memmap，保存可复现的 token 元数据。"""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        hidden_path = cache_dir / "history_hidden.npy"
        metadata_path = cache_dir / "metadata.json"
        if hidden_path.exists() and metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        values = np.asarray(history_normalized, dtype=np.float32)
        if values.ndim not in {2, 3}:
            raise ValueError(f"Toto2 history must be [N,T] or [N,T,C], got {values.shape}")
        n_samples = values.shape[0]
        n_channels = 1 if values.ndim == 2 else values.shape[-1]
        cache_dtype = np.float16 if self.config.model.cache_dtype == "float16" else np.float32
        memmap = None
        metadata = None
        with torch.no_grad():
            for start in range(0, n_samples, self.config.runtime.batch_size):
                stop = min(n_samples, start + self.config.runtime.batch_size)
                batch = torch.from_numpy(values[start:stop]).to(self.device, dtype=torch.float32)
                extracted = self._extract_batch(batch).float().cpu().numpy()
                if memmap is None:
                    shape = (n_samples,) + extracted.shape[1:]
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
                        "num_channels": int(n_channels),
                        "channel_aggregation": self.config.model.channel_aggregation,
                        "patch_len": self.patch_len,
                        "patch_stride": self.patch_stride,
                        "seq_len": self.config.data.seq_len,
                        "pred_len_conditioning": None,
                        "causal_history_only": True,
                        "final_layer_post_norm": True,
                        "model_id": self.config.model.model_id,
                        "model_revision": self.config.model.revision,
                    }
                memmap[start:stop] = extracted.astype(cache_dtype, copy=False)
                memmap.flush()
        if metadata is None:
            raise RuntimeError("No Toto2 hidden states were extracted.")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
