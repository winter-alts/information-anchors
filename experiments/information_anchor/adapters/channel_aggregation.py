from __future__ import annotations

import torch


SUPPORTED_CHANNEL_AGGREGATIONS = {
    "concat_same_time_patch",
    "mean",
    "mean_std",
}


def aggregate_channel_hidden(
    stacked: torch.Tensor,
    *,
    batch_size: int,
    n_channels: int,
    strategy: str,
) -> torch.Tensor:
    """按同一时间 patch 聚合多变量 hidden，避免缓存维度随变量数线性爆炸。"""
    if strategy not in SUPPORTED_CHANNEL_AGGREGATIONS:
        raise ValueError(
            f"Unsupported channel aggregation {strategy!r}; "
            f"expected one of {sorted(SUPPORTED_CHANNEL_AGGREGATIONS)}."
        )
    if stacked.ndim != 4:
        raise ValueError(f"Expected [batch*channels, layers, patches, hidden], got {stacked.shape}.")
    if batch_size < 1 or n_channels < 1 or stacked.shape[0] != batch_size * n_channels:
        raise ValueError(
            "The first hidden dimension must equal batch_size*n_channels: "
            f"shape={tuple(stacked.shape)}, batch_size={batch_size}, n_channels={n_channels}."
        )
    if n_channels == 1:
        return stacked

    _bc, n_layers, n_patches, hidden_size = stacked.shape
    grouped = stacked.reshape(batch_size, n_channels, n_layers, n_patches, hidden_size)
    # 统一到 [batch, layer, patch, channel, hidden]，后续统计只跨变量维度。
    grouped = grouped.permute(0, 2, 3, 1, 4)
    if strategy == "concat_same_time_patch":
        return grouped.reshape(batch_size, n_layers, n_patches, n_channels * hidden_size)
    mean = grouped.mean(dim=3)
    if strategy == "mean":
        return mean
    std = grouped.float().std(dim=3, unbiased=False).to(dtype=stacked.dtype)
    return torch.cat((mean, std), dim=-1)
