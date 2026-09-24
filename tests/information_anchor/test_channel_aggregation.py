from __future__ import annotations

import torch

from experiments.information_anchor.adapters.channel_aggregation import aggregate_channel_hidden


def test_channel_aggregation_shapes_and_values() -> None:
    """验证高维通道聚合的形状、均值和标准差协议。"""
    stacked = torch.arange(2 * 3 * 3 * 4 * 5, dtype=torch.float32).reshape(6, 3, 4, 5)
    mean = aggregate_channel_hidden(
        stacked, batch_size=2, n_channels=3, strategy="mean"
    )
    mean_std = aggregate_channel_hidden(
        stacked, batch_size=2, n_channels=3, strategy="mean_std"
    )
    concat = aggregate_channel_hidden(
        stacked, batch_size=2, n_channels=3, strategy="concat_same_time_patch"
    )
    assert mean.shape == (2, 3, 4, 5)
    assert mean_std.shape == (2, 3, 4, 10)
    assert concat.shape == (2, 3, 4, 15)
    grouped = stacked.reshape(2, 3, 3, 4, 5).permute(0, 2, 3, 1, 4)
    expected_mean = grouped.mean(dim=3)
    expected_std = grouped.std(dim=3, unbiased=False)
    assert torch.equal(mean, expected_mean)
    assert torch.equal(mean_std[..., :5], expected_mean)
    assert torch.equal(mean_std[..., 5:], expected_std)
