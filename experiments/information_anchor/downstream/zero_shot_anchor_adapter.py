"""Small trainable adapters for a frozen Chronos-Bolt zero-shot anchor."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


AdapterKind = Literal["linear", "film"]


class LinearResidualAdapter(nn.Module):
    """Predict an output residual from query decoder state and retrieval summary."""

    kind = "linear"

    def __init__(self, hidden_size: int, summary_size: int, pred_len: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.summary_size = int(summary_size)
        self.pred_len = int(pred_len)
        self.projection = nn.Linear(self.hidden_size + self.summary_size, self.pred_len)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or summary.ndim != 2:
            raise ValueError("Linear adapter inputs must be [B,D] and [B,P]")
        if hidden.shape[0] != summary.shape[0]:
            raise ValueError("Linear adapter batch sizes disagree")
        return self.projection(torch.cat((hidden, summary), dim=-1))


class RetrievalConditionedFiLM(nn.Module):
    """Generate an identity-initialized FiLM modulation from retrieval summary."""

    kind = "film"

    def __init__(self, hidden_size: int, summary_size: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.summary_size = int(summary_size)
        self.bottleneck = int(bottleneck)
        self.network = nn.Sequential(
            nn.Linear(self.summary_size, self.bottleneck),
            nn.GELU(),
            nn.Linear(self.bottleneck, 2 * self.hidden_size),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if summary.ndim != 2:
            raise ValueError("FiLM summary must be [B,P]")
        modulation = self.network(summary)
        delta_gamma, beta = modulation.chunk(2, dim=-1)
        return 1.0 + delta_gamma, beta


def build_adapter(
    kind: AdapterKind,
    *,
    hidden_size: int,
    summary_size: int,
    pred_len: int,
    film_bottleneck: int = 128,
) -> nn.Module:
    if kind == "linear":
        return LinearResidualAdapter(hidden_size, summary_size, pred_len)
    if kind == "film":
        if pred_len != summary_size:
            raise ValueError("FiLM uses the pred_len retrieval summary")
        return RetrievalConditionedFiLM(hidden_size, summary_size, film_bottleneck)
    raise ValueError(f"unknown adapter kind: {kind!r}")


def load_adapter_checkpoint(
    path: str,
    *,
    device: torch.device,
    kind: AdapterKind | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("adapter checkpoint must contain state_dict and config")
    config = dict(payload.get("config", {}))
    checkpoint_kind = str(config.get("kind", ""))
    selected_kind = checkpoint_kind if kind is None else kind
    if selected_kind not in {"linear", "film"} or selected_kind != checkpoint_kind:
        raise ValueError(
            f"adapter kind mismatch: checkpoint={checkpoint_kind!r}, requested={kind!r}"
        )
    adapter = build_adapter(
        selected_kind,  # type: ignore[arg-type]
        hidden_size=int(config["hidden_size"]),
        summary_size=int(config["summary_size"]),
        pred_len=int(config["pred_len"]),
        film_bottleneck=int(config.get("film_bottleneck", 128)),
    )
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter.to(device).eval(), config


def _self_check() -> None:
    hidden = torch.randn(3, 8)
    summary = torch.randn(3, 4)
    linear = LinearResidualAdapter(8, 4, 4)
    assert torch.equal(linear(hidden, summary), torch.zeros(3, 4))
    film = RetrievalConditionedFiLM(8, 4)
    gamma, beta = film(summary)
    assert torch.equal(gamma, torch.ones(3, 8))
    assert torch.equal(beta, torch.zeros(3, 8))


if __name__ == "__main__":
    _self_check()
