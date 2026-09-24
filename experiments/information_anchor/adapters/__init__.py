from __future__ import annotations

from experiments.information_anchor.config import ExperimentConfig


_ADAPTER_PROTOCOLS = {
    "ChronosBolt": "chronos-bolt-encoder-history-v1",
    "TimesFM2.5": "timesfm25-prefill-v1",
    "TimesFM3": "timesfm3-prefill-v1",
    "Toto2": "toto2-post-norm-v1",
    "TTM": "ttm-r2-common-grid-stages-v1",
}


def adapter_protocol(model_name: str) -> str | None:
    return _ADAPTER_PROTOCOLS.get(model_name)


def build_adapter(config: ExperimentConfig):
    if config.model.name == "TimesFM":
        from experiments.information_anchor.adapters.timesfm import TimesFMAdapter

        return TimesFMAdapter(config)
    if config.model.name == "TimesFM2.5":
        from experiments.information_anchor.adapters.timesfm25 import TimesFM25Adapter

        return TimesFM25Adapter(config)
    if config.model.name == "TimesFM3":
        from experiments.information_anchor.adapters.timesfm3 import TimesFM3Adapter

        return TimesFM3Adapter(config)
    if config.model.name == "Chronos2":
        from experiments.information_anchor.adapters.chronos2 import Chronos2Adapter

        return Chronos2Adapter(config)
    if config.model.name == "ChronosBolt":
        from experiments.information_anchor.adapters.chronos_bolt import ChronosBoltAdapter

        return ChronosBoltAdapter(config)
    if config.model.name == "Moirai2":
        from experiments.information_anchor.adapters.moirai2 import Moirai2Adapter

        return Moirai2Adapter(config)
    if config.model.name == "Toto2":
        from experiments.information_anchor.adapters.toto2 import Toto2Adapter

        return Toto2Adapter(config)
    if config.model.name == "TTM":
        from experiments.information_anchor.adapters.ttm import TTMAdapter

        return TTMAdapter(config)
    raise ValueError(f"No information-anchor adapter for model={config.model.name!r}.")


def representation_depends_on_pred_len(model_name: str) -> bool:
    return model_name == "Chronos2"
