from __future__ import annotations


def experiment_signature(config: dict) -> tuple:
    """Return the analysis-defining signature used for artifact reuse audits."""
    model = config["model"]
    data = config["data"]
    future = config["future_target"]
    mi = config["mi"]
    return (
        model["name"],
        model["model_id"],
        model.get("revision", ""),
        model.get("channel_aggregation", "concat_same_time_patch"),
        data["dataset"],
        data["seq_len"],
        data["pred_len"],
        data.get("features", "S"),
        tuple(data.get("target_columns", [])),
        data.get("max_origins"),
        future.get("bins"),
        future.get("spectral_bands"),
        future.get("pca_dim"),
        future.get("normalization"),
        future.get("scope"),
        mi.get("hidden_pca_dim"),
        mi.get("k"),
        mi.get("null_permutations"),
        mi.get("null_mode"),
        mi.get("seed"),
        mi.get("jitter"),
    )
