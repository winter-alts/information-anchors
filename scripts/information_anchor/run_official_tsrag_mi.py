#!/usr/bin/env python3
"""Run official TS-RAG with causal MI candidate selection.

The retrieved CSV is loaded with ``top_k=20``.  MI methods replace the
official first-ten slice with a sidecar-selected subset; the ARM checkpoint and
Chronos-Bolt weights are otherwise unchanged.

``mi_select`` is the membership-only variant: MI participates in choosing the
subset, but selected ranks are restored to official order before FrozenARM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RESIDUAL_PROTOTYPE_FUSIONS = {
    "residual_prototype",
    "residual_prototype_median",
    "residual_prototype_confident",
    "residual_prototype_mi_reliable",
    "residual_prototype_high_mi",
    "residual_prototype_hybrid",
    "residual_prototype_hybrid_median",
    "residual_prototype_hybrid_confident",
    "residual_prototype_hybrid_pool",
    "residual_prototype_mi_gra_plus_high_mi",
}
_HORIZON_RESIDUAL_FUSIONS = {"horizon_residual_transport"}
_COUPLED_RECENT_MI_FUSIONS = {"coupled_recent_mi"}
_UNIT_CONFIDENCE_FUSIONS = {"high_mi_linear", "bound_high_mi_linear_mix"}
_RESIDUAL_CONFIDENCE_FUSIONS = {"high_mi_consistency"}
_HISTORY_CONFIDENCE_FUSIONS = {"mi_rank_agreement", "mi_rank_entropy", "dual_rank_disagreement"}
_HISTORY_CONFIDENCE_SUPPORTED = {
    "high_mi",
    "high_mi_linear",
    "high_mi_consistency",
    "bound_high_mi_ablation",
}
_FULL_POOL_FORECAST_FUSIONS = {"high_mi_pool", "residual_prototype_hybrid_pool"}
_OFFICIAL_MI_RESIDUAL_METHODS = {"official_mi_residual"}
_MI_GRA_BIAS_TARGETS = {
    "mi_gra", "mi_gra_no_mi", "mi_gra_shuffled", "mi_gra_reversed",
    "mi_gra_uniform", "mi_gra_residual", "mi_gra_residual_no_mi",
    "mi_gra_residual_shuffled", "mi_gra_residual_reversed",
    "mi_gra_residual_uniform",
}
HORIZON_METHODS = {
    "horizon_mi", "horizon_global_mi", "horizon_uniform", "horizon_random"
}

_RETRIEVAL_AUGMENT_MODES = {"rcam", "trfa", "hera", "ipsra", "csea", "csea_fair", "bfa", "bfa_fair", "bfa_hidden", "bfa_hidden_fair", "oera", "oera_fair", "heca", "heca_fair", "htfa_fair", "htfa_forecast_fair", "chsfa_fair", "hcta_fair", "rqca_fair", "rgfa_fair", "raem_fair", "hrca_fair", "brea_fair", "sqem_fair", "bcsqm_fair", "qcra_fair", "rsm_fair", "hcsm_fair", "dqr_fair", "rem_fair"}
_NEW_ARM_METHODS = {"crossrag_official", "trfa_official", "hera_official", "ipsra_official", "csea_official", "bfa_official", "bfa_hidden_official", "oera_official", "heca_official", "htfa_official", "htfa_forecast_official", "chsfa_official", "hcta_official", "rqca_official", "rgfa_official", "raem_official", "hrca_official", "brea_official", "sqem_official", "bcsqm_official", "qcra_official", "rsm_official", "hcsm_official", "dqr_official", "rem_official"}


def _coupled_recent_selection(
    sidecar: dict[str, np.ndarray],
    *,
    method: str,
    top_k: int,
    local_offset: int,
    batch_size: int,
    candidate_span: int = 64,
) -> np.ndarray:
    """Return the paired recent/MI candidate ranks for one batch.

    The no-MI control is causal recency Top-k.  The MI arm changes both the
    membership and the distance source to the history-only MI ranking.  This
    keeps the MI switch paired instead of silently comparing two different
    meanings of ``recent``.
    """

    start = int(local_offset)
    end = start + int(batch_size)
    if method == "recent":
        if "recency_distances" in sidecar:
            distances = np.asarray(sidecar["recency_distances"])[start:end]
            valid_shape = distances.ndim == 2 and distances.shape[0] == int(batch_size)
        elif "query_origins" in sidecar and "candidate_starts" in sidecar:
            distances = causal_recency_distances(
                np.asarray(sidecar["query_origins"])[start:end],
                np.asarray(sidecar["candidate_starts"])[start:end],
                candidate_span=int(candidate_span),
            )
            valid_shape = distances.ndim == 2 and distances.shape[0] == int(batch_size)
        else:
            distances = np.asarray(None)
            valid_shape = False
        if not valid_shape:
            raise ValueError(
                "coupled_recent_mi recent arm requires recency_distances or "
                "aligned query_origins/candidate_starts"
            )
        selected = np.argsort(distances, axis=1, kind="stable")[:, : int(top_k)]
    elif method == "mi_recent":
        selected = np.asarray(sidecar.get("selected_ranks_mi_recent"))[
            start:end
        ]
    else:
        raise ValueError(
            "coupled_recent_mi supports only recent and mi_recent"
        )
    expected = (int(batch_size), int(top_k))
    if selected.shape != expected:
        raise ValueError(
            f"coupled_recent_mi {method} selection shape {selected.shape} "
            f"!= expected {expected}"
        )
    return np.asarray(selected, dtype=np.int64)


def _coupled_recent_distance_key(method: str) -> str:
    """Return the distance source paired with a coupled candidate selector."""

    if method == "recent":
        return "recency_distances"
    if method == "mi_recent":
        return "high_mi_distances"
    raise ValueError(f"unknown coupled_recent_mi method: {method}")


def _forecast_fusion_mi_branch_gate_applies(
    method: str,
    forecast_fusion: str,
) -> bool:
    """Return whether a history-only MI gate should scale this forecast arm."""

    return method in {"high_mi", "high_mi_anchor"} or (
        forecast_fusion == "coupled_recent_mi" and method == "mi_recent"
    )

# Controlled 2x2 ablation for the full high-MI forecast intervention.  The
# high-MI switch is deliberately bound to both high-MI Top-10 membership and
# high-MI distance weighting; the corresponding control uses official Top-10
# membership and official retrieval-distance weighting.
_BOUND_HIGH_MI_ABLATION_METHODS = {
    "official_top10_official_dist_no_alignment": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "none",
    },
    "official_top10_official_dist_recent_mean": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "recent_mean",
    },
    "high_mi_top10_mi_dist_no_alignment": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "none",
    },
    "high_mi_top10_mi_dist_recent_mean": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "recent_mean",
    },
}
_BOUND_HIGH_MI_ABLATION_METHOD_SET = set(_BOUND_HIGH_MI_ABLATION_METHODS)
_BOUND_HIGH_MI_OFFICIAL_BASELINE_METHOD_SET = {
    "official_top10_official_dist_recent_mean",
    "high_mi_top10_mi_dist_recent_mean",
    "high_mi_top10_mi_dist_no_alignment",
}
_BOUND_MI_SELECT_ABLATION_METHODS = {
    "mi_select_mi_l2_recent_mean": {
        "selection": "mi_select",
        "distance": "mi_l2_distances",
        "alignment": "recent_mean",
    },
    "mi_select_mi_l2_no_alignment": {
        "selection": "mi_select",
        "distance": "mi_l2_distances",
        "alignment": "none",
    },
}
_BOUND_MI_SELECT_ABLATION_METHOD_SET = {
    "official_top10_official_dist_no_alignment",
    "official_top10_official_dist_recent_mean",
    *_BOUND_MI_SELECT_ABLATION_METHODS,
}
_BOUND_HIGH_MI_L2_ABLATION_METHODS = {
    "high_mi_top10_mi_l2_recent_mean": {
        "selection": "high_mi_top10",
        "distance": "mi_l2_distances",
        "alignment": "recent_mean",
    },
    "high_mi_top10_mi_l2_no_alignment": {
        "selection": "high_mi_top10",
        "distance": "mi_l2_distances",
        "alignment": "none",
    },
}
_BOUND_HIGH_MI_L2_ABLATION_METHOD_SET = {
    "official_top10_official_dist_no_alignment",
    "official_top10_official_dist_recent_mean",
    *_BOUND_HIGH_MI_L2_ABLATION_METHODS,
}
_BOUND_MI_L2_HIGH_MI_ABLATION_METHODS = {
    "mi_select_high_mi_recent_mean": {
        "selection": "mi_select",
        "distance": "high_mi_distances",
        "alignment": "recent_mean",
    },
    "mi_select_high_mi_no_alignment": {
        "selection": "mi_select",
        "distance": "high_mi_distances",
        "alignment": "none",
    },
}
_BOUND_MI_L2_HIGH_MI_ABLATION_METHOD_SET = {
    "official_top10_official_dist_no_alignment",
    "official_top10_official_dist_recent_mean",
    *_BOUND_MI_L2_HIGH_MI_ABLATION_METHODS,
}
_BOUND_HIGH_MI_ALIGNED_ABLATION_METHODS = {
    "official_top10_official_dist_aligned": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "aligned",
    },
    "high_mi_top10_mi_dist_aligned": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "aligned",
    },
}
_BOUND_HIGH_MI_ALIGNED_ABLATION_METHOD_SET = {
    "official_top10_official_dist_no_alignment",
    "high_mi_top10_mi_dist_no_alignment",
    *_BOUND_HIGH_MI_ALIGNED_ABLATION_METHODS,
}
_BOUND_HIGH_MI_ABLATION_METHODS = {
    **_BOUND_HIGH_MI_ABLATION_METHODS,
    **_BOUND_HIGH_MI_ALIGNED_ABLATION_METHODS,
}
_BOUND_HIGH_MI_ALL_ABLATION_METHOD_SET = (
    _BOUND_HIGH_MI_ABLATION_METHOD_SET
    | _BOUND_HIGH_MI_ALIGNED_ABLATION_METHOD_SET
    | _BOUND_MI_SELECT_ABLATION_METHOD_SET
    | _BOUND_HIGH_MI_L2_ABLATION_METHOD_SET
    | _BOUND_MI_L2_HIGH_MI_ABLATION_METHOD_SET
)
_BOUND_HIGH_MI_LINEAR_MIX_METHODS = {
    "official_top10_official_dist_linear_mix_no_alignment": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "none",
    },
    "official_top10_official_dist_linear_mix_recent_mean": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "recent_mean",
    },
    "high_mi_top10_mi_dist_linear_mix_no_alignment": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "none",
    },
    "high_mi_top10_mi_dist_linear_mix_recent_mean": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "recent_mean",
    },
}
_BOUND_HIGH_MI_LINEAR_MIX_METHOD_SET = set(_BOUND_HIGH_MI_LINEAR_MIX_METHODS)
_BOUND_HIGH_MI_LINEAR_MIX_OFFICIAL_BASELINE_METHOD_SET = {
    "official_top10_official_dist_linear_mix_recent_mean",
    "high_mi_top10_mi_dist_linear_mix_recent_mean",
    "high_mi_top10_mi_dist_linear_mix_no_alignment",
}

# User-facing 2x2 ablation for the residual-prototype intervention.  The
# untouched ``official`` method is kept as a separate reference arm; these
# four names are the actual residual-fusion methods reported in the table.
_BOUND_HIGH_MI_RESIDUAL_METHODS = {
    "official_residual_no_alignment": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "none",
    },
    "official_residual_recent_mean96": {
        "selection": "official_top10",
        "distance": "official_distances",
        "alignment": "recent_mean",
    },
    "high_mi_residual_recent_mean96": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "recent_mean",
    },
    "high_mi_residual_no_alignment": {
        "selection": "high_mi_top10",
        "distance": "high_mi_distances",
        "alignment": "none",
    },
}
_BOUND_HIGH_MI_RESIDUAL_METHOD_SET = set(_BOUND_HIGH_MI_RESIDUAL_METHODS)
_BOUND_HIGH_MI_ALL_METHOD_SET = (
    _BOUND_HIGH_MI_ALL_ABLATION_METHOD_SET
    | _BOUND_HIGH_MI_LINEAR_MIX_METHOD_SET
    | _BOUND_HIGH_MI_RESIDUAL_METHOD_SET
)

def _bound_method_spec(method: str) -> dict[str, str] | None:
    return (
        _BOUND_HIGH_MI_ABLATION_METHODS.get(method)
        or _BOUND_MI_SELECT_ABLATION_METHODS.get(method)
        or _BOUND_HIGH_MI_L2_ABLATION_METHODS.get(method)
        or _BOUND_MI_L2_HIGH_MI_ABLATION_METHODS.get(method)
        or _BOUND_HIGH_MI_LINEAR_MIX_METHODS.get(method)
        or _BOUND_HIGH_MI_RESIDUAL_METHODS.get(method)
    )


def _is_official_baseline_four_method_protocol(
    methods: list[str], bound_methods: set[str], *, linear_mix: bool = False,
) -> bool:
    """Recognize the four-arm comparison with the true official baseline.

    The historical explicit 2x2 ablation names its no-alignment official arm
    as a fusion method, which is useful for a factorized ablation but is not
    the untouched TS-RAG baseline.  The user-facing four-arm experiment keeps
    ``official`` as the baseline and evaluates only the three non-baseline
    explicit arms in the bound protocol.
    """

    expected = (
        _BOUND_HIGH_MI_LINEAR_MIX_OFFICIAL_BASELINE_METHOD_SET
        if linear_mix
        else _BOUND_HIGH_MI_OFFICIAL_BASELINE_METHOD_SET
    )
    return (
        len(methods) == 4
        and set(methods) == {"official"} | expected
        and bound_methods == expected
    )


def _validate_bound_high_mi_method_set(
    methods: list[str], *, forecast_fusion: str,
) -> None:
    """Validate the explicit four-arm bound-ablation method protocol.

    The user-facing comparison uses the untouched ``official`` method as
    arm 1.  The other three explicit methods are the recent-aligned official
    control, MI without alignment, and MI with recent-mean alignment.
    """

    bound_methods = set(methods) & _BOUND_HIGH_MI_ALL_METHOD_SET
    if forecast_fusion == "bound_high_mi_ablation":
        valid_bound_sets = (
            _BOUND_HIGH_MI_ABLATION_METHOD_SET,
            _BOUND_HIGH_MI_ALIGNED_ABLATION_METHOD_SET,
            _BOUND_MI_SELECT_ABLATION_METHOD_SET,
            _BOUND_HIGH_MI_L2_ABLATION_METHOD_SET,
            _BOUND_MI_L2_HIGH_MI_ABLATION_METHOD_SET,
        )
        if not (
            _is_official_baseline_four_method_protocol(methods, bound_methods)
            or any(bound_methods == valid for valid in valid_bound_sets)
            and len(methods) == 4
        ):
            raise ValueError(
                "bound_high_mi_ablation requires exactly one of the supported "
                "explicit four-arm method sets"
            )
    elif bound_methods and forecast_fusion not in {
        "bound_high_mi_linear_mix",
        "bound_high_mi_residual_ablation",
    }:
        raise ValueError(
            "the explicit bound high-MI ablation methods require "
            "--forecast-fusion bound_high_mi_ablation"
        )


def _validate_bound_high_mi_residual_method_set(
    methods: list[str], *, forecast_fusion: str,
) -> None:
    """Validate the four residual arms, optionally plus raw official reference."""

    if forecast_fusion != "bound_high_mi_residual_ablation":
        if set(methods) & _BOUND_HIGH_MI_RESIDUAL_METHOD_SET:
            raise ValueError(
                "the explicit residual bound methods require "
                "--forecast-fusion bound_high_mi_residual_ablation"
            )
        return
    expected = {"official"} | _BOUND_HIGH_MI_RESIDUAL_METHOD_SET
    residual_four = _BOUND_HIGH_MI_RESIDUAL_METHOD_SET
    valid = (
        len(methods) == 4 and set(methods) == residual_four
    ) or (
        len(methods) == 5 and methods[0] == "official" and set(methods) == expected
    )
    if not valid:
        raise ValueError(
            "bound_high_mi_residual_ablation requires exactly the four residual "
            "arms, optionally preceded by the raw official reference"
        )


def _residual_prototype_score_key(fusion: str, method: str) -> str:
    """Return the history-only distance source for a residual prototype."""

    if fusion == "residual_prototype_high_mi":
        if method != "high_mi":
            raise ValueError("residual_prototype_high_mi requires the high_mi selector")
        return "high_mi_distances"
    if method == "mi_residual_prototype":
        return "selection_scores_mi_residual_prototype"
    return "mi_l2_distances"


def _hybrid_residual_fusion_scores(
    method: str,
    selected_mi_scores: "torch.Tensor",
    mi_gra_scores: "torch.Tensor | None",
) -> "torch.Tensor":
    """Select the MI/control scores used by the hybrid residual branch."""

    if str(method).startswith("mi_gra"):
        if mi_gra_scores is None:
            raise ValueError("MI-GRA hybrid residual methods require MI-GRA scores")
        return mi_gra_scores
    return selected_mi_scores


def _forecast_fusion_uses_full_pool(fusion: str) -> bool:
    return fusion in _FULL_POOL_FORECAST_FUSIONS


def _blend_dual_forecasts(
    primary_prediction: "torch.Tensor",
    high_mi_prediction: "torch.Tensor",
    high_mi_weight: float,
) -> "torch.Tensor":
    """Blend the MI-GRA and high-MI forecast branches with one frozen weight."""

    import torch

    if primary_prediction.shape != high_mi_prediction.shape:
        raise ValueError("dual forecast branches must have the same shape")
    weight = float(high_mi_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("dual forecast weight must be finite and lie in [0,1]")
    if not torch.isfinite(primary_prediction).all() or not torch.isfinite(high_mi_prediction).all():
        raise ValueError("dual forecast branches must be finite")
    return (1.0 - weight) * primary_prediction + weight * high_mi_prediction

from experiments.information_anchor.artifacts import capture_run_context
from experiments.information_anchor.downstream.gpu_policy import validate_gpu_id
from experiments.information_anchor.downstream.mi_residual_prototype import (
    mi_reliability_gate as compute_mi_reliability_gate,
    hybrid_residual_prototype_correction,
    residual_prototype_confidence,
    residual_prototype_correction,
)
from experiments.information_anchor.downstream.mi_confidence_gate import (
    history_only_mi_branch_gate,
    history_only_mi_confidence,
    mi_distance_match_multiplier,
)
from experiments.information_anchor.downstream.tsrag_attention import attention_prior_scores
from experiments.information_anchor.downstream.tsrag_output_mi import (
    output_mi_prior_bias,
)
from experiments.information_anchor.downstream.tsrag_alpha_search import apply_residual_shrink
from experiments.information_anchor.downstream.info_transport_rag import (
    horizon_residual_transport,
    mi_concentration_gate,
)
from scripts.information_anchor.build_official_rag_mi_artifact import (
    causal_recency_distances,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--root-path", required=True, help="Directory containing the official retrieved CSV")
    parser.add_argument("--data-path", required=True, help="Official retrieved CSV with top-20 columns")
    parser.add_argument("--data", required=True, choices=("ett_h_retrieve", "ett_m_retrieve", "custom_retrieve"))
    parser.add_argument("--artifact", default=None,
                        help="official top-20 MI sidecar (still required for official methods)")
    parser.add_argument(
        "--mi-gra-artifact", default=None,
        help=(
            "optional second history-only sidecar for MI-GRA distances; use this "
            "when --artifact supplies high_mi_distances but lacks mi_l2_distances"
        ),
    )
    parser.add_argument("--global-artifact", default=None,
                        help="MI-global sidecar with selected channel/start windows")
    parser.add_argument("--retrieval-database-dir", required=True)
    parser.add_argument("--metadata-frequency", required=True)
    parser.add_argument("--lookback-length", type=int, default=512)
    parser.add_argument("--mode", default="only_self_train")
    parser.add_argument("--results-dir", default="results/information_anchor_official_rag_mi/ts_rag")
    parser.add_argument("--methods", default="base,official,uniform,random,low_mi,high_mi")
    parser.add_argument(
        "--high-mi-branch-distance",
        choices=("mi", "official"),
        default="mi",
        help=(
            "Distance input to the frozen TS-RAG branch when method=high_mi. "
            "Use official to match the paper; MI distances remain in the outer correction."
        ),
    )
    parser.add_argument(
        "--contamination-mode",
        choices=("none", "random_time", "wrong_channel", "far_distance", "season_mismatch"),
        default="none",
        help="Offline candidate-pool perturbation. Selection never sees future values.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=64)
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-start", type=int, default=0,
                        help="inclusive flattened test-window offset (pilot shards)")
    parser.add_argument("--query-end", type=int, default=None,
                        help="exclusive flattened test-window offset; defaults to all windows")
    parser.add_argument("--query-count", type=int, default=None,
                        help="deterministically subsample this many queries inside the requested range")
    parser.add_argument("--query-sampling", choices=("contiguous", "stratified"), default="stratified",
                        help="sampling rule used when --query-count is set")
    parser.add_argument(
        "--query-parity", type=int, choices=(0, 1), default=None,
        help=(
            "optional deterministic validation fold: keep query indices whose "
            "global flattened index has the selected parity"
        ),
    )
    parser.add_argument("--pretrained-model-path", required=True)
    parser.add_argument("--base-weights", default=None)
    parser.add_argument("--retrieval-checkpoint", required=True)
    parser.add_argument(
        "--forecast-anchor",
        choices=("tsrag", "tsrag_arm", "tsrag_bfa", "tsrag_htfa", "tsrag_htfa_forecast", "tsrag_chsfa", "tsrag_hcta", "tsrag_rqca", "zero_shot"),
        default="tsrag",
        help=(
            "Forecast anchor for residual fusion. `tsrag` is the frozen "
            "TS-RAG ARM path; `zero_shot` uses frozen Chronos-Bolt on the "
            "query history and leaves retrieved futures to the outer fusion."
        ),
    )
    parser.add_argument(
        "--anchor-adapter-checkpoint",
        default=None,
        help=(
            "optional train-only Linear Residual Adapter or retrieval-conditioned "
            "FiLM checkpoint for the zero-shot anchor"
        ),
    )
    parser.add_argument(
        "--anchor-adapter-type",
        choices=("linear", "film"),
        default=None,
        help="optional adapter type check; otherwise read it from the checkpoint",
    )
    parser.add_argument(
        "--anchor-adapter-residual-mode",
        choices=(
            "raw", "scaled", "bounded", "gated_scaled", "gated_bounded",
            "reliability_scaled", "power_scaled", "deadzone_scaled",
        ),
        default="raw",
        help=(
            "how to apply a Linear anchor residual; bounded uses query diff-MAD; "
            "gated modes additionally multiply by the forecast-disagreement confidence"
        ),
    )
    parser.add_argument(
        "--anchor-adapter-strength",
        type=float,
        default=1.0,
        help="lambda_A for scaled/bounded Linear anchor residuals (0..1)",
    )
    parser.add_argument(
        "--anchor-adapter-gamma",
        type=float,
        default=1.0,
        help="gamma for the power gate g_q**gamma",
    )
    parser.add_argument(
        "--anchor-adapter-reliability-scale",
        type=float,
        default=1.0,
        help="s_c for the candidate-residual reliability gate",
    )
    parser.add_argument(
        "--anchor-adapter-deadzone-threshold",
        type=float,
        default=0.0,
        help="t for the dead-zone gate (0 <= t < 1)",
    )
    parser.add_argument("--correction-gain", type=float, default=1.0,
                        help="Frozen interpolation toward the official prediction (0..1).")
    parser.add_argument("--correction-gain-file", default=None,
                        help="JSON with a scalar or horizon-wise frozen correction gain.")
    parser.add_argument(
        "--forecast-fusion",
        choices=(
            "none", "high_mi", "mi_prior", "low_mi", "random", "uniform",
            "coupled_recent_mi",
            "high_mi_pool",
            "high_mi_linear",
            "high_mi_consistency",
            "residual_prototype", "residual_prototype_median",
            "residual_prototype_confident", "residual_prototype_mi_reliable",
            "residual_prototype_high_mi", "residual_prototype_hybrid",
            "residual_prototype_hybrid_median",
            "residual_prototype_hybrid_confident",
            "residual_prototype_hybrid_pool",
            "residual_prototype_mi_gra_plus_high_mi",
            "horizon_residual_transport",
            "bound_high_mi_ablation",
            "bound_high_mi_linear_mix",
            "bound_high_mi_residual_ablation",
        ),
        default="none",
        help=(
            "Optional history-only weighting of retrieved futures in forecast "
            "space, or residual_prototype correction."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-strength", type=float, default=0.1,
        help=(
            "Legacy forecast-fusion interpolation strength (0..1), or non-negative "
            "lambda for residual_prototype."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-dual-high-mi-strength", type=float, default=0.15,
        help=(
            "bounded interpolation strength of the high-MI branch in "
            "residual_prototype_mi_gra_plus_high_mi"
        ),
    )
    parser.add_argument(
        "--forecast-fusion-strength-file", default=None,
        help="JSON produced by tune_tsrag_alpha.py; loads selected_alpha for test inference.",
    )
    parser.add_argument(
        "--forecast-fusion-correction-clip", type=float, default=0.0,
        help=(
            "Validation-frozen absolute clip for each residual-prototype "
            "correction component; zero disables clipping."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-temperature", type=float, default=1.0,
        help="Temperature for row-wise MI distance weights.",
    )
    parser.add_argument(
        "--forecast-fusion-hybrid-beta", type=float, default=0.5,
        help=(
            "MI weight in residual_prototype_hybrid after per-source distance "
            "normalization; 0 is official distance only and 1 is MI only. "
            "For residual_prototype_mi_gra_plus_high_mi, this is the high-MI "
            "forecast-branch weight."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-reverse-weights", action="store_true",
        help=(
            "Reverse the distance weighting direction for the high/low-MI "
            "comparison control: use softmax(+z/T) instead of softmax(-z/T)."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-confidence-scale", type=float, default=1.0,
        help="Scale for disagreement-based confidence of the retrieved-future expert.",
    )
    parser.add_argument(
        "--forecast-fusion-weighting",
        choices=("mi", "mi_bcsa_reliability"),
        default="mi",
        help="Candidate weighting; mi_bcsa_reliability multiplies MI weights by fixed BCSA reliability.",
    )
    parser.add_argument(
        "--forecast-fusion-confidence-floor", type=float, default=0.0,
        help=(
            "Validation-frozen confidence floor. For residual_prototype_confident it "
            "keeps the legacy residual gate; for MI history modes it maps agreement "
            "to floor + (1-floor)*agreement."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-confidence-mode",
        choices=("forecast_disagreement", "dual_disagreement", "dual_rank_disagreement", "mi_rank_agreement", "mi_rank_entropy"),
        default="forecast_disagreement",
        help=(
            "Confidence source for legacy forecast fusion. The MI modes use only "
            "history-side MI/official ranking agreement; mi_rank_entropy additionally "
            "requires a concentrated MI distance distribution."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-history-entropy-power", type=float, default=1.0,
        help=(
            "Non-negative exponent for the optional history-only MI entropy factor "
            "when --forecast-fusion-confidence-mode=mi_rank_entropy."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-mi-match-file", default=None,
        help=(
            "Validation-frozen JSON calibration for a history-only MI-match "
            "multiplier. The file must contain center, scale, slope, floor, "
            "and ceiling for min(high_mi_distances)."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-mi-branch-gate",
        choices=("none", "mi_rank_agreement"),
        default="none",
        help=(
            "History-only admission gate for the high-MI TS-RAG branch; "
            "low MI/official ranking agreement falls back to official."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-mi-branch-gate-threshold",
        type=float,
        default=0.0,
        help="Minimum history-only MI/official rank agreement for branch admission.",
    )
    parser.add_argument(
        "--forecast-fusion-mi-branch-gate-power",
        type=float,
        default=1.0,
        help="Positive power applied to normalized history-only branch admission.",
    )
    parser.add_argument(
        "--forecast-fusion-mi-reliability-threshold", type=float, default=0.05,
        help=(
            "Minimum normalized MI concentration before the residual correction "
            "is admitted; used by residual_prototype_mi_reliable."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-mi-reliability-power", type=float, default=1.0,
        help=(
            "Power applied to the MI reliability activation; values above one "
            "make the gate more conservative."
        ),
    )
    parser.add_argument(
        "--forecast-fusion-alignment",
        choices=("none", "zscore", "last", "mean_shift", "recent_mean", "recent_ewmean", "bda", "endpoint", "bcsa", "alignrag"),
        default="none",
        help=(
            "Candidate-future treatment before residual fusion. `none` is the "
            "official TS-RAG-compatible setting; the other values are legacy "
            "non-official analyses and must not be used for fair Table 1."
        ),
    )
    parser.add_argument(
        "--bound-high-mi-alignment",
        choices=("zscore", "last", "mean_shift", "recent_mean", "recent_ewmean"),
        default="recent_mean",
        help=(
            "Alignment used by the explicit *_aligned bound-ablation arms; "
            "the no-alignment arms remain unchanged."
        ),
    )
    parser.add_argument(
        "--bound-high-mi-alignment-window",
        type=int,
        default=96,
        help=(
            "Recent-history length for the residual bound arms; 96 means "
            "recent_mean96."
        ),
    )
    parser.add_argument(
        "--bound-high-mi-residual-estimator",
        choices=("mean", "median"),
        default="mean",
        help="Residual prototype estimator for the explicit four-arm protocol.",
    )
    parser.add_argument(
        "--bound-high-mi-residual-weighting",
        choices=("median_pool", "standardized"),
        default="median_pool",
        help=(
            "History-distance weighting for the explicit residual arms; "
            "standardized uses softmax(-z(distance)/temperature)."
        ),
    )
    parser.add_argument(
        "--bound-high-mi-residual-reference",
        choices=("base_model", "arm"),
        default="base_model",
        help=(
            "Forecast reference subtracted from each candidate future in the "
            "explicit residual arms; `arm` uses candidate_future-F_ARM."
        ),
    )
    parser.add_argument(
        "--bound-high-mi-residual-confidence",
        choices=("none", "direction", "disagreement"),
        default="none",
        help=(
            "Optional history-only residual gate for the explicit arms: "
            "direction, or retrieved-forecast disagreement."
        ),
    )
    parser.add_argument(
        "--tsrag-retrieved-alignment",
        choices=("none", "recent_ewmean"),
        default="none",
        help=(
            "Internal official TS-RAG alignment applied to retrieved_y before ARM; "
            "none preserves the official model exactly."
        ),
    )
    parser.add_argument(
        "--tsrag-retrieved-window",
        type=int,
        default=96,
        help="Recent history length for internal retrieved-future alignment.",
    )
    parser.add_argument(
        "--tsrag-retrieved-tau",
        type=float,
        default=32.0,
        help="Exponential-decay temperature for internal retrieved-future alignment.",
    )
    parser.add_argument(
        "--output-mi-gamma", type=float, default=1.0,
        help=(
            "MI interpolation coefficient for the output-level TS-RAG future "
            "prior (0..1); frozen from validation before test."
        ),
    )
    parser.add_argument(
        "--output-mi-prior-strength", type=float, default=1.0,
        help=(
            "Non-negative scale applied to log(K*w) before it enters the "
            "existing TS-RAG retrieved-expert gate; freeze on validation."
        ),
    )
    parser.add_argument(
        "--output-mi-prior-centering", choices=("raw", "centered"), default="raw",
        help=(
            "Whether to center log(K*w) across retrieved experts so the "
            "prior changes only within-pool preference, not average retrieved "
            "versus query gate mass."
        ),
    )
    parser.add_argument(
        "--output-mi-distance-key",
        choices=("high_mi_distances", "mi_l2_distances"),
        default="high_mi_distances",
        help="history-only MI distance array used by the output-level prior",
    )
    parser.add_argument(
        "--mi-gra",
        action="store_true",
        help="enable the opt-in MI-Gated Residual ARM branch",
    )
    parser.add_argument("--mi-gra-strength", type=float, default=0.0)
    parser.add_argument("--mi-gra-mi-strength", type=float, default=0.0)
    parser.add_argument("--mi-gra-temperature", type=float, default=1.0)
    parser.add_argument(
        "--mi-gra-feature", choices=("att_output", "retrieved_y"), default="att_output",
        help="expert representation E used by the MI-GRA gate",
    )
    parser.add_argument("--mi-gra-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--mi-gra-checkpoint", default=None,
        help="optional validation-trained MI-GRA state dict loaded after the official TS-RAG checkpoint",
    )
    parser.add_argument(
        "--mi-gra-distance-key",
        choices=("mi_l2_distances", "high_mi_distances"),
        default="mi_l2_distances",
        help="history-only distance array passed to MI-GRA",
    )
    parser.add_argument(
        "--save-mi-gra-diagnostics",
        action="store_true",
        help="save MI-GRA candidate weights and reliability values",
    )
    parser.add_argument(
        "--augment-mode", default="moe", choices=("moe", "gate", "rcam", "trfa", "hera", "ipsra", "csea", "csea_fair", "bfa", "bfa_fair", "bfa_hidden", "bfa_hidden_fair", "oera", "oera_fair", "heca", "heca_fair", "htfa_fair", "htfa_forecast_fair", "chsfa_fair", "hcta_fair", "rqca_fair", "rgfa_fair", "raem_fair", "hrca_fair", "brea_fair", "sqem_fair", "bcsqm_fair", "qcra_fair", "rsm_fair", "hcsm_fair", "dqr_fair", "rem_fair")
    )
    parser.add_argument(
        "--official-arm-checkpoint",
        default=None,
        help=(
            "Optional frozen official TS-RAG ARM checkpoint. When the active "
            "augment mode is RCAM/TRFA, the official row is evaluated with this "
            "separate moe model so it remains a true ARM baseline."
        ),
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--arm-bias", action="store_true")
    parser.add_argument("--arm-bias-strength", type=float, default=0.5)
    parser.add_argument(
        "--arm-bias-method",
        choices=("high_mi", "mi_prior", "mi_bias", "both", "all_priors", "mi_gra"),
        default="high_mi",
                        help="selector whose logits receive the optional ARM bias")
    parser.add_argument(
        "--arm-attention-prior",
        action="store_true",
        help="add a history-only MI prior to query-to-retrieved ARM attention",
    )
    parser.add_argument(
        "--arm-attention-prior-method",
        choices=("high_mi", "mi_prior", "low_mi", "random", "uniform", "null_mi_prior"),
        default="high_mi",
        help="history-only score source used by the optional ARM attention prior",
    )
    parser.add_argument("--arm-attention-prior-strength", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument(
        "--stream-metrics",
        action="store_true",
        help=(
            "Accumulate only scalar MSE/MAE sufficient statistics instead of "
            "retaining all multivariate predictions; incompatible with saved preds "
            "and correction-analysis artifacts."
        ),
    )
    parser.add_argument(
        "--save-correction-analysis",
        action="store_true",
        help="Save raw TS-RAG, MI consensus, fused forecasts, and confidence for alignment analysis.",
    )
    parser.add_argument(
        "--weighting-ablation-alignments",
        default=None,
        help=(
            "Comma-separated candidate-future alignments for a weighting "
            "ablation. Uniform and MI-distance arms use the high-MI Top-10; "
            "the official-distance arm uses the ordinary-distance Top-10. "
            "TS-RAG prediction, confidence, and fusion strength are frozen. "
            "Passing none,zscore additionally writes the 2x2 factorial table "
            "(ordinary distance / standardization / MI weighting / both)."
        ),
    )
    parser.add_argument(
        "--mi-attribution-ablation",
        action="store_true",
        help=(
            "Save w/o MI candidate selection, w/o MI distance weighting, and "
            "w/o MI confidence-gate predictions around one fixed Full anchor."
        ),
    )
    parser.add_argument(
        "--system-ablation-suite",
        action="store_true",
        help="Evaluate the nine locked Table-2 arms on matched windows and save per-window losses.",
    )
    return parser.parse_args()


def _data_provider_flag(split: str) -> str:
    """Map the orchestration split name to the official TS-RAG loader flag."""
    if split == "validation":
        return "val"
    if split == "test":
        return "test"
    raise ValueError(f"unknown split: {split!r}")


def _rebuild_ordered_eval_loader(split: str) -> bool:
    """Validation is exposed as ``val`` by TS-RAG, whose loader shuffles."""
    return split == "validation"


def _prepend_external_path() -> Path:
    root = ROOT / "repository_packages" / "external_rag" / "TS-RAG" / "TS-RAG"
    if not root.exists():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    return root


def _query_indices(total: int, start: int, end: int | None, count: int | None,
                   sampling: str, parity: int | None = None) -> np.ndarray:
    lo = max(0, int(start))
    hi = int(total) if end is None else min(int(total), int(end))
    if lo >= hi:
        raise ValueError("query-start must be smaller than query-end")
    available = hi - lo
    if count is None or int(count) >= available:
        indices = np.arange(lo, hi, dtype=np.int64)
    else:
        if int(count) <= 0:
            raise ValueError("query-count must be positive")
        if sampling == "contiguous":
            indices = np.arange(lo, lo + int(count), dtype=np.int64)
        else:
            # Midpoints of equal-width strata avoid endpoint bias and are
            # unique when count <= available. Flattened TS-RAG order is
            # channel-major, so the sample covers both channels and origins.
            edges = np.linspace(lo, hi, int(count) + 1)
            indices = np.floor((edges[:-1] + edges[1:]) * 0.5).astype(np.int64)
    if parity is not None:
        if int(parity) not in {0, 1}:
            raise ValueError("query parity must be 0 or 1")
        indices = indices[np.mod(indices, 2) == int(parity)]
        if indices.size == 0:
            raise ValueError("query parity produced an empty query fold")
    return indices


def _shard_bounds(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_(\d+)_(\d+)\.npz$", path.name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _load_indexed_sidecar(path: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"no sidecar NPZ files found under {path}")
    requested = np.asarray(indices, dtype=np.int64)
    pieces: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for shard in paths:
        bounds = _shard_bounds(shard)
        if bounds is None:
            if len(paths) != 1:
                raise ValueError(f"cannot infer row range from sidecar shard {shard.name}")
            bounds = (0, int(requested.max()) + 1)
        lo, hi = bounds
        positions = np.flatnonzero((requested >= lo) & (requested < hi))
        if not len(positions):
            continue
        local = requested[positions] - lo
        with np.load(shard, allow_pickle=False) as artifact:
            rows = {key: artifact[key][local] for key in artifact.files}
        pieces.append((positions, rows))
    if not pieces:
        raise ValueError("requested query indices do not overlap the sidecar")
    keys = set(pieces[0][1])
    if any(set(rows) != keys for _, rows in pieces[1:]):
        raise ValueError("sidecar shards do not share the same array keys")
    arrays = {
        key: np.empty((len(requested), *pieces[0][1][key].shape[1:]), dtype=pieces[0][1][key].dtype)
        for key in keys
    }
    filled = np.zeros(len(requested), dtype=bool)
    for positions, rows in pieces:
        for key in keys:
            arrays[key][positions] = rows[key]
        filled[positions] = True
    if not filled.all():
        missing = requested[~filled][:5].tolist()
        raise ValueError(f"sidecar is missing requested query rows, first missing: {missing}")
    return arrays


def _load_sidecar(path: Path, n_windows: int, pool_k: int, top_k: int,
                  methods: list[str], *, query_start: int = 0,
                  query_end: int | None = None,
                  query_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if query_indices is not None:
        arrays = _load_indexed_sidecar(path, query_indices)
        paths: list[Path] = []
    else:
        paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f"no sidecar NPZ files found under {path}")
        chunks = []
        for shard in paths:
            with np.load(shard, allow_pickle=False) as artifact:
                chunks.append({key: artifact[key] for key in artifact.files})
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise ValueError("sidecar shards do not share the same array keys")
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
    total_windows = len(np.asarray(arrays.get("official_distances")))
    query_end = int(query_start + n_windows) if query_end is None else int(query_end)
    if total_windows != n_windows:
        # Reuse a complete sidecar for a pilot range while retaining support
        # for exact-length query shards.
        if query_start < 0 or query_end > total_windows or query_start >= query_end:
            raise ValueError(
                f"artifact has {total_windows} rows; cannot slice "
                f"[{query_start}, {query_end})"
            )
        arrays = {
            key: value[query_start:query_end] if getattr(value, "ndim", 0) > 0
            and len(value) == total_windows else value
            for key, value in arrays.items()
        }
    official = np.asarray(arrays.get("official_distances"), dtype=np.float32)
    if official.shape != (n_windows, pool_k):
        raise ValueError(f"artifact shape {official.shape} != expected {(n_windows, pool_k)}")
    for method in methods:
        # Branch modules (official ARM replacements) do not require a
        # separate retrieval-ranking sidecar.  Keep this check tied to the
        # canonical method set so newly added branch names cannot accidentally
        # be interpreted as sidecar selection keys.
        if method in _NEW_ARM_METHODS:
            continue
        # MI-GRA is a matched post-selection intervention: it reuses the
        # official Top-k membership/order and therefore must not require a
        # separate selected_ranks_mi_gra array in the sidecar.
        if (
            method in {
                "base", "official", "crossrag_official", "trfa_official", "hera_official", "ipsra_official", "hcta_official", "mi_prior_shuffled", "output_mi",
                "official_mi_residual", "mi_select", "high_mi",
            }
            or method.startswith("mi_global_")
            or method.startswith("mi_gra")
            or method in _BOUND_HIGH_MI_ALL_METHOD_SET
        ):
            continue
        key = f"selected_ranks_{method}"
        selected = np.asarray(arrays.get(key))
        if selected.shape != (n_windows, top_k):
            raise ValueError(f"{key} shape {selected.shape} != expected {(n_windows, top_k)}")
        if selected.min(initial=0) < 0 or selected.max(initial=-1) >= pool_k:
            raise ValueError(f"{key} leaves top-{pool_k} pool")
        if any(len(set(row.tolist())) != top_k for row in selected):
            raise ValueError(f"{key} contains duplicate candidates")
        arrays[key] = selected.astype(np.int64)
    return arrays


def _sidecar_metadata(path: Path) -> dict[str, object]:
    if path.is_file():
        candidates = [path.with_suffix(".json")]
        candidates.extend(sorted(path.parent.glob("*.json")))
    else:
        # Score shards are commonly passed as ``.../<dataset>/scores`` while
        # their target metadata lives in the dataset-level manifest.
        candidates = sorted(path.glob("*.json"))
        candidates.extend(sorted(path.parent.glob("*.json")))
    for candidate in candidates:
        if candidate.exists():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Some legacy shard directories contain concatenated audit
                # records.  The NPZ arrays remain valid; metadata is optional
                # for the forecast runner, so ignore only the malformed
                # sidecar metadata and continue with the next candidate.
                continue
            if isinstance(value, dict):
                return value
    return {}


def _sidecar_mi_target(metadata: dict[str, object]) -> str | None:
    """Resolve the public MI target label from current and legacy metadata."""
    target = metadata.get("mi_target")
    if target is not None:
        value = str(target)
        if value == "I(H;Y)":
            return value
    source = str(metadata.get("mi_target_source", ""))
    if source == "future_truth":
        return "I(H;Y)"
    return None


def _require_ihy_sidecar(metadata: dict[str, object], context: str) -> None:
    """Fail closed unless an MI sidecar is explicitly I(H;Y)."""
    target = _sidecar_mi_target(metadata)
    if target != "I(H;Y)":
        raise ValueError(
            f"{context} requires an I(H;Y) MI sidecar; got {target!r}"
        )


def _validate_sidecar_dataset(metadata: dict[str, object], dataset: str) -> None:
    """Reject a MI sidecar built for a different dataset.

    Sidecars are intentionally dataset-local because hidden MI distances are
    not calibrated across series.  Older artifacts may not contain a dataset
    field, so absence remains backward compatible; an explicit mismatch is a
    hard error rather than a silent cross-dataset experiment.
    """
    observed = metadata.get("dataset")
    if observed is not None and str(observed) != str(dataset):
        raise ValueError(
            f"MI sidecar dataset={observed!r} does not match requested dataset={dataset!r}"
        )


def _array_hash(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _validate_horizon_protocol(
    *,
    methods: list[str],
    pool_k: int,
    top_k: int,
    pred_len: int,
    mi_target: str,
    mi_condition: str,
    arm_bias: bool,
    arm_attention_prior: bool,
    forecast_fusion: str,
    correction_gain: np.ndarray,
) -> None:
    """Fail closed for the pure horizon-MI comparison protocol."""
    if not HORIZON_METHODS.intersection(str(method) for method in methods):
        return
    violations: list[str] = []
    if int(pool_k) != 20:
        violations.append("pool_k must be 20")
    if int(top_k) != 10:
        violations.append("top_k must be 10")
    if int(pred_len) != 64:
        violations.append("pred_len must be 64")
    if str(mi_target) != "I(H;Y)":
        violations.append("mi_target must be I(H;Y)")
    if str(mi_condition) != "none":
        violations.append("mi_condition must be none")
    if bool(arm_bias):
        violations.append("arm_bias is not allowed")
    if bool(arm_attention_prior):
        violations.append("arm_attention_prior is not allowed")
    if str(forecast_fusion) not in {"none", "horizon_residual_transport"}:
        violations.append(
            "forecast_fusion must be none or horizon_residual_transport"
        )
    gain = np.asarray(correction_gain, dtype=np.float64)
    if gain.ndim not in {0, 1} or not np.isfinite(gain).all() or not np.all(gain == 1.0):
        violations.append("correction_gain must be exactly 1")
    if violations:
        raise ValueError("horizon MI protocol violation: " + "; ".join(violations))


def _horizon_block_metrics(
    prediction: np.ndarray, truth: np.ndarray, *, block_size: int = 16
) -> dict[str, list[float]]:
    """Return MSE/MAE for each contiguous forecast block."""
    pred = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    if (
        pred.shape != target.shape
        or pred.ndim != 2
        or int(block_size) < 1
        or pred.shape[1] % int(block_size)
        or not np.isfinite(pred).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("prediction/truth must share finite [N,H] with divisible horizon")
    blocks = pred.shape[1] // int(block_size)
    error = (pred - target).reshape(len(pred), blocks, int(block_size))
    return {
        "block_mse": np.square(error).mean(axis=(0, 2)).astype(float).tolist(),
        "block_mae": np.abs(error).mean(axis=(0, 2)).astype(float).tolist(),
    }


def _horizon_sidecar_diagnostics(
    sidecar: dict[str, np.ndarray],
    sidecar_metadata: dict[str, object],
    methods: list[str],
    *,
    pool_k: int,
    top_k: int,
) -> dict[str, object] | None:
    """Summarize horizon membership and assignment without touching futures."""
    requested = [method for method in methods if method in HORIZON_METHODS]
    if not requested:
        return None
    diagnostics: dict[str, object] = {
        "horizon_mi_protocol": True,
        "horizon_block_size": int(sidecar_metadata.get("horizon_block_size", 16)),
        "horizon_block_count": int(sidecar_metadata.get("horizon_block_count", 4)),
        "horizon_mi_observed": sidecar_metadata.get("horizon_mi_observed"),
        "horizon_mi_weights": sidecar_metadata.get("horizon_mi_weights"),
        "global_mi_observed": sidecar_metadata.get("global_mi_observed"),
        "global_mi_weights": sidecar_metadata.get("global_mi_weights"),
        "methods": {},
    }
    official = set(range(int(top_k)))
    for method in requested:
        selected_key = f"selected_ranks_{method}"
        assigned_key = f"assigned_ranks_{method}"
        if selected_key not in sidecar or assigned_key not in sidecar:
            raise ValueError(f"horizon sidecar is missing {selected_key} or {assigned_key}")
        selected = np.asarray(sidecar[selected_key], dtype=np.int64)
        assigned = np.asarray(sidecar[assigned_key], dtype=np.int64)
        if selected.ndim != 2 or selected.shape[1] != int(top_k):
            raise ValueError(f"{selected_key} has invalid shape {selected.shape}")
        if assigned.ndim != 2 or assigned.shape[0] != selected.shape[0]:
            raise ValueError(f"{assigned_key} has invalid shape {assigned.shape}")
        if assigned.shape[1] != int(sidecar_metadata.get("horizon_block_count", 4)):
            raise ValueError(f"{assigned_key} has invalid block count {assigned.shape[1]}")
        changed = np.asarray(
            [int(top_k) - len(set(row.tolist()).intersection(official)) for row in selected],
            dtype=np.int64,
        )
        counts = np.zeros((assigned.shape[1], int(pool_k)), dtype=np.int64)
        for block in range(assigned.shape[1]):
            values = assigned[:, block]
            if np.any(values < 0) or np.any(values >= int(pool_k)):
                raise ValueError(f"{assigned_key} leaves the official candidate pool")
            counts[block] = np.bincount(values, minlength=int(pool_k))
        diagnostics["methods"][method] = {
            "changed_candidates_mean": float(changed.mean()),
            "changed_candidates_min": int(changed.min(initial=0)),
            "changed_candidates_max": int(changed.max(initial=0)),
            "assignment_count_matrix": counts.tolist(),
            "selected_ranks_sha256": _array_hash(selected),
            "assigned_ranks_sha256": _array_hash(assigned),
        }
    return diagnostics


def _validate_sidecar_alignment(arrays: dict[str, np.ndarray], test_data, pool_k: int,
                                query_start: int = 0, query_end: int | None = None,
                                query_indices: np.ndarray | None = None) -> None:
    per_channel = int(test_data.tot_len)
    channels = int(test_data.enc_in)
    total = per_channel * channels
    query_end = total if query_end is None else int(query_end)
    flat = (np.asarray(query_indices, dtype=np.int64) if query_indices is not None
            else np.arange(int(query_start), query_end, dtype=np.int64))
    expected_n = len(flat)
    if len(arrays["official_distances"]) != expected_n:
        raise ValueError(f"sidecar has {len(arrays['official_distances'])} rows, expected {expected_n}")
    expected_channels = flat // per_channel
    if "channel_ids" in arrays and not np.array_equal(np.asarray(arrays["channel_ids"]), expected_channels):
        raise ValueError("MI sidecar channel_ids do not match TS-RAG channel-major order")
    # Official TS-RAG stores [time, top-k, channel]; the DataLoader flattens it
    # to [channel, time] in __getitem__, matching Align-RAG's order.
    if "candidate_starts" in arrays:
        expected = np.asarray(test_data.timestamp_idx[:per_channel, :pool_k, :], dtype=np.int64)
        # DataLoader rows are time-major [time, top-k, channel], while both
        # official runners evaluate flattened channel-major [channel, time].
        expected = np.transpose(expected, (2, 0, 1)).reshape(total, pool_k)
        expected = expected[flat]
        if not np.array_equal(np.asarray(arrays["candidate_starts"]), expected):
            raise ValueError("MI sidecar candidate_starts do not match TS-RAG retrieved CSV")


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path} is not a state dict")
    return {str(key).replace("module.", "", 1): tensor for key, tensor in value.items()}


def _gather(values: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    idx = torch.from_numpy(np.asarray(indices, dtype=np.int64)).to(values.device)
    if values.ndim == 3:
        return torch.gather(values, 1, idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    return torch.gather(values, 1, idx)


def _split_branch_and_correction_distances(
    method: str,
    branch_distance_source: str,
    official_pool: torch.Tensor,
    selected: np.ndarray,
    correction_distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate the frozen-branch distance from the outer correction distance."""
    import torch

    if branch_distance_source not in {"mi", "official"}:
        raise ValueError("branch_distance_source must be 'mi' or 'official'")
    branch_distance = correction_distance
    if method == "high_mi" and branch_distance_source == "official":
        index = torch.from_numpy(np.asarray(selected, dtype=np.int64)).to(
            official_pool.device
        )
        branch_distance = torch.gather(official_pool, 1, index)
    return branch_distance, correction_distance


def _bias_from_scores(scores: np.ndarray, strength: float) -> torch.Tensor:
    values = np.asarray(scores, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    z = centered / np.maximum(values.std(axis=1, keepdims=True), 1e-6)
    return torch.from_numpy(np.clip(-z * float(strength), -2.0, 2.0).astype(np.float32))


def _mi_gra_control_distances(
    values: np.ndarray,
    *,
    control: str,
    seed: int,
    global_offset: int,
) -> np.ndarray:
    """Return same-row MI values for Real/Shuffle/Reverse/Uniform controls."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < -1e-6):
        raise ValueError("MI-GRA control distances must be finite non-negative rank-2 values")
    if control == "real":
        return values.copy()
    if control == "uniform":
        return np.ones_like(values)
    output = np.empty_like(values)
    if control == "shuffled":
        for row, source in enumerate(values):
            rng = np.random.default_rng(int(seed) + int(global_offset) + row)
            output[row] = source[rng.permutation(source.shape[0])]
        return output
    if control == "reversed":
        order = np.argsort(values, axis=1, kind="stable")
        rows = np.arange(values.shape[0])[:, None]
        output[rows, order] = values[rows, order[:, ::-1]]
        return output
    raise ValueError(f"unknown MI-GRA control: {control!r}")


def _mi_fusion_weights(
    scores: "torch.Tensor",
    temperature: float = 1.0,
    reverse_weights: bool = False,
) -> "torch.Tensor":
    """Convert history-only candidate distances to a row-wise MI prior.

    Distances are lower-is-better.  Standardising within each query keeps the
    fusion invariant to dataset-specific MI scales, while ``temperature`` is
    the only frozen calibration parameter exposed to the experiment.
    """
    import torch

    if scores.ndim != 2:
        raise ValueError(f"fusion scores must be [batch,candidates], got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError("fusion scores contain non-finite values")
    if float(temperature) <= 0.0:
        raise ValueError("fusion temperature must be positive")
    centered = scores - scores.mean(dim=1, keepdim=True)
    z = centered / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    direction = 1.0 if reverse_weights else -1.0
    return torch.softmax(direction * z / float(temperature), dim=1)


def _bcsa_shape_scale(context: "torch.Tensor", candidate_history: "torch.Tensor") -> "torch.Tensor":
    """Return the fixed eta=0.5 BCSA dynamics scale for each candidate."""

    import torch

    if context.ndim != 2 or candidate_history.ndim != 3:
        raise ValueError("BCSA scale inputs must be [batch,seq] and [batch,candidates,seq]")
    eps = 1e-6
    query_delta = context[:, 1:] - context[:, :-1]
    candidate_delta = candidate_history[..., 1:] - candidate_history[..., :-1]
    query_mad = (query_delta - query_delta.median(dim=1).values[:, None]).abs().median(dim=1).values
    candidate_mad = (candidate_delta - candidate_delta.median(dim=2).values[..., None]).abs().median(dim=2).values
    return torch.sqrt((query_mad[:, None] + eps) / (candidate_mad + eps))


def _query_history_scale(context: "torch.Tensor") -> "torch.Tensor":
    """Return a robust per-query scale from first-difference MAD."""

    import torch

    if context.ndim != 2:
        raise ValueError("query history scale input must be [batch,seq]")
    if context.shape[1] < 2:
        raise ValueError("query history must contain at least two points")
    delta = context[:, 1:] - context[:, :-1]
    median = delta.median(dim=1).values
    return (delta - median[:, None]).abs().median(dim=1).values


def _align_candidate_futures(
    context: "torch.Tensor",
    candidate_history: "torch.Tensor",
    candidate_future: "torch.Tensor",
    *,
    alignment: str = "none",
    recent_window: int = 96,
    recent_tau: float = 32.0,
) -> "torch.Tensor":
    """Apply the selected candidate-future treatment.

    Official TS-RAG does not apply a separate forecast-space alignment after
    retrieval.  ``none`` therefore returns the retrieved future unchanged;
    the official model's own instance normalization/ARM remains untouched.
    The named alternatives are retained only for backwards-compatible audit
    artifacts and are not part of the fair reproduction protocol.
    """

    import torch

    if alignment not in {
        "none", "zscore", "last", "mean_shift", "recent_mean", "recent_ewmean", "bda", "endpoint", "bcsa", "alignrag",
    }:
        raise ValueError(f"unknown forecast alignment {alignment!r}")
    if context.ndim != 2 or candidate_history.ndim != 3 or candidate_future.ndim != 3:
        raise ValueError("alignment inputs have incompatible ranks")
    if context.shape[0] != candidate_history.shape[0]:
        raise ValueError("alignment batch dimensions do not match")
    if candidate_history.shape[:2] != candidate_future.shape[:2]:
        raise ValueError("alignment candidate dimensions do not match")
    if int(recent_window) <= 0 or float(recent_tau) <= 0.0:
        raise ValueError("recent alignment window and tau must be positive")

    if alignment == "none":
        return candidate_future

    if alignment == "endpoint":
        # Boundary Shape Alignment: only translate each candidate future so
        # its candidate-history endpoint matches the query-history endpoint.
        return context[:, None, -1:] + (
            candidate_future - candidate_history[..., -1:].contiguous()
        )


    if alignment == "alignrag":
        # Official Align-RAG: Wiener-shrunk affine amplitude alignment
        # followed by centered cross-correlation integer-lag phase shift.
        eps = 1e-6
        max_scale_ratio = 5.0
        batch, candidates, seq_len = candidate_history.shape
        total_len = candidate_history.shape[-1] + candidate_future.shape[-1]
        query_mean = context.mean(dim=1, keepdim=True).unsqueeze(1)
        query_std = context.std(dim=1, keepdim=True).clamp_min(eps).unsqueeze(1)
        history_mean = candidate_history.mean(dim=2, keepdim=True)
        history_std = candidate_history.std(dim=2, keepdim=True).clamp_min(eps)
        shrink = query_std / max_scale_ratio
        scale = (query_std * history_std) / (
            history_std * history_std + shrink * shrink
        )
        path = (torch.cat((candidate_history, candidate_future), dim=-1) - history_mean) * scale + query_mean

        query_centered = (context - context.mean(dim=1, keepdim=True)).unsqueeze(1)
        history_centered = path[:, :, :seq_len]
        history_centered = history_centered - history_centered.mean(dim=2, keepdim=True)
        max_lag = seq_len // 4
        best_corr = torch.full(
            (batch, candidates), -float("inf"), device=path.device, dtype=path.dtype
        )
        best_lag = torch.zeros(
            (batch, candidates), dtype=torch.long, device=path.device
        )
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                query_slice = query_centered[:, :, : seq_len + lag]
                history_slice = history_centered[:, :, -lag:]
            elif lag > 0:
                query_slice = query_centered[:, :, lag:]
                history_slice = history_centered[:, :, :-lag]
            else:
                query_slice = query_centered
                history_slice = history_centered
            corr = (query_slice * history_slice).sum(dim=-1)
            update = corr > best_corr
            best_corr = torch.where(update, corr, best_corr)
            best_lag = torch.where(
                update, torch.full_like(best_lag, lag), best_lag
            )
        positions = torch.arange(
            total_len, device=path.device, dtype=torch.long
        ).view(1, 1, total_len).expand(batch, candidates, total_len)
        positions = (positions - best_lag.unsqueeze(-1)).clamp(0, total_len - 1)
        aligned_path = torch.gather(path, 2, positions)
        return aligned_path[..., seq_len:seq_len + candidate_future.shape[-1]]

    if alignment == "bcsa":
        # Boundary-Conditioned Shape Alignment: scale only the candidate
        # endpoint displacement using fixed eta=0.5 in log space.
        ratio = _bcsa_shape_scale(context, candidate_history)
        displacement = candidate_future - candidate_history[..., -1:]
        return context[:, None, -1:] + ratio[..., None] * displacement

    if alignment == "bda":
        # History-only BDA: transfer robust first-order dynamics while using
        # the query endpoint as the absolute state.  No candidate future is
        # used to estimate the alignment statistics.
        eps = 1e-6
        query_delta = context[:, 1:] - context[:, :-1]
        candidate_delta = candidate_history[..., 1:] - candidate_history[..., :-1]
        query_median = query_delta.median(dim=1).values
        candidate_median = candidate_delta.median(dim=2).values
        query_mad = (query_delta - query_median[:, None]).abs().median(dim=1).values
        candidate_mad = (candidate_delta - candidate_median[..., None]).abs().median(dim=2).values
        query_scale_raw = 1.4826 * query_mad
        candidate_scale_raw = 1.4826 * candidate_mad
        query_scale = query_scale_raw + eps
        candidate_scale = candidate_scale_raw + eps
        ratio = torch.where(
            candidate_scale_raw < eps,
            torch.ones_like(candidate_scale),
            query_scale[:, None] / candidate_scale,
        )

        future_delta = torch.cat(
            (
                candidate_future[..., :1] - candidate_history[..., -1:],
                candidate_future[..., 1:] - candidate_future[..., :-1],
            ),
            dim=-1,
        )
        aligned_delta = query_median[:, None, None] + ratio[..., None] * (
            future_delta - candidate_median[..., None]
        )
        return context[:, None, -1:] + aligned_delta.cumsum(dim=-1)

    query_mean = context.mean(dim=1, keepdim=True)
    history_mean = candidate_history.mean(dim=2, keepdim=True)
    if alignment == "zscore":
        query_std = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        history_std = candidate_history.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized_future = (
            (candidate_future - history_mean) / history_std
        ).clamp(-5.0, 5.0)
        return query_mean[:, None, :] + query_std[:, None, :] * normalized_future
    if alignment == "last":
        return context[:, None, -1:] + (
            candidate_future - candidate_history[..., -1:].contiguous()
        )
    if alignment == "mean_shift":
        return candidate_future + query_mean[:, None, :] - history_mean

    recent = min(int(recent_window), context.shape[1], candidate_history.shape[-1])
    if alignment == "recent_ewmean":
        age = torch.arange(
            recent - 1, -1, -1, device=context.device, dtype=context.dtype,
        )
        weights = torch.exp(-age / float(recent_tau))
        weights = weights / weights.sum().clamp_min(1e-6)
        query_recent_mean = (context[..., -recent:] * weights).sum(
            dim=1, keepdim=True
        )
        candidate_recent_mean = (
            candidate_history[..., -recent:] * weights.view(1, 1, -1)
        ).sum(dim=2, keepdim=True)
    else:
        query_recent_mean = context[:, -recent:].mean(dim=1, keepdim=True)
        candidate_recent_mean = candidate_history[..., -recent:].mean(dim=2, keepdim=True)
    return candidate_future + query_recent_mean[:, None, :] - candidate_recent_mean


def _bcsa_candidate_residual_dispersion(
    context: "torch.Tensor",
    retrieved_selected: "torch.Tensor",
    *,
    seq_len: int,
    pred_len: int,
) -> "torch.Tensor":
    """Return normalized unweighted dispersion of BCSA candidate displacements.

    The displacement is measured after BCSA endpoint anchoring.  Dividing by
    the query-history variance makes the resulting ``s_c`` dimensionless and
    comparable across datasets; the candidate average itself remains uniform,
    matching the reliability definition rather than the MI weighting prior.
    """

    import torch

    if context.ndim != 2 or retrieved_selected.ndim != 3:
        raise ValueError("candidate reliability inputs have incompatible ranks")
    candidate_history = retrieved_selected[..., :int(seq_len)]
    candidate_future = retrieved_selected[
        ..., int(seq_len):int(seq_len) + int(pred_len)
    ]
    aligned = _align_candidate_futures(
        context,
        candidate_history,
        candidate_future,
        alignment="bcsa",
    )
    displacement = aligned - context[:, None, -1:]
    center = displacement.mean(dim=1, keepdim=True)
    dispersion = torch.square(displacement - center).mean(dim=-1).mean(dim=1)
    query_variance = context.var(dim=1, unbiased=False).clamp_min(1e-6)
    return dispersion / query_variance


def _bound_residual_reference(
    arm_prediction: "torch.Tensor",
    candidate_base_prediction: "torch.Tensor",
    *,
    reference: str = "base_model",
) -> "torch.Tensor":
    """Return the forecast reference subtracted from each candidate future.

    ``base_model`` preserves the original residual-prototype definition.  The
    ``arm`` reference makes the correction an explicit residual around the
    current ARM forecast: ``candidate_future - F_ARM``.  Both references use
    only the candidate history and the current arm prediction; neither can
    access the query future.
    """

    import torch

    if reference not in {"base_model", "arm"}:
        raise ValueError(f"unknown bound residual reference {reference!r}")
    if arm_prediction.ndim != 2 or candidate_base_prediction.ndim != 3:
        raise ValueError("bound residual reference inputs must be [B,H] and [B,K,H]")
    if arm_prediction.shape[0] != candidate_base_prediction.shape[0]:
        raise ValueError("bound residual reference batch dimensions do not match")
    if arm_prediction.shape[1] != candidate_base_prediction.shape[2]:
        raise ValueError("bound residual reference horizon dimensions do not match")
    if not torch.isfinite(arm_prediction).all() or not torch.isfinite(candidate_base_prediction).all():
        raise ValueError("bound residual reference inputs contain non-finite values")
    return arm_prediction[:, None, :] if reference == "arm" else candidate_base_prediction


def _residual_forecast_disagreement_confidence(
    candidate_residuals: "torch.Tensor",
    fusion_scores: "torch.Tensor",
    context: "torch.Tensor",
    *,
    normalization_distances: "torch.Tensor | None" = None,
    weighting: str = "median_pool",
    temperature: float = 1.0,
    confidence_scale: float = 1.0,
) -> "torch.Tensor":
    """Return a forecast-disagreement gate for the residual prototype.

    The gate uses only retrieved candidate forecasts, their history-only
    distance weights, and the query history scale.  It never reads the query
    future.  The residual prototype itself owns the weighting implementation
    so the gate and correction cannot silently use different candidate priors.
    """

    import torch

    if candidate_residuals.ndim not in {3, 4} or fusion_scores.ndim != 2:
        raise ValueError("residual disagreement inputs have incompatible ranks")
    if context.ndim != 2 or context.shape[0] != candidate_residuals.shape[0]:
        raise ValueError("residual disagreement context must be [batch,seq]")
    if float(confidence_scale) <= 0.0 or not np.isfinite(float(confidence_scale)):
        raise ValueError("residual disagreement confidence scale must be positive")
    result = residual_prototype_correction(
        candidate_residuals.detach().cpu().numpy(),
        fusion_scores.detach().cpu().numpy(),
        reference_distances=(
            normalization_distances.detach().cpu().numpy()
            if normalization_distances is not None else None
        ),
        estimator="mean",
        temperature=float(temperature),
        weighting=weighting,
    )
    weights = torch.as_tensor(
        result.weights,
        device=candidate_residuals.device,
        dtype=candidate_residuals.dtype,
    )
    prototype = torch.as_tensor(
        result.correction,
        device=candidate_residuals.device,
        dtype=candidate_residuals.dtype,
    )
    residual_error = candidate_residuals - prototype.unsqueeze(1)
    expanded_weights = weights[(...,) + (None,) * (candidate_residuals.ndim - 2)]
    dispersion = (expanded_weights * torch.square(residual_error)).sum(dim=1)
    dispersion = dispersion.reshape(dispersion.shape[0], -1).mean(dim=1)
    history_scale = context.std(dim=1, unbiased=False).clamp_min(1e-6)
    return torch.exp(
        -dispersion / torch.square(history_scale) / float(confidence_scale)
    ).clamp(0.0, 1.0)


def _apply_mi_forecast_fusion(
    official_prediction: "torch.Tensor",
    context: "torch.Tensor",
    retrieved_selected: "torch.Tensor",
    fusion_scores: "torch.Tensor",
    *,
    seq_len: int,
    pred_len: int,
    strength: float | list[float] = 0.1,
    temperature: float = 1.0,
    reverse_weights: bool = False,
    confidence_scale: float = 1.0,
    alignment: str = "none",
    weighting: str = "mi",
    confidence_mode: str = "forecast_disagreement",
    use_confidence: bool = True,
    confidence_override: "torch.Tensor | None" = None,
    confidence_multiplier: "torch.Tensor | None" = None,
    alignment_window: int = 96,
    alignment_tau: float = 32.0,
    return_consensus: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor"] | tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Fuse a TS-RAG forecast with a history-aligned retrieved-future expert.

    The retrieved futures are already part of the official TS-RAG input.  We
    align each one using only query/candidate histories, weight them with the
    supplied history-only MI distances, and apply a bounded residual update.
    A disagreement-based confidence shrinks the update when retrieved futures
    disagree, preserving the frozen TS-RAG forecast in uncertain cases.
    """
    import torch

    if float(confidence_scale) < 0.0:
        raise ValueError("forecast fusion confidence scale must be non-negative")
    if weighting not in {"mi", "mi_bcsa_reliability"}:
        raise ValueError(f"unknown forecast fusion weighting {weighting!r}")
    if confidence_mode not in {
        "forecast_disagreement",
        "dual_disagreement",
        "dual_rank_disagreement",
        "mi_rank_agreement",
        "mi_rank_entropy",
    }:
        raise ValueError(f"unknown forecast fusion confidence mode {confidence_mode!r}")
    if weighting == "mi_bcsa_reliability" and alignment != "bcsa":
        raise ValueError("BCSA reliability weighting requires alignment=bcsa")
    if confidence_mode in {"dual_disagreement", "dual_rank_disagreement"} and alignment != "bcsa":
        raise ValueError("dual disagreement confidence requires alignment=bcsa")
    if official_prediction.ndim != 2 or context.ndim != 2:
        raise ValueError("official_prediction and context must be rank-2 tensors")
    if retrieved_selected.ndim != 3 or fusion_scores.ndim != 2:
        raise ValueError("retrieved_selected and fusion_scores have incompatible ranks")
    batch, candidates, total_len = retrieved_selected.shape
    if official_prediction.shape != (batch, int(pred_len)):
        raise ValueError(
            f"official_prediction shape {tuple(official_prediction.shape)} "
            f"!= {(batch, int(pred_len))}"
        )
    if context.shape != (batch, int(seq_len)):
        raise ValueError(
            f"context shape {tuple(context.shape)} != {(batch, int(seq_len))}"
        )
    if fusion_scores.shape != (batch, candidates):
        raise ValueError(
            f"fusion_scores shape {tuple(fusion_scores.shape)} "
            f"!= {(batch, candidates)}"
        )
    if total_len < int(seq_len) + int(pred_len):
        raise ValueError("retrieved candidates are shorter than seq_len + pred_len")
    if not torch.isfinite(official_prediction).all() or not torch.isfinite(context).all():
        raise ValueError("forecast fusion inputs contain non-finite values")
    strength_array = torch.as_tensor(
        strength, device=official_prediction.device, dtype=official_prediction.dtype
    )
    if strength_array.ndim == 0:
        valid_strength = torch.isfinite(strength_array) and float(strength_array) >= 0.0
    elif strength_array.ndim == 1:
        valid_strength = (
            strength_array.shape[0] == int(pred_len)
            and bool(torch.isfinite(strength_array).all())
            and bool(torch.all(strength_array >= 0.0))
        )
    else:
        valid_strength = False
    if not valid_strength:
        raise ValueError("forecast fusion strength must be a scalar or pred-len vector that is finite and non-negative")

    query_std = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    candidate_history = retrieved_selected[..., :int(seq_len)]
    candidate_future = retrieved_selected[
        ..., int(seq_len):int(seq_len) + int(pred_len)
    ]
    aligned = _align_candidate_futures(
        context,
        candidate_history,
        candidate_future,
        alignment=alignment,
        recent_window=alignment_window,
        recent_tau=alignment_tau,
    )
    weights = _mi_fusion_weights(
        fusion_scores,
        temperature=temperature,
        reverse_weights=reverse_weights,
    )
    if weighting == "mi_bcsa_reliability":
        shape_scale = _bcsa_shape_scale(context, candidate_history)
        reliability = torch.exp(-torch.abs(torch.log(shape_scale.clamp_min(1e-6))))
        weights = weights * reliability
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    consensus = torch.sum(weights.unsqueeze(-1) * aligned, dim=1)
    # Average over forecast horizons rather than summing all H values; the
    # latter would make confidence collapse exponentially with pred_len.
    dispersion = torch.sum(
        weights.unsqueeze(-1) * torch.square(aligned - consensus[:, None, :]), dim=1
    ).mean(dim=1) / torch.square(query_std.squeeze(-1)).clamp_min(1e-6)
    if confidence_mode in {"dual_disagreement", "dual_rank_disagreement"}:
        model_disagreement = torch.square(consensus - official_prediction).mean(dim=1)
        model_disagreement = model_disagreement / torch.square(query_std.squeeze(-1)).clamp_min(1e-6)
        dispersion = dispersion + model_disagreement
    if float(confidence_scale) == 0.0:
        confidence = torch.ones_like(dispersion)
    else:
        confidence = torch.exp(-dispersion / float(confidence_scale)).clamp(0.0, 1.0)
    if confidence_override is not None:
        if confidence_override.ndim != 1 or confidence_override.shape[0] != batch:
            raise ValueError("confidence override must have one value per query")
        if not torch.isfinite(confidence_override).all() or torch.any(
            (confidence_override < 0.0) | (confidence_override > 1.0)
        ):
            raise ValueError("confidence override must be finite and in [0,1]")
        override = confidence_override.to(
            device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
        confidence = (
            confidence * override
            if confidence_mode == "dual_rank_disagreement"
            else override
        )
    elif not use_confidence:
        confidence = torch.ones_like(confidence)
    if confidence_multiplier is not None:
        if confidence_multiplier.ndim != 1 or confidence_multiplier.shape[0] != batch:
            raise ValueError("confidence multiplier must have one value per query")
        if not torch.isfinite(confidence_multiplier).all() or torch.any(
            confidence_multiplier < 0.0
        ):
            raise ValueError("confidence multiplier must be finite and non-negative")
        confidence = confidence * confidence_multiplier.to(
            device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
    fused = apply_residual_shrink(
        official_prediction,
        consensus,
        confidence,
        alpha=strength_array,
    )
    if return_consensus:
        return fused, confidence, consensus
    return fused, confidence


def _mi_attribution_ablation(
    anchor_prediction: "torch.Tensor",
    context: "torch.Tensor",
    mi_retrieved_selected: "torch.Tensor",
    mi_scores: "torch.Tensor",
    official_retrieved_selected: "torch.Tensor",
    mi_scores_on_official: "torch.Tensor",
    official_scores_on_mi: "torch.Tensor",
    *,
    seq_len: int,
    pred_len: int,
    strength: float | list[float] = 0.1,
    temperature: float = 1.0,
    confidence_scale: float = 1.0,
) -> tuple[dict[str, "torch.Tensor"], dict[str, "torch.Tensor"]]:
    """Return single-factor MI attribution arms around one fixed Full anchor.

    The anchor is the already-computed Full MI-TS-RAG branch prediction.  Only
    the outer candidate selection, distance weighting, or confidence gate is
    changed, so the three arms do not silently re-estimate the TS-RAG anchor.
    """

    predictions: dict[str, "torch.Tensor"] = {}
    confidences: dict[str, "torch.Tensor"] = {}

    predictions["w/o_mi_selection"], confidences["w/o_mi_selection"] = (
        _apply_mi_forecast_fusion(
            anchor_prediction,
            context,
            official_retrieved_selected,
            mi_scores_on_official,
            seq_len=seq_len,
            pred_len=pred_len,
            strength=strength,
            temperature=temperature,
            confidence_scale=confidence_scale,
            alignment="bcsa",
            weighting="mi",
            confidence_mode="forecast_disagreement",
            use_confidence=True,
        )
    )
    predictions["w/o_mi_weighting"], confidences["w/o_mi_weighting"] = (
        _apply_mi_forecast_fusion(
            anchor_prediction,
            context,
            mi_retrieved_selected,
            official_scores_on_mi,
            seq_len=seq_len,
            pred_len=pred_len,
            strength=strength,
            temperature=temperature,
            confidence_scale=confidence_scale,
            alignment="bcsa",
            weighting="mi",
            confidence_mode="forecast_disagreement",
            use_confidence=True,
        )
    )
    predictions["w/o_mi_gate"], confidences["w/o_mi_gate"] = (
        _apply_mi_forecast_fusion(
            anchor_prediction,
            context,
            mi_retrieved_selected,
            mi_scores,
            seq_len=seq_len,
            pred_len=pred_len,
            strength=strength,
            temperature=temperature,
            confidence_scale=confidence_scale,
            alignment="bcsa",
            weighting="mi",
            confidence_mode="forecast_disagreement",
            use_confidence=False,
        )
    )
    return predictions, confidences


def _apply_horizon_residual_transport(
    official_prediction: "torch.Tensor",
    candidate_residuals: "torch.Tensor",
    selected_horizon_distances: "torch.Tensor",
    reference_horizon_distances: "torch.Tensor",
    *,
    block_size: int,
    strength: float | list[float] = 1.0,
    temperature: float = 1.0,
    gate_threshold: float = 0.05,
    gate_power: float = 1.0,
    correction_clip: float = 0.0,
    return_consensus: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor"] | tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Apply history-only, blockwise MI residual transport."""

    import torch

    if official_prediction.ndim != 2 or candidate_residuals.ndim != 3:
        raise ValueError("horizon residual transport expects rank-2/3 forecasts")
    if selected_horizon_distances.ndim != 3 or reference_horizon_distances.ndim != 3:
        raise ValueError("horizon MI distances must be rank-3")
    if candidate_residuals.shape[0] != official_prediction.shape[0]:
        raise ValueError("horizon residual transport batch sizes disagree")
    if candidate_residuals.shape[2:] != official_prediction.shape[1:]:
        raise ValueError("horizon residual transport forecast shapes disagree")
    if selected_horizon_distances.shape[0] != candidate_residuals.shape[0]:
        raise ValueError("selected horizon distances have a different batch size")
    if selected_horizon_distances.shape[2] != candidate_residuals.shape[1]:
        raise ValueError("selected horizon distances do not match candidates")
    if reference_horizon_distances.shape[:2] != selected_horizon_distances.shape[:2]:
        raise ValueError("reference horizon distances do not match block dimensions")
    if not all(
        torch.isfinite(value).all()
        for value in (
            official_prediction,
            candidate_residuals,
            selected_horizon_distances,
            reference_horizon_distances,
        )
    ):
        raise ValueError("horizon residual transport inputs contain non-finite values")
    if torch.any(selected_horizon_distances < 0.0) or torch.any(
        reference_horizon_distances < 0.0
    ):
        raise ValueError("horizon MI distances must be non-negative")

    selected_np = selected_horizon_distances.detach().cpu().numpy()
    reference_np = reference_horizon_distances.detach().cpu().numpy()
    gate = mi_concentration_gate(
        selected_np,
        reference_distances=reference_np,
        threshold=float(gate_threshold),
        power=float(gate_power),
        temperature=float(temperature),
    )
    result = horizon_residual_transport(
        candidate_residuals.detach().cpu().numpy(),
        selected_np,
        block_size=int(block_size),
        reference_distances=reference_np,
        temperature=float(temperature),
        estimator="mean",
        gate=gate,
    )
    correction = np.asarray(result.correction, dtype=np.float32)
    clip_value = float(correction_clip)
    if not np.isfinite(clip_value) or clip_value < 0.0:
        raise ValueError("correction_clip must be finite and non-negative")
    if clip_value:
        correction = np.clip(correction, -clip_value, clip_value)
    strength_values = np.asarray(strength, dtype=np.float64)
    horizon = correction.shape[1]
    blocks = selected_np.shape[1]
    if strength_values.ndim == 0:
        expanded_strength = np.full(horizon, float(strength_values))
    elif strength_values.ndim == 1 and strength_values.shape[0] == blocks:
        expanded_strength = np.repeat(strength_values, int(block_size))
    elif strength_values.ndim == 1 and strength_values.shape[0] == horizon:
        expanded_strength = strength_values
    else:
        raise ValueError("strength must be scalar, block-wise, or horizon-wise")
    if not np.isfinite(expanded_strength).all() or np.any(expanded_strength < 0.0):
        raise ValueError("strength must be finite and non-negative")
    base_np = official_prediction.detach().cpu().numpy()
    fused_np = base_np + correction * expanded_strength[None, :]
    consensus_np = base_np + correction
    fused = torch.as_tensor(
        fused_np, device=official_prediction.device, dtype=official_prediction.dtype
    )
    confidence = torch.as_tensor(
        gate.mean(axis=1), device=official_prediction.device, dtype=official_prediction.dtype
    )
    if return_consensus:
        consensus = torch.as_tensor(
            consensus_np, device=official_prediction.device, dtype=official_prediction.dtype
        )
        return fused, confidence, consensus
    return fused, confidence


def _fixed_confidence_weighting_ablation(
    baseline_prediction: "torch.Tensor",
    context: "torch.Tensor",
    retrieved_selected: "torch.Tensor",
    mi_scores: "torch.Tensor",
    official_scores: "torch.Tensor",
    fixed_confidence: "torch.Tensor",
    *,
    seq_len: int,
    pred_len: int,
    strength: float | list[float] = 0.1,
    temperature: float = 1.0,
    alignment: str = "none",
    ordinary_retrieved_selected: "torch.Tensor | None" = None,
    ordinary_official_scores: "torch.Tensor | None" = None,
) -> dict[str, "torch.Tensor"]:
    """Isolate candidate weighting with selection matched to each weight."""

    import torch

    if mi_scores.shape != official_scores.shape:
        raise ValueError("MI and official ablation scores must have identical shapes")
    if ordinary_retrieved_selected is None:
        ordinary_retrieved_selected = retrieved_selected
    if ordinary_official_scores is None:
        ordinary_official_scores = official_scores
    if ordinary_retrieved_selected.shape != retrieved_selected.shape:
        raise ValueError("ordinary and high-MI candidate tensors must have identical shapes")
    if ordinary_official_scores.shape != official_scores.shape:
        raise ValueError("ordinary and high-MI distance tensors must have identical shapes")
    arm_inputs = {
        "uniform_weight": (retrieved_selected, torch.zeros_like(mi_scores)),
        "official_distance_weight": (
            ordinary_retrieved_selected,
            ordinary_official_scores,
        ),
        "mi_distance_weight": (retrieved_selected, mi_scores),
    }
    predictions = {"no_future": baseline_prediction}
    for arm, (candidates, scores) in arm_inputs.items():
        candidate_history = candidates[..., :int(seq_len)]
        candidate_future = candidates[
            ..., int(seq_len):int(seq_len) + int(pred_len)
        ]
        aligned = _align_candidate_futures(
            context,
            candidate_history,
            candidate_future,
            alignment=alignment,
        )
        weights = _mi_fusion_weights(scores, temperature=temperature)
        consensus = torch.sum(weights.unsqueeze(-1) * aligned, dim=1)
        predictions[arm] = apply_residual_shrink(
            baseline_prediction,
            consensus,
            fixed_confidence,
            alpha=torch.as_tensor(
                strength,
                device=baseline_prediction.device,
                dtype=baseline_prediction.dtype,
            ),
        )
    return predictions


def _factorial_confidence_weighting_ablation(
    baseline_prediction: "torch.Tensor",
    context: "torch.Tensor",
    retrieved_selected: "torch.Tensor",
    mi_scores: "torch.Tensor",
    official_scores: "torch.Tensor",
    fixed_confidence: "torch.Tensor",
    *,
    seq_len: int,
    pred_len: int,
    strength: float | list[float] = 0.1,
    temperature: float = 1.0,
    ordinary_retrieved_selected: "torch.Tensor | None" = None,
    ordinary_official_scores: "torch.Tensor | None" = None,
) -> dict[str, "torch.Tensor"]:
    """Return the 2x2 alignment-by-weighting factorial ablation.

    ``ordinary_distance`` and ``standardization_only`` use ordinary-distance
    Top-10 candidates.  The two remaining arms use high-MI Top-10 candidates
    and replace the official distance with the history-only MI distance, with
    and without that same standardization.  TS-RAG prediction, confidence,
    strength, and temperature are frozen.
    """

    ordinary = _fixed_confidence_weighting_ablation(
        baseline_prediction,
        context,
        retrieved_selected,
        mi_scores,
        official_scores,
        fixed_confidence,
        seq_len=seq_len,
        pred_len=pred_len,
        strength=strength,
        temperature=temperature,
        alignment="none",
        ordinary_retrieved_selected=ordinary_retrieved_selected,
        ordinary_official_scores=ordinary_official_scores,
    )
    standardized = _fixed_confidence_weighting_ablation(
        baseline_prediction,
        context,
        retrieved_selected,
        mi_scores,
        official_scores,
        fixed_confidence,
        seq_len=seq_len,
        pred_len=pred_len,
        strength=strength,
        temperature=temperature,
        alignment="zscore",
        ordinary_retrieved_selected=ordinary_retrieved_selected,
        ordinary_official_scores=ordinary_official_scores,
    )
    return {
        "ordinary_distance": ordinary["official_distance_weight"],
        "standardization_only": standardized["official_distance_weight"],
        "weighting_only": ordinary["mi_distance_weight"],
        "standardized_weighted": standardized["mi_distance_weight"],
        "no_future": baseline_prediction,
    }


def _apply_residual_prototype_correction(
    official_prediction: "torch.Tensor",
    candidate_residuals: "torch.Tensor",
    fusion_scores: "torch.Tensor",
    *,
    lambda_: float,
    correction_clip: float | None = 0.0,
    normalization_distances: "torch.Tensor | None" = None,
    hybrid_beta: float | None = None,
    hybrid_official_distances: "torch.Tensor | None" = None,
    hybrid_official_normalization_distances: "torch.Tensor | None" = None,
    estimator: str = "mean",
    temperature: float = 1.0,
    weighting: str = "median_pool",
    confidence_gate: bool = False,
    confidence_override: "torch.Tensor | None" = None,
    confidence_floor: float = 0.0,
    mi_reliability_gate: bool = False,
    mi_reliability_official_distances: "torch.Tensor | None" = None,
    mi_reliability_threshold: float = 0.05,
    mi_reliability_power: float = 1.0,
    return_consensus: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor"] | tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Apply the gate-free residual-prototype correction to a TS-RAG forecast.

    ``normalization_distances`` is the complete MI pool row when the caller
    wants the protocol's Top-20 median scale instead of the selected Top-K
    median fallback.  When ``hybrid_beta`` is provided, ``fusion_scores`` are
    MI distances and ``hybrid_official_distances`` are the matched official
    distances; each source is normalized separately before mixing.
    """

    import torch

    if official_prediction.ndim not in {2, 3}:
        raise ValueError("official_prediction must be rank-2 or rank-3")
    if candidate_residuals.ndim not in {3, 4} or fusion_scores.ndim != 2:
        raise ValueError("candidate residuals and scores have incompatible ranks")
    if candidate_residuals.shape[:2] != fusion_scores.shape:
        raise ValueError("candidate residuals and fusion scores have incompatible shapes")
    if candidate_residuals.shape[0] != official_prediction.shape[0]:
        raise ValueError("official prediction and candidate residuals have different batches")
    if candidate_residuals.shape[2:] != official_prediction.shape[1:]:
        raise ValueError("candidate residuals and official prediction have incompatible shapes")
    if normalization_distances is not None:
        if normalization_distances.ndim != 2:
            raise ValueError("normalization distances must be rank-2")
        if normalization_distances.shape[0] != official_prediction.shape[0]:
            raise ValueError("normalization distances have a different batch size")
    if hybrid_beta is not None:
        if not np.isfinite(float(hybrid_beta)) or not 0.0 <= float(hybrid_beta) <= 1.0:
            raise ValueError("hybrid_beta must be finite and lie in [0,1]")
        if hybrid_official_distances is None:
            raise ValueError("hybrid official distances are required")
        if hybrid_official_distances.ndim != 2:
            raise ValueError("hybrid official distances must be rank-2")
        if hybrid_official_distances.shape != fusion_scores.shape:
            raise ValueError("hybrid official distances must match fusion scores")
        if hybrid_official_normalization_distances is not None:
            if hybrid_official_normalization_distances.ndim != 2:
                raise ValueError("hybrid official normalization distances must be rank-2")
            if hybrid_official_normalization_distances.shape[0] != official_prediction.shape[0]:
                raise ValueError(
                    "hybrid official normalization distances have a different batch size"
                )
    lambda_array = np.asarray(lambda_, dtype=np.float64)
    if (
        lambda_array.ndim not in {0, 1}
        or not np.isfinite(lambda_array).all()
        or np.any(lambda_array < 0.0)
    ):
        raise ValueError("lambda_ must be a finite scalar or vector and non-negative")
    horizon = int(candidate_residuals.shape[2])
    if lambda_array.ndim == 1 and lambda_array.shape[0] != horizon:
        raise ValueError(
            f"lambda_ vector must match the residual horizon ({horizon})"
        )
    clip_value = 0.0 if correction_clip is None else float(correction_clip)
    if not np.isfinite(clip_value) or clip_value < 0.0:
        raise ValueError("correction_clip must be finite and non-negative")
    finite_values = (official_prediction, candidate_residuals, fusion_scores)
    if normalization_distances is not None:
        finite_values += (normalization_distances,)
    if not all(torch.isfinite(value).all() for value in finite_values):
        raise ValueError("residual prototype inputs contain non-finite values")
    if weighting not in {"median_pool", "standardized"}:
        raise ValueError("residual prototype weighting must be 'median_pool' or 'standardized'")
    if confidence_override is not None:
        if confidence_override.ndim != 1 or confidence_override.shape[0] != official_prediction.shape[0]:
            raise ValueError("confidence override must have one value per query")
        if not torch.isfinite(confidence_override).all() or torch.any(
            (confidence_override < 0.0) | (confidence_override > 1.0)
        ):
            raise ValueError("confidence override must be finite and in [0,1]")
        if confidence_gate or mi_reliability_gate:
            raise ValueError("confidence override cannot be combined with another confidence gate")
    if not np.isfinite(float(confidence_floor)) or not 0.0 <= float(confidence_floor) < 1.0:
        raise ValueError("confidence_floor must be finite and in [0,1)")
    if not np.isfinite(float(mi_reliability_threshold)) or not 0.0 <= float(mi_reliability_threshold) < 1.0:
        raise ValueError("mi_reliability_threshold must be finite and in [0,1)")
    if not np.isfinite(float(mi_reliability_power)) or float(mi_reliability_power) <= 0.0:
        raise ValueError("mi_reliability_power must be finite and positive")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if mi_reliability_gate:
        if mi_reliability_official_distances is None:
            raise ValueError("official distances are required for MI reliability gating")
        if mi_reliability_official_distances.ndim != 2:
            raise ValueError("MI reliability official distances must be rank-2")
        if mi_reliability_official_distances.shape != fusion_scores.shape:
            raise ValueError("MI reliability official distances must match fusion scores")
        if not torch.isfinite(mi_reliability_official_distances).all() or torch.any(
            mi_reliability_official_distances < 0.0
        ):
            raise ValueError("MI reliability official distances must be finite and non-negative")

    residual_values = candidate_residuals.detach().cpu().numpy()
    fusion_values = fusion_scores.detach().cpu().numpy()
    mi_reference_values = (
        normalization_distances.detach().cpu().numpy()
        if normalization_distances is not None else None
    )
    if hybrid_beta is None:
        result = residual_prototype_correction(
            residual_values,
            fusion_values,
            reference_distances=mi_reference_values,
            estimator=estimator,
            temperature=float(temperature),
            weighting=weighting,
        )
    else:
        result = hybrid_residual_prototype_correction(
            residual_values,
            fusion_values,
            hybrid_official_distances.detach().cpu().numpy(),
            beta=float(hybrid_beta),
            reference_mi_distances=mi_reference_values,
            reference_official_distances=(
                hybrid_official_normalization_distances.detach().cpu().numpy()
                if hybrid_official_normalization_distances is not None else None
            ),
            estimator=estimator,
        )
    correction = torch.as_tensor(
        result.correction,
        device=official_prediction.device,
        dtype=official_prediction.dtype,
    )
    if clip_value > 0.0:
        correction = correction.clamp(min=-clip_value, max=clip_value)
    confidence = torch.as_tensor(
        result.confidence,
        device=official_prediction.device,
        dtype=official_prediction.dtype,
    )
    effective_confidence = torch.ones_like(confidence)
    if confidence_override is not None:
        effective_confidence = confidence_override.to(
            device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
    if confidence_gate:
        effective_confidence = torch.as_tensor(
            residual_prototype_confidence(
                candidate_residuals.detach().cpu().numpy(),
                fusion_scores.detach().cpu().numpy(),
                reference_distances=(
                    normalization_distances.detach().cpu().numpy()
                    if normalization_distances is not None else None
                ),
                weighting=weighting,
                temperature=float(temperature),
            ),
            device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
        if float(confidence_floor) > 0.0:
            effective_confidence = (
                (effective_confidence - float(confidence_floor))
                / (1.0 - float(confidence_floor))
            ).clamp(0.0, 1.0)
    if mi_reliability_gate:
        reliability = compute_mi_reliability_gate(
            fusion_scores.detach().cpu().numpy(),
            mi_reliability_official_distances.detach().cpu().numpy(),
            threshold=float(mi_reliability_threshold),
            power=float(mi_reliability_power),
        ).gate
        # MI reliability is an alternative history-only admission rule.  Do
        # not multiply it by the legacy residual-direction gate: a useful MI
        # ranking can be paired with heterogeneous residual directions, and
        # the MI-specific control must be able to keep that correction alive.
        effective_confidence = torch.as_tensor(
            reliability,
            device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
    if lambda_array.ndim == 0:
        lambda_tensor = torch.as_tensor(
            float(lambda_array), device=official_prediction.device,
            dtype=official_prediction.dtype,
        )
    elif correction.ndim == 2:
        lambda_tensor = torch.as_tensor(
            lambda_array, device=official_prediction.device,
            dtype=official_prediction.dtype,
        ).view(1, -1)
    else:
        lambda_tensor = torch.as_tensor(
            lambda_array, device=official_prediction.device,
            dtype=official_prediction.dtype,
        ).view(1, -1, 1)
    if confidence_override is not None or confidence_gate or mi_reliability_gate:
        fused = official_prediction + lambda_tensor * effective_confidence.unsqueeze(-1) * correction
        confidence = effective_confidence
    else:
        # The residual-prototype experiment deliberately removes g_q.  The
        # compatibility confidence tensor returned above is all ones, but is
        # not part of the forecast equation.
        fused = official_prediction + lambda_tensor * correction
    if return_consensus:
        return fused, confidence, official_prediction + correction
    return fused, confidence


def _bias_targets(selector: str) -> set[str]:
    """Map the public selector name to bounded and MI-global method names."""
    if selector == "mi_gra":
        return set(_MI_GRA_BIAS_TARGETS)
    if selector == "both":
        return {"high_mi", "mi_prior", "mi_global_high"}
    if selector == "high_mi":
        return {"high_mi", "mi_global_high"}
    if selector == "mi_bias":
        return {
            "official_mi_bias", "official_random_bias",
            "official_uniform_bias", "official_low_mi_bias",
        }
    if selector == "all_priors":
        return {
            "mi_prior", "no_mi_prior", "uniform_no_mi_prior",
        "recency_no_mi_prior", "mi_recency_prior",
        "mi_orthogonal_recency_prior",
            "reliability_gated_mi_recency", "null_mi_prior",
        }
    return {"mi_prior"}


def should_apply_attention_prior(method: str, prior_method: str) -> bool:
    """Apply the MI attention prior only to its matching MI prediction row.

    ``official`` is the fixed TS-RAG reference and ``base`` is the non-RAG
    control.  Neither may be changed by an optional MI prediction module.
    Requiring a method/source match also prevents, for example, a high-MI
    prior from silently modifying a low-MI control.
    """

    method_name = str(method)
    prior_name = str(prior_method)
    if prior_name == "high_mi":
        # The explicit residual protocol names its MI arm
        # ``high_mi_residual_*`` while using the same high-MI Top-10 selector.
        # Treat those names as the matching high-MI ARM row as well; official
        # and official-distance controls remain untouched.
        return method_name == "high_mi" or method_name.startswith("high_mi_")
    return method_name == prior_name


def _contaminate_pool(
    retrieved: "torch.Tensor",
    distances: "torch.Tensor",
    context: "torch.Tensor",
    sidecar: dict[str, np.ndarray],
    local_offset: int,
    seq_len: int,
    retriever_rawdata: np.ndarray,
    mode: str,
    period: int,
    seed: int = 2021,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Apply a history-only corruption to the top-20 candidate pool.

    The random-time and wrong-channel variants reconstruct candidates from the
    train-history retrieval database.  Far-distance and seasonal variants only
    reorder the already retrieved top-20 rows.  No candidate future is read by
    this function, so it is safe to use before selection.
    """
    if mode == "none":
        return retrieved, distances
    batch_size = int(retrieved.shape[0])
    pool_k = int(retrieved.shape[1])
    total_len = int(retrieved.shape[2])
    starts = np.asarray(sidecar["candidate_starts"])[local_offset:local_offset + batch_size]
    channels = np.asarray(sidecar["channel_ids"])[local_offset:local_offset + batch_size]
    if starts.shape != (batch_size, pool_k):
        raise ValueError(f"candidate_starts shape {starts.shape} does not match batch {(batch_size, pool_k)}")

    if mode in {"far_distance", "season_mismatch"}:
        if mode == "far_distance":
            order = np.tile(np.arange(pool_k, dtype=np.int64), (batch_size, 1))
            if pool_k >= 2 * (pool_k // 2):
                half = pool_k // 2
                order = np.concatenate([order[:, half:], order[:, :half]], axis=1)
        else:
            # Circular phase distance is computed from timestamps only.  It is
            # a deliberately conservative proxy for a season mismatch.
            period = max(int(period), 1)
            phase = np.mod(starts - np.asarray(sidecar["query_origins"])[local_offset:local_offset + batch_size, None], period)
            gap = np.minimum(phase, period - phase)
            order = np.argsort(-gap, axis=1, kind="stable")
        idx = torch.from_numpy(order).to(retrieved.device)
        idx3 = idx.unsqueeze(-1).expand(-1, -1, total_len)
        return torch.gather(retrieved, 1, idx3), torch.gather(distances, 1, idx)

    # Rebuild a candidate pool from the train-history database.  The database
    # is channel-major and standardized once in main(), matching the loader.
    raw = np.asarray(retriever_rawdata, dtype=np.float32)
    max_start = raw.shape[1] - total_len
    if max_start < 0:
        raise ValueError("retrieval database is shorter than seq_len + pred_len")
    rebuilt = np.empty((batch_size, pool_k, total_len), dtype=np.float32)
    rng = np.random.default_rng(int(seed) + int(local_offset))
    for row in range(batch_size):
        query_channel = int(channels[row])
        for rank in range(pool_k):
            if mode == "wrong_channel":
                channel = (query_channel + 1 + rank % max(raw.shape[0] - 1, 1)) % raw.shape[0]
                start = int(starts[row, rank]) % (max_start + 1)
            else:  # random_time
                channel = query_channel
                start = int(rng.integers(0, max_start + 1))
            rebuilt[row, rank] = raw[channel, start:start + total_len]
    rebuilt_t = torch.from_numpy(rebuilt).to(retrieved.device, dtype=retrieved.dtype)
    context_np = context.detach().cpu().numpy().astype(np.float32, copy=False)
    rebuilt_dist = np.mean(np.square(rebuilt[:, :, :seq_len] - context_np[:, None, :]), axis=2)
    return rebuilt_t, torch.from_numpy(rebuilt_dist).to(retrieved.device, dtype=distances.dtype)


def _update_stream_metric_stats(
    stats: dict[str, float], prediction: np.ndarray, truth: np.ndarray,
) -> None:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if prediction_array.shape != truth_array.shape:
        raise ValueError(
            f"stream metric shapes do not match: {prediction_array.shape} vs "
            f"{truth_array.shape}"
        )
    error = prediction_array - truth_array
    stats["count"] = stats.get("count", 0.0) + float(error.size)
    stats["sum_sq"] = stats.get("sum_sq", 0.0) + float(np.square(error).sum())
    stats["sum_abs"] = stats.get("sum_abs", 0.0) + float(np.abs(error).sum())


def _metric_rows(predictions: dict[str, np.ndarray], truth: np.ndarray,
                 retrieval_scores: dict[str, np.ndarray], *, mi_target: str,
                 mi_condition: str, arm_bias: bool = False,
                 augment_mode: str = "moe",
                 arm_bias_strength: float = 0.5,
                 arm_bias_method: str = "high_mi",
                 forecast_fusion: str = "none",
                 forecast_fusion_strength: float | list[float] = 0.1,
                 forecast_fusion_temperature: float = 1.0,
                 forecast_fusion_reverse_weights: bool = False,
                 forecast_fusion_confidence_scale: float = 1.0,
                 forecast_fusion_confidence_floor: float = 0.0,
                 forecast_fusion_confidence_mode: str = "forecast_disagreement",
                 forecast_fusion_alignment: str = "none",
                 output_mi_gamma: float = 1.0,
                 output_mi_prior_strength: float = 1.0,
                 output_mi_prior_centering: str = "raw",
                 output_mi_distance_key: str = "high_mi_distances",
                 forecast_confidence: dict[str, np.ndarray] | None = None,
                 retrieval_block_scores: dict[str, np.ndarray] | None = None,
                 oracle_recall: dict[str, np.ndarray] | None = None,
                 official_overlap: dict[str, np.ndarray] | None = None,
                 contamination_mode: str = "none",
                 oracle_best_mse: float | None = None,
                 seed: int = 2021,
                 stream_stats: dict[str, dict[str, float]] | None = None,
                 ) -> list[dict[str, object]]:
    # Method-one runs are used for large matrices to parallelize GPUs.  They
    # intentionally contain only one prediction key, so relative-to-base is
    # unavailable until the merge helper combines the method artifacts.
    base = predictions.get("base")
    base_stats = stream_stats.get("base") if stream_stats else None
    base_mse = (
        float(base_stats["sum_sq"] / base_stats["count"])
        if base_stats and base_stats.get("count", 0.0) > 0.0
        else float(np.square(base - truth).mean()) if base is not None else float("nan")
    )
    rows = []
    bias_methods = _bias_targets(arm_bias_method)
    for method, prediction in predictions.items():
        method_stats = stream_stats.get(method) if stream_stats else None
        if method_stats and method_stats.get("count", 0.0) > 0.0:
            mse = float(method_stats["sum_sq"] / method_stats["count"])
            mae = float(method_stats["sum_abs"] / method_stats["count"])
        else:
            mse = float(np.square(prediction - truth).mean())
            mae = float(np.abs(prediction - truth).mean())
        retrieval = retrieval_scores.get(method)
        if method_stats:
            block_values = {"block_mse": [], "block_mae": []}
            block_count = 0
        elif prediction.ndim == 2 and prediction.shape[1] % 16 == 0:
            block_values = _horizon_block_metrics(prediction, truth, block_size=16)
            block_count = len(block_values["block_mse"])
        else:
            block_values = {"block_mse": [], "block_mae": []}
            block_count = 0
        retrieval_blocks = (
            np.asarray(retrieval_block_scores[method], dtype=np.float64).mean(axis=0).tolist()
            if retrieval_block_scores is not None and method in retrieval_block_scores
            else [float("nan")] * block_count
        )
        if method == "base":
            retriever_name = "none"
        elif method == "official":
            retriever_name = "official_top10"
        elif method in _NEW_ARM_METHODS:
            retriever_name = "official_top10"
        elif _bound_method_spec(method) is not None:
            retriever_name = _bound_method_spec(method)["selection"]
        elif method == "output_mi":
            retriever_name = "official_top10"
        elif method.startswith("mi_gra"):
            retriever_name = "official_top10_mi_gra"
        elif method in HORIZON_METHODS:
            retriever_name = "horizon_balanced_top20"
        elif method == "mi_select":
            retriever_name = "mi_select_top20"
        elif method == "recency_no_mi_prior":
            retriever_name = "recency_top10"
        elif method == "mi_recency_prior":
            retriever_name = "mi_recency_top10"
        elif method == "recent":
            retriever_name = "causal_recency_top10"
        elif method == "mi_recent":
            retriever_name = "high_mi_top10"
        elif method == "mi_gate":
            retriever_name = "mi_gate_top20"
        elif method == "official_mi_bias":
            retriever_name = "official_top10_mi_bias"
        elif method in {"official_random_bias", "official_uniform_bias", "official_low_mi_bias"}:
            retriever_name = "official_top10_bias_control"
        elif method in {"mi_insert", "low_mi_insert", "random_insert", "all_patch_insert", "residual_insert"}:
            retriever_name = "mi_insert_top20"
        elif method == "residual_mi":
            retriever_name = "residual_mi_top20"
        elif method == "random":
            retriever_name = "random_top20"
        elif method == "mi_prior_shuffled":
            retriever_name = "mi_sidecar_shuffled"
        elif method.startswith("mi_global_"):
            retriever_name = "global_mi_history"
        elif method.endswith("_mmr"):
            retriever_name = "mi_sidecar_mmr"
        else:
            retriever_name = "mi_sidecar"
        if method == "base":
            fusion_name = "none"
        elif method == "official":
            fusion_name = "official_arm"
        elif method == "output_mi":
            fusion_name = "official_output_mi_prior"
        elif method.startswith("mi_gra"):
            fusion_name = "official_mi_gated_residual_arm"
        elif method == "mi_gate":
            fusion_name = "mi_membership_overlap_gate"
        elif method == "mi_select":
            fusion_name = "mi_membership_official_order"
        elif method == "recency_no_mi_prior":
            fusion_name = "recency_arm"
        elif method == "mi_recency_prior":
            fusion_name = "mi_recency_arm"
        elif method == "recent":
            fusion_name = "recent_arm"
        elif method == "mi_recent":
            fusion_name = "mi_recent_arm"
        elif method == "official_mi_bias":
            fusion_name = "official_membership_mi_bias"
        elif method in {"official_random_bias", "official_uniform_bias", "official_low_mi_bias"}:
            fusion_name = "official_membership_bias_control"
        elif method in {"mi_insert", "low_mi_insert", "random_insert", "all_patch_insert", "residual_insert"}:
            fusion_name = "mi_membership_insert"
        elif method == "residual_mi":
            fusion_name = "residual_mi_membership"
        elif method.startswith("mi_global_") and arm_bias and method in bias_methods:
            fusion_name = "global_mi_arm_mi_bias"
        elif arm_bias and method in bias_methods:
            fusion_name = "official_arm_mi_bias"
        elif method.startswith("mi_global_"):
            fusion_name = "global_mi_arm"
        elif method == "crossrag_official":
            fusion_name = "crossrag_cross_attention"
        elif method == "trfa_official":
            fusion_name = "trfa_temporal_relational"
        elif method == "hera_official":
            fusion_name = "hera_horizon_evidence"
        elif method == "ipsra_official":
            fusion_name = "ipsra_identity_set_residual"
        elif method == "csea_official":
            fusion_name = "csea_kernel_set_evidence"
        elif method == "bfa_official":
            fusion_name = "bfa_boundary_forecast"
        elif method == "bfa_hidden_official":
            fusion_name = "bfa_hidden_residual"
        elif method == "oera_official":
            fusion_name = "oera_output_residual"
        elif method == "heca_official":
            fusion_name = "heca_horizon_cross_attention"
        elif method == "htfa_official":
            fusion_name = "htfa_latent_transport_fusion"
        elif method == "htfa_forecast_official":
            fusion_name = "htfa_forecast_only_transport_fusion"
        elif method == "chsfa_official":
            fusion_name = "chsfa_cross_horizon_selective_fusion"
        elif method == "hcta_official":
            fusion_name = "hcta_horizon_cross_transport"
        elif method == "rqca_official":
            fusion_name = "rqca_robust_quantile_consensus"
        elif method == "rgfa_official":
            fusion_name = "rgfa_reliability_gated_future_fusion"
        elif method == "raem_official":
            fusion_name = "raem_relation_aware_expert_mixer"
        elif method == "hrca_official":
            fusion_name = "hrca_horizon_relational_cross_attention"
        elif method == "brea_official":
            fusion_name = "brea_boundary_relational_evidence"
        elif method == "sqem_official":
            fusion_name = "sqem_setwise_query_evidence"
        elif method == "bcsqm_official":
            fusion_name = "bcsqm_boundary_conditioned_setwise_query_mixer"
        elif method == "qcra_official":
            fusion_name = "qcra_query_centered_relational"
        elif method == "rsm_official":
            fusion_name = "rsm_relational_set_mixer"
        elif method == "hcsm_official":
            fusion_name = "hcsm_history_conditioned_set_mixer"
        elif method == "dqr_official":
            fusion_name = "dqr_distance_aware_query_router"
        elif method == "rem_official":
            fusion_name = "rem_relational_expert_mixer"
        else:
            fusion_name = (
                "rcam"
                if augment_mode == "rcam"
                    else (
                    "trfa" if augment_mode == "trfa"
                    else "hera" if augment_mode == "hera"
                    else "ipsra" if augment_mode == "ipsra"
                    else "csea_fair" if augment_mode == "csea_fair"
                    else "csea" if augment_mode == "csea"
                    else "bfa_fair" if augment_mode == "bfa_fair"
                    else "bfa" if augment_mode == "bfa"
                    else "bfa_hidden" if augment_mode in {"bfa_hidden", "bfa_hidden_fair"}
                    else "oera" if augment_mode in {"oera", "oera_fair"}
                    else "heca" if augment_mode in {"heca", "heca_fair"}
                    else "htfa_forecast" if augment_mode == "htfa_forecast_fair"
                    else "htfa" if augment_mode == "htfa_fair"
                    else "chsfa" if augment_mode == "chsfa_fair"
                    else "hcta" if augment_mode == "hcta_fair"
                    else "rqca" if augment_mode == "rqca_fair"
                    else "rgfa" if augment_mode == "rgfa_fair"
                    else "raem" if augment_mode == "raem_fair"
                    else "hrca" if augment_mode == "hrca_fair"
                    else "brea" if augment_mode == "brea_fair"
                    else "bcsqm" if augment_mode == "bcsqm_fair"
                    else "sqem" if augment_mode == "sqem_fair"
                    else "qcra" if augment_mode == "qcra_fair"
                    else "rsm" if augment_mode == "rsm_fair"
                    else "hcsm" if augment_mode == "hcsm_fair"
                    else "dqr" if augment_mode == "dqr_fair"
                    else "rem" if augment_mode == "rem_fair"
                    else "official_arm"
                )
            )
        bound_spec = _bound_method_spec(method)
        if bound_spec is not None:
            fusion_name = (
                f"{bound_spec['selection']}_{bound_spec['distance']}"
                f"_forecast_{bound_spec['alignment']}"
            )
        # Keep the official TS-RAG row untouched.  Forecast fusion is the
        # proposed MI extension and must not silently redefine the baseline.
        fusion_applied = forecast_fusion != "none" and method not in {
            "base", "official", *_NEW_ARM_METHODS
        }
        if fusion_applied:
            fusion_name = f"{fusion_name}+forecast_{forecast_fusion}"
        row_forecast_alignment = (
            bound_spec["alignment"]
            if bound_spec is not None and fusion_applied
            else forecast_fusion_alignment if fusion_applied else "none"
        )
        rows.append({
            "method": method,
            "mse": mse,
            "mae": mae,
            "relative_mse_vs_base": (float(mse / max(base_mse, 1e-12) - 1.0)
                                      if np.isfinite(base_mse) else float("nan")),
            "mi_target": mi_target,
            "mi_condition": mi_condition,
            "retriever": retriever_name,
            "fusion": fusion_name,
            "forecast_fusion": forecast_fusion if fusion_applied else "none",
            "forecast_fusion_alignment": row_forecast_alignment,
            "bound_ablation_selection": (
                bound_spec["selection"] if bound_spec is not None else "none"
            ),
            "bound_ablation_distance": (
                bound_spec["distance"] if bound_spec is not None else "none"
            ),
            "bound_ablation_alignment": row_forecast_alignment,
            "forecast_fusion_strength": (
                _serialize_scalar_or_vector(forecast_fusion_strength) if fusion_applied else 0.0
            ),
            "forecast_fusion_temperature": (
                1.0
                if fusion_applied and forecast_fusion == "bound_high_mi_linear_mix"
                else float(forecast_fusion_temperature) if fusion_applied else 0.0
            ),
            "forecast_fusion_reverse_weights": (
                bool(forecast_fusion_reverse_weights) if fusion_applied else False
            ),
            "forecast_fusion_confidence_scale": (
                0.0
                if fusion_applied and forecast_fusion == "bound_high_mi_linear_mix"
                else float(forecast_fusion_confidence_scale) if fusion_applied else 0.0
            ),
            "forecast_fusion_confidence_floor": (
                float(forecast_fusion_confidence_floor) if fusion_applied else 0.0
            ),
            "forecast_fusion_confidence_mode": (
                "none"
                if fusion_applied and forecast_fusion == "bound_high_mi_linear_mix"
                else str(forecast_fusion_confidence_mode) if fusion_applied else "none"
            ),
            "output_mi_gamma": (
                float(output_mi_gamma) if method == "output_mi" else 0.0
            ),
            "output_mi_prior_strength": (
                float(output_mi_prior_strength) if method == "output_mi" else 0.0
            ),
            "output_mi_prior_centering": (
                str(output_mi_prior_centering) if method == "output_mi" else "none"
            ),
            "output_mi_distance_key": (
                str(output_mi_distance_key) if method == "output_mi" else "none"
            ),
            "forecast_confidence_mean": (
                float(np.asarray(forecast_confidence[method]).mean())
                if fusion_applied and forecast_confidence and method in forecast_confidence else float("nan")
            ),
            "arm_bias": bool(arm_bias and method in bias_methods),
            "arm_bias_strength": float(arm_bias_strength) if arm_bias and method in bias_methods else 0.0,
            "retrieval_future_mse": float(np.asarray(retrieval).mean()) if retrieval is not None else float("nan"),
            "retrieval_residual_mse": float(np.asarray(retrieval).mean()) if retrieval is not None else float("nan"),
            "retrieval_future_mse_by_block": retrieval_blocks,
            "block_mse": block_values["block_mse"],
            "block_mae": block_values["block_mae"],
            "oracle_recall_at_10": (float(np.asarray(oracle_recall[method]).mean())
                                    if oracle_recall and method in oracle_recall else float("nan")),
            "official_overlap_at_10": (float(np.asarray(official_overlap[method]).mean())
                                        if official_overlap and method in official_overlap else float("nan")),
            "contamination_mode": contamination_mode,
            "offline_only": False,
            "seed": int(seed),
            "leakage_ok": True,
        })
    if oracle_best_mse is not None:
        rows.append({
            "method": "oracle",
            "mse": float("nan"), "mae": float("nan"),
            "relative_mse_vs_base": float("nan"),
            "mi_target": mi_target, "mi_condition": mi_condition,
            "retriever": "offline_future_oracle",
            "fusion": "none", "arm_bias": False, "arm_bias_strength": 0.0,
            "forecast_fusion": "none", "forecast_fusion_strength": 0.0,
            "forecast_fusion_temperature": 0.0,
            "forecast_fusion_confidence_scale": 0.0,
            "forecast_fusion_confidence_floor": 0.0,
            "forecast_fusion_confidence_mode": "none",
            "output_mi_gamma": 0.0, "output_mi_distance_key": "none",
            "output_mi_prior_strength": 0.0,
            "output_mi_prior_centering": "none",
            "forecast_confidence_mean": float("nan"),
            "retrieval_future_mse": float(oracle_best_mse),
            "retrieval_residual_mse": float(oracle_best_mse),
            "retrieval_future_mse_by_block": [],
            "block_mse": [],
            "block_mae": [],
            "oracle_recall_at_10": 1.0,
            "official_overlap_at_10": float("nan"),
            "contamination_mode": contamination_mode,
            "offline_only": True,
            "seed": int(seed),
            "leakage_ok": True,
        })
    return rows


def _serialize_scalar_or_vector(value: float | list[float] | np.ndarray) -> float | list[float]:
    """Keep scalar metadata scalar and make horizon vectors JSON/CSV-safe."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return array.tolist()
    raise ValueError("expected a scalar or rank-1 numeric value")


def _expand_block_strength(value: object, payload: dict[str, object], pred_len: int) -> object:
    """Expand compact block metadata into one frozen strength per horizon."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] == int(pred_len):
        return value
    if payload.get("alpha_mode") != "blocks":
        return value
    num_blocks = int(payload.get("num_blocks", array.shape[0]))
    if num_blocks != array.shape[0] or num_blocks <= 0 or num_blocks > int(pred_len):
        raise ValueError("block alpha metadata is inconsistent with pred-len")
    expanded = np.empty(int(pred_len), dtype=np.float64)
    for block_index, horizon_block in enumerate(np.array_split(np.arange(int(pred_len)), num_blocks)):
        expanded[horizon_block] = array[block_index]
    return expanded.tolist()


def _load_mi_match_config(path: str | None) -> dict[str, float | str] | None:
    """Load a validation-frozen history-only MI-match calibration."""

    if not path:
        return None
    calibration_path = Path(path).resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("forecast-fusion-mi-match-file must contain a JSON object")
    if payload.get("feature", "min_high_mi_distance") != "min_high_mi_distance":
        raise ValueError(
            "forecast-fusion-mi-match-file supports only min_high_mi_distance"
        )
    required = ("center", "scale", "slope", "floor", "ceiling")
    if any(key not in payload for key in required):
        raise ValueError(
            "forecast-fusion-mi-match-file must contain "
            + ", ".join(required)
        )
    values = {key: float(payload[key]) for key in required}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("MI-match calibration values must be finite")
    if values["scale"] <= 0.0:
        raise ValueError("MI-match calibration scale must be positive")
    if values["floor"] < 0.0 or values["floor"] > values["ceiling"]:
        raise ValueError("MI-match calibration bounds must be ordered and non-negative")
    return {
        "feature": "min_high_mi_distance",
        **values,
        "path": str(calibration_path),
    }


def main() -> None:
    global torch
    args = _args()
    args.gpu = validate_gpu_id(args.gpu)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not np.isfinite(float(args.arm_attention_prior_strength)) or args.arm_attention_prior_strength < 0.0:
        raise ValueError("arm-attention-prior-strength must be finite and non-negative")
    import torch
    # The TS-RAG checkpoint leaves several retrieval layers randomly
    # initialized.  Seed every RNG before constructing the model so paired
    # method comparisons are reproducible across processes and GPUs.
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if args.forecast_fusion_strength_file:
        alpha_path = Path(args.forecast_fusion_strength_file).resolve()
        alpha_payload = json.loads(alpha_path.read_text(encoding="utf-8"))
        if not isinstance(alpha_payload, dict):
            raise ValueError("forecast-fusion-strength-file must contain a JSON object")
        selected_alpha = alpha_payload.get("selected_alpha", alpha_payload.get("alpha"))
        if selected_alpha is None:
            raise ValueError("forecast-fusion-strength-file must contain 'selected_alpha' or 'alpha'")
        args.forecast_fusion_strength = _expand_block_strength(
            selected_alpha, alpha_payload, args.pred_len
        )
    mi_match_config = _load_mi_match_config(args.forecast_fusion_mi_match_file)
    if mi_match_config is not None and args.forecast_fusion != "high_mi":
        raise ValueError(
            "forecast-fusion-mi-match-file currently requires "
            "--forecast-fusion high_mi"
        )
    strength_array = np.asarray(args.forecast_fusion_strength, dtype=np.float64)
    if strength_array.ndim not in {0, 1} or not np.isfinite(strength_array).all():
        raise ValueError("forecast-fusion-strength must be a finite scalar or vector")
    if strength_array.ndim == 1 and strength_array.shape[0] != args.pred_len:
        raise ValueError("forecast-fusion-strength vector must match pred-len")
    if np.any(strength_array < 0.0):
        raise ValueError("forecast-fusion-strength must be non-negative")
    args.forecast_fusion_strength = _serialize_scalar_or_vector(strength_array)
    if (
        not np.isfinite(float(args.forecast_fusion_correction_clip))
        or float(args.forecast_fusion_correction_clip) < 0.0
    ):
        raise ValueError(
            "forecast-fusion-correction-clip must be finite and non-negative"
        )
    if float(args.forecast_fusion_temperature) <= 0.0:
        raise ValueError("forecast-fusion-temperature must be positive")
    if (
        not np.isfinite(float(args.forecast_fusion_hybrid_beta))
        or not 0.0 <= float(args.forecast_fusion_hybrid_beta) <= 1.0
    ):
        raise ValueError("forecast-fusion-hybrid-beta must be finite and lie in [0,1]")
    if float(args.forecast_fusion_confidence_scale) < 0.0:
        raise ValueError("forecast-fusion-confidence-scale must be non-negative")
    if (
        not np.isfinite(float(args.forecast_fusion_history_entropy_power))
        or float(args.forecast_fusion_history_entropy_power) < 0.0
    ):
        raise ValueError(
            "forecast-fusion-history-entropy-power must be finite and non-negative"
        )
    if (
        args.forecast_fusion_confidence_mode in _HISTORY_CONFIDENCE_FUSIONS
        and args.forecast_fusion not in (
            _HISTORY_CONFIDENCE_SUPPORTED
            | _BOUND_HIGH_MI_ALL_METHOD_SET
        )
    ):
        raise ValueError(
            "history-only MI confidence requires forecast-fusion to be one of "
            f"{sorted(_HISTORY_CONFIDENCE_SUPPORTED)}"
        )
    if args.forecast_fusion in _COUPLED_RECENT_MI_FUSIONS:
        required = {"recent", "mi_recent"}
        unsupported = set(methods) - {"base", "official", "recent", "mi_recent"}
        if not required.issubset(methods) or unsupported:
            raise ValueError(
                "coupled_recent_mi requires only official/base plus recent+mi_recent; "
                "candidate selection and distance weighting are fixed as one pair"
            )
        if args.forecast_fusion_confidence_mode != "forecast_disagreement":
            raise ValueError(
                "coupled_recent_mi uses the same forecast-disagreement confidence "
                "for the recent and MI+recent arms"
            )
    if (
        not np.isfinite(float(args.forecast_fusion_mi_branch_gate_threshold))
        or not 0.0 <= float(args.forecast_fusion_mi_branch_gate_threshold) < 1.0
    ):
        raise ValueError(
            "forecast-fusion-mi-branch-gate-threshold must be finite and lie in [0,1)"
        )
    if (
        not np.isfinite(float(args.forecast_fusion_mi_branch_gate_power))
        or float(args.forecast_fusion_mi_branch_gate_power) <= 0.0
    ):
        raise ValueError(
            "forecast-fusion-mi-branch-gate-power must be finite and positive"
        )
    if (
        not np.isfinite(float(args.forecast_fusion_confidence_floor))
        or not 0.0 <= float(args.forecast_fusion_confidence_floor) < 1.0
    ):
        raise ValueError("forecast-fusion-confidence-floor must be finite and lie in [0,1)")
    if (
        not np.isfinite(float(args.forecast_fusion_mi_reliability_threshold))
        or not 0.0 <= float(args.forecast_fusion_mi_reliability_threshold) < 1.0
    ):
        raise ValueError("forecast-fusion-mi-reliability-threshold must be finite and lie in [0,1)")
    if (
        not np.isfinite(float(args.forecast_fusion_mi_reliability_power))
        or float(args.forecast_fusion_mi_reliability_power) <= 0.0
    ):
        raise ValueError("forecast-fusion-mi-reliability-power must be finite and positive")
    if not np.isfinite(float(args.output_mi_gamma)) or not 0.0 <= float(args.output_mi_gamma) <= 1.0:
        raise ValueError("output-mi-gamma must be finite and lie in [0,1]")
    if not np.isfinite(float(args.output_mi_prior_strength)) or args.output_mi_prior_strength < 0.0:
        raise ValueError("output-mi-prior-strength must be finite and non-negative")
    if not np.isfinite(float(args.mi_gra_strength)) or args.mi_gra_strength < 0.0:
        raise ValueError("mi-gra-strength must be finite and non-negative")
    if not np.isfinite(float(args.mi_gra_mi_strength)):
        raise ValueError("mi-gra-mi-strength must be finite")
    if not np.isfinite(float(args.mi_gra_temperature)) or args.mi_gra_temperature <= 0.0:
        raise ValueError("mi-gra-temperature must be finite and positive")
    if int(args.mi_gra_hidden_dim) < 1:
        raise ValueError("mi-gra-hidden-dim must be positive")
    if args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS:
        if strength_array.ndim == 1 and strength_array.shape[0] != args.pred_len:
            raise ValueError(
                "residual-prototype fusion strength vector must match pred-len"
            )
        if float(args.forecast_fusion_temperature) != 1.0:
            raise ValueError("residual-prototype fusion fixes temperature at 1")
        if float(args.forecast_fusion_confidence_scale) != 1.0:
            raise ValueError("residual-prototype fusion fixes confidence scale at 1")
        if args.forecast_fusion_alignment != "none":
            raise ValueError("residual-prototype fusion does not support alignment")
    if int(args.bound_high_mi_alignment_window) <= 0:
        raise ValueError("bound-high-mi-alignment-window must be positive")
    if args.forecast_fusion == "bound_high_mi_residual_ablation":
        if (
            args.bound_high_mi_residual_confidence != "disagreement"
            and float(args.forecast_fusion_confidence_scale) != 1.0
        ):
            raise ValueError(
                "bound_high_mi_residual_ablation fixes non-disagreement residual "
                "confidence scale at 1"
            )
    if args.forecast_fusion in _HORIZON_RESIDUAL_FUSIONS:
        if set(methods) - ({"base", "official"} | HORIZON_METHODS):
            raise ValueError(
                "horizon_residual_transport only supports base, official, and horizon MI methods"
            )
        if "horizon_mi" not in methods:
            raise ValueError(
                "horizon_residual_transport requires the horizon_mi selector"
            )
        if "official" not in methods:
            raise ValueError(
                "horizon_residual_transport requires the official reference method"
            )
        if float(args.forecast_fusion_confidence_scale) != 1.0:
            raise ValueError(
                "horizon_residual_transport fixes confidence_scale at 1"
            )
        if args.forecast_fusion_alignment != "none":
            raise ValueError(
                "horizon_residual_transport does not support candidate alignment"
            )
    gain_map: dict[str, np.ndarray] = {}
    if args.correction_gain_file:
        gain_path = Path(args.correction_gain_file).resolve()
        payload = json.loads(gain_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("correction-gain-file must contain a JSON object")
        if "gains" in payload:
            gain_map = {
                str(method): np.asarray(value, dtype=np.float32)
                for method, value in dict(payload["gains"]).items()
            }
            args.correction_gain = payload.get("gain", 1.0)
        elif "gain" in payload:
            args.correction_gain = payload["gain"]
        else:
            raise ValueError("correction-gain-file must contain 'gain' or 'gains'")
    gain_array = np.asarray(args.correction_gain, dtype=np.float32)
    def validate_gain(value: np.ndarray, name: str) -> None:
        if value.ndim not in {0, 1} or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite scalar or horizon vector")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(f"{name} must lie in [0,1]")
        if value.ndim == 1 and value.shape[0] != args.pred_len:
            raise ValueError(f"{name} must match pred-len")
    validate_gain(gain_array, "correction gain")
    for method, value in gain_map.items():
        validate_gain(value, f"correction gain for {method}")

    def gain_for(method: str) -> np.ndarray:
        return gain_map.get(method, gain_array)
    weighting_ablation_alignments = [
        item.strip()
        for item in (args.weighting_ablation_alignments or "").split(",")
        if item.strip()
    ]
    if args.stream_metrics and (args.save_preds or args.save_correction_analysis):
        raise ValueError(
            "--stream-metrics cannot be combined with --save-preds or "
            "--save-correction-analysis"
        )
    if args.stream_metrics and weighting_ablation_alignments:
        raise ValueError("--stream-metrics cannot be combined with weighting ablations")
    if len(weighting_ablation_alignments) != len(set(weighting_ablation_alignments)):
        raise ValueError("weighting-ablation alignments must be unique")
    unknown_alignments = set(weighting_ablation_alignments) - {
        "none", "zscore", "last", "mean_shift", "recent_mean", "recent_ewmean", "bda", "endpoint", "bcsa", "alignrag"
    }
    if unknown_alignments:
        raise ValueError(
            f"unknown weighting-ablation alignments: {sorted(unknown_alignments)}"
        )
    factorial_ablation_enabled = {"none", "zscore"}.issubset(
        set(weighting_ablation_alignments)
    )
    allowed = {
        "base", "official", *_NEW_ARM_METHODS, "official_permuted", "uniform", "random", "low_mi", "high_mi",
        "mi_select", "mi_gate", "mi_anchor_gate", "low_mi_anchor_gate",
        "random_patch_gate", "all_patch_gate",
        "official_mi_bias", "mi_insert", "low_mi_insert", "random_insert",
        "all_patch_insert", "residual_mi",
        "residual_insert",
        "mi_residual_prototype",
        "official_mi_residual",
        "official_random_bias", "official_uniform_bias", "official_low_mi_bias",
        "high_mi_boundary", "low_mi_boundary",
        "random_patch_boundary", "all_patch_boundary",
        "high_mi_anchor", "low_mi_anchor", "random_patch", "all_patch", "mi_prior",
        "mi_prior_shuffled",
        "mi_student", "student_no_mi", "no_mi_prior", "null_mi_prior",
        "uniform_no_mi_prior", "recency_no_mi_prior", "mi_recency_prior",
        "recent", "mi_recent",
        "mi_orthogonal_recency_prior",
        "quota_recency_mi",
        "reliability_gated_mi_recency",
        "mi_prior_mmr", "no_mi_prior_mmr",
        "uniform_no_mi_prior_mmr", "null_mi_prior_mmr", "pairwise_gain",
        "conformal_mi_prior",
        "output_mi",
        "mi_gra", "mi_gra_no_mi", "mi_gra_shuffled", "mi_gra_reversed", "mi_gra_uniform",
        "mi_gra_residual", "mi_gra_residual_no_mi", "mi_gra_residual_shuffled",
        "mi_gra_residual_reversed", "mi_gra_residual_uniform",
        "mi_global_high", "mi_global_low", "mi_global_uniform", "mi_global_random",
        "horizon_mi", "horizon_global_mi", "horizon_uniform", "horizon_random",
        *_BOUND_HIGH_MI_ALL_METHOD_SET,
    }
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    _validate_bound_high_mi_method_set(
        methods, forecast_fusion=args.forecast_fusion
    )
    _validate_bound_high_mi_residual_method_set(
        methods, forecast_fusion=args.forecast_fusion
    )
    if (
        args.forecast_fusion == "bound_high_mi_ablation"
        and args.forecast_fusion_alignment != "none"
    ):
        raise ValueError(
            "bound_high_mi_ablation stores alignment per method; pass "
            "--forecast-fusion-alignment none"
        )
    if (
        args.forecast_fusion == "bound_high_mi_residual_ablation"
        and args.forecast_fusion_alignment != "none"
    ):
        raise ValueError(
            "bound_high_mi_residual_ablation stores alignment per method; "
            "pass --forecast-fusion-alignment none"
        )
    if args.forecast_fusion == "bound_high_mi_linear_mix":
        valid_linear_mix = (
            _is_official_baseline_four_method_protocol(
                methods, bound_methods, linear_mix=True
            )
            or (
                bound_methods == _BOUND_HIGH_MI_LINEAR_MIX_METHOD_SET
                and len(methods) == 4
            )
        )
        if not valid_linear_mix:
            raise ValueError(
                "bound_high_mi_linear_mix requires the official baseline plus "
                "three explicit arms, or exactly the four explicit 2x2 methods"
            )
        if args.forecast_fusion_alignment != "none":
            raise ValueError(
                "bound_high_mi_linear_mix stores alignment per method; pass "
                "--forecast-fusion-alignment none"
            )
    if weighting_ablation_alignments:
        if args.forecast_fusion != "high_mi":
            raise ValueError(
                "weighting ablation requires --forecast-fusion high_mi"
            )
        if "official" not in methods or "high_mi" not in methods:
            raise ValueError(
                "weighting ablation requires official and high_mi methods"
            )
        if methods.index("official") > methods.index("high_mi"):
            raise ValueError(
                "weighting ablation requires official before high_mi"
            )
        if args.forecast_fusion_confidence_mode not in (
            _HISTORY_CONFIDENCE_FUSIONS | {"forecast_disagreement"}
        ):
            raise ValueError(
                "weighting ablation freezes one confidence across all arms; use "
                "mi_rank_agreement, mi_rank_entropy, or forecast_disagreement"
            )
    if args.mi_attribution_ablation:
        if args.forecast_fusion != "high_mi":
            raise ValueError("MI attribution ablation requires --forecast-fusion high_mi")
        if not {"official", "high_mi"}.issubset(methods):
            raise ValueError("MI attribution ablation requires official and high_mi methods")
        if args.forecast_fusion_alignment != "bcsa":
            raise ValueError("MI attribution ablation requires --forecast-fusion-alignment bcsa")
        if args.forecast_fusion_weighting != "mi":
            raise ValueError("MI attribution ablation requires MI weighting")
        if args.forecast_fusion_confidence_mode != "forecast_disagreement":
            raise ValueError(
                "MI attribution ablation requires forecast-disagreement confidence"
            )
        if args.forecast_fusion_mi_branch_gate != "none":
            raise ValueError("MI attribution ablation does not combine with a branch gate")
    if args.system_ablation_suite:
        required = (
            methods == ["official", "high_mi_anchor"]
            and args.forecast_fusion == "high_mi"
            and args.forecast_fusion_alignment == "bcsa"
            and args.forecast_fusion_weighting == "mi"
            and args.forecast_fusion_confidence_mode == "forecast_disagreement"
            and args.forecast_fusion_mi_branch_gate == "none"
            and args.augment_mode == "moe"
        )
        if not required:
            raise ValueError("system ablation suite requires official,high_mi_anchor with the locked BCSA/MoE/MI forecast-fusion configuration")
    if args.forecast_fusion_mi_branch_gate != "none":
        if "official" not in methods:
            raise ValueError("MI branch gating requires the official method")
        if not any(
            _forecast_fusion_mi_branch_gate_applies(method, args.forecast_fusion)
            for method in methods
        ):
            raise ValueError(
                "MI branch gating requires high_mi/high_mi_anchor, or mi_recent "
                "with coupled_recent_mi, in --methods"
            )
    if args.forecast_fusion != "none" and any(method.startswith("mi_global_") for method in methods):
        # The global selector reconstructs its own history/future candidates
        # from the train-only knowledge bank.  It can therefore feed the same
        # outer BCSA/MI-weighted correction shell as a fixed-pool selector,
        # but only the main high-MI protocol is currently defined for this
        # path.  Keep the restriction explicit so other fusion variants do
        # not silently mix global and official-pool semantics.
        if args.forecast_fusion != "high_mi":
            raise ValueError(
                "MI-global forecast fusion currently supports only high_mi"
            )
        if args.forecast_fusion_alignment != "bcsa":
            raise ValueError(
                "MI-global high_mi forecast fusion requires alignment=bcsa"
            )
    if "official_mi_residual" in methods and args.forecast_fusion not in {
        "residual_prototype_hybrid",
        "residual_prototype_hybrid_median",
        "residual_prototype_hybrid_confident",
        "residual_prototype_hybrid_pool",
    }:
        raise ValueError(
            "official_mi_residual requires a hybrid residual-prototype fusion"
        )
    if "output_mi" in methods:
        if args.augment_mode != "moe":
            raise ValueError("output_mi requires the TS-RAG moe ARM output gate")
        if args.forecast_fusion != "none":
            raise ValueError("output_mi cannot be combined with forecast-space fusion")
        if args.arm_bias:
            raise ValueError("output_mi cannot be combined with legacy arm-bias")
        if args.arm_attention_prior:
            raise ValueError("output_mi cannot be combined with arm-attention-prior")
    mi_gra_methods = {
        "mi_gra", "mi_gra_no_mi", "mi_gra_shuffled", "mi_gra_reversed", "mi_gra_uniform",
        "mi_gra_residual", "mi_gra_residual_no_mi", "mi_gra_residual_shuffled",
        "mi_gra_residual_reversed", "mi_gra_residual_uniform",
    }
    if args.mi_gra and args.augment_mode != "moe":
        raise ValueError("mi-gra requires the TS-RAG moe ARM")
    if any(method in mi_gra_methods for method in methods) and args.augment_mode != "moe":
        raise ValueError("mi_gra methods require the TS-RAG moe ARM")
    if any(method in mi_gra_methods for method in methods):
        args.mi_gra = True
    if args.pool_k < args.top_k:
        raise ValueError("pool-k must be >= top-k")
    _validate_horizon_protocol(
        methods=methods,
        pool_k=args.pool_k,
        top_k=args.top_k,
        pred_len=args.pred_len,
        mi_target="I(H;Y)",
        mi_condition="none",
        arm_bias=args.arm_bias,
        arm_attention_prior=args.arm_attention_prior,
        forecast_fusion=args.forecast_fusion,
        correction_gain=gain_array,
    )
    for method in (item for item in methods if item in HORIZON_METHODS):
        _validate_horizon_protocol(
            methods=[method],
            pool_k=args.pool_k,
            top_k=args.top_k,
            pred_len=args.pred_len,
            mi_target="I(H;Y)",
            mi_condition="none",
            arm_bias=args.arm_bias,
            arm_attention_prior=args.arm_attention_prior,
            forecast_fusion=args.forecast_fusion,
            correction_gain=np.asarray(gain_for(method)),
        )
    _prepend_external_path()
    from data_provider.data_factory import data_provider
    from retrieve import load_database
    from models.ChronosBolt import (
        ChronosBoltModelForForecasting,
        ChronosBoltModelForForecastingWithRetrieval,
    )
    from transformers import AutoConfig
    from sklearn.preprocessing import StandardScaler

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    retrieval_dir = Path(args.retrieval_database_dir).resolve()
    database_path = retrieval_dir / f"{args.dataset}_{args.metadata_frequency}_{args.lookback_length}.pkl"
    if not database_path.exists():
        lowercase_path = retrieval_dir / (
            f"{args.dataset.lower()}_{args.metadata_frequency}_{args.lookback_length}.pkl"
        )
        if lowercase_path.exists():
            database_path = lowercase_path
        else:
            raise FileNotFoundError(f"official TS-RAG retrieval database is missing: {database_path}")
    database = load_database(str(database_path))
    raw = np.asarray([database[key]["raw_data"] for key in database], dtype=np.float32).T
    raw_scaler = StandardScaler().fit(raw)
    retriever_rawdata = raw_scaler.transform(raw).T

    class Args:
        pass

    data_args = Args()
    data_args.model_id = "tsrag_mi_retrieve"
    data_args.root_path = str(Path(args.root_path).resolve()) + "/"
    data_args.data_path = args.data_path
    data_args.data = args.data
    data_args.features = "M"
    data_args.target = "OT"
    data_args.embed = "timeF"
    data_args.freq = "h"
    data_args.percent = 100
    data_args.max_len = -1
    data_args.seq_len = args.seq_len
    data_args.label_len = 0
    data_args.pred_len = args.pred_len
    data_args.batch_size = args.batch_size
    data_args.num_workers = args.num_workers
    data_args.top_k = args.pool_k
    data_args.mode = args.mode
    data_args.return_feature_id = False
    data_flag = _data_provider_flag(args.split)
    test_data, test_loader = data_provider(data_args, data_flag, retriever_rawdata=retriever_rawdata)
    if _rebuild_ordered_eval_loader(args.split):
        # ``data_provider(..., 'val')`` intentionally shuffles for training
        # workflows, while sidecar alignment requires deterministic
        # channel-major order.
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
        )
    query_start = max(0, int(args.query_start))
    query_end = len(test_data) if args.query_end is None else min(len(test_data), int(args.query_end))
    query_indices = _query_indices(
        len(test_data), query_start, query_end, args.query_count, args.query_sampling,
        args.query_parity,
    )
    contiguous = np.array_equal(query_indices, np.arange(query_start, query_end, dtype=np.int64))
    contiguous_subset = contiguous and (
        query_start > 0 or query_end < len(test_data)
    )
    if contiguous_subset:
        # A contiguous shard should not iterate through and discard a large
        # prefix of the test set.  Keeping the logical offset at query_start
        # preserves sidecar alignment while letting multi-GPU shard jobs start
        # directly at their assigned window range.
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                test_data, range(query_start, query_end)
            ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
        )
    elif not contiguous:
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_data, query_indices.tolist()),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
        )
    global_methods = [method for method in methods if method.startswith("mi_global_")]
    needs_official_sidecar = (
        args.forecast_fusion != "none"
        or any(method in mi_gra_methods for method in methods)
        or (args.arm_attention_prior and any(method != "base" for method in methods))
        or any(
            method not in {"base", "official"} and not method.startswith("mi_global_")
            for method in methods
        )
    )
    if needs_official_sidecar and not args.artifact:
        raise ValueError("--artifact is required for non-global MI methods")
    if args.artifact:
        artifact_path = Path(args.artifact).resolve()
        # Large test artifacts are stored as disjoint NPZ shards.  Even for a
        # contiguous query interval, load only the requested rows from each
        # shard instead of concatenating every shard before slicing.  This
        # keeps the full-test protocol unchanged while avoiding a large
        # transient memory spike on datasets such as Electricity.
        sidecar_query_indices = (
            query_indices
            if query_indices is not None
            else np.arange(query_start, query_end, dtype=np.int64)
        ) if artifact_path.is_dir() else None
        sidecar = _load_sidecar(
            artifact_path, len(query_indices),
            args.pool_k, args.top_k, methods,
            query_start=query_start, query_end=query_end,
            query_indices=sidecar_query_indices,
        )
        _validate_sidecar_alignment(
            sidecar, test_data, args.pool_k, query_start, query_end,
            query_indices=None if contiguous else query_indices,
        )
        sidecar_meta = _sidecar_metadata(Path(args.artifact).resolve())
        _validate_sidecar_dataset(sidecar_meta, args.dataset)
        _require_ihy_sidecar(sidecar_meta, "MI forecast runner")
        if args.forecast_fusion in _COUPLED_RECENT_MI_FUSIONS:
            target = _sidecar_mi_target(sidecar_meta)
            if target != "I(H;Y)":
                raise ValueError(
                    "coupled_recent_mi is restricted to an I(H;Y) sidecar; "
                    f"got {target!r}"
                )
        if set(methods) & _BOUND_MI_SELECT_ABLATION_METHOD_SET:
            mi_target = _sidecar_mi_target(sidecar_meta)
            if mi_target != "I(H;Y)":
                raise ValueError(
                    "MI-L2 coupled ablation is restricted to an I(H;Y) sidecar; "
                    f"got {mi_target!r}"
                )
    else:
        sidecar = {}
        sidecar_meta = {}
    mi_gra_sidecar = sidecar
    mi_gra_sidecar_meta = sidecar_meta
    if args.mi_gra_artifact:
        mi_gra_sidecar = _load_sidecar(
            Path(args.mi_gra_artifact).resolve(), len(query_indices),
            args.pool_k, args.top_k, ["mi_gra_residual"],
            query_start=query_start, query_end=query_end,
            query_indices=None if contiguous else query_indices,
        )
        _validate_sidecar_alignment(
            mi_gra_sidecar, test_data, args.pool_k, query_start, query_end,
            query_indices=None if contiguous else query_indices,
        )
        mi_gra_sidecar_meta = _sidecar_metadata(Path(args.mi_gra_artifact).resolve())
        _validate_sidecar_dataset(mi_gra_sidecar_meta, args.dataset)
        _require_ihy_sidecar(mi_gra_sidecar_meta, "MI-GRA forecast runner")
        primary_hash = sidecar_meta.get("official_distance_hash")
        mi_gra_hash = mi_gra_sidecar_meta.get("official_distance_hash")
        if primary_hash is not None and mi_gra_hash is not None and primary_hash != mi_gra_hash:
            raise ValueError(
                "--artifact and --mi-gra-artifact disagree on official distance hash"
            )
        primary_target = _sidecar_mi_target(sidecar_meta)
        mi_gra_target = _sidecar_mi_target(mi_gra_sidecar_meta)
        if primary_target is not None and mi_gra_target is not None and primary_target != mi_gra_target:
            raise ValueError(
                "--artifact and --mi-gra-artifact disagree on MI target"
            )
        if not np.array_equal(
            np.asarray(sidecar.get("official_distances")),
            np.asarray(mi_gra_sidecar.get("official_distances")),
        ):
            raise ValueError(
                "--artifact and --mi-gra-artifact disagree on official candidate distances"
            )
    horizon_diagnostics = None
    horizon_requested = [method for method in methods if method in HORIZON_METHODS]
    if horizon_requested:
        if not bool(sidecar_meta.get("horizon_mi_protocol", False)):
            raise ValueError("horizon MI methods require a horizon_mi_protocol sidecar")
        if int(sidecar_meta.get("horizon_block_size", -1)) != 16:
            raise ValueError("horizon MI sidecar requires horizon_block_size=16")
        if int(sidecar_meta.get("horizon_block_count", -1)) != 4:
            raise ValueError("horizon MI sidecar requires horizon_block_count=4")
        for key in (
            "horizon_mi_observed", "horizon_mi_weights",
            "global_mi_observed", "global_mi_weights",
        ):
            if key not in sidecar_meta or sidecar_meta[key] is None:
                raise ValueError(f"horizon MI sidecar metadata is missing {key}")
        _validate_horizon_protocol(
            methods=methods,
            pool_k=args.pool_k,
            top_k=args.top_k,
            pred_len=args.pred_len,
            mi_target=_sidecar_mi_target(sidecar_meta) or "",
            mi_condition=str(sidecar_meta.get("mi_condition", "")),
            arm_bias=args.arm_bias,
            arm_attention_prior=args.arm_attention_prior,
            forecast_fusion=args.forecast_fusion,
            correction_gain=gain_array,
        )
        horizon_diagnostics = _horizon_sidecar_diagnostics(
            sidecar, sidecar_meta, methods,
            pool_k=args.pool_k, top_k=args.top_k,
        )
    if args.forecast_fusion == "horizon_residual_transport":
        expected_horizon_shape = (len(query_indices), 4, args.pool_k)
        for horizon_method in horizon_requested:
            horizon_key = f"horizon_distances_{horizon_method}"
            horizon_values = np.asarray(sidecar.get(horizon_key))
            if horizon_values.shape != expected_horizon_shape:
                raise ValueError(
                    f"{horizon_key} shape {horizon_values.shape} != "
                    f"expected {expected_horizon_shape}"
                )
            if not np.isfinite(horizon_values).all() or np.any(horizon_values < -1e-6):
                raise ValueError(f"{horizon_key} must be finite and non-negative")
    if "output_mi" in methods:
        mi_target = _sidecar_mi_target(sidecar_meta)
        if mi_target != "I(H;Y)":
            raise ValueError(
                "output_mi requires an I(H;Y) sidecar"
            )
        mi_distance_values = np.asarray(sidecar.get(args.output_mi_distance_key))
        expected_shape = (len(query_indices), args.pool_k)
        if mi_distance_values.shape != expected_shape:
            raise ValueError(
                f"{args.output_mi_distance_key} shape {mi_distance_values.shape} "
                f"!= expected {expected_shape}"
            )
        if not np.isfinite(mi_distance_values).all() or np.any(mi_distance_values < -1e-6):
            raise ValueError(f"{args.output_mi_distance_key} must be finite and non-negative")
    if any(method in mi_gra_methods for method in methods):
        mi_gra_values = np.asarray(mi_gra_sidecar.get(args.mi_gra_distance_key))
        expected_shape = (len(query_indices), args.pool_k)
        if mi_gra_values.shape != expected_shape:
            raise ValueError(
                f"{args.mi_gra_distance_key} shape {mi_gra_values.shape} "
                f"!= expected {expected_shape}"
            )
        if not np.isfinite(mi_gra_values).all() or np.any(mi_gra_values < -1e-6):
            raise ValueError(f"{args.mi_gra_distance_key} must be finite and non-negative")
    global_sidecar: dict[str, np.ndarray] | None = None
    if global_methods:
        if not contiguous:
            raise ValueError("stratified query sampling is not yet supported for mi_global_* methods")
        if not args.global_artifact:
            raise ValueError("--global-artifact is required for mi_global_* methods")
        with np.load(Path(args.global_artifact).resolve(), allow_pickle=False) as loaded:
            global_sidecar = {key: loaded[key] for key in loaded.files}
        expected = query_end - query_start
        for method in global_methods:
            for key in (
                f"selected_channel_ids_{method}",
                f"selected_starts_{method}",
                f"selection_scores_{method}",
            ):
                if key not in global_sidecar:
                    raise ValueError(f"global sidecar is missing {key}")
                value = np.asarray(global_sidecar[key])
                if value.shape != (expected, args.top_k):
                    raise ValueError(f"{key} shape {value.shape} != {(expected, args.top_k)}")
        if "channel_ids" in global_sidecar:
            expected_channels = np.arange(query_start, query_end, dtype=np.int64) // int(test_data.tot_len)
            if not np.array_equal(np.asarray(global_sidecar["channel_ids"]), expected_channels):
                raise ValueError("MI-global sidecar channel_ids do not match test order")
        if "query_origins" in global_sidecar:
            if len(np.asarray(global_sidecar["query_origins"])) != expected:
                raise ValueError("MI-global query_origins length mismatch")
        global_meta_path = Path(args.global_artifact).with_suffix(".json")
        if global_meta_path.exists() and not sidecar_meta:
            sidecar_meta = json.loads(global_meta_path.read_text(encoding="utf-8"))

    model_config = AutoConfig.from_pretrained(args.pretrained_model_path)
    pretrained_files = (
        Path(args.pretrained_model_path) / "pytorch_model.bin",
        Path(args.pretrained_model_path) / "model.safetensors",
    )
    if any(path.exists() for path in pretrained_files):
        rag_model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
            args.pretrained_model_path, config=model_config, augment=args.augment_mode,
            enable_mi_gra=bool(args.mi_gra), mi_gra_hidden_dim=args.mi_gra_hidden_dim,
        )
    else:
        # The official TS-RAG checkpoint is stored separately from the
        # Chronos config in this workspace.  Constructing from the config
        # keeps RCAM checkpoints compatible without duplicating an 840-MB
        # base model file into every experiment directory.
        rag_model = ChronosBoltModelForForecastingWithRetrieval(
            model_config, augment=args.augment_mode,
            enable_mi_gra=bool(args.mi_gra), mi_gra_hidden_dim=args.mi_gra_hidden_dim,
        )
    retrieval_state = _state_dict(Path(args.retrieval_checkpoint).resolve())
    # RQCA can be warm-started from the already-pretrained HCTA routing
    # blocks for a causal validation probe.  Its dedicated pretraining entry
    # point writes rqca.* keys, so this compatibility path is only used when a
    # caller intentionally supplies an HCTA checkpoint.
    if args.augment_mode == "rqca_fair":
        has_rqca = any(str(key).startswith("rqca.") for key in retrieval_state)
        has_hcta = any(str(key).startswith("hcta.") for key in retrieval_state)
        if not has_rqca and has_hcta:
            retrieval_state = {
                ("rqca." + str(key)[len("hcta."):]
                 if str(key).startswith("hcta.") else key): value
                for key, value in retrieval_state.items()
            }
    state_result = rag_model.load_state_dict(
        retrieval_state,
        strict=not bool(args.mi_gra),
    )
    if args.mi_gra:
        bad_missing = [key for key in state_result.missing_keys if not key.startswith("mi_gra.")]
        if bad_missing or state_result.unexpected_keys:
            raise RuntimeError(
                "official TS-RAG checkpoint mismatch outside MI-GRA keys: "
                f"missing={bad_missing}, unexpected={state_result.unexpected_keys}"
            )
        # The official checkpoint predates MI-GRA.  Transformers may leave
        # checkpoint-missing modules non-finite after from_pretrained; reset
        # only the new branch, never the official TS-RAG parameters.  A future
        # checkpoint containing all MI-GRA keys is loaded without resetting.
        if state_result.missing_keys:
            rag_model.mi_gra.reset_parameters()
        if args.mi_gra_checkpoint:
            mi_gra_path = Path(args.mi_gra_checkpoint).resolve()
            if not mi_gra_path.exists():
                raise FileNotFoundError(mi_gra_path)
            mi_gra_state = torch.load(mi_gra_path, map_location="cpu")
            if isinstance(mi_gra_state, dict) and "mi_gra" in mi_gra_state:
                mi_gra_state = mi_gra_state["mi_gra"]
            if not isinstance(mi_gra_state, dict):
                raise ValueError("mi-gra-checkpoint must contain a state dict")
            if any(str(key).startswith("mi_gra.") for key in mi_gra_state):
                mi_gra_state = {
                    str(key).removeprefix("mi_gra."): value
                    for key, value in mi_gra_state.items()
                }
            rag_model.mi_gra.load_state_dict(mi_gra_state, strict=True)
    rag_model.to(device).eval()

    official_model = None
    if args.official_arm_checkpoint:
        official_model = ChronosBoltModelForForecastingWithRetrieval(
            model_config, augment="moe"
        )
        official_model.load_state_dict(
            _state_dict(Path(args.official_arm_checkpoint).resolve()),
            strict=True,
        )
        official_model.to(device).eval()
    elif args.augment_mode in _RETRIEVAL_AUGMENT_MODES and "official" in methods:
        raise ValueError(
            "new retrieval-adapter runs that report the official baseline must pass "
            "--official-arm-checkpoint"
        )
    if any(path.exists() for path in pretrained_files):
        base_model = ChronosBoltModelForForecasting.from_pretrained(
            args.pretrained_model_path, config=model_config
        )
    else:
        base_model = ChronosBoltModelForForecasting(model_config)
    base_path = Path(args.base_weights).resolve() if args.base_weights else Path(args.pretrained_model_path) / "autogluon_model.pth"
    base_source = "huggingface_pretrained"
    if base_path.exists():
        base_model.load_state_dict(_state_dict(base_path), strict=False)
        base_source = str(base_path)
    base_model.to(device).eval()
    anchor_adapter = None
    anchor_adapter_config: dict[str, object] | None = None
    anchor_adapter_kind = None
    if args.anchor_adapter_checkpoint:
        if args.forecast_anchor != "zero_shot":
            raise ValueError("anchor adapters require --forecast-anchor zero_shot")
        if args.forecast_fusion != "high_mi" or args.forecast_fusion_alignment != "bcsa":
            raise ValueError(
                "anchor adapters currently require high_mi forecast fusion with BCSA"
            )
        if not 0.0 <= float(args.anchor_adapter_strength) <= 1.0:
            raise ValueError("anchor-adapter-strength must be in [0,1]")
        if (
            args.anchor_adapter_residual_mode == "power_scaled"
            and (not np.isfinite(float(args.anchor_adapter_gamma))
                 or float(args.anchor_adapter_gamma) <= 0.0)
        ):
            raise ValueError("anchor-adapter-gamma must be finite and positive")
        if (
            args.anchor_adapter_residual_mode == "reliability_scaled"
            and (
                float(args.anchor_adapter_reliability_scale) <= 0.0
                or not (
                    np.isfinite(float(args.anchor_adapter_reliability_scale))
                    or np.isposinf(float(args.anchor_adapter_reliability_scale))
                )
            )
        ):
            raise ValueError(
                "anchor-adapter-reliability-scale must be positive (finite or inf)"
            )
        if (
            args.anchor_adapter_residual_mode == "deadzone_scaled"
            and (
                not np.isfinite(float(args.anchor_adapter_deadzone_threshold))
                or not 0.0 <= float(args.anchor_adapter_deadzone_threshold) < 1.0
            )
        ):
            raise ValueError(
                "anchor-adapter-deadzone-threshold must be finite in [0,1)"
            )
        from experiments.information_anchor.downstream.zero_shot_anchor_adapter import (
            load_adapter_checkpoint,
        )
        anchor_adapter, anchor_adapter_config = load_adapter_checkpoint(
            str(Path(args.anchor_adapter_checkpoint).resolve()),
            device=device,
            kind=args.anchor_adapter_type,
        )
        anchor_adapter_kind = str(anchor_adapter_config["kind"])
        if int(anchor_adapter_config["pred_len"]) != int(args.pred_len):
            raise ValueError("anchor adapter pred_len does not match the runner")
    quantiles = torch.as_tensor(rag_model.quantiles, device=device)
    median_index = int(torch.abs(quantiles - 0.5).argmin())

    predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    stream_metric_stats: dict[str, dict[str, float]] | None = (
        {method: {} for method in methods} if args.stream_metrics else None
    )
    retrieval_scores: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    retrieval_block_scores: dict[str, list[np.ndarray]] = {
        method: [] for method in methods if method in HORIZON_METHODS
    }
    oracle_recalls: dict[str, list[np.ndarray]] = {method: [] for method in methods if method != "base"}
    official_overlaps: dict[str, list[np.ndarray]] = {method: [] for method in methods if method != "base"}
    forecast_confidences: dict[str, list[np.ndarray]] = {
        method: [] for method in methods
        if method not in {"base", "official", *_NEW_ARM_METHODS} and args.forecast_fusion != "none"
    }
    correction_analysis: dict[str, dict[str, list[np.ndarray]]] = {
        method: {
            "official_reference_prediction": [],
            "tsrag_prediction": [],
            "mi_consensus": [],
            "fused_prediction": [],
            "forecast_confidence": [],
            "query_history_scale": [],
            "candidate_residual_dispersion": [],
        }
        for method in methods
        if method not in {"base", "official", *_NEW_ARM_METHODS} and args.forecast_fusion != "none"
    }
    weighting_ablation_predictions: dict[str, list[np.ndarray]] = {
        "no_future": [],
        **{
            f"{alignment}__{arm}": []
            for alignment in weighting_ablation_alignments
            for arm in (
                "uniform_weight",
                "official_distance_weight",
                "mi_distance_weight",
            )
        },
    }
    if factorial_ablation_enabled:
        weighting_ablation_predictions.update({
            f"factorial__{arm}": []
            for arm in (
                "ordinary_distance",
                "standardization_only",
                "weighting_only",
                "standardized_weighted",
            )
        })
    weighting_ablation_confidence: list[np.ndarray] = []
    mi_attribution_predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("w/o_mi_selection", "w/o_mi_weighting", "w/o_mi_gate")
    } if args.mi_attribution_ablation else {}
    mi_attribution_confidences: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("w/o_mi_selection", "w/o_mi_weighting", "w/o_mi_gate")
    } if args.mi_attribution_ablation else {}
    system_ablation_names = (
        "full", "tsrag", "without_mi_rank", "without_mi_distance",
        "without_mi_gate", "without_bcsa", "without_mi",
        "without_mi_bcsa", "without_residual_correction",
    )
    system_ablation_losses = {
        name: {"mse": [], "mae": []} for name in system_ablation_names
    } if args.system_ablation_suite else {}
    mi_gra_diagnostics: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"alpha": [], "reliability": []}
        for method in methods if method in mi_gra_methods
    }
    truths: list[np.ndarray] = []
    oracle_best_values: list[np.ndarray] = []
    shuffled_rows = None
    if "mi_prior_shuffled" in methods:
        if "selected_ranks_mi_prior" not in sidecar:
            raise ValueError("mi_prior_shuffled requires selected_ranks_mi_prior in the sidecar")
        shuffled_rows = np.random.default_rng(args.seed).permutation(len(query_indices))
    offset = query_start if contiguous_subset else 0
    local_offset = 0
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, _batch_x_mark, _batch_y_mark, retrieved, distances = batch
            batch_size = int(batch_x.shape[0])
            stop = offset + batch_size
            keep_lo = 0 if not contiguous else max(query_start - offset, 0)
            keep_hi = batch_size if not contiguous else min(query_end - offset, batch_size)
            offset = stop
            if keep_lo >= keep_hi:
                if contiguous and offset >= query_end:
                    break
                continue
            batch_x = batch_x[keep_lo:keep_hi]
            batch_y = batch_y[keep_lo:keep_hi]
            retrieved = retrieved[keep_lo:keep_hi]
            distances = distances[keep_lo:keep_hi]
            batch_size = int(batch_x.shape[0])
            context = batch_x.float().to(device).squeeze(-1)
            truth = batch_y.float().to(device).squeeze(-1)[..., -args.pred_len:]
            retrieved = retrieved.float().to(device)
            distances = distances.float().to(device)
            retrieved, distances = _contaminate_pool(
                retrieved, distances, context, sidecar, local_offset, args.seq_len,
                retriever_rawdata, args.contamination_mode,
                period={"ETTm2": 96, "ETTm1": 96, "ETTh1": 24, "ETTh2": 24,
                        "weather": 144, "electricity": 24, "exchange_rate": 24}.get(args.dataset, 24),
                seed=args.seed,
            )
            ordinary_retrieved_selected = None
            ordinary_distance_selected = None
            if weighting_ablation_alignments:
                ordinary_selected = np.argsort(
                    distances.detach().cpu().numpy(), axis=1, kind="stable"
                )[:, :args.top_k]
                ordinary_retrieved_selected = _gather(retrieved, ordinary_selected)
                ordinary_distance_selected = _gather(distances, ordinary_selected)
            truth_numpy = truth.cpu().numpy()
            if not args.stream_metrics:
                truths.append(truth_numpy)
            reference_prediction = None
            needs_reference = args.save_correction_analysis or any(
                method not in {"base", "official", *_NEW_ARM_METHODS}
                and np.any(gain_for(method) < 1.0)
                for method in methods
            )
            if needs_reference and "official" not in methods:
                reference_indices = np.tile(np.arange(args.top_k), (batch_size, 1))
                reference_retrieved = _gather(retrieved, reference_indices)
                reference_distances = _gather(distances, reference_indices)
                reference_model = official_model or rag_model
                reference_prediction = reference_model(
                    context=context,
                    retrieved_seq=reference_retrieved,
                    distances=reference_distances,
                    retrieval_bias=None,
                ).quantile_preds[:, median_index, :args.pred_len]
            # Every MI-GRA residual control uses the same official Top-10
            # candidates.  Cache their frozen Chronos baseline once per batch
            # so Real/Shuffle/Reverse/Uniform differ only in the distance
            # assignment, not in redundant model evaluation.
            candidate_base_cache = None
            candidate_base_cache_selection = None
            history_confidence_override = None
            mi_branch_gate_override = None
            mi_match_multiplier = None
            if args.forecast_fusion_mi_branch_gate != "none":
                if "high_mi_distances" not in sidecar:
                    raise ValueError(
                        "MI branch gating requires high_mi_distances in the artifact"
                    )
                official_history = np.asarray(sidecar["official_distances"])[
                    local_offset:local_offset + batch_size
                ]
                mi_history = np.asarray(sidecar["high_mi_distances"])[
                    local_offset:local_offset + batch_size
                ]
                branch_gate = history_only_mi_branch_gate(
                    mi_history,
                    official_history,
                    threshold=args.forecast_fusion_mi_branch_gate_threshold,
                    power=args.forecast_fusion_mi_branch_gate_power,
                )
                mi_branch_gate_override = torch.from_numpy(branch_gate).to(
                    device=device, dtype=torch.float32
                )
            if args.forecast_fusion_confidence_mode in _HISTORY_CONFIDENCE_FUSIONS:
                official_history = np.asarray(sidecar["official_distances"])[
                    local_offset:local_offset + batch_size
                ]
                mi_history = np.asarray(sidecar["high_mi_distances"])[
                    local_offset:local_offset + batch_size
                ]
                history_confidence = history_only_mi_confidence(
                    mi_history,
                    official_history,
                    entropy_power=(
                        args.forecast_fusion_history_entropy_power
                    if args.forecast_fusion_confidence_mode == "mi_rank_entropy"
                        else 0.0
                    ),
                    temperature=args.forecast_fusion_temperature,
                    floor=(
                        args.forecast_fusion_confidence_floor
                        if args.forecast_fusion_confidence_mode
                        in _HISTORY_CONFIDENCE_FUSIONS
                        else 0.0
                    ),
                )
                history_confidence_override = torch.from_numpy(
                    history_confidence
                ).to(device=device, dtype=torch.float32)
            if mi_match_config is not None:
                if "high_mi_distances" not in sidecar:
                    raise ValueError(
                        "forecast-fusion-mi-match-file requires high_mi_distances"
                    )
                mi_history = np.asarray(sidecar["high_mi_distances"])[
                    local_offset:local_offset + batch_size
                ]
                match_multiplier = mi_distance_match_multiplier(
                    mi_history,
                    center=float(mi_match_config["center"]),
                    scale=float(mi_match_config["scale"]),
                    slope=float(mi_match_config["slope"]),
                    floor=float(mi_match_config["floor"]),
                    ceiling=float(mi_match_config["ceiling"]),
                )
                mi_match_multiplier = torch.from_numpy(match_multiplier).to(
                    device=device, dtype=torch.float32
                )
            for method in methods:
                weighting_ablation_batch = None
                mi_attribution_batch = None
                adapter_candidate_dispersion = None
                if method == "base":
                    output = base_model(context=context).quantile_preds[:, median_index, :args.pred_len]
                else:
                    is_global = method.startswith("mi_global_")
                    bound_spec = _bound_method_spec(method)
                    candidate_future = retrieved[..., args.seq_len:args.seq_len + args.pred_len]
                    if method in {"official", *_NEW_ARM_METHODS, "official_mi_residual"} or (
                        bound_spec is not None
                        and bound_spec["selection"] == "official_top10"
                    ):
                        selected = np.tile(np.arange(args.top_k), (batch_size, 1))
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    elif method == "output_mi":
                        # Output-level MI keeps the official Top-k membership
                        # and order exactly fixed.  Only the final ARM
                        # retrieved-expert gate receives the history-only
                        # future prior below.
                        selected = np.tile(np.arange(args.top_k), (batch_size, 1))
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    elif method in mi_gra_methods:
                        # MI-GRA keeps the official Top-k membership/order;
                        # only the post-ARM MI assignment changes below.
                        selected = np.tile(np.arange(args.top_k), (batch_size, 1))
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    elif method == "mi_select":
                        # Compact MI sidecars store the history-only MI
                        # distances for the full official Top-20 pool.  Use
                        # them directly for membership selection, then sort
                        # the selected ranks back to pool order so ARM sees a
                        # deterministic order-only intervention.
                        mi_pool = np.asarray(sidecar.get("mi_l2_distances"))[
                            local_offset:local_offset + batch_size
                        ]
                        if mi_pool.shape != (batch_size, args.pool_k):
                            raise ValueError(
                                "mi_select requires mi_l2_distances with shape "
                                f"{(batch_size, args.pool_k)}, got {mi_pool.shape}"
                            )
                        selected = np.argsort(mi_pool, axis=1, kind="stable")[:, :args.top_k]
                        selected = np.sort(selected, axis=1)
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    elif method == "high_mi":
                        # Prefer an explicit rank map when present.  Compact
                        # distance-only sidecars are also valid: compute the
                        # Top-10 inside the already retrieved official pool,
                        # then bind the model distance source to the same MI
                        # scores.  This is the RCAM pilot's strict
                        # Official-Top20 -> MI-Top10 protocol.
                        if "selected_ranks_high_mi" in sidecar:
                            selected = np.asarray(sidecar["selected_ranks_high_mi"])[
                                local_offset:local_offset + batch_size
                            ]
                        else:
                            high_mi_pool = np.asarray(sidecar["high_mi_distances"])[
                                local_offset:local_offset + batch_size
                            ]
                            selected = np.argsort(
                                high_mi_pool, axis=1, kind="stable"
                            )[:, :args.top_k]
                        retrieved_selected = _gather(retrieved, selected)
                        high_mi_pool = np.asarray(sidecar["high_mi_distances"])[
                            local_offset:local_offset + batch_size
                        ]
                        high_mi_selected = np.take_along_axis(
                            high_mi_pool, np.asarray(selected, dtype=np.int64), axis=1
                        )
                        distance_selected = torch.from_numpy(
                            high_mi_selected.astype(np.float32, copy=False)
                        ).to(device=device, dtype=distances.dtype)
                        selected_future = _gather(candidate_future, selected)
                    elif args.forecast_fusion == "coupled_recent_mi":
                        selected = _coupled_recent_selection(
                            sidecar,
                            method=method,
                            top_k=args.top_k,
                            local_offset=local_offset,
                            batch_size=batch_size,
                            candidate_span=args.pred_len,
                        )
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    elif is_global:
                        assert global_sidecar is not None
                        # Global candidates do not have positions in the
                        # official retrieved Top-20 tensor.  Use an identity
                        # rank map for downstream code that expects a
                        # selected-rank matrix; the actual windows and scores
                        # are reconstructed from the global sidecar below.
                        selected = np.tile(np.arange(args.top_k), (batch_size, 1))
                        method_starts = np.asarray(global_sidecar[f"selected_starts_{method}"])[local_offset:local_offset + batch_size]
                        method_channels = np.asarray(global_sidecar[f"selected_channel_ids_{method}"])[local_offset:local_offset + batch_size]
                        selected_future = torch.empty(
                            batch_size, args.top_k, args.seq_len + args.pred_len,
                            dtype=torch.float32, device=device,
                        )
                        for row in range(batch_size):
                            for rank, start in enumerate(method_starts[row]):
                                channel = int(method_channels[row, rank])
                                begin = int(start); end = begin + args.seq_len + args.pred_len
                                if (channel < 0 or channel >= retriever_rawdata.shape[0]
                                        or begin < 0 or end > retriever_rawdata.shape[1]):
                                    raise ValueError("MI-global candidate leaves retrieval database")
                                selected_future[row, rank] = torch.from_numpy(
                                    np.asarray(retriever_rawdata[channel, begin:end], dtype=np.float32)
                                ).to(device)
                        retrieved_selected = selected_future
                        selected_future = retrieved_selected[..., args.seq_len:args.seq_len + args.pred_len]
                        distance_selected = torch.from_numpy(
                            np.asarray(global_sidecar[f"selection_scores_{method}"])[local_offset:local_offset + batch_size],
                        ).to(device=device, dtype=torch.float32)
                    elif bound_spec is not None and bound_spec["selection"] == "high_mi_top10":
                        selected = sidecar["selected_ranks_high_mi"][
                            local_offset:local_offset + batch_size
                        ]
                        retrieved_selected = _gather(retrieved, selected)
                        # The explicit four-arm protocol binds the high-MI
                        # candidate set and its distance source together.
                        # Use high_mi_distances for the high-MI ARM itself as
                        # well as for the residual prototype below; otherwise
                        # the reported ``high_mi_top10 + high_mi_distances``
                        # arm would still feed official distances to ARM.
                        high_mi_pool = np.asarray(sidecar["high_mi_distances"])[
                            local_offset:local_offset + batch_size
                        ]
                        high_mi_selected = np.take_along_axis(
                            high_mi_pool, selected, axis=1
                        )
                        distance_selected = torch.from_numpy(
                            high_mi_selected
                        ).to(device=device, dtype=distances.dtype)
                        selected_future = _gather(candidate_future, selected)
                    elif bound_spec is not None and bound_spec["selection"] == "mi_select":
                        mi_pool = np.asarray(sidecar.get("mi_l2_distances"))[
                            local_offset:local_offset + batch_size
                        ]
                        if mi_pool.shape != (batch_size, args.pool_k):
                            raise ValueError(
                                "MI-L2 bound selection requires mi_l2_distances with shape "
                                f"{(batch_size, args.pool_k)}, got {mi_pool.shape}"
                            )
                        selected = np.argsort(mi_pool, axis=1, kind="stable")[:, :args.top_k]
                        selected = np.sort(selected, axis=1)
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    else:
                        if method == "mi_prior_shuffled":
                            selected = sidecar["selected_ranks_mi_prior"][shuffled_rows[local_offset:local_offset + batch_size]]
                        else:
                            selected = sidecar[f"selected_ranks_{method}"][local_offset:local_offset + batch_size]
                        retrieved_selected = _gather(retrieved, selected)
                        distance_selected = _gather(distances, selected)
                        selected_future = _gather(candidate_future, selected)
                    branch_distance_selected, distance_selected = (
                        _split_branch_and_correction_distances(
                            method,
                            args.high_mi_branch_distance,
                            distances,
                            np.asarray(selected, dtype=np.int64),
                            distance_selected,
                        )
                    )
                    retrieval_scores[method].append(
                        torch.square(selected_future - truth[:, None, :]).mean(dim=(1, 2)).cpu().numpy()
                    )
                    if method in HORIZON_METHODS:
                        selected_error = torch.square(
                            selected_future - truth[:, None, :]
                        ).reshape(batch_size, args.top_k, 4, 16)
                        retrieval_block_scores[method].append(
                            selected_error.mean(dim=(1, 3)).cpu().numpy()
                        )
                    candidate_mse = torch.square(candidate_future - truth[:, None, :]).mean(dim=2)
                    oracle_best_values.append(candidate_mse.min(dim=1).values.cpu().numpy())
                    oracle_top10 = torch.topk(candidate_mse, k=min(args.top_k, args.pool_k), dim=1, largest=False).indices
                    selected_np = np.asarray(selected, dtype=np.int64)
                    oracle_np = oracle_top10.cpu().numpy()
                    oracle_recalls[method].append(np.asarray([
                        len(set(row.tolist()).intersection(set(best.tolist()))) / float(args.top_k)
                        for row, best in zip(selected_np, oracle_np)
                    ], dtype=np.float32))
                    if is_global:
                        # Global-bank positions are not ranks in the official
                        # Top-20 pool, so an overlap score would be meaningless.
                        official_overlaps[method].append(
                            np.full(batch_size, np.nan, dtype=np.float32)
                        )
                    else:
                        official_set = set(range(min(args.top_k, args.pool_k)))
                        official_overlaps[method].append(np.asarray([
                            len(set(row.tolist()).intersection(official_set)) / float(args.top_k)
                            for row in selected_np
                        ], dtype=np.float32))
                    attention_prior = None
                    if args.arm_attention_prior and should_apply_attention_prior(
                        method, args.arm_attention_prior_method
                    ):
                        attention_scores = attention_prior_scores(
                            sidecar,
                            method=method,
                            prior_method=args.arm_attention_prior_method,
                            selected=selected_np,
                            local_offset=local_offset,
                            batch_size=batch_size,
                        )
                        attention_prior = torch.from_numpy(attention_scores).to(device)
                    bias = None
                    mi_gra_selected = None
                    if method in mi_gra_methods:
                        mi_pool = np.asarray(mi_gra_sidecar[args.mi_gra_distance_key])[
                            local_offset:local_offset + batch_size
                        ]
                        mi_pool_selected = np.take_along_axis(
                            mi_pool, np.asarray(selected, dtype=np.int64), axis=1
                        )
                        control = {
                            "mi_gra": "real",
                            "mi_gra_no_mi": "uniform",
                            "mi_gra_shuffled": "shuffled",
                            "mi_gra_reversed": "reversed",
                            "mi_gra_uniform": "uniform",
                            "mi_gra_residual": "real",
                            "mi_gra_residual_no_mi": "uniform",
                            "mi_gra_residual_shuffled": "shuffled",
                            "mi_gra_residual_reversed": "reversed",
                            "mi_gra_residual_uniform": "uniform",
                        }[method]
                        mi_gra_selected = torch.from_numpy(
                            _mi_gra_control_distances(
                                mi_pool_selected,
                                control=control,
                                seed=args.seed,
                                global_offset=local_offset,
                            )
                        ).to(device=device, dtype=distance_selected.dtype)
                    if method == "output_mi":
                        mi_pool = np.asarray(sidecar[args.output_mi_distance_key])[
                            local_offset:local_offset + batch_size
                        ]
                        mi_selected = torch.from_numpy(
                            np.take_along_axis(mi_pool, selected, axis=1)
                        ).to(device=device, dtype=distance_selected.dtype)
                        bias = output_mi_prior_bias(
                            distance_selected,
                            mi_selected,
                            gamma=args.output_mi_gamma,
                            strength=args.output_mi_prior_strength,
                            center=args.output_mi_prior_centering == "centered",
                        )
                    bias_methods = _bias_targets(args.arm_bias_method)
                    if args.arm_bias and method in bias_methods:
                        if method in _MI_GRA_BIAS_TARGETS:
                            if mi_gra_selected is None:
                                raise RuntimeError(
                                    "MI-GRA ARM bias requires MI-GRA candidate distances"
                                )
                            # The MI-GRA branch has already produced the
                            # history-only distance/control values aligned to
                            # the fixed official Top-k candidates.  Reusing
                            # them avoids a second score definition and makes
                            # Real/Shuffle/Reverse/Uniform controls matched.
                            bias_scores = mi_gra_selected.detach().cpu().numpy()
                        else:
                            score_key = f"selection_scores_{method}"
                            source_scores = sidecar if score_key in sidecar else global_sidecar
                            if source_scores is None or score_key not in source_scores:
                                raise ValueError(f"artifact is missing {score_key} for ARM bias")
                            bias_scores = source_scores[score_key][
                                local_offset:local_offset + batch_size
                            ]
                        bias = _bias_from_scores(
                            bias_scores, args.arm_bias_strength,
                        ).to(device)
                    anchor_hidden = None
                    if (
                        args.forecast_anchor == "zero_shot"
                        and method not in {"official", *_NEW_ARM_METHODS, "official_mi_residual"}
                    ):
                        # The zero-shot experiment deliberately bypasses the
                        # TS-RAG ARM. Retrieved futures remain available only
                        # to the history-only outer consensus correction below.
                        base_output = base_model(
                            context=context,
                            return_hidden=(
                                anchor_adapter is not None and method == "high_mi"
                            ),
                        )
                        output = base_output.quantile_preds[
                            :, median_index, :args.pred_len
                        ]
                        if anchor_adapter is not None and method == "high_mi":
                            anchor_hidden = base_output.hidden_state
                    else:
                        model_for_method = (
                            official_model
                            if method == "official" and official_model is not None
                            else rag_model
                        )
                        output_object = model_for_method(
                            context=context,
                            retrieved_seq=retrieved_selected,
                            distances=branch_distance_selected,
                            retrieval_bias=bias,
                            mi_attention_prior=attention_prior,
                            mi_attention_strength=args.arm_attention_prior_strength,
                            mi_gra_distances=mi_gra_selected,
                            mi_gra_scale=args.mi_gra_strength,
                            mi_gra_mi_strength=(
                                0.0 if method.endswith("_no_mi") else args.mi_gra_mi_strength
                            ),
                            mi_gra_temperature=args.mi_gra_temperature,
                            mi_gra_feature=args.mi_gra_feature,
                            return_mi_gra_diagnostics=(
                                args.save_mi_gra_diagnostics and method in mi_gra_methods
                            ),
                            retrieved_alignment=(
                                args.tsrag_retrieved_alignment
                                if method == "high_mi"
                                or (
                                    bound_spec is not None
                                    and bound_spec["selection"] == "high_mi_top10"
                                )
                                else "none"
                            ),
                            retrieved_alignment_window=args.tsrag_retrieved_window,
                            retrieved_alignment_tau=args.tsrag_retrieved_tau,
                        )
                        output = output_object.quantile_preds[:, median_index, :args.pred_len]
                        if args.save_mi_gra_diagnostics and method in mi_gra_methods:
                            # The diagnostics are in candidate space and are
                            # intentionally saved before any metric aggregation.
                            if output_object.mi_gra_alpha is None or output_object.mi_gra_reliability is None:
                                raise RuntimeError("MI-GRA diagnostics were requested but not returned")
                            mi_gra_diagnostics[method]["alpha"].append(
                                output_object.mi_gra_alpha.float().cpu().numpy()
                            )
                            mi_gra_diagnostics[method]["reliability"].append(
                                output_object.mi_gra_reliability.float().cpu().numpy()
                            )
                    if (
                        mi_branch_gate_override is not None
                        and method in {"high_mi", "high_mi_anchor"}
                    ):
                        if reference_prediction is None:
                            raise RuntimeError(
                                "MI branch gating requires an official reference prediction"
                            )
                        output = reference_prediction + mi_branch_gate_override.unsqueeze(-1) * (
                            output - reference_prediction
                        )
                    if anchor_adapter is not None and method == "high_mi":
                        if anchor_adapter_kind not in {"linear", "film"}:
                            raise RuntimeError("unknown loaded anchor adapter kind")
                        if "high_mi_distances" not in sidecar:
                            raise ValueError(
                                "anchor adapters require high_mi_distances in the artifact"
                            )
                        adapter_score_pool = np.asarray(sidecar["high_mi_distances"])[
                            local_offset:local_offset + batch_size
                        ]
                        adapter_scores = torch.from_numpy(
                            np.take_along_axis(
                                adapter_score_pool,
                                np.asarray(selected, dtype=np.int64),
                                axis=1,
                            )
                        ).to(device=device, dtype=torch.float32)
                        _, adapter_confidence, adapter_consensus = _apply_mi_forecast_fusion(
                            output,
                            context,
                            retrieved_selected,
                            adapter_scores,
                            seq_len=args.seq_len,
                            pred_len=args.pred_len,
                            strength=0.0,
                            temperature=args.forecast_fusion_temperature,
                            confidence_scale=args.forecast_fusion_confidence_scale,
                            alignment="bcsa",
                            weighting=args.forecast_fusion_weighting,
                            confidence_mode=args.forecast_fusion_confidence_mode,
                            return_consensus=True,
                        )
                        if (
                            args.save_correction_analysis
                            or args.anchor_adapter_residual_mode == "reliability_scaled"
                        ):
                            adapter_candidate_dispersion = _bcsa_candidate_residual_dispersion(
                                context,
                                retrieved_selected,
                                seq_len=args.seq_len,
                                pred_len=args.pred_len,
                            )
                        if anchor_adapter_kind == "linear":
                            if anchor_hidden is None:
                                raise RuntimeError(
                                    "Linear anchor adapter requires Chronos hidden state"
                                )
                            adapter_delta = anchor_adapter(
                                anchor_hidden, adapter_consensus
                            )
                            if args.anchor_adapter_residual_mode == "raw":
                                applied_delta = adapter_delta
                            elif args.anchor_adapter_residual_mode == "scaled":
                                applied_delta = (
                                    float(args.anchor_adapter_strength) * adapter_delta
                                )
                            elif args.anchor_adapter_residual_mode == "bounded":
                                query_scale = _query_history_scale(context).unsqueeze(-1)
                                applied_delta = (
                                    float(args.anchor_adapter_strength)
                                    * query_scale
                                    * torch.tanh(adapter_delta / (query_scale + 1e-6))
                                )
                            elif args.anchor_adapter_residual_mode in {
                                "gated_scaled", "gated_bounded"
                            }:
                                if adapter_confidence is None:
                                    raise RuntimeError(
                                        "gated anchor adapter requires forecast confidence"
                                    )
                                gate = adapter_confidence.unsqueeze(-1)
                                if args.anchor_adapter_residual_mode == "gated_scaled":
                                    applied_delta = (
                                        float(args.anchor_adapter_strength)
                                        * gate
                                        * adapter_delta
                                    )
                                else:
                                    query_scale = _query_history_scale(context).unsqueeze(-1)
                                    applied_delta = (
                                        float(args.anchor_adapter_strength)
                                        * gate
                                        * query_scale
                                        * torch.tanh(
                                            adapter_delta / (query_scale + 1e-6)
                                        )
                                    )
                            elif args.anchor_adapter_residual_mode in {
                                "reliability_scaled", "power_scaled", "deadzone_scaled"
                            }:
                                if adapter_confidence is None:
                                    raise RuntimeError(
                                        "adaptive anchor gate requires forecast confidence"
                                    )
                                gate = adapter_confidence
                                if args.anchor_adapter_residual_mode == "reliability_scaled":
                                    if adapter_candidate_dispersion is None:
                                        raise RuntimeError(
                                            "reliability gate requires candidate dispersion"
                                        )
                                    if np.isposinf(float(args.anchor_adapter_reliability_scale)):
                                        reliability = torch.ones_like(adapter_confidence)
                                    else:
                                        reliability = torch.exp(
                                            -adapter_candidate_dispersion
                                            / float(args.anchor_adapter_reliability_scale)
                                        ).clamp(0.0, 1.0)
                                    gate = gate * reliability
                                elif args.anchor_adapter_residual_mode == "power_scaled":
                                    gate = gate.pow(float(args.anchor_adapter_gamma))
                                else:
                                    threshold = float(args.anchor_adapter_deadzone_threshold)
                                    gate = (
                                        (gate - threshold) / (1.0 - threshold)
                                    ).clamp(0.0, 1.0)
                                applied_delta = (
                                    float(args.anchor_adapter_strength)
                                    * gate.unsqueeze(-1)
                                    * adapter_delta
                                )
                            else:
                                raise RuntimeError(
                                    f"unknown anchor adapter residual mode: "
                                    f"{args.anchor_adapter_residual_mode}"
                                )
                            output = output + applied_delta
                        else:
                            film_gamma, film_beta = anchor_adapter(adapter_consensus)
                            film_output = base_model(
                                context=context,
                                film_gamma=film_gamma,
                                film_beta=film_beta,
                            )
                            output = film_output.quantile_preds[
                                :, median_index, :args.pred_len
                            ]
                    raw_prediction = output
                    consensus = None
                    confidence = None
                    if (
                        args.forecast_fusion == "horizon_residual_transport"
                        and method in HORIZON_METHODS
                    ):
                        horizon_pool = np.asarray(
                            sidecar[f"horizon_distances_{method}"]
                        )[
                            local_offset:local_offset + batch_size
                        ]
                        horizon_selected = np.take_along_axis(
                            horizon_pool,
                            np.asarray(selected, dtype=np.int64)[:, None, :],
                            axis=2,
                        )
                        correction_retrieved = retrieved_selected
                        if (
                            candidate_base_cache is None
                            or candidate_base_cache_selection is None
                            or not np.array_equal(
                                candidate_base_cache_selection, np.asarray(selected, dtype=np.int64)
                            )
                        ):
                            candidate_history = correction_retrieved[..., :args.seq_len]
                            candidate_base_cache = base_model(
                                context=candidate_history.reshape(-1, args.seq_len)
                            ).quantile_preds[:, median_index, :args.pred_len]
                            candidate_base_cache = candidate_base_cache.reshape(
                                batch_size, correction_retrieved.shape[1], args.pred_len
                            )
                            candidate_base_cache_selection = np.asarray(
                                selected, dtype=np.int64
                            ).copy()
                        candidate_residuals = selected_future - candidate_base_cache
                        fusion_result = _apply_horizon_residual_transport(
                            output,
                            candidate_residuals,
                            torch.from_numpy(horizon_selected).to(
                                device=device, dtype=torch.float32
                            ),
                            torch.from_numpy(horizon_pool).to(
                                device=device, dtype=torch.float32
                            ),
                            block_size=16,
                            strength=args.forecast_fusion_strength,
                            temperature=args.forecast_fusion_temperature,
                            gate_threshold=args.forecast_fusion_mi_reliability_threshold,
                            gate_power=args.forecast_fusion_mi_reliability_power,
                            correction_clip=args.forecast_fusion_correction_clip,
                            return_consensus=args.save_correction_analysis,
                        )
                        if args.save_correction_analysis:
                            output, confidence, consensus = fusion_result
                        else:
                            output, confidence = fusion_result
                        forecast_confidences[method].append(confidence.cpu().numpy())
                    elif args.forecast_fusion != "none" and method not in {"base", "official", *_NEW_ARM_METHODS}:
                        if args.forecast_fusion == "bound_high_mi_residual_ablation":
                            bound_spec = _bound_method_spec(method)
                            if (
                                method not in _BOUND_HIGH_MI_RESIDUAL_METHOD_SET
                                or bound_spec is None
                            ):
                                raise ValueError(
                                    "bound_high_mi_residual_ablation only supports "
                                    "the four explicit residual arms"
                                )
                            score_pool = np.asarray(sidecar[bound_spec["distance"]])[
                                local_offset:local_offset + batch_size
                            ]
                            # Distance sidecars are mathematically
                            # non-negative, but a few normalized I(H;Y)
                            # artifacts contain tiny float32 round-off such as
                            # -2e-7.  Preserve the strict invariant for real
                            # invalid distances while removing only this
                            # numerical noise before residual weighting.
                            if np.any(score_pool < -1e-6):
                                raise ValueError(
                                    f"{bound_spec['distance']} contains materially "
                                    "negative distances"
                                )
                            score_pool = np.maximum(score_pool, 0.0)
                            expected_pool_shape = (batch_size, args.pool_k)
                            if score_pool.shape != expected_pool_shape:
                                raise ValueError(
                                    f"{bound_spec['distance']} shape {score_pool.shape} "
                                    f"!= expected {expected_pool_shape}"
                                )
                            selected_np = np.asarray(selected, dtype=np.int64)
                            fusion_scores = torch.from_numpy(
                                np.take_along_axis(score_pool, selected_np, axis=1)
                            ).to(device=device, dtype=torch.float32)
                            normalization_distances = torch.from_numpy(score_pool).to(
                                device=device, dtype=torch.float32
                            )
                            correction_retrieved = retrieved_selected
                            candidate_history = correction_retrieved[..., :args.seq_len]
                            if (
                                candidate_base_cache is None
                                or candidate_base_cache_selection is None
                                or not np.array_equal(
                                    candidate_base_cache_selection, selected_np
                                )
                            ):
                                candidate_base_cache = base_model(
                                    context=candidate_history.reshape(-1, args.seq_len)
                                ).quantile_preds[:, median_index, :args.pred_len]
                                candidate_base_cache = candidate_base_cache.reshape(
                                    batch_size, correction_retrieved.shape[1], args.pred_len
                                )
                                candidate_base_cache_selection = selected_np.copy()
                            aligned_future = _align_candidate_futures(
                                context,
                                candidate_history,
                                selected_future,
                                alignment=bound_spec["alignment"],
                                recent_window=args.bound_high_mi_alignment_window,
                                recent_tau=args.tsrag_retrieved_tau,
                            )
                            residual_reference = _bound_residual_reference(
                                output,
                                candidate_base_cache,
                                reference=args.bound_high_mi_residual_reference,
                            )
                            candidate_residuals = aligned_future - residual_reference
                            confidence_override = None
                            if args.bound_high_mi_residual_confidence == "disagreement":
                                confidence_override = _residual_forecast_disagreement_confidence(
                                    candidate_residuals,
                                    fusion_scores,
                                    context,
                                    normalization_distances=normalization_distances,
                                    weighting=args.bound_high_mi_residual_weighting,
                                    temperature=args.forecast_fusion_temperature,
                                    confidence_scale=args.forecast_fusion_confidence_scale,
                                )
                            fusion_result = _apply_residual_prototype_correction(
                                output,
                                candidate_residuals,
                                fusion_scores,
                                lambda_=args.forecast_fusion_strength,
                                correction_clip=args.forecast_fusion_correction_clip,
                                normalization_distances=normalization_distances,
                                estimator=args.bound_high_mi_residual_estimator,
                                temperature=args.forecast_fusion_temperature,
                                weighting=args.bound_high_mi_residual_weighting,
                                confidence_gate=(
                                    args.bound_high_mi_residual_confidence == "direction"
                                ),
                                confidence_override=confidence_override,
                                confidence_floor=args.forecast_fusion_confidence_floor,
                                return_consensus=args.save_correction_analysis,
                            )
                            if args.save_correction_analysis:
                                output, confidence, consensus = fusion_result
                            else:
                                output, confidence = fusion_result
                            forecast_confidences[method].append(confidence.cpu().numpy())
                        elif args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS:
                            if (
                                args.forecast_fusion == "residual_prototype_high_mi"
                                and method != "high_mi"
                            ):
                                raise ValueError(
                                    "residual_prototype_high_mi requires the high_mi selector"
                                )
                            if (
                                args.forecast_fusion != "residual_prototype_high_mi"
                                and method != "mi_residual_prototype"
                                and method != "mi_select"
                                and method not in _OFFICIAL_MI_RESIDUAL_METHODS
                                and not method.startswith("mi_gra_residual")
                            ):
                                raise ValueError(
                                    "residual_prototype fusion requires the "
                                    "mi_residual_prototype, high_mi, or mi_gra_residual selector"
                                )
                            if args.forecast_fusion in {
                                "residual_prototype_hybrid",
                                "residual_prototype_hybrid_median",
                                "residual_prototype_hybrid_confident",
                                "residual_prototype_hybrid_pool",
                            }:
                                score_key = "mi_l2_distances"
                                score_values = np.asarray(sidecar.get(score_key))[
                                    local_offset:local_offset + batch_size
                                ]
                                if args.forecast_fusion == "residual_prototype_hybrid_pool":
                                    selected_mi_scores = torch.from_numpy(score_values).to(
                                        device=device, dtype=torch.float32
                                    )
                                else:
                                    selected_mi_scores = torch.from_numpy(
                                        np.take_along_axis(
                                            score_values,
                                            np.asarray(selected, dtype=np.int64),
                                            axis=1,
                                        )
                                    ).to(device=device, dtype=torch.float32)
                                fusion_scores = _hybrid_residual_fusion_scores(
                                    method, selected_mi_scores, mi_gra_selected
                                )
                            elif method == "mi_residual_prototype" or args.forecast_fusion == "residual_prototype_high_mi":
                                score_key = _residual_prototype_score_key(
                                    args.forecast_fusion, method
                                )
                                if score_key not in sidecar:
                                    raise ValueError(
                                        f"artifact is missing {score_key} for forecast fusion"
                                    )
                                score_values = np.asarray(sidecar[score_key])[
                                    local_offset:local_offset + batch_size
                                ]
                                if args.forecast_fusion == "residual_prototype_high_mi":
                                    score_values = np.take_along_axis(
                                        score_values,
                                        np.asarray(selected, dtype=np.int64),
                                        axis=1,
                                    )
                                fusion_scores = torch.from_numpy(score_values).to(
                                    device=device, dtype=torch.float32
                                )
                            else:
                                # MI-GRA residual uses the same fixed official
                                # Top-10 candidates and the history-only
                                # distance/control values as correction weights.
                                # Reusing the exact tensor passed to the ARM
                                # keeps the Real/Shuffle/Reverse/Uniform
                                # residual controls matched to the same gate.
                                fusion_scores = mi_gra_selected.to(
                                    device=device, dtype=torch.float32
                                )
                            normalization_key = (
                                "high_mi_distances"
                                if args.forecast_fusion == "residual_prototype_high_mi"
                                else "mi_l2_distances"
                                if args.forecast_fusion in {
                                    "residual_prototype_hybrid",
                                    "residual_prototype_hybrid_median",
                                    "residual_prototype_hybrid_confident",
                                    "residual_prototype_hybrid_pool",
                                }
                                else args.mi_gra_distance_key
                            )
                            normalization_source = (
                                mi_gra_sidecar
                                if method.startswith("mi_gra")
                                and args.forecast_fusion != "residual_prototype_high_mi"
                                else sidecar
                            )
                            pool_distance_values = np.asarray(
                                normalization_source.get(normalization_key)
                            )
                            expected_pool_shape = (len(sidecar["official_distances"]), args.pool_k)
                            if pool_distance_values.shape != expected_pool_shape:
                                raise ValueError(
                                    "artifact mi_l2_distances must have shape "
                                    f"{expected_pool_shape}, got {pool_distance_values.shape}"
                                )
                            normalization_distances = torch.from_numpy(
                                pool_distance_values[
                                    local_offset:local_offset + batch_size
                                ]
                            ).to(device=device, dtype=torch.float32)
                            hybrid_official_normalization_distances = None
                            if args.forecast_fusion in {
                                "residual_prototype_hybrid",
                                "residual_prototype_hybrid_median",
                                "residual_prototype_hybrid_confident",
                                "residual_prototype_hybrid_pool",
                            }:
                                official_pool_values = np.asarray(
                                    sidecar.get("official_distances")
                                )
                                if official_pool_values.shape != expected_pool_shape:
                                    raise ValueError(
                                        "artifact official_distances must have shape "
                                        f"{expected_pool_shape}, got {official_pool_values.shape}"
                                    )
                                hybrid_official_normalization_distances = torch.from_numpy(
                                    official_pool_values[
                                        local_offset:local_offset + batch_size
                                    ]
                                ).to(device=device, dtype=torch.float32)
                            correction_selection = (
                                np.tile(np.arange(args.pool_k), (batch_size, 1))
                                if args.forecast_fusion == "residual_prototype_hybrid_pool"
                                else np.asarray(selected, dtype=np.int64)
                            )
                            correction_retrieved = (
                                retrieved
                                if args.forecast_fusion == "residual_prototype_hybrid_pool"
                                else retrieved_selected
                            )
                            correction_future = (
                                candidate_future
                                if args.forecast_fusion == "residual_prototype_hybrid_pool"
                                else selected_future
                            )
                            if (
                                candidate_base_cache is None
                                or candidate_base_cache_selection is None
                                or not np.array_equal(
                                    candidate_base_cache_selection, correction_selection
                                )
                            ):
                                candidate_history = correction_retrieved[..., :args.seq_len]
                                candidate_base_cache = base_model(
                                    context=candidate_history.reshape(-1, args.seq_len)
                                ).quantile_preds[:, median_index, :args.pred_len]
                                candidate_base_cache = candidate_base_cache.reshape(
                                    batch_size,
                                    correction_retrieved.shape[1],
                                    args.pred_len,
                                )
                                candidate_base_cache_selection = correction_selection.copy()
                            candidate_base = candidate_base_cache
                            candidate_residuals = correction_future - candidate_base
                            fusion_result = _apply_residual_prototype_correction(
                                output,
                                candidate_residuals,
                                fusion_scores,
                                lambda_=args.forecast_fusion_strength,
                                correction_clip=args.forecast_fusion_correction_clip,
                                normalization_distances=normalization_distances,
                                hybrid_beta=(
                                    args.forecast_fusion_hybrid_beta
                                    if args.forecast_fusion in {
                                        "residual_prototype_hybrid",
                                        "residual_prototype_hybrid_median",
                                        "residual_prototype_hybrid_confident",
                                        "residual_prototype_hybrid_pool",
                                    }
                                    else None
                                ),
                                hybrid_official_distances=(
                                    distances
                                    if args.forecast_fusion == "residual_prototype_hybrid_pool"
                                    else distance_selected
                                    if args.forecast_fusion in {
                                        "residual_prototype_hybrid",
                                        "residual_prototype_hybrid_median",
                                        "residual_prototype_hybrid_confident",
                                        "residual_prototype_hybrid_pool",
                                    }
                                    else None
                                ),
                                hybrid_official_normalization_distances=(
                                    hybrid_official_normalization_distances
                                    if args.forecast_fusion in {
                                        "residual_prototype_hybrid",
                                        "residual_prototype_hybrid_median",
                                        "residual_prototype_hybrid_confident",
                                        "residual_prototype_hybrid_pool",
                                    }
                                    else None
                                ),
                                estimator=(
            "median"
            if args.forecast_fusion in {
                "residual_prototype_median",
                                        "residual_prototype_hybrid_median",
                                    }
                                    else "mean"
                                ),
                                confidence_gate=(
                                    args.forecast_fusion in {
                                        "residual_prototype_confident",
                                        "residual_prototype_hybrid_confident",
                                        "residual_prototype_mi_reliable",
                                        "residual_prototype_high_mi",
                                        "residual_prototype_mi_gra_plus_high_mi",
                                    }
                                ),
                                confidence_floor=args.forecast_fusion_confidence_floor,
                                mi_reliability_gate=(
                                    args.forecast_fusion == "residual_prototype_mi_reliable"
                                ),
                                mi_reliability_official_distances=distance_selected,
                                mi_reliability_threshold=args.forecast_fusion_mi_reliability_threshold,
                                mi_reliability_power=args.forecast_fusion_mi_reliability_power,
                                return_consensus=args.save_correction_analysis,
                            )
                            if args.forecast_fusion == "residual_prototype_mi_gra_plus_high_mi":
                                if not method.startswith("mi_gra_residual"):
                                    raise ValueError(
                                        "residual_prototype_mi_gra_plus_high_mi requires "
                                        "a mi_gra_residual method"
                                    )
                                if reference_prediction is None:
                                    raise ValueError(
                                        "dual MI forecast fusion requires official to be "
                                        "evaluated before the MI-GRA method"
                                    )
                                if "high_mi_distances" not in sidecar:
                                    raise ValueError(
                                        "residual_prototype_mi_gra_plus_high_mi requires "
                                        "an artifact containing high_mi_distances"
                                    )
                                if "selected_ranks_high_mi" not in sidecar:
                                    raise ValueError(
                                        "residual_prototype_mi_gra_plus_high_mi requires "
                                        "selected_ranks_high_mi in the high-MI artifact"
                                    )
                                high_mi_pool = np.asarray(sidecar["high_mi_distances"])
                                if high_mi_pool.shape != expected_pool_shape:
                                    raise ValueError(
                                        "artifact high_mi_distances must have shape "
                                        f"{expected_pool_shape}, got {high_mi_pool.shape}"
                                    )
                                high_mi_indices = np.asarray(
                                    sidecar["selected_ranks_high_mi"]
                                )[local_offset:local_offset + batch_size]
                                high_mi_retrieved_selected = _gather(
                                    retrieved, high_mi_indices
                                )
                                high_mi_selected = torch.from_numpy(
                                    np.take_along_axis(
                                        high_mi_pool[
                                            local_offset:local_offset + batch_size
                                        ],
                                        high_mi_indices,
                                        axis=1,
                                    )
                                ).to(device=device, dtype=torch.float32)
                                high_mi_result = _apply_mi_forecast_fusion(
                                    reference_prediction,
                                    context,
                                    high_mi_retrieved_selected,
                                    high_mi_selected,
                                    seq_len=args.seq_len,
                                    pred_len=args.pred_len,
                                    strength=args.forecast_fusion_dual_high_mi_strength,
                                    temperature=args.forecast_fusion_temperature,
                                    confidence_scale=args.forecast_fusion_confidence_scale,
                                    alignment=args.forecast_fusion_alignment,
                                    weighting=args.forecast_fusion_weighting,
                                    confidence_mode=args.forecast_fusion_confidence_mode,
                                    return_consensus=args.save_correction_analysis,
                                )
                                high_mi_weight = float(args.forecast_fusion_hybrid_beta)
                                if args.save_correction_analysis:
                                    primary_output, primary_confidence, primary_consensus = fusion_result
                                    high_output, high_confidence, high_consensus = high_mi_result
                                    output = _blend_dual_forecasts(
                                        primary_output,
                                        high_output,
                                        high_mi_weight,
                                    )
                                    confidence = _blend_dual_forecasts(
                                        primary_confidence[:, None],
                                        high_confidence[:, None],
                                        high_mi_weight,
                                    ).squeeze(1)
                                    consensus = _blend_dual_forecasts(
                                        primary_consensus,
                                        high_consensus,
                                        high_mi_weight,
                                    )
                                    fusion_result = output, confidence, consensus
                                else:
                                    primary_output, primary_confidence = fusion_result
                                    high_output, high_confidence = high_mi_result
                                    output = _blend_dual_forecasts(
                                        primary_output,
                                        high_output,
                                        high_mi_weight,
                                    )
                                    confidence = _blend_dual_forecasts(
                                        primary_confidence[:, None],
                                        high_confidence[:, None],
                                        high_mi_weight,
                                    ).squeeze(1)
                                    fusion_result = output, confidence
                        else:
                            fusion_retrieved = retrieved_selected
                            if args.forecast_fusion == "mi_prior":
                                if method != "mi_prior":
                                    raise ValueError("mi_prior forecast fusion requires methods to include only the mi_prior MI variant")
                                score_key = "selection_scores_mi_prior"
                            elif args.forecast_fusion in {
                                "bound_high_mi_ablation",
                                "bound_high_mi_linear_mix",
                            }:
                                bound_spec = _bound_method_spec(method)
                                if bound_spec is None:
                                    raise ValueError(
                                        "bound high-MI special fusion only supports the "
                                        "explicit bound-ablation methods"
                                    )
                                score_key = bound_spec["distance"]
                            elif args.forecast_fusion == "coupled_recent_mi":
                                score_key = _coupled_recent_distance_key(method)
                            else:
                                score_key = {
                                    "high_mi": "high_mi_distances",
                                    "high_mi_pool": "high_mi_distances",
                                    "high_mi_linear": "high_mi_distances",
                                    "high_mi_consistency": "high_mi_distances",
                                    "low_mi": "low_mi_distances",
                                    "random": "random_distances",
                                    "uniform": "uniform_distances",
                                }[args.forecast_fusion]
                            if is_global:
                                assert global_sidecar is not None
                                global_score_key = f"selection_scores_{method}"
                                score_values = np.asarray(
                                    global_sidecar[global_score_key]
                                )[local_offset:local_offset + batch_size]
                            elif score_key not in sidecar:
                                if (
                                    args.forecast_fusion == "coupled_recent_mi"
                                    and method == "recent"
                                    and score_key == "recency_distances"
                                    and "query_origins" in sidecar
                                    and "candidate_starts" in sidecar
                                ):
                                    score_values = causal_recency_distances(
                                        np.asarray(sidecar["query_origins"])[
                                            local_offset:local_offset + batch_size
                                        ],
                                        np.asarray(sidecar["candidate_starts"])[
                                            local_offset:local_offset + batch_size
                                        ],
                                        candidate_span=args.pred_len,
                                    )
                                else:
                                    raise ValueError(
                                        f"artifact is missing {score_key} for forecast fusion"
                                    )
                            else:
                                score_values = np.asarray(sidecar[score_key])[
                                    local_offset:local_offset + batch_size
                                ]
                            if args.forecast_fusion == "mi_prior":
                                fusion_scores = torch.from_numpy(score_values)
                            elif _forecast_fusion_uses_full_pool(args.forecast_fusion):
                                if method != "high_mi":
                                    raise ValueError(
                                        "high_mi_pool forecast fusion requires the high_mi selector"
                                    )
                                fusion_retrieved = retrieved
                                fusion_scores = torch.from_numpy(score_values)
                            else:
                                fusion_scores = torch.from_numpy(
                                    np.take_along_axis(
                                        score_values,
                                        np.asarray(selected, dtype=np.int64), axis=1,
                                    )
                                )
                            fusion_scores = fusion_scores.to(device=device, dtype=torch.float32)
                            confidence_override = None
                            if args.forecast_fusion_confidence_mode in _HISTORY_CONFIDENCE_FUSIONS:
                                confidence_override = history_confidence_override
                            if args.forecast_fusion in _RESIDUAL_CONFIDENCE_FUSIONS:
                                candidate_history = retrieved_selected[..., :args.seq_len]
                                candidate_base = base_model(
                                    context=candidate_history.reshape(-1, args.seq_len)
                                ).quantile_preds[:, median_index, :args.pred_len]
                                candidate_base = candidate_base.reshape(
                                    batch_size, args.top_k, args.pred_len
                                )
                                candidate_residuals = selected_future - candidate_base
                                confidence_values = residual_prototype_confidence(
                                    candidate_residuals.detach().cpu().numpy(),
                                    fusion_scores.detach().cpu().numpy(),
                                    reference_distances=score_values,
                                )
                                confidence_override = torch.from_numpy(
                                    confidence_values
                                ).to(device=device, dtype=torch.float32)
                            fusion_result = _apply_mi_forecast_fusion(
                                output,
                                context,
                                fusion_retrieved,
                                fusion_scores,
                                seq_len=args.seq_len,
                                pred_len=args.pred_len,
                                strength=args.forecast_fusion_strength,
                                temperature=(
                                    1.0
                                    if args.forecast_fusion == "bound_high_mi_linear_mix"
                                    else args.forecast_fusion_temperature
                                ),
                                reverse_weights=args.forecast_fusion_reverse_weights,
                                confidence_scale=args.forecast_fusion_confidence_scale,
                                weighting=args.forecast_fusion_weighting,
                                confidence_mode=args.forecast_fusion_confidence_mode,
                                alignment=(
                                    args.bound_high_mi_alignment
                                    if (
                                        args.forecast_fusion == "bound_high_mi_ablation"
                                        and _bound_method_spec(method)["alignment"]
                                        == "aligned"
                                    )
                                    else _bound_method_spec(method)["alignment"]
                                    if args.forecast_fusion in {
                                        "bound_high_mi_ablation",
                                        "bound_high_mi_linear_mix",
                                    }
                                    else args.forecast_fusion_alignment
                                ),
                                use_confidence=(
                                    args.forecast_fusion not in (
                                        _UNIT_CONFIDENCE_FUSIONS
                                        | _RESIDUAL_CONFIDENCE_FUSIONS
                                    )
                                ),
                                confidence_override=confidence_override,
                                confidence_multiplier=(
                                    mi_match_multiplier
                                    if method == "high_mi"
                                    else (
                                        mi_branch_gate_override
                                        if _forecast_fusion_mi_branch_gate_applies(
                                            method, args.forecast_fusion
                                        )
                                        and method == "mi_recent"
                                        else None
                                    )
                                ),
                                return_consensus=args.save_correction_analysis,
                            )
                        if args.save_correction_analysis:
                            output, confidence, consensus = fusion_result
                        else:
                            output, confidence = fusion_result
                        forecast_confidences[method].append(confidence.cpu().numpy())
                        if weighting_ablation_alignments and method == "high_mi":
                            weighting_ablation_batch = {}
                            for alignment in weighting_ablation_alignments:
                                arm_predictions = _fixed_confidence_weighting_ablation(
                                    raw_prediction,
                                    context,
                                    fusion_retrieved,
                                    fusion_scores,
                                    distance_selected,
                                    confidence,
                                    seq_len=args.seq_len,
                                    pred_len=args.pred_len,
                                    strength=args.forecast_fusion_strength,
                                    temperature=args.forecast_fusion_temperature,
                                    alignment=alignment,
                                    ordinary_retrieved_selected=ordinary_retrieved_selected,
                                    ordinary_official_scores=ordinary_distance_selected,
                                )
                                if alignment == args.forecast_fusion_alignment:
                                    if not torch.allclose(
                                        arm_predictions["mi_distance_weight"],
                                        output,
                                        rtol=1e-5,
                                        atol=1e-6,
                                    ):
                                        raise RuntimeError(
                                            "MI weighting ablation does not reproduce the "
                                            "configured high_mi forecast"
                                        )
                                weighting_ablation_batch.update({
                                    f"{alignment}__{arm}": prediction
                                    for arm, prediction in arm_predictions.items()
                                    if arm != "no_future"
                                })
                            if factorial_ablation_enabled:
                                factorial_predictions = _factorial_confidence_weighting_ablation(
                                    raw_prediction,
                                    context,
                                    fusion_retrieved,
                                    fusion_scores,
                                    distance_selected,
                                    confidence,
                                    seq_len=args.seq_len,
                                    pred_len=args.pred_len,
                                    strength=args.forecast_fusion_strength,
                                    temperature=args.forecast_fusion_temperature,
                                    ordinary_retrieved_selected=ordinary_retrieved_selected,
                                    ordinary_official_scores=ordinary_distance_selected,
                                )
                                weighting_ablation_batch.update({
                                    f"factorial__{arm}": prediction
                                    for arm, prediction in factorial_predictions.items()
                                    if arm != "no_future"
                                })
                            weighting_ablation_batch["no_future"] = raw_prediction
                        if args.mi_attribution_ablation and method == "high_mi":
                            selected_np = np.asarray(selected, dtype=np.int64)
                            official_selected = np.tile(
                                np.arange(args.top_k, dtype=np.int64),
                                (batch_size, 1),
                            )
                            mi_pool = np.asarray(sidecar["high_mi_distances"])[
                                local_offset:local_offset + batch_size
                            ]
                            official_pool = np.asarray(sidecar["official_distances"])[
                                local_offset:local_offset + batch_size
                            ]
                            official_retrieved_selected = _gather(
                                retrieved, official_selected
                            )
                            mi_scores_on_official = torch.from_numpy(
                                np.take_along_axis(
                                    mi_pool, official_selected, axis=1
                                )
                            ).to(device=device, dtype=torch.float32)
                            official_scores_on_mi = torch.from_numpy(
                                np.take_along_axis(
                                    official_pool, selected_np, axis=1
                                )
                            ).to(device=device, dtype=torch.float32)
                            mi_scores_on_mi = torch.from_numpy(
                                np.take_along_axis(
                                    mi_pool, selected_np, axis=1
                                )
                            ).to(device=device, dtype=torch.float32)
                            attribution_predictions, attribution_confidences = (
                                _mi_attribution_ablation(
                                    raw_prediction,
                                    context,
                                    retrieved_selected,
                                    mi_scores_on_mi,
                                    official_retrieved_selected,
                                    mi_scores_on_official,
                                    official_scores_on_mi,
                                    seq_len=args.seq_len,
                                    pred_len=args.pred_len,
                                    strength=args.forecast_fusion_strength,
                                    temperature=args.forecast_fusion_temperature,
                                    confidence_scale=args.forecast_fusion_confidence_scale,
                                )
                            )
                            mi_attribution_batch = {
                                name: (
                                    attribution_predictions[name],
                                    attribution_confidences[name],
                                )
                                for name in attribution_predictions
                            }
                        if reference_prediction is not None:
                            method_gain = gain_for(method)
                            gain_tensor = torch.as_tensor(
                                method_gain,
                                device=output.device,
                                dtype=output.dtype,
                            )
                            output = reference_prediction + gain_tensor * (
                                output - reference_prediction
                            )
                            if weighting_ablation_batch is not None:
                                weighting_ablation_batch = {
                                    arm: reference_prediction + gain_tensor * (
                                        prediction - reference_prediction
                                    )
                                    for arm, prediction in weighting_ablation_batch.items()
                                }
                            if mi_attribution_batch is not None:
                                mi_attribution_batch = {
                                    name: (
                                        reference_prediction + gain_tensor * (
                                            prediction - reference_prediction
                                        ),
                                        confidence_value,
                                    )
                                    for name, (prediction, confidence_value)
                                    in mi_attribution_batch.items()
                                }
                        if weighting_ablation_batch is not None:
                            for arm, prediction in weighting_ablation_batch.items():
                                weighting_ablation_predictions[arm].append(
                                    prediction.float().cpu().numpy()
                                )
                            weighting_ablation_confidence.append(
                                confidence.float().cpu().numpy()
                            )
                        if mi_attribution_batch is not None:
                            for name, (prediction, confidence_value) in mi_attribution_batch.items():
                                mi_attribution_predictions[name].append(
                                    prediction.float().cpu().numpy()
                                )
                                mi_attribution_confidences[name].append(
                                    confidence_value.float().cpu().numpy()
                                )
                        if args.system_ablation_suite and method == "high_mi_anchor":
                            if reference_prediction is None:
                                raise RuntimeError("system ablations require the official prediction first")
                            selected_np = np.asarray(selected, dtype=np.int64)
                            official_selected = np.tile(
                                np.arange(args.top_k, dtype=np.int64), (batch_size, 1)
                            )
                            mi_pool = np.asarray(sidecar["high_mi_distances"])[
                                local_offset:local_offset + batch_size
                            ]
                            official_pool = np.asarray(sidecar["official_distances"])[
                                local_offset:local_offset + batch_size
                            ]
                            def selected_scores(pool, indices):
                                return torch.from_numpy(
                                    np.take_along_axis(pool, indices, axis=1)
                                ).to(device=device, dtype=torch.float32)
                            mi_scores = selected_scores(mi_pool, selected_np)
                            official_scores = selected_scores(official_pool, selected_np)
                            mi_on_official = selected_scores(mi_pool, official_selected)
                            official_on_official = selected_scores(official_pool, official_selected)
                            official_retrieved = _gather(retrieved, official_selected)
                            attribution, _ = _mi_attribution_ablation(
                                raw_prediction, context, retrieved_selected, mi_scores,
                                official_retrieved, mi_on_official, official_scores,
                                seq_len=args.seq_len, pred_len=args.pred_len,
                                strength=args.forecast_fusion_strength,
                                temperature=args.forecast_fusion_temperature,
                                confidence_scale=args.forecast_fusion_confidence_scale,
                            )
                            def correction(anchor, candidates, scores, alignment):
                                prediction, _ = _apply_mi_forecast_fusion(
                                    anchor, context, candidates, scores,
                                    seq_len=args.seq_len, pred_len=args.pred_len,
                                    strength=args.forecast_fusion_strength,
                                    temperature=args.forecast_fusion_temperature,
                                    confidence_scale=args.forecast_fusion_confidence_scale,
                                    alignment=alignment, weighting="mi",
                                    confidence_mode="forecast_disagreement",
                                )
                                return prediction
                            arms = {
                                "full": output,
                                "tsrag": reference_prediction,
                                "without_mi_rank": attribution["w/o_mi_selection"],
                                "without_mi_distance": attribution["w/o_mi_weighting"],
                                "without_mi_gate": attribution["w/o_mi_gate"],
                                "without_bcsa": correction(raw_prediction, retrieved_selected, mi_scores, "none"),
                                "without_mi": correction(reference_prediction, official_retrieved, official_on_official, "bcsa"),
                                "without_mi_bcsa": correction(reference_prediction, official_retrieved, official_on_official, "none"),
                                "without_residual_correction": raw_prediction,
                            }
                            for name, prediction in arms.items():
                                error = prediction - truth
                                system_ablation_losses[name]["mse"].append(
                                    error.square().mean(dim=1).float().cpu().numpy()
                                )
                                system_ablation_losses[name]["mae"].append(
                                    error.abs().mean(dim=1).float().cpu().numpy()
                                )
                    if args.save_correction_analysis and method in correction_analysis:
                        if consensus is None or confidence is None:
                            raise RuntimeError("forecast fusion did not produce correction-analysis details")
                        if reference_prediction is None:
                            raise RuntimeError(
                                "correction analysis requires the official method "
                                "to be evaluated before the fused MI method"
                            )
                        correction_analysis[method]["official_reference_prediction"].append(
                            reference_prediction.float().cpu().numpy()
                        )
                        correction_analysis[method]["tsrag_prediction"].append(
                            raw_prediction.float().cpu().numpy()
                        )
                        correction_analysis[method]["mi_consensus"].append(
                            consensus.float().cpu().numpy()
                        )
                        correction_analysis[method]["fused_prediction"].append(
                            output.float().cpu().numpy()
                        )
                        correction_analysis[method]["forecast_confidence"].append(
                            confidence.float().cpu().numpy()
                        )
                        correction_analysis[method]["query_history_scale"].append(
                            _query_history_scale(context).float().cpu().numpy()
                        )
                        if adapter_candidate_dispersion is None:
                            correction_analysis[method]["candidate_residual_dispersion"].append(
                                np.full((batch_size,), np.nan, dtype=np.float32)
                            )
                        else:
                            correction_analysis[method]["candidate_residual_dispersion"].append(
                                adapter_candidate_dispersion.float().cpu().numpy()
                            )
                if method == "official":
                    reference_prediction = output
                output_numpy = output.float().cpu().numpy()
                if args.stream_metrics:
                    if stream_metric_stats is None:
                        raise RuntimeError("stream metric state was not initialized")
                    _update_stream_metric_stats(
                        stream_metric_stats[method], output_numpy, truth_numpy
                    )
                else:
                    predictions[method].append(output_numpy)
            local_offset += batch_size
    truth = (
        np.empty((0,), dtype=np.float32)
        if args.stream_metrics
        else np.concatenate(truths, axis=0)
    )
    merged = (
        {method: np.empty((0,), dtype=np.float32) for method in methods}
        if args.stream_metrics
        else {method: np.concatenate(values, axis=0) for method, values in predictions.items()}
    )
    merged_retrieval = {method: np.concatenate(values, axis=0) for method, values in retrieval_scores.items() if values}
    merged_oracle_recall = {method: np.concatenate(values, axis=0) for method, values in oracle_recalls.items() if values}
    merged_official_overlap = {method: np.concatenate(values, axis=0) for method, values in official_overlaps.items() if values}
    oracle_best_mse = None
    if merged_retrieval:
        # Recompute the offline oracle from the candidate-level diagnostics
        # accumulated above.  The scalar is populated below from the batch
        # values so no oracle candidate is ever passed to the forecaster.
        oracle_best_mse = float(np.mean(np.concatenate(oracle_best_values, axis=0)))
    rows = _metric_rows(
        merged, truth, merged_retrieval,
        mi_target=_sidecar_mi_target(sidecar_meta) or "I(H;Y)",
        mi_condition=str(sidecar_meta.get("mi_condition", "none")),
        arm_bias=args.arm_bias, augment_mode=args.augment_mode,
        arm_bias_strength=args.arm_bias_strength,
        arm_bias_method=args.arm_bias_method,
        forecast_fusion=args.forecast_fusion,
        forecast_fusion_strength=args.forecast_fusion_strength,
        forecast_fusion_temperature=args.forecast_fusion_temperature,
        forecast_fusion_reverse_weights=args.forecast_fusion_reverse_weights,
        forecast_fusion_confidence_scale=args.forecast_fusion_confidence_scale,
        forecast_fusion_confidence_floor=args.forecast_fusion_confidence_floor,
        forecast_fusion_confidence_mode=args.forecast_fusion_confidence_mode,
        forecast_fusion_alignment=args.forecast_fusion_alignment,
        output_mi_gamma=args.output_mi_gamma,
        output_mi_prior_strength=args.output_mi_prior_strength,
        output_mi_prior_centering=args.output_mi_prior_centering,
        output_mi_distance_key=args.output_mi_distance_key,
        forecast_confidence={
            method: np.concatenate(values, axis=0)
            for method, values in forecast_confidences.items() if values
        },
        retrieval_block_scores={
            method: np.concatenate(values, axis=0)
            for method, values in retrieval_block_scores.items() if values
        },
        oracle_recall=merged_oracle_recall,
        official_overlap=merged_official_overlap,
        contamination_mode=args.contamination_mode,
        oracle_best_mse=oracle_best_mse,
        seed=args.seed,
        stream_stats=stream_metric_stats,
    )
    hash_source = global_sidecar if global_sidecar is not None else sidecar
    output = Path(args.results_dir).resolve() / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if args.save_preds:
        np.savez_compressed(output / "predictions.npz", truth=truth, **merged)
    system_ablation_rows: list[dict[str, object]] = []
    if args.system_ablation_suite:
        loss_arrays = {
            f"{name}_{metric}": np.concatenate(system_ablation_losses[name][metric])
            for name in system_ablation_names for metric in ("mse", "mae")
        }
        np.savez_compressed(output / "system_ablation_losses.npz", **loss_arrays)
        system_ablation_rows = [
            {"arm": name, "mse": float(loss_arrays[f"{name}_mse"].mean()),
             "mae": float(loss_arrays[f"{name}_mae"].mean())}
            for name in system_ablation_names
        ]
        with (output / "system_ablation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["arm", "mse", "mae"])
            writer.writeheader()
            writer.writerows(system_ablation_rows)
    if args.save_correction_analysis:
        analysis_values: dict[str, np.ndarray] = {"truth": truth}
        for method, fields in correction_analysis.items():
            for field, values in fields.items():
                if not values:
                    raise RuntimeError(f"no correction-analysis values collected for {method}/{field}")
                analysis_values[f"{method}_{field}"] = np.concatenate(values, axis=0)
        np.savez_compressed(output / "correction_alignment_inputs.npz", **analysis_values)
    mi_attribution_rows: list[dict[str, object]] = []
    if args.mi_attribution_ablation:
        merged_attribution = {
            name: np.concatenate(values, axis=0)
            for name, values in mi_attribution_predictions.items()
        }
        merged_attribution_confidence = {
            name: np.concatenate(values, axis=0)
            for name, values in mi_attribution_confidences.items()
        }
        if args.save_preds:
            np.savez_compressed(
                output / "mi_attribution_predictions.npz",
                truth=truth,
                full_anchor=merged["high_mi"],
                **merged_attribution,
            )
        full_mse = float(np.square(merged["high_mi"] - truth).mean())
        full_mae = float(np.abs(merged["high_mi"] - truth).mean())
        for name in (
            "w/o_mi_selection",
            "w/o_mi_weighting",
            "w/o_mi_gate",
        ):
            prediction = merged_attribution[name]
            mse = float(np.square(prediction - truth).mean())
            mae = float(np.abs(prediction - truth).mean())
            mi_attribution_rows.append({
                "variant": name,
                "mse": mse,
                "mae": mae,
                "mse_delta_vs_full_pct": 100.0 * (mse - full_mse) / full_mse,
                "mae_delta_vs_full_pct": 100.0 * (mae - full_mae) / full_mae,
                "full_mse": full_mse,
                "full_mae": full_mae,
                "confidence_mean": float(
                    merged_attribution_confidence[name].mean()
                ),
            })
        with (output / "mi_attribution_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mi_attribution_rows[0]))
            writer.writeheader()
            writer.writerows(mi_attribution_rows)
    weighting_ablation_rows: list[dict[str, object]] = []
    factorial_ablation_rows: list[dict[str, object]] = []
    if weighting_ablation_alignments:
        merged_weighting = {
            arm: np.concatenate(values, axis=0)
            for arm, values in weighting_ablation_predictions.items()
        }
        confidence_values = np.concatenate(weighting_ablation_confidence, axis=0)
        np.savez_compressed(
            output / "weighting_ablation_predictions.npz",
            truth=truth,
            official_reference=merged["official"],
            fixed_history_confidence=confidence_values,
            **merged_weighting,
        )
        no_future_mse = float(np.square(merged_weighting["no_future"] - truth).mean())
        no_future_mae = float(np.abs(merged_weighting["no_future"] - truth).mean())
        selection_by_arm = {
            "no_future": "none",
            "uniform_weight": "high_mi",
            "official_distance_weight": "ordinary_distance",
            "mi_distance_weight": "high_mi",
        }
        for alignment in weighting_ablation_alignments:
            uniform_prediction = merged_weighting[f"{alignment}__uniform_weight"]
            uniform_mse = float(np.square(uniform_prediction - truth).mean())
            uniform_mae = float(np.abs(uniform_prediction - truth).mean())
            for arm in (
                "no_future",
                "uniform_weight",
                "official_distance_weight",
                "mi_distance_weight",
            ):
                key = "no_future" if arm == "no_future" else f"{alignment}__{arm}"
                prediction = merged_weighting[key]
                mse = float(np.square(prediction - truth).mean())
                mae = float(np.abs(prediction - truth).mean())
                weighting_ablation_rows.append({
                    "alignment": alignment,
                    "arm": arm,
                    "selection": selection_by_arm[arm],
                    "mse": mse,
                    "mae": mae,
                    "mse_gain_vs_no_future_pct": 100.0 * (no_future_mse - mse) / no_future_mse,
                    "mae_gain_vs_no_future_pct": 100.0 * (no_future_mae - mae) / no_future_mae,
                    "mse_gain_vs_uniform_pct": 100.0 * (uniform_mse - mse) / uniform_mse,
                    "mae_gain_vs_uniform_pct": 100.0 * (uniform_mae - mae) / uniform_mae,
                    "fixed_confidence_mean": float(confidence_values.mean()),
                })
        if factorial_ablation_enabled:
            factorial = {
                arm: merged_weighting[f"factorial__{arm}"]
                for arm in (
                    "ordinary_distance",
                    "standardization_only",
                    "weighting_only",
                    "standardized_weighted",
                )
            }
            factorial_metrics = {
                arm: {
                    "mse": float(np.square(prediction - truth).mean()),
                    "mae": float(np.abs(prediction - truth).mean()),
                }
                for arm, prediction in factorial.items()
            }
            ordinary_mse = factorial_metrics["ordinary_distance"]["mse"]
            ordinary_mae = factorial_metrics["ordinary_distance"]["mae"]
            standardized_mse = factorial_metrics["standardization_only"]["mse"]
            standardized_mae = factorial_metrics["standardization_only"]["mae"]
            official_mse = float(np.square(merged["official"] - truth).mean())
            official_mae = float(np.abs(merged["official"] - truth).mean())
            for arm, metrics in factorial_metrics.items():
                factorial_ablation_rows.append({
                    "arm": arm,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "mse_gain_vs_official_tsrag_pct": 100.0 * (
                        official_mse - metrics["mse"]
                    ) / official_mse,
                    "mae_gain_vs_official_tsrag_pct": 100.0 * (
                        official_mae - metrics["mae"]
                    ) / official_mae,
                    "mse_gain_vs_ordinary_distance_pct": 100.0 * (
                        ordinary_mse - metrics["mse"]
                    ) / ordinary_mse,
                    "mae_gain_vs_ordinary_distance_pct": 100.0 * (
                        ordinary_mae - metrics["mae"]
                    ) / ordinary_mae,
                    "mse_gain_vs_standardization_only_pct": 100.0 * (
                        standardized_mse - metrics["mse"]
                    ) / standardized_mse,
                    "mae_gain_vs_standardization_only_pct": 100.0 * (
                        standardized_mae - metrics["mae"]
                    ) / standardized_mae,
                    "fixed_confidence_mean": float(confidence_values.mean()),
                })
        with (output / "weighting_ablation_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(weighting_ablation_rows[0])
            )
            writer.writeheader()
            writer.writerows(weighting_ablation_rows)
        if factorial_ablation_rows:
            with (output / "weighting_ablation_factorial.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(factorial_ablation_rows[0])
                )
                writer.writeheader()
                writer.writerows(factorial_ablation_rows)
    if args.save_mi_gra_diagnostics:
        diagnostic_values: dict[str, np.ndarray] = {}
        for method, fields in mi_gra_diagnostics.items():
            for field, values in fields.items():
                if not values:
                    raise RuntimeError(f"no MI-GRA diagnostics collected for {method}/{field}")
                diagnostic_values[f"{method}_{field}"] = np.concatenate(values, axis=0)
        np.savez_compressed(output / "mi_gra_diagnostics.npz", **diagnostic_values)
    summary = {
        "backend": "TS-RAG",
        "dataset": args.dataset,
        "split": args.split,
        "methods": methods,
        "query_start": query_start,
        "query_end": query_end,
        "query_windows": len(query_indices),
        "query_count": args.query_count,
        "query_sampling": args.query_sampling if args.query_count is not None else "all",
        "query_parity": args.query_parity,
        "stream_metrics": bool(args.stream_metrics),
        "query_indices_hash": _array_hash(query_indices),
        "artifact": str(Path(args.artifact).resolve()) if args.artifact else None,
        "mi_gra_artifact": (
            str(Path(args.mi_gra_artifact).resolve())
            if args.mi_gra_artifact else None
        ),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "official_candidate_pool": args.pool_k,
        "neighbors": args.top_k,
        "horizon_diagnostics": horizon_diagnostics,
        "contamination_mode": args.contamination_mode,
        "augment_mode": args.augment_mode,
        "forecast_anchor": args.forecast_anchor,
        "anchor_adapter_checkpoint": (
            str(Path(args.anchor_adapter_checkpoint).resolve())
            if args.anchor_adapter_checkpoint else None
        ),
        "anchor_adapter_type": anchor_adapter_kind,
        "anchor_adapter_config": anchor_adapter_config,
        "anchor_adapter_residual_mode": args.anchor_adapter_residual_mode,
        "anchor_adapter_strength": float(args.anchor_adapter_strength),
        "arm_bias": bool(args.arm_bias),
        "arm_bias_strength": float(args.arm_bias_strength),
        "arm_bias_method": args.arm_bias_method,
        "arm_attention_prior": bool(args.arm_attention_prior),
        "arm_attention_prior_method": args.arm_attention_prior_method,
        "arm_attention_prior_strength": float(args.arm_attention_prior_strength),
        "forecast_fusion": args.forecast_fusion,
        "high_mi_branch_distance": args.high_mi_branch_distance,
        "forecast_fusion_mi_branch_gate": args.forecast_fusion_mi_branch_gate,
        "forecast_fusion_mi_branch_gate_threshold": float(
            args.forecast_fusion_mi_branch_gate_threshold
        ),
        "forecast_fusion_mi_branch_gate_power": float(
            args.forecast_fusion_mi_branch_gate_power
        ),
        "forecast_fusion_coupled_protocol": (
            {
                "recent_arm": {
                    "method": "recent",
                    "candidate_selection": "causal_recency_top10",
                    "distance_weight": "recency_distances",
                },
                "mi_recent_arm": {
                    "method": "mi_recent",
                    "candidate_selection": "high_mi_top10",
                    "distance_weight": "high_mi_distances",
                },
                "confidence": "forecast_disagreement",
                "coupling": "MI is enabled in both candidate selection and distance weighting, or neither",
            }
            if args.forecast_fusion == "coupled_recent_mi" else None
        ),
        "bound_high_mi_ablation_protocol": (
            {
                method: {
                    "candidate_selection": spec["selection"],
                    "distance_weight": spec["distance"],
                    "candidate_alignment": spec["alignment"],
                }
                for method, spec in {
                    **_BOUND_HIGH_MI_ABLATION_METHODS,
                    **_BOUND_MI_SELECT_ABLATION_METHODS,
                }.items()
            }
            if args.forecast_fusion == "bound_high_mi_ablation" else None
        ),
        "bound_high_mi_linear_mix_protocol": (
            {
                method: {
                    "candidate_selection": spec["selection"],
                    "distance_weight": spec["distance"],
                    "candidate_alignment": spec["alignment"],
                }
                for method, spec in _BOUND_HIGH_MI_LINEAR_MIX_METHODS.items()
            }
            if args.forecast_fusion == "bound_high_mi_linear_mix" else None
        ),
        "forecast_fusion_strength": _serialize_scalar_or_vector(args.forecast_fusion_strength),
        "forecast_fusion_dual_high_mi_strength": float(
            args.forecast_fusion_dual_high_mi_strength
        ),
        "forecast_fusion_correction_clip": float(
            args.forecast_fusion_correction_clip
        ),
        "forecast_fusion_temperature": float(args.forecast_fusion_temperature),
        "forecast_fusion_hybrid_beta": float(args.forecast_fusion_hybrid_beta),
        "forecast_fusion_reverse_weights": bool(args.forecast_fusion_reverse_weights),
        "forecast_fusion_confidence_scale": float(args.forecast_fusion_confidence_scale),
        "forecast_fusion_confidence_floor": float(args.forecast_fusion_confidence_floor),
        "forecast_fusion_confidence_mode": args.forecast_fusion_confidence_mode,
        "forecast_fusion_mi_match": mi_match_config,
        "forecast_fusion_mi_match_formula": (
            "clip(1+slope*(min(high_mi_distances)-center)/scale, floor, ceiling)"
            if mi_match_config is not None else None
        ),
        "forecast_fusion_history_entropy_power": float(
            args.forecast_fusion_history_entropy_power
        ),
        "forecast_fusion_mi_reliability_threshold": float(
            args.forecast_fusion_mi_reliability_threshold
        ),
        "forecast_fusion_mi_reliability_power": float(
            args.forecast_fusion_mi_reliability_power
        ),
        "forecast_fusion_alignment": args.forecast_fusion_alignment,
        "mi_attribution_ablation": mi_attribution_rows,
        "bound_high_mi_alignment": args.bound_high_mi_alignment,
        "bound_high_mi_alignment_window": int(args.bound_high_mi_alignment_window),
        "tsrag_retrieved_alignment": args.tsrag_retrieved_alignment,
        "tsrag_retrieved_window": int(args.tsrag_retrieved_window),
        "tsrag_retrieved_tau": float(args.tsrag_retrieved_tau),
        "output_mi_gamma": float(args.output_mi_gamma),
        "output_mi_prior_strength": float(args.output_mi_prior_strength),
        "output_mi_prior_centering": args.output_mi_prior_centering,
        "output_mi_distance_key": args.output_mi_distance_key,
        "mi_gra": bool(args.mi_gra),
        "mi_gra_strength": float(args.mi_gra_strength),
        "mi_gra_mi_strength": float(args.mi_gra_mi_strength),
        "mi_gra_temperature": float(args.mi_gra_temperature),
        "mi_gra_hidden_dim": int(args.mi_gra_hidden_dim),
        "mi_gra_distance_key": args.mi_gra_distance_key,
        "mi_gra_controls": {
            "mi_gra": "real",
            "mi_gra_no_mi": "uniform_and_mi_strength_zero",
            "mi_gra_shuffled": "within_query_permutation",
            "mi_gra_reversed": "within_query_rank_reversal",
            "mi_gra_uniform": "uniform",
            "mi_gra_residual": "real_residual_prototype",
            "mi_gra_residual_no_mi": "uniform_residual_prototype_and_mi_strength_zero",
            "mi_gra_residual_shuffled": "within_query_permutation_residual_prototype",
            "mi_gra_residual_reversed": "within_query_rank_reversal_residual_prototype",
            "mi_gra_residual_uniform": "uniform_residual_prototype",
        } if any(method in mi_gra_methods for method in methods) else None,
        "mi_gra_formula": (
            "e_final=e_off+scale*rho*sum_i((alpha_mi-alpha_off)*LN(E_i-q)); "
            "logit_mi=logit_off+(W_g[query,candidate,candidate-query,abs(candidate-query)]+b_g)"
            "+mi_strength*rank_norm(-d_mi)"
            if args.mi_gra else None
        ),
        "output_mi_prior_formula": (
            "log(K*((1-gamma)*softmax(-d_off/(median(d_off)/5)) "
            "+ gamma*softmax(-d_mi_cal/(median(d_off)/5))))"
            if "output_mi" in methods else None
        ),
        "output_mi_prior_strength_applied": (
            "strength*log(K*w)"
            if "output_mi" in methods else None
        ),
        "output_mi_mi_median_calibration": (
            "d_mi_cal=d_mi*median(d_off)/median(d_mi)"
            if "output_mi" in methods else None
        ),
        "forecast_fusion_uses_retrieved_future": args.forecast_fusion != "none",
        "forecast_fusion_candidate_future_used_for_correction": (
            args.forecast_fusion != "none"
        ),
        "forecast_fusion_estimator": (
            "blockwise_mean"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "median"
            if args.forecast_fusion == "residual_prototype_median"
            else "mean"
            if args.forecast_fusion in {
                "residual_prototype",
                "residual_prototype_confident",
                "residual_prototype_mi_reliable",
                "residual_prototype_high_mi",
                "residual_prototype_hybrid",
                "residual_prototype_hybrid_confident",
                "residual_prototype_hybrid_pool",
                "residual_prototype_mi_gra_plus_high_mi",
                "bound_high_mi_residual_ablation",
            }
            else None
        ),
        "forecast_fusion_distance": (
            "recency_distances for recent arm; high_mi_distances for MI+recent arm"
            if args.forecast_fusion == "coupled_recent_mi"
            else
            "horizon_block_mi_normalized"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "hybrid_normalized_official_mi"
            if args.forecast_fusion in {
                "residual_prototype_hybrid",
                "residual_prototype_hybrid_median",
                "residual_prototype_hybrid_confident",
                "residual_prototype_hybrid_pool",
            }
            else "weighted_squared_euclidean"
            if args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS
            else "weighted_residual_prototype"
            if args.forecast_fusion == "bound_high_mi_residual_ablation" else None
        ),
        "forecast_fusion_formula": (
            "recent: recency Top-10 + recent_mean with softmax(-z(recency_distances)/T); "
            "MI+recent: High-MI Top-10 + recent_mean with softmax(-z(high_mi_distances)/T); "
            "both use the same forecast-disagreement confidence"
            if args.forecast_fusion == "coupled_recent_mi"
            else
            "per-block softmax(-z(horizon_MI_distance))*candidate_residual; "
            "history-only concentration gate"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "softmax(-((1-beta)*z(d_official)+beta*z(d_mi)))"
            if args.forecast_fusion in {
                "residual_prototype_hybrid",
                "residual_prototype_hybrid_median",
                "residual_prototype_hybrid_confident",
                "residual_prototype_hybrid_pool",
            }
            else "(1-beta)*[MI-GRA+lambda*g_MI*e_MI]"
            "+beta*[official+alpha_highMI*g_disagreement*retrieved_future_consensus]"
            if args.forecast_fusion == "residual_prototype_mi_gra_plus_high_mi"
            else
            "y=F_ARM(x,C_method)+lambda*sum_i(softmax(-z(d_method))_i*"
            "(aligned_future_i-reference_i)); reference_i is either "
            "base_model(history_i) or F_ARM(x,C_method); recent_mean96 uses "
            "the last 96 history values; MI selection and MI distance weights "
            "are enabled together"
            if args.forecast_fusion == "bound_high_mi_residual_ablation"
            else None
        ),
        "bound_high_mi_ablation_formula": (
            "TS-RAG(method-specific candidate set) + alpha * confidence * "
            "weighted_candidate_future; official_distances are bound to official_top10 "
            "and high_mi_distances are bound to high_mi_top10; alignment is method-specific"
            if args.forecast_fusion == "bound_high_mi_ablation" else None
        ),
        "bound_high_mi_linear_mix_formula": (
            "y=(1-a)*y_tsrag+a*y_MI; y_MI is the fixed-temperature-1 "
            "distance-weighted candidate-future consensus; no confidence or gate; "
            "official_distances are bound to official_top10 and high_mi_distances "
            "are bound to high_mi_top10"
            if args.forecast_fusion == "bound_high_mi_linear_mix" else None
        ),
        "bound_high_mi_residual_protocol": (
            {
                method: {
                    "candidate_selection": spec["selection"],
                    "distance_weight": spec["distance"],
                    "candidate_alignment": spec["alignment"],
                    "recent_window": int(args.bound_high_mi_alignment_window),
                    "fusion": "residual_prototype",
                    "estimator": args.bound_high_mi_residual_estimator,
                    "weighting": args.bound_high_mi_residual_weighting,
                    "residual_reference": args.bound_high_mi_residual_reference,
                    "temperature": float(args.forecast_fusion_temperature),
                    "confidence": args.bound_high_mi_residual_confidence,
                }
                for method, spec in _BOUND_HIGH_MI_RESIDUAL_METHODS.items()
            }
            if args.forecast_fusion == "bound_high_mi_residual_ablation" else None
        ),
        "forecast_fusion_confidence_formula": (
            "mean_block(mi_concentration_gate)"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "rank_agreement(high_mi_distances,official_distances)"
            if args.forecast_fusion_confidence_mode == "mi_rank_agreement"
            else "rank_agreement(high_mi_distances,official_distances)*"
            "normalized_entropy(high_mi_distances)"
            if args.forecast_fusion_confidence_mode == "mi_rank_entropy"
            else "max(0,C_q)*exp(-D_q)"
            if args.forecast_fusion in {
                "residual_prototype_confident",
                "residual_prototype_hybrid_confident",
            }
            else "R_MI(q)"
            if args.forecast_fusion == "residual_prototype_mi_reliable"
            else "1"
            if args.forecast_fusion == "bound_high_mi_residual_ablation"
            else "dual blend of MI-GRA residual gate and high-MI residual gate"
            if args.forecast_fusion == "residual_prototype_mi_gra_plus_high_mi"
            else "1"
            if args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS
            else "1"
            if args.forecast_fusion in _UNIT_CONFIDENCE_FUSIONS
            else "max(0,C_q)*exp(-D_q)"
            if args.forecast_fusion in _RESIDUAL_CONFIDENCE_FUSIONS
            else None
        ),
        "forecast_fusion_confidence_formula_legacy": (
            "mean_block(mi_concentration_gate)"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "max(0,C_q)*exp(-D_q)"
            if args.forecast_fusion in {
                "residual_prototype_confident",
                "residual_prototype_hybrid_confident",
            }
            else "R_MI(q)"
            if args.forecast_fusion == "residual_prototype_mi_reliable"
            else "1"
            if args.forecast_fusion == "bound_high_mi_residual_ablation"
            else "1"
            if args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS
            else "1"
            if args.forecast_fusion in _UNIT_CONFIDENCE_FUSIONS
            else "max(0,C_q)*exp(-D_q)"
            if args.forecast_fusion in _RESIDUAL_CONFIDENCE_FUSIONS
            else None
        ),
        "forecast_fusion_gate": (
            "history_only_horizon_mi_concentration"
            if args.forecast_fusion == "horizon_residual_transport"
            else
            "residual_direction_distance"
            if args.forecast_fusion in {
                "residual_prototype_confident",
                "residual_prototype_hybrid_confident",
            }
            else "mi_reliability_abstention"
            if args.forecast_fusion == "residual_prototype_mi_reliable"
            else "none"
            if args.forecast_fusion == "bound_high_mi_residual_ablation"
            else "dual_history_only_residual_gates"
            if args.forecast_fusion == "residual_prototype_mi_gra_plus_high_mi"
            else "none" if args.forecast_fusion in _RESIDUAL_PROTOTYPE_FUSIONS else None
        ),
        "forecast_fusion_selection_history_only": True,
        "forecast_fusion_candidate_pool": (
            "official_top20"
            if _forecast_fusion_uses_full_pool(args.forecast_fusion)
            else "selected_top10"
            if args.forecast_fusion != "none"
            else None
        ),
        "correction_alignment_saved": bool(args.save_correction_analysis),
        "system_ablation_metrics": system_ablation_rows,
        "weighting_ablation_alignments": weighting_ablation_alignments,
        "weighting_ablation_protocol": (
            "high-MI Top-10 for uniform/MI arms and ordinary-distance Top-10 "
            "for the official-distance arm; fixed branch prediction, "
            "confidence, and alpha; only candidate-future alignment and "
            "uniform/official/MI weighting plus its matched selector change"
            if weighting_ablation_alignments else None
        ),
        "weighting_ablation_metrics": weighting_ablation_rows,
        "weighting_ablation_factorial": factorial_ablation_rows,
        "correction_gain": (
            {method: (float(value) if value.ndim == 0 else value.tolist())
             for method, value in gain_map.items()}
            if gain_map else
            (float(gain_array) if gain_array.ndim == 0 else gain_array.tolist())
        ),
        "correction_gain_file": (str(Path(args.correction_gain_file).resolve())
                                  if args.correction_gain_file else None),
        "mi_target": _sidecar_mi_target(sidecar_meta) or "I(H;Y)",
        "mi_condition": str(sidecar_meta.get("mi_condition", "none")),
        "mi_condition_transform": sidecar_meta.get("mi_condition_transform"),
        "origin_hash": _array_hash(hash_source.get("query_origins")),
        "query_origin_hash": _array_hash(hash_source.get("query_origins")),
        "candidate_pool_hash": _array_hash(
            hash_source.get("candidate_starts")
            if hash_source.get("candidate_starts") is not None
            else hash_source.get("candidate_bank_starts")
        ),
        "official_distance_hash": _array_hash(
            hash_source.get("official_distances")
        ),
        "knowledge_origin": "train_history_discovery",
        "query_origin": "chronological_test_origin",
        "discovery_hyperparameters": sidecar_meta.get("hyperparameters", {}),
        "attribution_controls": bool(sidecar_meta.get("student_capacity_matched", False)),
        "student_capacity_matched": bool(sidecar_meta.get("student_capacity_matched", False)),
        "global_artifact": str(Path(args.global_artifact).resolve()) if args.global_artifact else None,
        "global_candidate_scope": (
            sidecar_meta.get("candidate_scope")
            if global_sidecar is None
            else "same_channel_train_history"
        ),
        "retrieval_checkpoint": str(Path(args.retrieval_checkpoint).resolve()),
        "base_weights_source": base_source,
        "seconds": round(time.time() - started, 3),
        "peak_memory_mb": (round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3)
                           if torch.cuda.is_available() else 0.0),
        "seed": int(args.seed),
        "leakage_checks": {
            "query_future_loaded": False,
            "candidate_future_used_for_selection": False,
            "candidate_future_used_for_correction": (
                args.forecast_fusion != "none"
            ),
            "leakage_ok": True,
        },
        "metrics": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    capture_run_context(output, " ".join(sys.argv))
    print(json.dumps({"output": str(output), "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
