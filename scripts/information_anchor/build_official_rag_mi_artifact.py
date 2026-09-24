#!/usr/bin/env python3
"""Build a causal MI selector sidecar for an official top-20 pool.

The expensive hidden-state extraction is intentionally supplied as one NPZ so
the same selector can be reused by Align-RAG and TS-RAG without recomputing
scores.  The NPZ must contain only history-side arrays:

``official_distances [N,K]``
``query_tokens [N,P,D]``
``candidate_tokens [N,K,P,D]``
``query_mi_scores [N,P]``
``candidate_mi_scores [N,K,P]``

``student_scores [N,K]`` is optional and enables the MI-prior variant.  No
future array is accepted by this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.downstream.mi_candidate_selector import (
    select_candidates,
    select_horizon_balanced_candidates,
    weighted_hidden_distances,
)
from experiments.information_anchor.downstream.mi_retrieval import (
    HorizonMIProfile,
    horizon_control_weights,
    horizon_weighted_hidden_distances,
)
from experiments.information_anchor.downstream.mi_residual_prototype import (
    mi_weighted_squared_hidden_distances,
)
from experiments.information_anchor.downstream.selective_residual_mi import (
    route_candidate_indices,
)


BASE_METHODS = ("official", "uniform", "random", "low_mi", "high_mi", "mi_select")
MI_ANCHOR_METHODS = ("high_mi_anchor", "low_mi_anchor", "random_patch", "all_patch")
P0_METHODS = (
    "mi_gate",
    "mi_anchor_gate",
    "low_mi_anchor_gate",
    "random_patch_gate",
    "all_patch_gate",
    "official_mi_bias",
    "official_random_bias",
    "official_uniform_bias",
    "official_low_mi_bias",
    "mi_insert",
    "low_mi_insert",
    "random_insert",
    "all_patch_insert",
    "residual_insert",
    "residual_mi",
    "mi_residual_prototype",
)
ATTRIBUTION_METHODS = (
    "mi_student", "student_no_mi", "no_mi_prior", "null_mi_prior",
    "uniform_no_mi_prior", "mi_prior_mmr", "no_mi_prior_mmr",
    "uniform_no_mi_prior_mmr", "null_mi_prior_mmr",
)


def causal_recency_distances(
    query_origins: np.ndarray,
    candidate_starts: np.ndarray,
    *,
    candidate_span: int = 64,
) -> np.ndarray:
    """Return candidate-future ages relative to the query forecast origin.

    Both stored origins are context starts.  Their context lengths cancel, so
    strict causality is ``candidate_start + pred_len <= query_start``.
    """
    query = np.asarray(query_origins, dtype=np.int64)
    starts = np.asarray(candidate_starts, dtype=np.int64)
    if query.ndim != 1 or starts.ndim != 2 or len(query) != len(starts):
        raise ValueError("query_origins and candidate_starts are not row-aligned")
    gaps = query[:, None] - (starts + int(candidate_span))
    if np.any(gaps < 0):
        row, column = np.argwhere(gaps < 0)[0]
        raise ValueError(
            "recency control found a candidate whose future overlaps the query: "
            f"row={int(row)}, candidate={int(column)}, gap={int(gaps[row, column])}"
        )
    return gaps.astype(np.float32)


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def build_artifact(
    arrays: dict[str, np.ndarray],
    *,
    dataset: str,
    split: str,
    top_k: int = 10,
    mi_insert_count: int = 1,
    gamma: float = 0.7,
    mi_prior_weight: float = 1.0,
    mmr_lambda: float = 0.3,
    seed: int = 2021,
    mi_target: str = "I(H;E)",
    mi_condition: str = "none",
    mi_target_source: str = "residual",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create selections and score matrices from history-only inputs."""
    required = {
        "official_distances",
        "query_tokens",
        "candidate_tokens",
        "query_mi_scores",
        "candidate_mi_scores",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"missing required history arrays: {missing}")
    forbidden = sorted(name for name in arrays if "future" in name.lower() or name in {"truth", "targets"})
    if forbidden:
        raise ValueError(f"future/target arrays are forbidden in selector input: {forbidden}")

    official = np.asarray(arrays["official_distances"], dtype=np.float32)
    if official.ndim != 2:
        raise ValueError(f"official_distances must be [N,K], got {official.shape}")
    if top_k < 1 or top_k > official.shape[1]:
        raise ValueError(f"top_k={top_k} exceeds candidate count={official.shape[1]}")
    if int(mi_insert_count) < 1:
        raise ValueError("mi_insert_count must be positive")
    if top_k > 1 and int(mi_insert_count) >= int(top_k):
        raise ValueError("mi_insert_count must be in [1, top_k-1]")
    for optional, expected_rank in (("query_origins", 1), ("channel_ids", 1), ("candidate_starts", 2)):
        if optional in arrays:
            value = np.asarray(arrays[optional])
            if value.ndim != expected_rank or value.shape[0] != len(official):
                raise ValueError(f"{optional} must align with {len(official)} queries, got {value.shape}")
            if optional == "candidate_starts" and value.shape[1] != official.shape[1]:
                raise ValueError("candidate_starts must have one entry per official candidate")
    high = weighted_hidden_distances(
        arrays["query_tokens"],
        arrays["candidate_tokens"],
        arrays["query_mi_scores"],
        arrays["candidate_mi_scores"],
        mode="high",
        seed=seed,
    )
    low = weighted_hidden_distances(
        arrays["query_tokens"],
        arrays["candidate_tokens"],
        arrays["query_mi_scores"],
        arrays["candidate_mi_scores"],
        mode="low",
        seed=seed,
    )
    uniform = weighted_hidden_distances(
        arrays["query_tokens"],
        arrays["candidate_tokens"],
        arrays["query_mi_scores"],
        arrays["candidate_mi_scores"],
        mode="uniform",
        seed=seed,
    )
    random = weighted_hidden_distances(
        arrays["query_tokens"],
        arrays["candidate_tokens"],
        arrays["query_mi_scores"],
        arrays["candidate_mi_scores"],
        mode="random",
        seed=seed,
    )
    mi_l2 = mi_weighted_squared_hidden_distances(
        arrays["query_tokens"],
        arrays["candidate_tokens"],
        arrays["query_mi_scores"],
    )
    student = arrays.get("student_scores")
    if student is not None:
        student = np.asarray(student, dtype=np.float32)
        if student.shape != official.shape:
            raise ValueError(f"student_scores must have shape {official.shape}, got {student.shape}")
    student_no_mi = arrays.get("student_no_mi_scores")
    student_null_mi = arrays.get("student_null_mi_scores")
    null_high = None
    recency = None
    recency_control_skipped = False
    if student_no_mi is not None:
        student_no_mi = np.asarray(student_no_mi, dtype=np.float32)
        if student_no_mi.shape != official.shape:
            raise ValueError("student_no_mi_scores must match official_distances")
    if student_null_mi is not None:
        student_null_mi = np.asarray(student_null_mi, dtype=np.float32)
        if student_null_mi.shape != official.shape:
            raise ValueError("student_null_mi_scores must match official_distances")
    if "query_null_mi_scores" in arrays or "candidate_null_mi_scores" in arrays:
        if "query_null_mi_scores" not in arrays or "candidate_null_mi_scores" not in arrays:
            raise ValueError("both query_null_mi_scores and candidate_null_mi_scores are required")
        null_high = weighted_hidden_distances(
            arrays["query_tokens"], arrays["candidate_tokens"],
            arrays["query_null_mi_scores"], arrays["candidate_null_mi_scores"],
            mode="high", seed=seed,
        )
    if "query_origins" in arrays and "candidate_starts" in arrays:
        try:
            recency = causal_recency_distances(
                arrays["query_origins"], arrays["candidate_starts"]
            )
        except ValueError as error:
            # Validation retrieval CSVs may contain candidates whose future
            # overlaps the validation query.  That candidate is still valid
            # for the history-only MI selectors (which never read its future),
            # but it cannot participate in the optional recency control.  Do
            # not discard the whole sidecar merely because that control is
            # undefined for this split.
            if "future overlaps the query" not in str(error):
                raise
            recency_control_skipped = True

    horizon_keys = {
        "horizon_mi_observed", "horizon_mi_weights",
        "global_mi_observed", "global_mi_weights",
    }
    horizon_present = horizon_keys.intersection(arrays)
    horizon_profile: HorizonMIProfile | None = None
    if horizon_present:
        missing_profile = sorted(horizon_keys - set(arrays))
        if missing_profile:
            raise ValueError(f"incomplete horizon MI profile: missing {missing_profile}")
        if mi_target != "I(H;Y)" or mi_condition != "none" or mi_target_source != "future_truth":
            raise ValueError(
                "horizon MI sidecars require mi_target=I(H;Y), mi_condition=none, "
                "and mi_target_source=future_truth"
            )
        horizon_profile = HorizonMIProfile(
            observed=np.asarray(arrays["horizon_mi_observed"], dtype=np.float32),
            weights=np.asarray(arrays["horizon_mi_weights"], dtype=np.float32),
            global_observed=np.asarray(arrays["global_mi_observed"], dtype=np.float32),
            global_weights=np.asarray(arrays["global_mi_weights"], dtype=np.float32),
        )
        if (
            horizon_profile.observed.ndim != 2
            or horizon_profile.weights.shape != horizon_profile.observed.shape
            or horizon_profile.global_observed.ndim != 1
            or horizon_profile.global_weights.shape != horizon_profile.global_observed.shape
            or horizon_profile.observed.shape[1] != np.asarray(arrays["query_tokens"]).shape[1]
            or not all(
                np.isfinite(value).all()
                for value in (
                    horizon_profile.observed,
                    horizon_profile.weights,
                    horizon_profile.global_observed,
                    horizon_profile.global_weights,
                )
            )
        ):
            raise ValueError("horizon MI profile arrays have incompatible or non-finite shapes")
        if not np.allclose(horizon_profile.weights.sum(axis=1), 1.0):
            raise ValueError("horizon MI weights must sum to one per future block")
        if not np.isclose(horizon_profile.global_weights.sum(), 1.0):
            raise ValueError("global MI weights must sum to one")

    distances = {
        "official": official,
        "uniform": uniform,
        "random": random,
        "low_mi": low,
        "high_mi": high,
        "mi_residual_prototype": mi_l2,
    }
    outputs: dict[str, np.ndarray] = {
        "official_distances": official,
        "uniform_distances": uniform,
        "random_distances": random,
        "low_mi_distances": low,
        "high_mi_distances": high,
        "mi_l2_distances": mi_l2,
    }
    if student is not None:
        distances["mi_prior"] = high
        outputs["student_scores"] = student
    # Keep the historical methods intact and add explicit names for the
    # minimal MI-Anchor protocol.  The aliases make the paper comparison
    # self-documenting while preserving old ``high_mi`` rank-fusion results.
    # Insert variants need at least one official position to retain and one
    # outside candidate to insert.  Keep the non-insert P0 methods available
    # for the top_k=1 compatibility path used by unit tests and imputation.
    p0_methods = tuple(
        method for method in P0_METHODS
        if top_k > 1 or not method.endswith("_insert")
    )
    methods = list(BASE_METHODS) + list(MI_ANCHOR_METHODS) + list(p0_methods)
    if student is not None:
        methods.extend(["mi_prior", "mi_student", "mi_prior_mmr"])
    if student_no_mi is not None:
        methods.append("student_no_mi")
        outputs["student_no_mi_scores"] = student_no_mi
        if recency is not None:
            methods.extend([
                "recency_no_mi_prior", "mi_recency_prior",
                "mi_orthogonal_recency_prior",
            ])
            outputs["recency_distances"] = recency
    if student_no_mi is not None and high is not None:
        methods.extend([
            "no_mi_prior", "no_mi_prior_mmr",
            "uniform_no_mi_prior", "uniform_no_mi_prior_mmr",
        ])
    if student_null_mi is not None and null_high is not None:
        methods.extend(["null_mi_prior", "null_mi_prior_mmr"])
        outputs["student_null_mi_scores"] = student_null_mi
        outputs["null_high_mi_distances"] = null_high
        outputs["query_null_mi_scores"] = np.asarray(arrays["query_null_mi_scores"], dtype=np.float32)
        outputs["candidate_null_mi_scores"] = np.asarray(arrays["candidate_null_mi_scores"], dtype=np.float32)
    for method in methods:
        selection = select_candidates(
            official,
            top_k=top_k,
            method=method,
            high_mi_distances=high,
            mi_residual_prototype_distances=mi_l2,
            low_mi_distances=low,
            uniform_distances=uniform,
            random_mi_distances=random,
            student_scores=student,
            student_no_mi_scores=student_no_mi,
            student_null_mi_scores=student_null_mi,
            null_mi_distances=null_high,
            recency_distances=recency,
            candidate_tokens=np.asarray(arrays["candidate_tokens"], dtype=np.float32),
            gamma=gamma,
            mi_prior_weight=mi_prior_weight,
            mi_insert_count=mi_insert_count,
            mmr_lambda=mmr_lambda,
            seed=seed,
        )
        outputs[f"selected_ranks_{method}"] = selection.indices
        outputs[f"selection_scores_{method}"] = selection.scores

    horizon_methods: tuple[str, ...] = ()
    if horizon_profile is not None:
        horizon_methods = (
            "horizon_mi", "horizon_global_mi", "horizon_uniform", "horizon_random"
        )
        for method, mode in (
            ("horizon_mi", "horizon"),
            ("horizon_global_mi", "global"),
            ("horizon_uniform", "uniform"),
            ("horizon_random", "random"),
        ):
            block_distances = horizon_weighted_hidden_distances(
                arrays["query_tokens"],
                arrays["candidate_tokens"],
                horizon_control_weights(horizon_profile, mode=mode, seed=seed),
            )
            selection = select_horizon_balanced_candidates(
                official, block_distances, top_k=top_k, method=method
            )
            outputs[f"horizon_distances_{method}"] = block_distances
            outputs[f"selected_ranks_{method}"] = selection.indices
            outputs[f"selection_scores_{method}"] = selection.scores
            outputs[f"assigned_ranks_{method}"] = selection.assigned_indices
            outputs[f"assignment_costs_{method}"] = selection.assignment_costs
        methods.extend(horizon_methods)

    selective_use = arrays.get("selective_gate_use_mi")
    if selective_use is not None:
        selective_use = np.asarray(selective_use, dtype=bool)
        if selective_use.shape != (len(official),):
            raise ValueError("selective_gate_use_mi must have one value per query")
        if "selected_ranks_mi_prior" not in outputs:
            raise ValueError("selective MI routing requires student-backed mi_prior")
        selective_indices = route_candidate_indices(
            outputs["selected_ranks_mi_prior"],
            outputs["selected_ranks_official"],
            selective_use,
        )
        outputs["selected_ranks_selective_mi_prior"] = selective_indices
        # The selected ranks index the full official pool, whereas the stored
        # method scores are only the already-selected top-k positions.  The
        # selective runner does not use these scores unless score-aware
        # blending is explicitly requested, so retain a shape-correct neutral
        # placeholder rather than indexing the reduced arrays incorrectly.
        outputs["selection_scores_selective_mi_prior"] = np.zeros_like(
            selective_indices, dtype=np.float32
        )
        methods.append("selective_mi_prior")
        for key in (
            "selective_gate_use_mi",
            "selective_gate_predicted_gain",
            "selective_gate_lower_bound",
        ):
            if key in arrays:
                outputs[key] = np.asarray(arrays[key])

    for optional in ("query_origins", "channel_ids", "candidate_starts"):
        if optional in arrays:
            outputs[optional] = np.asarray(arrays[optional])
    metadata: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "seq_len": 512,
        "pred_len": 64,
        "candidate_count": int(official.shape[1]),
        "neighbors": int(top_k),
        "mi_insert_count": int(mi_insert_count),
        "gamma": float(gamma),
        "mi_prior_weight": float(mi_prior_weight),
        "mmr_lambda": float(mmr_lambda),
        "seed": int(seed),
        "mi_target": mi_target,
        "mi_condition": mi_condition,
        "mi_target_source": str(mi_target_source),
        "selector_input": "history_only",
        "query_future_loaded": False,
        "candidate_future_used_for_selection": False,
        "leakage_ok": True,
        "official_distance_sha256": _hash_array(official),
        "n_queries": int(len(official)),
        "methods": methods,
        "mi_anchor_protocol": student is None,
        "mi_anchor_methods": list(MI_ANCHOR_METHODS),
        "mi_anchor_selection": "pure_high_mi_distance_within_official_pool",
        "mi_select_order": "official_rank",
        "mi_prior_available": student is not None,
        "selective_gate_enabled": selective_use is not None,
        "selective_gate_fallback": "official" if selective_use is not None else None,
        "selective_gate_use_treatment_rate": (
            float(np.mean(selective_use)) if selective_use is not None else None
        ),
        "recency_control_skipped": bool(recency_control_skipped),
        "horizon_mi_protocol": horizon_profile is not None,
        "horizon_mi_estimator": (
            "gaussian_copula_pairwise_I(H_p;Y_b)"
            if horizon_profile is not None else None
        ),
        "horizon_block_size": 16 if horizon_profile is not None else None,
        "horizon_block_count": (
            int(horizon_profile.weights.shape[0]) if horizon_profile is not None else None
        ),
        "horizon_preserved_count": 6 if horizon_profile is not None else None,
        "horizon_assignment": (
            "minimum_rank_cost_lexicographic" if horizon_profile is not None else None
        ),
        "horizon_mi_observed": (
            horizon_profile.observed.tolist() if horizon_profile is not None else None
        ),
        "horizon_mi_weights": (
            horizon_profile.weights.tolist() if horizon_profile is not None else None
        ),
        "global_mi_observed": (
            horizon_profile.global_observed.tolist() if horizon_profile is not None else None
        ),
        "global_mi_weights": (
            horizon_profile.global_weights.tolist() if horizon_profile is not None else None
        ),
    }
    for optional in ("query_origins", "channel_ids", "candidate_starts"):
        if optional in arrays:
            metadata[f"{optional}_sha256"] = _hash_array(np.asarray(arrays[optional]))
    for method in horizon_methods:
        metadata[f"selected_ranks_{method}_sha256"] = _hash_array(
            outputs[f"selected_ranks_{method}"]
        )
        metadata[f"assigned_ranks_{method}_sha256"] = _hash_array(
            outputs[f"assigned_ranks_{method}"]
        )
    return outputs, metadata


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, help="History-only hidden/MI arrays")
    parser.add_argument("--output", required=True, help="Output candidate_scores.npz")
    parser.add_argument("--summary", default=None, help="Optional JSON metadata path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("discovery", "validation", "test"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mi-insert-count", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--mi-prior-weight", type=float, default=1.0)
    parser.add_argument("--mmr-lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--mi-target", default="I(H;E)")
    parser.add_argument("--mi-condition", default="none")
    parser.add_argument("--mi-target-source", choices=("residual", "future_truth"), default="residual")
    return parser.parse_args()


def main() -> None:
    args = _args()
    input_path = Path(args.input_npz).resolve()
    with np.load(input_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    outputs, metadata = build_artifact(
        arrays,
        dataset=args.dataset,
        split=args.split,
        top_k=args.top_k,
        mi_insert_count=args.mi_insert_count,
        gamma=args.gamma,
        mi_prior_weight=args.mi_prior_weight,
        mmr_lambda=args.mmr_lambda,
        seed=args.seed,
        mi_target=args.mi_target,
        mi_condition=args.mi_condition,
        mi_target_source=args.mi_target_source,
    )
    input_summary_path = input_path.with_suffix(".json")
    if input_summary_path.exists():
        input_summary = json.loads(input_summary_path.read_text(encoding="utf-8"))
        if isinstance(input_summary, dict):
            for key in (
                "channels", "train_end", "discovery_samples", "student_score_source",
                "student_enabled", "mi_anchor_protocol",
                "student_discovery_future_used_for_fit", "student_test_future_loaded",
                "student_candidate_count", "recent_length",
                "student_feature_count", "student_capacity_matched",
                "student_target",
                "student_oof", "student_control_oof", "mi_critic_oof",
                "hyperparameters", "query_start", "query_end", "query_windows",
            ):
                if key in input_summary:
                    metadata[key] = input_summary[key]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **outputs)
    summary = Path(args.summary).resolve() if args.summary else output.with_suffix(".json")
    summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(summary), **metadata}, indent=2))


if __name__ == "__main__":
    main()
