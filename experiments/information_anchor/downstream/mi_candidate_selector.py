"""Causal MI candidate selection shared by Align-RAG and TS-RAG.

The selector only consumes query/candidate history representations and scores
fitted on discovery data.  Candidate futures are deliberately absent from the
API; a backend may gather them *after* selection for its own RAG mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np


@dataclass(frozen=True)
class CandidateSelection:
    """Selected candidate positions, ordered from best to worst."""

    indices: np.ndarray
    scores: np.ndarray
    method: str


def _finite(name: str, values: np.ndarray, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _row_rank(values: np.ndarray) -> np.ndarray:
    """Stable ascending ranks scaled to ``[0, 1]`` per query."""
    values = _finite("values", values, 2)
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(values, dtype=np.float32)
    ranks[np.arange(len(values))[:, None], order] = np.arange(values.shape[1], dtype=np.float32)
    if values.shape[1] > 1:
        ranks /= np.float32(values.shape[1] - 1)
    return ranks


def _row_zscore(values: np.ndarray) -> np.ndarray:
    values = _finite("values", values, 2)
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(values.std(axis=1, keepdims=True), 1e-6)


def _orthogonalized_distance(high: np.ndarray, recency: np.ndarray) -> np.ndarray:
    """Remove the row-wise recency component from a High-MI distance."""
    high_z = _row_zscore(high)
    recency_z = _row_zscore(recency)
    denominator = np.sum(recency_z * recency_z, axis=1, keepdims=True)
    beta = np.sum(high_z * recency_z, axis=1, keepdims=True) / np.maximum(denominator, 1e-6)
    return (high_z - beta * recency_z).astype(np.float32)


def token_weights(
    mi_scores: np.ndarray,
    mode: str = "high",
    *,
    temperature: float = 1.0,
    seed: int = 2021,
) -> np.ndarray:
    """Convert history-predicted patch MI profiles to positive weights.

    ``mi_scores`` is ``[batch, patches]``.  ``random`` permutes each row with
    a deterministic generator and is therefore a matched negative control.
    """
    scores = _finite("mi_scores", mi_scores, 2)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode not in {"high", "low", "random", "uniform"}:
        raise ValueError(f"unknown MI weighting mode: {mode}")
    if mode == "uniform":
        return np.full_like(scores, 1.0 / scores.shape[1], dtype=np.float32)
    logits = _row_zscore(scores)
    if mode == "low":
        logits = -logits
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        logits = np.stack([row[rng.permutation(len(row))] for row in logits], axis=0)
    logits = logits / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 60.0))
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def weighted_hidden_distances(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
    query_mi_scores: np.ndarray,
    candidate_mi_scores: np.ndarray,
    *,
    mode: str = "high",
    temperature: float = 1.0,
    seed: int = 2021,
) -> np.ndarray:
    """Return weighted cosine distances for a fixed candidate pool.

    Shapes are ``query_tokens=[B,P,D]``, ``candidate_tokens=[B,K,P,D]``,
    ``query_mi_scores=[B,P]`` and ``candidate_mi_scores=[B,K,P]``.  Only
    history-derived hidden tokens and MI profiles are used.
    """
    query = _finite("query_tokens", query_tokens, 3)
    candidates = _finite("candidate_tokens", candidate_tokens, 4)
    q_scores = _finite("query_mi_scores", query_mi_scores, 2)
    c_scores = _finite("candidate_mi_scores", candidate_mi_scores, 3)
    if query.shape[0] != candidates.shape[0] or query.shape[1:] != candidates.shape[2:]:
        raise ValueError("query/candidate token shapes are incompatible")
    if q_scores.shape != query.shape[:2] or c_scores.shape != candidates.shape[:3]:
        raise ValueError("MI profile shapes do not match token shapes")
    q_norm = query / np.maximum(np.linalg.norm(query, axis=2, keepdims=True), 1e-8)
    c_norm = candidates / np.maximum(np.linalg.norm(candidates, axis=3, keepdims=True), 1e-8)
    q_weights = token_weights(q_scores, mode, temperature=temperature, seed=seed)
    c_weights = token_weights(
        c_scores.reshape(-1, c_scores.shape[-1]),
        mode,
        temperature=temperature,
        seed=seed,
    ).reshape(c_scores.shape)
    pair_weights = np.sqrt(q_weights[:, None, :] * c_weights)
    cosine = np.sum(q_norm[:, None] * c_norm, axis=3)
    weighted_cosine = np.sum(pair_weights * cosine, axis=2)
    normalizer = np.maximum(pair_weights.sum(axis=2), 1e-12)
    return (1.0 - weighted_cosine / normalizer).astype(np.float32)


def global_weighted_hidden_distances(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
    query_mi_scores: np.ndarray,
    candidate_mi_scores: np.ndarray,
    *,
    mode: str = "high",
    temperature: float = 1.0,
    seed: int = 2021,
    query_block_size: int = 8,
    candidate_block_size: int = 1024,
) -> np.ndarray:
    """Compute MI distances against an unrestricted history candidate set.

    Unlike :func:`weighted_hidden_distances`, this function accepts a global
    candidate bank ``[N,P,D]`` instead of a per-query ``[B,K,P,D]`` pool.  It
    deliberately has no official-retriever distance input: callers can use it
    for the ``MI-global`` ablation that searches every train-only history
    window.  The two block sizes bound the temporary ``[Q,C,P,D]`` tensor.
    Candidate futures are not part of this API.
    """
    query = _finite("query_tokens", query_tokens, 3)
    candidates = _finite("candidate_tokens", candidate_tokens, 3)
    q_scores = _finite("query_mi_scores", query_mi_scores, 2)
    c_scores = _finite("candidate_mi_scores", candidate_mi_scores, 2)
    if query.shape[1:] != candidates.shape[1:]:
        raise ValueError("query/candidate token shapes are incompatible")
    if q_scores.shape != query.shape[:2] or c_scores.shape != candidates.shape[:2]:
        raise ValueError("MI profile shapes do not match token shapes")
    if query_block_size < 1 or candidate_block_size < 1:
        raise ValueError("global distance block sizes must be positive")

    output = np.empty((len(query), len(candidates)), dtype=np.float32)
    for q_start in range(0, len(query), int(query_block_size)):
        q_stop = min(len(query), q_start + int(query_block_size))
        q_tokens = query[q_start:q_stop]
        q_mi = q_scores[q_start:q_stop]
        q_count = q_stop - q_start
        for c_start in range(0, len(candidates), int(candidate_block_size)):
            c_stop = min(len(candidates), c_start + int(candidate_block_size))
            c_tokens = candidates[c_start:c_stop]
            c_mi = c_scores[c_start:c_stop]
            # Broadcasting avoids materialising the complete global bank in
            # one temporary tensor while preserving the exact pairwise metric.
            q_view = np.broadcast_to(
                q_tokens[:, None, :, :],
                (q_count, c_stop - c_start, q_tokens.shape[1], q_tokens.shape[2]),
            )
            c_view = np.broadcast_to(
                c_tokens[None, :, :, :],
                (q_count, c_stop - c_start, c_tokens.shape[1], c_tokens.shape[2]),
            )
            q_mi_view = np.broadcast_to(
                q_mi[:, None, :], (q_count, c_stop - c_start, q_mi.shape[1])
            )
            c_mi_view = np.broadcast_to(
                c_mi[None, :, :], (q_count, c_stop - c_start, c_mi.shape[1])
            )
            # weighted_hidden_distances expects one query token bank per row;
            # flatten the query/candidate block and restore its matrix shape.
            flat_q = np.broadcast_to(q_tokens[:, None, :, :], q_view.shape).reshape(
                -1, q_tokens.shape[1], q_tokens.shape[2]
            )
            flat_c = np.broadcast_to(c_tokens[None, :, :, :], c_view.shape).reshape(
                -1, c_tokens.shape[1], c_tokens.shape[2]
            )
            flat_q_mi = np.broadcast_to(q_mi[:, None, :], q_mi_view.shape).reshape(
                -1, q_mi.shape[1]
            )
            flat_c_mi = np.broadcast_to(c_mi[None, :, :], c_mi_view.shape).reshape(
                -1, c_mi.shape[1]
            )
            pair = weighted_hidden_distances(
                flat_q,
                flat_c[:, None, :, :],
                flat_q_mi,
                flat_c_mi[:, None, :],
                mode=mode,
                temperature=temperature,
                seed=seed,
            )[:, 0].reshape(q_count, c_stop - c_start)
            output[q_start:q_stop, c_start:c_stop] = pair
    return output


def select_global_candidates(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
    query_mi_scores: np.ndarray,
    candidate_mi_scores: np.ndarray,
    *,
    top_k: int,
    mode: str = "high",
    temperature: float = 1.0,
    seed: int = 2021,
    query_block_size: int = 8,
    candidate_block_size: int = 1024,
) -> CandidateSelection:
    """Select windows by searching the complete supplied history bank.

    The returned indices are positions in ``candidate_tokens``.  The bank is
    expected to contain only train/discovery history windows; this function
    does not inspect or accept candidate futures.
    """
    candidates = _finite("candidate_tokens", candidate_tokens, 3)
    if top_k < 1 or top_k > len(candidates):
        raise ValueError(f"top_k={top_k} must be in [1,{len(candidates)}]")
    if mode not in {"high", "low", "random", "uniform"}:
        raise ValueError(f"unknown global MI weighting mode: {mode}")
    distances = global_weighted_hidden_distances(
        query_tokens,
        candidates,
        query_mi_scores,
        candidate_mi_scores,
        mode=mode,
        temperature=temperature,
        seed=seed,
        query_block_size=query_block_size,
        candidate_block_size=candidate_block_size,
    )
    chosen = np.argsort(distances, axis=1, kind="stable")[:, :top_k]
    scores = np.take_along_axis(distances, chosen, axis=1).astype(np.float32)
    return CandidateSelection(
        indices=chosen.astype(np.int64), scores=scores, method=f"mi_global_{mode}"
    )


def validate_candidate_pool(
    official_distances: np.ndarray,
    *,
    top_k: int,
    selected_indices: np.ndarray | None = None,
) -> tuple[int, int]:
    """Validate a fixed ``[query,candidate]`` pool and optional selection."""
    distances = _finite("official_distances", official_distances, 2)
    if top_k < 1 or top_k > distances.shape[1]:
        raise ValueError(f"top_k={top_k} must be in [1,{distances.shape[1]}]")
    if selected_indices is not None:
        selected = np.asarray(selected_indices)
        if selected.ndim != 2 or selected.shape[0] != distances.shape[0] or selected.shape[1] != top_k:
            raise ValueError("selected_indices has incompatible shape")
        if not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("selected_indices must be integer positions")
        if selected.min(initial=0) < 0 or selected.max(initial=-1) >= distances.shape[1]:
            raise ValueError("selected_indices leaves the official candidate pool")
        if any(len(set(row.tolist())) != top_k for row in selected):
            raise ValueError("selected_indices contains duplicate candidates")
    return distances.shape


def _mmr_from_objective(
    objective: np.ndarray,
    candidate_tokens: np.ndarray,
    top_k: int,
    lam: float,
    method: str,
) -> CandidateSelection:
    """Greedy relevance/diversity selection over one immutable candidate pool.

    ``objective`` is lower-is-better (the same convention as the selector),
    while ``candidate_tokens`` supplies history-only diversity features.  The
    implementation deliberately mirrors the official Align-RAG MMR contract:
    the first item is the most relevant one, and later items trade relevance
    against maximum similarity to already selected candidates.  No future
    values are accepted.
    """
    scores = _finite("mmr_objective", objective, 2)
    tokens = _finite("candidate_tokens", candidate_tokens, 4)
    if scores.shape[:2] != tokens.shape[:2]:
        raise ValueError("MMR objective and candidate token shapes are incompatible")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("mmr_lambda must be in [0,1]")
    if not 1 <= int(top_k) <= scores.shape[1]:
        raise ValueError("top_k must lie inside the candidate pool")
    # Mean-pool hidden tokens and use cosine similarity as a history-only
    # diversity proxy.  The official Align implementation uses Pearson
    # similarity on raw histories; hidden cosine is the only representation
    # available at this history-only artifact boundary.
    pooled = tokens.mean(axis=2)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=2, keepdims=True), 1e-8)
    similarity = np.einsum("nkd,njd->nkj", pooled, pooled)
    relevance = -_row_zscore(scores)
    n_rows, n_candidates = scores.shape
    selected = np.empty((n_rows, int(top_k)), dtype=np.int64)
    picked = np.zeros((n_rows, n_candidates), dtype=bool)
    first = np.argmax(relevance, axis=1)
    selected[:, 0] = first
    picked[np.arange(n_rows), first] = True
    for rank in range(1, int(top_k)):
        diversity_penalty = np.max(
            np.where(picked[:, None, :], similarity, -np.inf), axis=2
        )
        mmr_score = float(lam) * relevance - (1.0 - float(lam)) * diversity_penalty
        mmr_score[picked] = -np.inf
        chosen = np.argmax(mmr_score, axis=1)
        selected[:, rank] = chosen
        picked[np.arange(n_rows), chosen] = True
    chosen_scores = np.take_along_axis(scores, selected, axis=1).astype(np.float32)
    return CandidateSelection(
        indices=selected,
        scores=chosen_scores,
        method=method,
    )


def select_candidates(
    official_distances: np.ndarray,
    *,
    top_k: int,
    method: str = "official",
    high_mi_distances: np.ndarray | None = None,
    mi_residual_prototype_distances: np.ndarray | None = None,
    low_mi_distances: np.ndarray | None = None,
    uniform_distances: np.ndarray | None = None,
    random_mi_distances: np.ndarray | None = None,
    student_scores: np.ndarray | None = None,
    student_no_mi_scores: np.ndarray | None = None,
    student_null_mi_scores: np.ndarray | None = None,
    pairwise_scores: np.ndarray | None = None,
    null_mi_distances: np.ndarray | None = None,
    recency_distances: np.ndarray | None = None,
    candidate_tokens: np.ndarray | None = None,
    gamma: float = 0.7,
    candidate_limit: int | None = None,
    mi_prior_weight: float = 1.0,
    mi_gate_overlap_max: float = 0.3,
    mi_insert_count: int = 1,
    mmr_lambda: float = 0.3,
    seed: int = 2021,
) -> CandidateSelection:
    """Select top-k positions from one immutable official candidate pool.

    All distance methods are lower-is-better.  Student scores are
    higher-is-better.  The function never
    receives candidate futures, making accidental query-future leakage hard to
    introduce at the backend boundary.
    """
    official = _finite("official_distances", official_distances, 2)
    validate_candidate_pool(official, top_k=top_k)
    method = str(method).lower()
    if candidate_limit is not None:
        if not int(top_k) <= int(candidate_limit) <= official.shape[1]:
            raise ValueError(
                "candidate_limit must lie between top_k and the official candidate count"
            )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if mi_prior_weight < 0:
        raise ValueError("mi_prior_weight must be non-negative")
    if not 0.0 <= float(mi_gate_overlap_max) <= 1.0:
        raise ValueError("mi_gate_overlap_max must be in [0,1]")
    if int(mi_insert_count) < 1:
        raise ValueError("mi_insert_count must be positive")
    bias_score_source: np.ndarray | None = None

    def require(name: str, values: np.ndarray | None) -> np.ndarray:
        if values is None:
            raise ValueError(f"{name} is required for method={method}")
        array = _finite(name, values, 2)
        if array.shape != official.shape:
            raise ValueError(f"{name} must have shape {official.shape}, got {array.shape}")
        return array

    if method in {"official", "official_permuted"}:
        objective = official
    elif method in {
        "official_mi_bias", "official_random_bias",
        "official_uniform_bias", "official_low_mi_bias",
    }:
        # Keep the official membership exactly fixed.  The selected scores
        # are replaced with the requested history-only control distance below
        # so the runner can use them as a continuous expert-gate bias without
        # changing candidates.  Random/uniform/low-MI variants are matched
        # controls for the High-MI intervention on the exact same set.
        if method == "official_mi_bias":
            bias_score_source = require("high_mi_distances", high_mi_distances)
        elif method == "official_random_bias":
            bias_score_source = require("random_mi_distances", random_mi_distances)
        elif method == "official_uniform_bias":
            bias_score_source = require("uniform_distances", uniform_distances)
        else:
            bias_score_source = require("low_mi_distances", low_mi_distances)
        objective = official
    elif method == "mi_residual_prototype":
        objective = require(
            "mi_residual_prototype_distances", mi_residual_prototype_distances
        )
    elif method in {"high_mi", "mi_select", "mi_gate"}:
        high = require("high_mi_distances", high_mi_distances)
        objective = (1.0 - gamma) * _row_rank(official) + gamma * _row_rank(high)
    elif method in {"high_mi_anchor", "mi_anchor", "mi_anchor_gate"}:
        # MI-Anchor Retrieval deliberately removes the legacy official/MI
        # rank fusion: within the immutable official pool, rank only by the
        # High-MI hidden distance.  ``high_mi`` remains unchanged for
        # backwards-compatible reproduction of the original protocol.
        objective = require("high_mi_distances", high_mi_distances)
    elif method == "low_mi":
        objective = require("low_mi_distances", low_mi_distances)
    elif method == "high_mi_boundary":
        objective = require("high_mi_distances", high_mi_distances)
    elif method == "low_mi_boundary":
        objective = require("low_mi_distances", low_mi_distances)
    elif method in {"low_mi_anchor", "low_mi_anchor_gate"}:
        objective = require("low_mi_distances", low_mi_distances)
    elif method == "uniform":
        objective = require("uniform_distances", uniform_distances)
    elif method in {"all_patch", "all_patch_gate"}:
        objective = require("uniform_distances", uniform_distances)
    elif method == "all_patch_boundary":
        objective = require("uniform_distances", uniform_distances)
    elif method == "random":
        # Random is a matched token-weight control, not a new candidate pool.
        objective = require("random_mi_distances", random_mi_distances)
    elif method in {"random_patch", "random_patch_gate"}:
        objective = require("random_mi_distances", random_mi_distances)
    elif method == "random_patch_boundary":
        objective = require("random_mi_distances", random_mi_distances)
    elif method in {
        "mi_insert", "low_mi_insert", "random_insert", "all_patch_insert",
        "residual_insert",
    }:
        if method == "mi_insert":
            objective = require("high_mi_distances", high_mi_distances)
        elif method == "low_mi_insert":
            objective = require("low_mi_distances", low_mi_distances)
        elif method == "random_insert":
            objective = require("random_mi_distances", random_mi_distances)
        elif method == "residual_insert":
            high = require("high_mi_distances", high_mi_distances)
            objective = _orthogonalized_distance(high, official)
        else:
            objective = require("uniform_distances", uniform_distances)
    elif method == "residual_mi":
        high = require("high_mi_distances", high_mi_distances)
        objective = _orthogonalized_distance(high, official)
    elif method == "mi_prior":
        high = require("high_mi_distances", high_mi_distances)
        student = require("student_scores", student_scores)
        objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(high)
    elif method == "mi_student":
        student = require("student_scores", student_scores)
        objective = -_row_zscore(student)
    elif method == "student_no_mi":
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = -_row_zscore(student)
    elif method == "no_mi_prior":
        high = require("high_mi_distances", high_mi_distances)
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(high)
    elif method == "uniform_no_mi_prior":
        uniform = require("uniform_distances", uniform_distances)
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(uniform)
    elif method == "recency_no_mi_prior":
        # Exact-capacity temporal control for MI-prior: the same five-feature
        # no-MI student and row-standardized additive prior are retained, but
        # the MI-weighted hidden distance is replaced by a causal age gap.
        recency = require("recency_distances", recency_distances)
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(recency)
    elif method == "mi_recency_prior":
        # Incremental-value test: retain the strongest causal no-MI system
        # (student + recency) and add only the High-MI hidden distance.
        high = require("high_mi_distances", high_mi_distances)
        recency = require("recency_distances", recency_distances)
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = (
            -_row_zscore(student)
            + _row_zscore(recency)
            + float(mi_prior_weight) * _row_zscore(high)
        )
    elif method == "mi_orthogonal_recency_prior":
        # Keep the capacity-matched no-MI student and recency prior, but use
        # only the High-MI component not explained by candidate age.
        high = require("high_mi_distances", high_mi_distances)
        recency = require("recency_distances", recency_distances)
        student = require("student_no_mi_scores", student_no_mi_scores)
        objective = (
            -_row_zscore(student)
            + _row_zscore(recency)
            + float(mi_prior_weight) * _orthogonalized_distance(high, recency)
        )
    elif method == "null_mi_prior":
        null_high = require("null_mi_distances", null_mi_distances)
        student = require("student_null_mi_scores", student_null_mi_scores)
        objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(null_high)
    elif method == "pairwise_gain":
        # PairwiseGainRanker exposes predicted lower-is-better loss scores at
        # this boundary.  Abstention/fallback is intentionally handled by
        # the ranker object because it needs its discovery residual scale.
        objective = require("pairwise_scores", pairwise_scores)
    elif method in {"mi_prior_mmr", "no_mi_prior_mmr", "uniform_no_mi_prior_mmr",
                    "null_mi_prior_mmr"}:
        if candidate_tokens is None:
            raise ValueError(f"candidate_tokens is required for method={method}")
        if method == "mi_prior_mmr":
            high = require("high_mi_distances", high_mi_distances)
            student = require("student_scores", student_scores)
            objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(high)
        elif method == "no_mi_prior_mmr":
            high = require("high_mi_distances", high_mi_distances)
            student = require("student_no_mi_scores", student_no_mi_scores)
            objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(high)
        elif method == "uniform_no_mi_prior_mmr":
            uniform = require("uniform_distances", uniform_distances)
            student = require("student_no_mi_scores", student_no_mi_scores)
            objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(uniform)
        else:
            null_high = require("null_mi_distances", null_mi_distances)
            student = require("student_null_mi_scores", student_null_mi_scores)
            objective = -_row_zscore(student) + float(mi_prior_weight) * _row_zscore(null_high)
        return _mmr_from_objective(objective, candidate_tokens, top_k, mmr_lambda, method)
    else:
        raise ValueError(f"unknown candidate-selection method: {method}")

    boundary_methods = {
        "high_mi_boundary", "low_mi_boundary",
        "random_patch_boundary", "all_patch_boundary",
    }
    insert_methods = {
        "mi_insert", "low_mi_insert", "random_insert", "all_patch_insert",
        "residual_insert",
    }
    if method in insert_methods:
        insert_count = int(mi_insert_count)
        if insert_count >= int(top_k):
            raise ValueError("mi_insert_count must be smaller than top_k")
        if insert_count > official.shape[1] - int(top_k):
            raise ValueError("mi_insert_count exceeds candidates outside official Top-k")
        official_prefix = np.broadcast_to(
            np.arange(int(top_k - insert_count), dtype=np.int64),
            (len(official), int(top_k - insert_count)),
        )
        extras = np.argsort(
            objective[:, int(top_k):], axis=1, kind="stable"
        )[:, :insert_count] + int(top_k)
        chosen = np.sort(
            np.concatenate([official_prefix, extras.astype(np.int64)], axis=1),
            axis=1,
        )
        scores = np.take_along_axis(objective, chosen, axis=1).astype(np.float32)
        validate_candidate_pool(official, top_k=top_k, selected_indices=chosen)
        return CandidateSelection(
            indices=chosen.astype(np.int64), scores=scores, method=method
        )
    if method in boundary_methods:
        if candidate_limit is None:
            raise ValueError(f"candidate_limit is required for method={method}")
        objective = objective.copy()
        objective[:, int(candidate_limit):] = np.inf
    chosen = np.argsort(objective, axis=1, kind="stable")[:, :top_k]
    if method in {"mi_select", "mi_gate", "mi_anchor_gate",
                  "low_mi_anchor_gate", "random_patch_gate", "all_patch_gate"}:
        # Selection uses MI, while the downstream FrozenARM receives the
        # selected candidates in the canonical official-rank order.  The
        # retrieval MoE is permutation invariant here; keeping a canonical
        # order makes artifacts deterministic and keeps membership attribution
        # explicit.
        chosen = np.sort(chosen, axis=1)
    if method in {"mi_gate", "mi_anchor_gate", "low_mi_anchor_gate",
                  "random_patch_gate", "all_patch_gate"}:
        # Conservative history-only gate: adopt the MI membership only when it
        # makes a clear replacement of the official Top-k set.  Otherwise the
        # official selection is retained exactly, so MI cannot perturb nearly
        # identical candidate sets through a noisy small change.
        overlap = np.mean(chosen < int(top_k), axis=1)
        official_rows = np.broadcast_to(
            np.arange(int(top_k), dtype=np.int64), chosen.shape
        )
        use_mi = overlap <= float(mi_gate_overlap_max)
        chosen = np.where(use_mi[:, None], chosen, official_rows)
    elif method == "official_permuted":
        rng = np.random.default_rng(int(seed))
        canonical = np.arange(int(top_k), dtype=np.int64)
        for row in range(len(chosen)):
            permutation = rng.permutation(int(top_k))
            if int(top_k) > 1 and np.array_equal(permutation, canonical):
                permutation = np.roll(permutation, 1)
            chosen[row] = chosen[row, permutation]
    if method in {
        "official_mi_bias", "official_random_bias",
        "official_uniform_bias", "official_low_mi_bias",
    }:
        assert bias_score_source is not None
        score_source = bias_score_source
    else:
        score_source = objective
    scores = np.take_along_axis(score_source, chosen, axis=1).astype(np.float32)
    validate_candidate_pool(official, top_k=top_k, selected_indices=chosen)
    return CandidateSelection(indices=chosen.astype(np.int64), scores=scores, method=method)


@lru_cache(maxsize=None)
def _exact_rank_layers(width: int, blocks: int) -> tuple[np.ndarray, ...]:
    """Cache complete rank assignments grouped by their total rank cost."""
    width = int(width)
    blocks = int(blocks)
    if width < 1 or blocks < 1:
        raise ValueError("rank-layer dimensions must be positive")
    grouped = [[] for _ in range(blocks * (width - 1) + 1)]
    for ranks in product(range(width), repeat=blocks):
        grouped[sum(ranks)].append(ranks)
    return tuple(
        np.asarray(layer, dtype=np.int16).reshape(-1, blocks)
        for layer in grouped
    )


@dataclass(frozen=True)
class HorizonBalancedSelection:
    """Top-k history candidates with one distinct horizon representative per block."""

    indices: np.ndarray
    scores: np.ndarray
    assigned_indices: np.ndarray
    assignment_costs: np.ndarray
    method: str


def select_horizon_balanced_candidates(
    official_distances: np.ndarray,
    horizon_distances: np.ndarray,
    *,
    top_k: int,
    method: str = "horizon_mi",
) -> HorizonBalancedSelection:
    """Preserve an official prefix and assign distinct candidates per horizon block.

    The assignment objective is the sum of per-block candidate ranks within the
    remaining official pool. Complete equal-cost rank layers are enumerated so
    the result is exact; ties use the lexicographically smallest tuple of
    original candidate ranks. Candidate futures never enter this boundary.
    """
    official = _finite("official_distances", official_distances, 2)
    horizon = _finite("horizon_distances", horizon_distances, 3)
    top_k = int(top_k)
    if horizon.shape[0] != official.shape[0] or horizon.shape[2] != official.shape[1]:
        raise ValueError("horizon distances must align with official queries and candidates")
    blocks = horizon.shape[1]
    pool = official.shape[1]
    if not 0 < blocks < top_k:
        raise ValueError("future block count must be positive and smaller than top_k")
    if not top_k <= pool or pool - (top_k - blocks) < blocks:
        raise ValueError("candidate pool cannot satisfy the horizon assignment budget")

    prefix_stop = top_k - blocks
    available = np.arange(prefix_stop, pool, dtype=np.int64)
    selected = np.empty((len(official), top_k), dtype=np.int64)
    assigned = np.empty((len(official), blocks), dtype=np.int64)
    assignment_costs = np.empty((len(official), blocks), dtype=np.float32)
    rank_layers = _exact_rank_layers(len(available), blocks)
    block_indices = np.arange(blocks, dtype=np.int64)[None, :]
    lex_base = pool + 1
    lex_factors = lex_base ** np.arange(blocks - 1, -1, -1, dtype=np.int64)

    for row in range(len(official)):
        # Index the query first so NumPy keeps the block axis leading; mixed
        # advanced indexing (``horizon[row, :, available]``) would transpose
        # this logical [blocks, candidates] layout to [candidates, blocks].
        raw = horizon[row][:, available]
        order = np.argsort(raw, axis=1, kind="stable")
        columns = None
        # The first layer containing a distinct assignment is globally optimal
        # under the sum-of-ranks objective.
        for rank_layer in rank_layers:
            proposed = order[block_indices, rank_layer]
            distinct = np.asarray(
                [len(set(candidate_tuple.tolist())) == blocks for candidate_tuple in proposed]
            )
            feasible = proposed[distinct]
            if len(feasible):
                original_ranks = available[feasible]
                lex_codes = original_ranks @ lex_factors
                columns = feasible[int(np.argmin(lex_codes))]
                break
        if columns is None:
            raise RuntimeError("exact assignment enumeration found no feasible tuple")
        assigned[row] = available[columns]
        assignment_costs[row] = raw[np.arange(blocks), columns]
        selected[row] = np.sort(
            np.concatenate([np.arange(prefix_stop, dtype=np.int64), assigned[row]])
        )

    scores = np.take_along_axis(official, selected, axis=1).astype(np.float32)
    return HorizonBalancedSelection(
        indices=selected,
        scores=scores,
        assigned_indices=assigned,
        assignment_costs=assignment_costs,
        method=str(method),
    )


def assert_selection_subset(selected_indices: np.ndarray, candidate_count: int) -> None:
    """Raise if a sidecar selection is not a unique subset of top-k candidates."""
    selected = np.asarray(selected_indices)
    if selected.ndim != 2 or selected.shape[1] < 1:
        raise ValueError(f"selection must be [query,k], got {selected.shape}")
    if selected.min(initial=0) < 0 or selected.max(initial=-1) >= int(candidate_count):
        raise ValueError("selection contains an index outside the candidate pool")
    if any(len(set(row.tolist())) != selected.shape[1] for row in selected):
        raise ValueError("selection contains duplicate positions")


if __name__ == "__main__":
    # ponytail: one runnable self-check is enough for this pure numerical module.
    rng = np.random.default_rng(2021)
    q = rng.normal(size=(3, 4, 5)).astype(np.float32)
    c = rng.normal(size=(3, 6, 4, 5)).astype(np.float32)
    qm = rng.normal(size=(3, 4)).astype(np.float32)
    cm = rng.normal(size=(3, 6, 4)).astype(np.float32)
    d = weighted_hidden_distances(q, c, qm, cm)
    result = select_candidates(np.arange(18, dtype=np.float32).reshape(3, 6), top_k=3,
                               method="high_mi", high_mi_distances=d)
    assert result.indices.shape == (3, 3)
    assert_selection_subset(result.indices, 6)
    print("mi_candidate_selector self-check passed")
