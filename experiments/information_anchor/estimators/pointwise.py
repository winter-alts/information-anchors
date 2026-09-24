"""Pointwise mutual-information scores from a cross-fitted density-ratio critic.

An MI scalar needs a sample distribution.  This module therefore does not claim
to estimate ``I(X;Y)`` from one origin.  It fits a classifier on discovery
origins to distinguish joint pairs ``(X,Y)`` from product-of-marginals pairs
``(X,Y')`` and returns the held-out classifier logit for each positive pair.
With balanced classes, that logit estimates the pointwise log density ratio
``log p(x,y) / (p(x)p(y))`` up to finite-sample/model error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PointwiseMIResult:
    """Cross-fitted per-origin/per-patch pointwise information scores."""

    scores: np.ndarray
    raw_scores: np.ndarray
    fold_accuracy: np.ndarray
    train_accuracy: np.ndarray


@dataclass
class PointwiseMICritic:
    """A fitted density-ratio critic for one patch position."""

    scaler: StandardScaler
    classifier: LogisticRegression
    x_dim: int
    y_dim: int

    def score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return pointwise log density-ratio scores for positive pairs."""
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        # A one-dimensional input to this public method is one feature vector,
        # unlike the fitting API where a 1-D array means scalar observations.
        if x_array.ndim == 1:
            x_array = x_array[None, :]
        if y_array.ndim == 1:
            y_array = y_array[None, :]
        features = _pair_features(x_array, y_array, x_dim=self.x_dim, y_dim=self.y_dim)
        return self.classifier.decision_function(self.scaler.transform(features)).astype(
            np.float32,
            copy=False,
        )


@dataclass
class ConditionalPointwiseMICritic:
    """A fitted critic for ``I(X;Y|R)`` using local product pairs."""

    scaler: StandardScaler
    classifier: LogisticRegression
    x_dim: int
    y_dim: int
    r_dim: int

    def score(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        r_array = np.asarray(r)
        if x_array.ndim == 1:
            x_array = x_array[None, :]
        if y_array.ndim == 1:
            y_array = y_array[None, :]
        if r_array.ndim == 1:
            r_array = r_array[None, :]
        features = _conditional_pair_features(
            x_array,
            y_array,
            r_array,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            r_dim=self.r_dim,
        )
        return self.classifier.decision_function(self.scaler.transform(features)).astype(
            np.float32,
            copy=False,
        )


def _as_2d(values: np.ndarray, name: str, *, min_samples: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < min_samples:
        raise ValueError(
            f"{name} must have shape [samples, features] with >={min_samples} samples, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _pair_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_dim: int | None = None,
    y_dim: int | None = None,
) -> np.ndarray:
    """Build a compact joint/product discriminator feature map."""
    x_array = _as_2d(x, "x")
    y_array = _as_2d(y, "y")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"x and y must share samples, got {x_array.shape} and {y_array.shape}")
    if x_dim is not None and x_array.shape[1] != x_dim:
        raise ValueError(f"Expected x dimension {x_dim}, got {x_array.shape[1]}")
    if y_dim is not None and y_array.shape[1] != y_dim:
        raise ValueError(f"Expected y dimension {y_dim}, got {y_array.shape[1]}")
    # The outer product supplies cross terms without requiring a neural critic.
    interaction = np.einsum("ni,nj->nij", x_array, y_array).reshape(len(x_array), -1)
    return np.concatenate([x_array, y_array, interaction], axis=1)


def _conditional_pair_features(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    *,
    x_dim: int | None = None,
    y_dim: int | None = None,
    r_dim: int | None = None,
) -> np.ndarray:
    """Build a joint-vs-local-product feature map with an observed condition."""
    x_array = _as_2d(x, "x")
    y_array = _as_2d(y, "y")
    r_array = _as_2d(r, "r")
    if not (x_array.shape[0] == y_array.shape[0] == r_array.shape[0]):
        raise ValueError(
            f"x, y, and r must share samples, got {x_array.shape}, {y_array.shape}, {r_array.shape}"
        )
    if x_dim is not None and x_array.shape[1] != x_dim:
        raise ValueError(f"Expected x dimension {x_dim}, got {x_array.shape[1]}")
    if y_dim is not None and y_array.shape[1] != y_dim:
        raise ValueError(f"Expected y dimension {y_dim}, got {y_array.shape[1]}")
    if r_dim is not None and r_array.shape[1] != r_dim:
        raise ValueError(f"Expected r dimension {r_dim}, got {r_array.shape[1]}")
    xy = np.einsum("ni,nj->nij", x_array, y_array).reshape(len(x_array), -1)
    xr = np.einsum("ni,nj->nij", x_array, r_array).reshape(len(x_array), -1)
    yr = np.einsum("ni,nj->nij", y_array, r_array).reshape(len(x_array), -1)
    return np.concatenate([x_array, y_array, r_array, xy, xr, yr], axis=1)


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation with no fixed points when size permits."""
    if size < 2:
        raise ValueError("At least two training samples are required for product pairs.")
    permutation = rng.permutation(size)
    if np.any(permutation == np.arange(size)):
        permutation = np.roll(np.arange(size), 1)
    return permutation


def _local_condition_matches(
    reference: np.ndarray,
    candidates: np.ndarray,
    rng: np.random.Generator,
    *,
    exclude_self: bool,
    neighbors: int = 8,
) -> np.ndarray:
    """Match each condition to a nearby candidate for a local product null."""
    reference_array = _as_2d(reference, "reference")
    candidate_array = _as_2d(candidates, "candidates")
    scale = np.maximum(candidate_array.std(axis=0, keepdims=True), 1e-6)
    reference_scaled = reference_array / scale
    candidate_scaled = candidate_array / scale
    distances = np.sum(
        np.square(reference_scaled[:, None, :] - candidate_scaled[None, :, :]), axis=2
    )
    if exclude_self and reference_array.shape == candidate_array.shape:
        distances[np.arange(len(reference_array)), np.arange(len(candidate_array))] = np.inf
    available = candidate_array.shape[0] - int(exclude_self)
    if available < 1:
        raise ValueError("No candidate remains for conditional local matching.")
    width = min(int(neighbors), available)
    nearest = np.argpartition(distances, kth=width - 1, axis=1)[:, :width]
    choices = rng.integers(0, width, size=len(reference_array))
    return nearest[np.arange(len(reference_array)), choices]


def _compress_condition(values: np.ndarray, max_dim: int = 16) -> np.ndarray:
    """Keep conditional MI tractable for long recent-history vectors.

    The conditional critic uses explicit ``X*R`` and ``Y*R`` interactions.  A
    96-step raw condition therefore creates thousands of logistic-regression
    features and makes a five-fold, patch-wise fit needlessly expensive.  We
    deterministically average contiguous bins down to at most ``max_dim``;
    this is a fixed history-only transform (no target-dependent fitting) and
    leaves short conditions untouched.
    """
    array = _as_2d(values, "condition")
    if array.shape[1] <= int(max_dim):
        return array
    edges = np.linspace(0, array.shape[1], int(max_dim) + 1, dtype=np.int64)
    pooled = [array[:, lo:hi].mean(axis=1) for lo, hi in zip(edges[:-1], edges[1:])
              if hi > lo]
    return np.stack(pooled, axis=1)


def fit_pointwise_mi_critic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 2021,
    regularization: float = 1.0,
    max_iter: int = 2000,
) -> PointwiseMICritic:
    """Fit a balanced joint-vs-product density-ratio critic for one patch."""
    x_array = _as_2d(x, "x", min_samples=4)
    y_array = _as_2d(y, "y", min_samples=4)
    rng = np.random.default_rng(seed)
    negative_y = y_array[_derangement(len(y_array), rng)]
    positive = _pair_features(x_array, y_array)
    negative = _pair_features(x_array, negative_y)
    features = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
    )
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(regularization),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(scaler.transform(features), labels)
    return PointwiseMICritic(
        scaler=scaler,
        classifier=classifier,
        x_dim=x_array.shape[1],
        y_dim=y_array.shape[1],
    )


def cross_fitted_pointwise_mi(
    hidden: np.ndarray,
    target: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 2021,
    regularization: float = 1.0,
    max_iter: int = 2000,
) -> PointwiseMIResult:
    """Return held-out pointwise scores with shape ``[sample, patch]``.

    ``hidden`` is ``[sample, patch, hidden_dim]`` and ``target`` is
    ``[sample, target_dim]``.  A separate critic is fitted for each patch,
    because the goal is to localize information back to the history timeline.
    """
    hidden_array = np.asarray(hidden, dtype=np.float64)
    target_array = _as_2d(target, "target")
    if hidden_array.ndim != 3:
        raise ValueError(f"hidden must have shape [samples, patches, features], got {hidden_array.shape}")
    if hidden_array.shape[0] != target_array.shape[0]:
        raise ValueError(
            f"hidden and target must share samples, got {hidden_array.shape} and {target_array.shape}"
        )
    if not np.isfinite(hidden_array).all():
        raise ValueError("hidden contains non-finite values.")
    n_samples, n_patches, _ = hidden_array.shape
    if n_samples < 8:
        raise ValueError("At least eight samples are required for cross-fitting.")
    # Keep at least four training origins in every fold for a stable critic.
    folds = max(2, min(int(folds), n_samples - 4))
    scores = np.empty((n_samples, n_patches), dtype=np.float32)
    raw_scores = np.empty_like(scores)
    fold_accuracy = np.empty(folds, dtype=np.float32)
    train_accuracy = np.empty(folds, dtype=np.float32)
    rng = np.random.default_rng(seed)
    fold_indices = np.array_split(np.arange(n_samples), folds)

    for fold_index, test_indices in enumerate(fold_indices):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        patch_accuracies = []
        patch_train_accuracies = []
        for patch_index in range(n_patches):
            critic = fit_pointwise_mi_critic(
                hidden_array[train_indices, patch_index],
                target_array[train_indices],
                seed=int(rng.integers(0, 2**31 - 1)),
                regularization=regularization,
                max_iter=max_iter,
            )
            positive_scores = critic.score(
                hidden_array[test_indices, patch_index], target_array[test_indices]
            )
            raw_scores[test_indices, patch_index] = positive_scores
            train_negative_target = target_array[train_indices][
                _derangement(len(train_indices), np.random.default_rng(critic.classifier.random_state))
            ]
            train_features = _pair_features(
                np.concatenate(
                    [hidden_array[train_indices, patch_index], hidden_array[train_indices, patch_index]],
                    axis=0,
                ),
                np.concatenate([target_array[train_indices], train_negative_target], axis=0),
            )
            labels = np.concatenate(
                [np.ones(len(train_indices), dtype=np.int64), np.zeros(len(train_indices), dtype=np.int64)]
            )
            patch_train_accuracies.append(
                float(np.mean(critic.classifier.predict(critic.scaler.transform(train_features)) == labels))
            )

            test_negative_target = target_array[test_indices][_derangement(len(test_indices), rng)]
            test_features = _pair_features(
                np.concatenate(
                    [hidden_array[test_indices, patch_index], hidden_array[test_indices, patch_index]],
                    axis=0,
                ),
                np.concatenate([target_array[test_indices], test_negative_target], axis=0),
            )
            test_labels = np.concatenate(
                [np.ones(len(test_indices), dtype=np.int64), np.zeros(len(test_indices), dtype=np.int64)]
            )
            patch_accuracies.append(
                float(np.mean(critic.classifier.predict(critic.scaler.transform(test_features)) == test_labels))
            )
            # Calibrate patch-specific logits against held-out product pairs so
            # different patch critics are comparable for local peak detection.
            null_logits = []
            for _ in range(8):
                null_target = target_array[test_indices][_derangement(len(test_indices), rng)]
                null_logits.append(
                    critic.score(hidden_array[test_indices, patch_index], null_target)
                )
            null_flat = np.concatenate(null_logits)
            null_center = float(null_flat.mean())
            null_scale = max(float(null_flat.std(ddof=1)), 1e-3)
            scores[test_indices, patch_index] = (
                (positive_scores - null_center) / null_scale
            ).astype(np.float32)
        fold_accuracy[fold_index] = float(np.mean(patch_accuracies))
        train_accuracy[fold_index] = float(np.mean(patch_train_accuracies))

    return PointwiseMIResult(
        scores=scores,
        raw_scores=raw_scores,
        fold_accuracy=fold_accuracy,
        train_accuracy=train_accuracy,
    )


def fit_conditional_pointwise_mi_critic(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    negative_y: np.ndarray,
    *,
    seed: int = 2021,
    regularization: float = 1.0,
    max_iter: int = 2000,
) -> ConditionalPointwiseMICritic:
    """Fit a critic from joint pairs and locally matched conditional products."""
    x_array = _as_2d(x, "x", min_samples=4)
    y_array = _as_2d(y, "y", min_samples=4)
    r_array = _as_2d(r, "r", min_samples=4)
    negative_array = _as_2d(negative_y, "negative_y", min_samples=4)
    if not (len(x_array) == len(y_array) == len(r_array) == len(negative_array)):
        raise ValueError("Conditional critic inputs must share samples.")
    positive = _conditional_pair_features(x_array, y_array, r_array)
    negative = _conditional_pair_features(x_array, negative_array, r_array)
    features = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
    )
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(regularization),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(scaler.transform(features), labels)
    return ConditionalPointwiseMICritic(
        scaler=scaler,
        classifier=classifier,
        x_dim=x_array.shape[1],
        y_dim=y_array.shape[1],
        r_dim=r_array.shape[1],
    )


def cross_fitted_conditional_pointwise_mi(
    hidden: np.ndarray,
    target: np.ndarray,
    condition: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 2021,
    regularization: float = 1.0,
    max_iter: int = 2000,
) -> PointwiseMIResult:
    """Return held-out local-conditional pointwise scores ``I(H;Y|R)``.

    The conditional product null pairs each origin with a future target from a
    nearby origin in ``R`` space.  This is a local conditional critic, not an
    exact nonlinear conditional-MI estimator; its null matching is explicit in
    the result protocol and should be reported as such.
    """
    hidden_array = np.asarray(hidden, dtype=np.float64)
    target_array = _as_2d(target, "target")
    # Keep the conditional interaction map compact.  In the registered
    # 512-to-64 protocol K_recent=96, so this reduces 96 raw steps to 16
    # deterministic mean-pooled bins without consulting any future target.
    condition_array = _compress_condition(condition, max_dim=16)
    if hidden_array.ndim != 3:
        raise ValueError(f"hidden must have shape [samples, patches, features], got {hidden_array.shape}")
    if not (hidden_array.shape[0] == len(target_array) == len(condition_array)):
        raise ValueError("hidden, target, and condition must share samples.")
    if not np.isfinite(hidden_array).all():
        raise ValueError("hidden contains non-finite values.")
    n_samples, n_patches, _ = hidden_array.shape
    if n_samples < 8:
        raise ValueError("At least eight samples are required for conditional cross-fitting.")
    folds = max(2, min(int(folds), n_samples - 4))
    scores = np.empty((n_samples, n_patches), dtype=np.float32)
    raw_scores = np.empty_like(scores)
    fold_accuracy = np.empty(folds, dtype=np.float32)
    train_accuracy = np.empty(folds, dtype=np.float32)
    rng = np.random.default_rng(seed)
    fold_indices = np.array_split(np.arange(n_samples), folds)

    for fold_index, test_indices in enumerate(fold_indices):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_negative_indices = _local_condition_matches(
            condition_array[train_indices],
            condition_array[train_indices],
            rng,
            exclude_self=True,
        )
        test_negative_indices = _local_condition_matches(
            condition_array[test_indices],
            condition_array[train_indices],
            rng,
            exclude_self=False,
        )
        patch_accuracies = []
        patch_train_accuracies = []
        for patch_index in range(n_patches):
            critic = fit_conditional_pointwise_mi_critic(
                hidden_array[train_indices, patch_index],
                target_array[train_indices],
                condition_array[train_indices],
                target_array[train_indices][train_negative_indices],
                seed=int(rng.integers(0, 2**31 - 1)),
                regularization=regularization,
                max_iter=max_iter,
            )
            positive_scores = critic.score(
                hidden_array[test_indices, patch_index],
                target_array[test_indices],
                condition_array[test_indices],
            )
            raw_scores[test_indices, patch_index] = positive_scores

            train_features = _conditional_pair_features(
                np.concatenate(
                    [hidden_array[train_indices, patch_index], hidden_array[train_indices, patch_index]],
                    axis=0,
                ),
                np.concatenate(
                    [target_array[train_indices], target_array[train_indices][train_negative_indices]],
                    axis=0,
                ),
                np.concatenate([condition_array[train_indices], condition_array[train_indices]], axis=0),
            )
            train_labels = np.concatenate(
                [np.ones(len(train_indices), dtype=np.int64), np.zeros(len(train_indices), dtype=np.int64)]
            )
            patch_train_accuracies.append(
                float(np.mean(critic.classifier.predict(critic.scaler.transform(train_features)) == train_labels))
            )

            test_negative_target = target_array[train_indices][test_negative_indices]
            test_features = _conditional_pair_features(
                np.concatenate(
                    [hidden_array[test_indices, patch_index], hidden_array[test_indices, patch_index]],
                    axis=0,
                ),
                np.concatenate([target_array[test_indices], test_negative_target], axis=0),
                np.concatenate([condition_array[test_indices], condition_array[test_indices]], axis=0),
            )
            test_labels = np.concatenate(
                [np.ones(len(test_indices), dtype=np.int64), np.zeros(len(test_indices), dtype=np.int64)]
            )
            patch_accuracies.append(
                float(np.mean(critic.classifier.predict(critic.scaler.transform(test_features)) == test_labels))
            )

            null_logits = []
            for _ in range(8):
                null_indices = _local_condition_matches(
                    condition_array[test_indices],
                    condition_array[train_indices],
                    rng,
                    exclude_self=False,
                )
                null_logits.append(
                    critic.score(
                        hidden_array[test_indices, patch_index],
                        target_array[train_indices][null_indices],
                        condition_array[test_indices],
                    )
                )
            null_flat = np.concatenate(null_logits)
            null_center = float(null_flat.mean())
            null_scale = max(float(null_flat.std(ddof=1)), 1e-3)
            scores[test_indices, patch_index] = (
                (positive_scores - null_center) / null_scale
            ).astype(np.float32)
        fold_accuracy[fold_index] = float(np.mean(patch_accuracies))
        train_accuracy[fold_index] = float(np.mean(patch_train_accuracies))

    return PointwiseMIResult(
        scores=scores,
        raw_scores=raw_scores,
        fold_accuracy=fold_accuracy,
        train_accuracy=train_accuracy,
    )
