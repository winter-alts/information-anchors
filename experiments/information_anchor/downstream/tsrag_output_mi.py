"""Output-level MI priors for the official TS-RAG ARM gate.

The helper in this module deliberately operates only on the already selected
candidate pool.  It does not select candidates, reorder them, or inspect
candidate futures.  The returned prior can be passed to TS-RAG's existing
``retrieval_bias`` argument as a log prior relative to a uniform candidate
prior.
"""

from __future__ import annotations

import torch


def _validate_distances(
    official_distances: torch.Tensor,
    mi_distances: torch.Tensor,
    gamma: float,
) -> None:
    if official_distances.ndim != 2 or mi_distances.ndim != 2:
        raise ValueError("output MI distances must be rank-2 [batch,candidates]")
    if official_distances.shape != mi_distances.shape:
        raise ValueError(
            "output MI distance shape mismatch: "
            f"official={tuple(official_distances.shape)}, "
            f"mi={tuple(mi_distances.shape)}"
        )
    if official_distances.shape[1] < 1:
        raise ValueError("output MI distances must contain at least one candidate")
    if not torch.isfinite(official_distances).all() or not torch.isfinite(mi_distances).all():
        raise ValueError("output MI distances must be finite")
    if torch.any(official_distances < -1e-6) or torch.any(mi_distances < -1e-6):
        raise ValueError("output MI distances must be non-negative")
    if not torch.isfinite(torch.as_tensor(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and lie in [0,1]")


def output_mi_weights(
    official_distances: torch.Tensor,
    mi_distances: torch.Tensor,
    *,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interpolated output-level weights and their calibration.

    ``official_distances`` and ``mi_distances`` must refer to the same
    already-selected TS-RAG candidates.  MI distances are query-wise median
    calibrated to the official scale.  Both distributions use the official
    query temperature ``median(official_distances) / 5``:

    ``w = (1-gamma) softmax(-d_off/tau) + gamma softmax(-d_mi_cal/tau)``.

    The return values are ``(w, d_mi_cal, tau)``.  The function is intentionally
    pure so validation and test runs can use exactly the same implementation.
    """
    _validate_distances(official_distances, mi_distances, gamma)
    if not torch.isfinite(torch.as_tensor(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    # A few legacy score shards contain tiny negative round-off residues from
    # weighted squared distances.  They are mathematically zero and are
    # clamped here; materially negative inputs remain rejected above.
    official_distances = official_distances.clamp_min(0.0)
    mi_distances = mi_distances.clamp_min(0.0)
    official_median = torch.quantile(official_distances, 0.5, dim=1, keepdim=True)
    mi_median = torch.quantile(mi_distances, 0.5, dim=1, keepdim=True)
    official_scale = official_median.clamp_min(float(eps))
    mi_scale = mi_median.clamp_min(float(eps))
    calibrated_mi = mi_distances * (official_scale / mi_scale)
    tau = (official_scale / 5.0).clamp_min(float(eps))

    official_weights = torch.softmax(-official_distances / tau, dim=1)
    mi_weights = torch.softmax(-calibrated_mi / tau, dim=1)
    mixed = (1.0 - float(gamma)) * official_weights + float(gamma) * mi_weights
    return mixed, calibrated_mi, tau


def output_mi_prior_bias(
    official_distances: torch.Tensor,
    mi_distances: torch.Tensor,
    *,
    gamma: float,
    strength: float = 1.0,
    center: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert output-level weights to an ARM log prior.

    The bias is ``log(K * w)`` rather than ``log(w)``.  Consequently a
    uniform prior maps to zero and leaves the existing TS-RAG gate unchanged;
    only relative candidate preference is injected into the retrieved expert
    branch.  The query expert continues to receive zero bias in ChronosBolt.
    """
    if not torch.isfinite(torch.as_tensor(strength)) or float(strength) < 0.0:
        raise ValueError("strength must be finite and non-negative")
    weights, _, _ = output_mi_weights(
        official_distances, mi_distances, gamma=gamma, eps=eps
    )
    candidates = weights.shape[1]
    bias = torch.log(weights.clamp_min(float(eps)) * float(candidates))
    if center:
        # Keep the retrieved-vs-query gate mass unchanged on average; only
        # the relative preference among retrieved experts is altered.
        bias = bias - bias.mean(dim=1, keepdim=True)
    return float(strength) * bias


__all__ = ["output_mi_prior_bias", "output_mi_weights"]
