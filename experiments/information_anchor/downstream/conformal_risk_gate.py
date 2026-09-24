"""Split-conformal routing between a treatment and a fallback forecaster.

The treatment is selected only when a discovery-trained model predicts a
positive treatment gain with a conformal lower confidence bound.  The module
accepts history-derived query features at inference; calibration targets are
used only during discovery fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _finite(name: str, value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    values = np.sort(_finite("calibration_residuals", values, 1))
    if not len(values):
        raise ValueError("at least one calibration residual is required")
    index = min(len(values) - 1, max(0, int(np.ceil(float(probability) * len(values))) - 1))
    return float(values[index])


@dataclass
class ConformalRiskGate:
    """History-only treatment router with split-conformal residual radius."""

    scaler: StandardScaler
    model: Ridge
    radius: float
    alpha: float
    feature_names: tuple[str, ...]
    fit_count: int
    calibration_count: int
    fit_rmse: float
    calibration_rmse: float

    def predict_gain(self, query_features: np.ndarray) -> np.ndarray:
        values = _finite("query_features", query_features, 2)
        if values.shape[1] != len(self.feature_names):
            raise ValueError("query feature width does not match fitted gate")
        return self.model.predict(self.scaler.transform(values)).astype(np.float32)

    def route(self, query_features: np.ndarray) -> "ConformalRoute":
        predicted = self.predict_gain(query_features)
        lower = predicted - np.float32(self.radius)
        return ConformalRoute(
            use_treatment=(lower > 0.0),
            predicted_gain=predicted,
            lower_bound=lower,
        )


@dataclass(frozen=True)
class ConformalRoute:
    use_treatment: np.ndarray
    predicted_gain: np.ndarray
    lower_bound: np.ndarray


def fit_conformal_risk_gate(
    query_features: np.ndarray,
    treatment_gain: np.ndarray,
    *,
    alpha: float = 0.1,
    fit_fraction: float = 0.6,
    ridge_alpha: float = 10.0,
    feature_names: tuple[str, ...] | None = None,
) -> ConformalRiskGate:
    """Fit a chronological split-conformal gain predictor.

    ``treatment_gain > 0`` means the treatment is better than fallback.  The
    first ``fit_fraction`` discovery rows fit the ridge model; the remaining
    rows calibrate absolute residuals.  No test/future values enter the route.
    """
    features = _finite("query_features", query_features, 2)
    gain = _finite("treatment_gain", treatment_gain, 1)
    if len(features) != len(gain) or len(features) < 8:
        raise ValueError("query_features and treatment_gain need >=8 aligned rows")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not 0.5 <= float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must lie in [0.5,1)")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    names = feature_names or tuple(f"feature_{i}" for i in range(features.shape[1]))
    if len(names) != features.shape[1]:
        raise ValueError("feature_names width does not match query_features")
    fit_count = max(4, min(len(features) - 1, int(len(features) * float(fit_fraction))))
    scaler = StandardScaler().fit(features[:fit_count])
    model = Ridge(alpha=float(ridge_alpha)).fit(
        scaler.transform(features[:fit_count]), gain[:fit_count]
    )
    fit_pred = model.predict(scaler.transform(features[:fit_count]))
    calibration_features = features[fit_count:]
    calibration_gain = gain[fit_count:]
    calibration_pred = model.predict(scaler.transform(calibration_features))
    residuals = np.abs(calibration_gain - calibration_pred).astype(np.float32)
    # Finite-sample split-conformal correction: ceil((n+1)(1-alpha))/n.
    probability = min(1.0, float(np.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals)))
    radius = _higher_quantile(residuals, probability)
    return ConformalRiskGate(
        scaler=scaler,
        model=model,
        radius=float(radius),
        alpha=float(alpha),
        feature_names=tuple(names),
        fit_count=int(fit_count),
        calibration_count=int(len(residuals)),
        fit_rmse=float(np.sqrt(np.mean(np.square(gain[:fit_count] - fit_pred)))),
        calibration_rmse=float(np.sqrt(np.mean(np.square(calibration_gain - calibration_pred)))),
    )


__all__ = ["ConformalRiskGate", "ConformalRoute", "fit_conformal_risk_gate"]

